"""Phase 1.5 step 1 sanity check (case-B custom tire-force injection).

Runs one of three scenarios and verifies the dynamics module:
  - straight:  forward accelerate / coast. Checks Fz balance, z drift.
  - brake:     accelerate to target speed, then brake hard. Stopping distance
               vs theoretical v^2/(2 mu g).
  - circle:    constant steer, constant target speed. Steady-state radius and
               lateral accel; vs theoretical limit speed sqrt(mu g R).

Run from repository root (venv python):
    PY="/c/work/isaac/env_isaaclab/Scripts/python.exe"
    $PY scripts/sim/run_phase1_5.py --scenario straight --mu 0.9 --headless
    $PY scripts/sim/run_phase1_5.py --scenario brake    --mu 0.9 --headless
    $PY scripts/sim/run_phase1_5.py --scenario circle   --mu 0.9 --steer 0.1 --headless
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 1.5 step 1 sanity check")
parser.add_argument("--scenario", choices=["straight", "brake", "circle"], default="straight")
parser.add_argument("--mu", type=float, default=0.9, help="Uniform tire friction coefficient")
parser.add_argument("--duration", type=float, default=12.0, help="Sim seconds")
parser.add_argument("--target_speed", type=float, default=15.0, help="m/s, used by brake/circle")
parser.add_argument("--steer", type=float, default=0.1,
                    help="circle scenario: constant steering angle [rad]")
parser.add_argument("--video_dir", type=str,
                    default="C:/Users/user/vehicle_rl/videos")
parser.add_argument("--metrics_dir", type=str,
                    default="C:/Users/user/vehicle_rl/metrics")
parser.add_argument("--no_video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.no_video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports that require the sim app to be running ---
import csv
import json
import math

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.sensors import CameraCfg, Camera

from vehicle_rl.assets import (
    SEDAN_CFG, WHEELBASE, TRACK, COG_Z_DEFAULT, TOTAL_MASS,
)
from vehicle_rl.dynamics import (
    VehicleState,
    FirstOrderLagActuator,
    StaticNormalLoadModel,
    LinearFrictionCircleTire,
    AttitudeDamper,
    aggregate_tire_forces_to_base_link,
    quat_wxyz_to_rotmat,
    quat_wxyz_to_rpy,
)


# Phase 1.5 step 1 hyperparameters (PLAN.md)
TAU_STEER = 0.05    # s
TAU_DRIVE = 0.20    # s
TAU_BRAKE = 0.07    # s

C_ALPHA = 60_000.0  # N/rad per wheel (mid-size sedan rough estimate)
H_CG = COG_Z_DEFAULT
A_FRONT = WHEELBASE / 2.0    # symmetric (a = b = L/2)

Z_DRIFT_KP = 50_000.0   # N/m  -- well below static stiffness, just to suppress integration drift
Z_DRIFT_KD = 5_000.0    # N s/m

K_ROLL = 80_000.0
C_ROLL = 8_000.0
K_PITCH = 80_000.0
C_PITCH = 8_000.0

GRAVITY = 9.81


def design_scene(record_video):
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    sedan_cfg = SEDAN_CFG.copy()
    sedan_cfg.prim_path = "/World/Sedan"
    sedan = Articulation(cfg=sedan_cfg)

    if not record_video:
        return {"sedan": sedan, "cam": None}

    cam_cfg = CameraCfg(
        prim_path="/World/SideCam",
        update_period=1.0 / 30.0,
        height=540,
        width=960,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=50.0,
            horizontal_aperture=20.955, clipping_range=(0.1, 1000.0),
        ),
        # Pose is set after sim.reset() via cam.set_world_poses_from_view().
    )
    cam = Camera(cfg=cam_cfg)
    return {"sedan": sedan, "cam": cam}


def write_mp4(frames_rgb, out_path, fps=30):
    import av
    if not frames_rgb:
        return
    h, w = frames_rgb[0].shape[:2]
    out = av.open(out_path, "w")
    stream = out.add_stream("libx264", rate=int(fps))
    stream.width, stream.height = w, h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "veryfast"}
    for fr in frames_rgb:
        vf = av.VideoFrame.from_ndarray(fr, format="rgb24")
        for pkt in stream.encode(vf):
            out.mux(pkt)
    for pkt in stream.encode():
        out.mux(pkt)
    out.close()


def build_state(sedan: Articulation, wheel_ids, delta_actual, a_x_actual, a_y_estimate, device):
    rs = sedan.data.root_state_w   # (1, 13): pos[3], quat[4] wxyz, lin_vel_w[3], ang_vel_w[3]
    pos_world = rs[:, 0:3]
    quat = rs[:, 3:7]
    vel_world = rs[:, 7:10]
    angvel_world = rs[:, 10:13]
    R = quat_wxyz_to_rotmat(quat)               # (1, 3, 3) body->world
    Rt = R.transpose(-1, -2)                    # world->body
    vel_body = (Rt @ vel_world.unsqueeze(-1)).squeeze(-1)
    angvel_body = (Rt @ angvel_world.unsqueeze(-1)).squeeze(-1)
    rpy = quat_wxyz_to_rpy(quat)

    jvel = sedan.data.joint_vel              # (1, num_joints)
    omega_wheel = jvel[:, wheel_ids]         # (1, 4) order set by caller
    return VehicleState(
        pos_world=pos_world,
        quat_wxyz_world=quat,
        rot_body_to_world=R,
        vel_world=vel_world,
        vel_body=vel_body,
        angvel_body=angvel_body,
        rpy=rpy,
        delta_actual=delta_actual,
        omega_wheel=omega_wheel,
        a_x_actual=a_x_actual,
        a_y_estimate=a_y_estimate,
    )


def scenario_command(scenario, t, v_long, v_target, steer):
    """Returns (a_x_target, delta_target) as Python floats."""
    if scenario == "straight":
        a_x = 2.0 if t < 5.0 else 0.0
        d = 0.0
    elif scenario == "brake":
        if t < 5.0:
            a_x = 3.0
        elif t < 6.0:
            a_x = 0.0
        else:
            # Friction-limited brake. Release once stopped — real brakes don't
            # accelerate the vehicle backward; they only oppose forward motion.
            a_x = -10.0 if v_long > 0.1 else 0.0
        d = 0.0
    elif scenario == "circle":
        # Simple P controller for speed; ramp up over first 4 s.
        v_ref = min(v_target, v_target * t / 4.0) if t < 4.0 else v_target
        a_x = max(-2.0, min(3.0, 0.8 * (v_ref - v_long)))
        d = steer
    else:
        raise ValueError(scenario)
    return a_x, d


def main():
    device = args_cli.device
    sim_cfg = sim_utils.SimulationCfg(
        device=device,
        dt=1.0 / 200.0,
        gravity=(0.0, 0.0, -GRAVITY),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([7.0, -10.0, 3.5], [0.0, 0.0, 0.5])

    record_video = not args_cli.no_video
    entities = design_scene(record_video=record_video)
    sim.reset()

    sedan: Articulation = entities["sedan"]
    cam: Camera = entities["cam"]

    if cam is not None:
        eye = torch.tensor([[35.0, -60.0, 25.0]], device=device)
        target = torch.tensor([[35.0, 0.0, 1.0]], device=device)
        cam.set_world_poses_from_view(eye, target)

    print(f"[INFO] scenario={args_cli.scenario} mu={args_cli.mu} duration={args_cli.duration}", flush=True)
    print(f"[INFO] joint_names = {sedan.joint_names}", flush=True)
    print(f"[INFO] body_names  = {sedan.body_names}", flush=True)

    # Reset
    root_state = sedan.data.default_root_state.clone()
    sedan.write_root_pose_to_sim(root_state[:, :7])
    sedan.write_root_velocity_to_sim(root_state[:, 7:])
    jp = sedan.data.default_joint_pos.clone()
    jv = sedan.data.default_joint_vel.clone()
    sedan.write_joint_state_to_sim(jp, jv)
    sedan.reset()

    # Joint / body indices (PLAN.md wheel order: FL, FR, RL, RR).
    fl_steer_id = sedan.find_joints("front_left_steer_joint")[0][0]
    fr_steer_id = sedan.find_joints("front_right_steer_joint")[0][0]
    fl_wheel_id = sedan.find_joints("front_left_wheel_joint")[0][0]
    fr_wheel_id = sedan.find_joints("front_right_wheel_joint")[0][0]
    rl_wheel_id = sedan.find_joints("rear_left_wheel_joint")[0][0]
    rr_wheel_id = sedan.find_joints("rear_right_wheel_joint")[0][0]
    base_id = sedan.find_bodies("base_link")[0][0]
    steer_ids = [fl_steer_id, fr_steer_id]
    wheel_ids = [fl_wheel_id, fr_wheel_id, rl_wheel_id, rr_wheel_id]   # FL, FR, RL, RR
    print(f"[INFO] base_id={base_id} steer_ids={steer_ids} wheel_ids={wheel_ids}", flush=True)

    # Dynamics modules
    steer_act = FirstOrderLagActuator(num_envs=1, device=device, tau_pos=TAU_STEER)
    drive_act = FirstOrderLagActuator(num_envs=1, device=device,
                                      tau_pos=TAU_DRIVE, tau_neg=TAU_BRAKE)
    normal_load = StaticNormalLoadModel(
        mass=TOTAL_MASS, wheelbase=WHEELBASE, track=TRACK,
        h_cg=H_CG, a_front=A_FRONT, z_ref=COG_Z_DEFAULT,
        gravity=GRAVITY, z_drift_kp=Z_DRIFT_KP, z_drift_kd=Z_DRIFT_KD,
    )
    tire = LinearFrictionCircleTire(
        cornering_stiffness=C_ALPHA,
        wheelbase=WHEELBASE, track=TRACK, a_front=A_FRONT, h_cg=H_CG,
    )
    attitude = AttitudeDamper(
        k_roll=K_ROLL, c_roll=C_ROLL, k_pitch=K_PITCH, c_pitch=C_PITCH,
    )

    mu_tensor = torch.full((1, 4), float(args_cli.mu), device=device)

    sim_dt = sim.get_physics_dt()
    n_steps = int(args_cli.duration / sim_dt)
    cam_period = max(1, int(round(1.0 / 30.0 / sim_dt)))
    a_y_estimate = torch.zeros(1, device=device)

    # Logging
    rows = [["t", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg",
             "vx_body", "vy_body", "vz_world",
             "wx_body", "wy_body", "wz_body",
             "delta_target", "delta_actual", "a_x_target", "a_x_actual",
             "Fz_FL", "Fz_FR", "Fz_RL", "Fz_RR", "sum_Fz",
             "Fx_FL", "Fx_FR", "Fx_RL", "Fx_RR",
             "Fy_FL", "Fy_FR", "Fy_RL", "Fy_RR",
             "alpha_FL_deg", "alpha_FR_deg", "alpha_RL_deg", "alpha_RR_deg"]]
    frames = []

    max_roll = 0.0
    max_pitch = 0.0
    max_z_err = 0.0
    max_sum_fz_err = 0.0
    brake_stopped_at = None
    brake_start_x = None
    brake_start_v = None
    brake_stop_x = None

    mg = TOTAL_MASS * GRAVITY

    for step in range(n_steps):
        t = step * sim_dt

        # Read state for command computation
        rs = sedan.data.root_state_w
        quat = rs[:, 3:7]
        R = quat_wxyz_to_rotmat(quat)
        Rt = R.transpose(-1, -2)
        v_world = rs[:, 7:10]
        v_body = (Rt @ v_world.unsqueeze(-1)).squeeze(-1)
        v_long = float(v_body[0, 0].item())

        a_x_t, d_t = scenario_command(args_cli.scenario, t, v_long,
                                       args_cli.target_speed, args_cli.steer)
        a_x_target = torch.full((1,), a_x_t, device=device)
        delta_target = torch.full((1,), d_t, device=device)

        delta_actual = steer_act.step(delta_target, sim_dt)
        a_x_actual = drive_act.step(a_x_target, sim_dt)

        # Build full state
        state = build_state(sedan, wheel_ids, delta_actual, a_x_actual, a_y_estimate, device)

        # Fz
        Fz = normal_load.compute(state)   # (1, 4)

        # Longitudinal force command per wheel.
        # Drive (a_x_actual >= 0): RWD -- rear wheels split the force.
        # Brake (a_x_actual <  0): all four wheels split equally.
        Fx_total = TOTAL_MASS * a_x_actual   # (1,)
        zero = torch.zeros_like(Fx_total)
        if a_x_actual.item() >= 0.0:
            half = Fx_total * 0.5
            fx_command = torch.stack([zero, zero, half, half], dim=-1)
        else:
            quarter = Fx_total * 0.25
            fx_command = torch.stack([quarter, quarter, quarter, quarter], dim=-1)

        # Tire forces (with friction-circle clip)
        Fx_tire, Fy_tire = tire.compute(state, Fz, mu_tensor, fx_command)

        # Aggregate to base_link
        delta_per_wheel = tire.per_wheel_steer(delta_actual)
        F_body, tau_body = aggregate_tire_forces_to_base_link(
            Fx_tire, Fy_tire, Fz, delta_per_wheel, tire.tire_positions_body()
        )
        # Add virtual roll/pitch damper torque
        tau_body = tau_body + attitude.compute(state)

        # Apply external force/torque on base_link
        forces = F_body.unsqueeze(1)     # (N, 1, 3)
        torques = tau_body.unsqueeze(1)
        sedan.set_external_force_and_torque(forces, torques, body_ids=[base_id])

        # Steering: write delta_actual to both front-wheel revolute joints(For visualization)
        steer_pos_target = torch.stack([delta_actual, delta_actual], dim=-1)   # (1, 2)
        sedan.set_joint_position_target(steer_pos_target, joint_ids=steer_ids)

        sedan.write_data_to_sim()
        sim.step()
        sedan.update(sim_dt)
        if cam is not None:
            cam.update(sim_dt)

        # Update a_y_estimate (centripetal approx: omega_z * v_long_body)
        wz = float(state.angvel_body[0, 2].item())
        a_y_estimate = torch.tensor([wz * v_long], device=device)

        # Log
        rs2 = sedan.data.root_state_w[0]
        rpy = quat_wxyz_to_rpy(rs2[3:7].unsqueeze(0))[0]
        v_body2 = state.vel_body[0]
        w_body2 = state.angvel_body[0]
        sum_fz = float(Fz.sum().item())
        # Slip angles: recompute for logging (cheap)
        v_tire_body = state.vel_body.unsqueeze(1).expand(-1, 4, -1) + torch.cross(
            state.angvel_body.unsqueeze(1).expand(-1, 4, -1),
            tire.tire_positions_body().to(device).unsqueeze(0).expand(1, -1, -1),
            dim=-1,
        )
        cos_d = torch.cos(delta_per_wheel)
        sin_d = torch.sin(delta_per_wheel)
        v_long_t = cos_d * v_tire_body[..., 0] + sin_d * v_tire_body[..., 1]
        v_lat_t = -sin_d * v_tire_body[..., 0] + cos_d * v_tire_body[..., 1]
        denom = torch.clamp(torch.abs(v_long_t), min=1e-2)
        alpha = torch.atan(v_lat_t / denom)[0]   # (4,)

        row = [
            t,
            float(rs2[0]), float(rs2[1]), float(rs2[2]),
            math.degrees(float(rpy[0])), math.degrees(float(rpy[1])), math.degrees(float(rpy[2])),
            float(v_body2[0]), float(v_body2[1]), float(rs2[9]),
            float(w_body2[0]), float(w_body2[1]), float(w_body2[2]),
            d_t, float(delta_actual[0]), a_x_t, float(a_x_actual[0]),
            float(Fz[0, 0]), float(Fz[0, 1]), float(Fz[0, 2]), float(Fz[0, 3]), sum_fz,
            float(Fx_tire[0, 0]), float(Fx_tire[0, 1]), float(Fx_tire[0, 2]), float(Fx_tire[0, 3]),
            float(Fy_tire[0, 0]), float(Fy_tire[0, 1]), float(Fy_tire[0, 2]), float(Fy_tire[0, 3]),
            math.degrees(float(alpha[0])), math.degrees(float(alpha[1])),
            math.degrees(float(alpha[2])), math.degrees(float(alpha[3])),
        ]
        rows.append(row)

        max_roll = max(max_roll, abs(math.degrees(float(rpy[0]))))
        max_pitch = max(max_pitch, abs(math.degrees(float(rpy[1]))))
        max_z_err = max(max_z_err, abs(float(rs2[2]) - COG_Z_DEFAULT))
        max_sum_fz_err = max(max_sum_fz_err, abs(sum_fz - mg) / mg)

        if args_cli.scenario == "brake":
            # Track the start of the brake phase (= moment a_x_target first
            # commands negative; uses target rather than actual so it triggers
            # on intent, not after the actuator has lagged into negative territory).
            if t >= 6.0 and brake_start_x is None:
                brake_start_x = float(rs2[0])
                brake_start_v = v_long
            # Stopping moment: vehicle has decelerated to ~0 forward speed.
            if brake_start_x is not None and brake_stopped_at is None and v_long < 0.1:
                brake_stopped_at = t
                brake_stop_x = float(rs2[0])

        if (cam is not None) and (step % cam_period == 0):
            rgb = cam.data.output["rgb"][0].cpu().numpy()
            if rgb.shape[2] == 4:
                rgb = rgb[..., :3]
            frames.append(rgb.astype(np.uint8))

    # Outputs
    os.makedirs(args_cli.video_dir, exist_ok=True)
    os.makedirs(args_cli.metrics_dir, exist_ok=True)
    tag = f"phase1_5_{args_cli.scenario}_mu{args_cli.mu:.2f}"
    csv_path = os.path.join(args_cli.metrics_dir, f"{tag}.csv")
    json_path = os.path.join(args_cli.metrics_dir, f"{tag}.json")
    video_path = os.path.join(args_cli.video_dir, f"{tag}.mp4")

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[INFO] wrote {csv_path}", flush=True)

    if not args_cli.no_video:
        write_mp4(frames, video_path, fps=30)
        print(f"[INFO] wrote {video_path} ({len(frames)} frames)", flush=True)

    summary = {
        "scenario": args_cli.scenario,
        "mu": args_cli.mu,
        "duration_sec": args_cli.duration,
        "max_roll_deg": max_roll,
        "max_pitch_deg": max_pitch,
        "max_z_drift_m": max_z_err,
        "max_sum_Fz_relative_error": max_sum_fz_err,
    }
    if args_cli.scenario == "brake" and brake_start_x is not None:
        # Distance from start of braking to stopping point (or final pos if never stopped).
        end_x = brake_stop_x if brake_stop_x is not None else rows[-1][1]
        d_actual = end_x - brake_start_x
        v0 = brake_start_v if brake_start_v is not None else 0.0
        d_theory = (v0 * v0) / (2.0 * args_cli.mu * GRAVITY) if args_cli.mu > 0 else float("inf")
        summary.update({
            "brake_start_v_mps": v0,
            "brake_distance_actual_m": d_actual,
            "brake_distance_theory_m": d_theory,
            "brake_distance_error_pct": (d_actual - d_theory) / d_theory * 100.0 if d_theory > 0 else None,
            "brake_stopped_t_sec": brake_stopped_at,
            "brake_stopped_completed": brake_stop_x is not None,
        })
    if args_cli.scenario == "circle":
        # Steady-state radius from yaw rate: R = v_long / |omega_z|
        # Take the last 2 seconds
        late = [r for r in rows[1:] if r[0] >= args_cli.duration - 2.0]
        if late:
            v_means = [r[7] for r in late]   # vx_body
            wz_means = [r[12] for r in late]
            v_ss = sum(v_means) / len(v_means)
            wz_ss = sum(wz_means) / len(wz_means)
            R_ss = v_ss / abs(wz_ss) if abs(wz_ss) > 1e-3 else float("inf")
            v_limit_theory = math.sqrt(args_cli.mu * GRAVITY * R_ss) if R_ss > 0 else 0.0
            summary.update({
                "circle_steady_v_mps": v_ss,
                "circle_steady_yaw_rate_rad_s": wz_ss,
                "circle_steady_radius_m": R_ss,
                "circle_limit_speed_theory_mps": v_limit_theory,
                "circle_target_speed_mps": args_cli.target_speed,
                "circle_target_above_limit": args_cli.target_speed > v_limit_theory,
            })

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] wrote {json_path}", flush=True)

    print("=" * 60, flush=True)
    print(f"[RESULT] scenario       = {args_cli.scenario}", flush=True)
    print(f"[RESULT] mu             = {args_cli.mu}", flush=True)
    print(f"[RESULT] max_roll       = {max_roll:.4f} deg", flush=True)
    print(f"[RESULT] max_pitch      = {max_pitch:.4f} deg", flush=True)
    print(f"[RESULT] max_z_drift    = {max_z_err*1000:.2f} mm", flush=True)
    print(f"[RESULT] sum_Fz_err     = {max_sum_fz_err*100:.3f} %  (target < 1%)", flush=True)
    if args_cli.scenario == "brake" and "brake_distance_actual_m" in summary:
        print(f"[RESULT] brake_dist     = actual {summary['brake_distance_actual_m']:.2f} m  "
              f"vs theory {summary['brake_distance_theory_m']:.2f} m  "
              f"err {summary['brake_distance_error_pct']:.1f}%", flush=True)
    if args_cli.scenario == "circle" and "circle_steady_radius_m" in summary:
        print(f"[RESULT] circle radius  = {summary['circle_steady_radius_m']:.2f} m", flush=True)
        print(f"[RESULT] limit speed    = {summary['circle_limit_speed_theory_mps']:.2f} m/s "
              f"(running at {summary['circle_steady_v_mps']:.2f} m/s)", flush=True)
    print("=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
