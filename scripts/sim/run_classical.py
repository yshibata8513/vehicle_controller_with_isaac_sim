"""Phase 2 classical baseline driver: Pure Pursuit + PI on speed.

Pipeline per step:
    plan, lat_err, hdg_err, s_proj, nearest_idx  <-  Path.project(
        pos_xy, yaw, nearest_idx, search_radius_samples=W, K=K)
    obs                     <-  build_observation(state_gt, plan, lat_err, hdg_err)
    pinion_target           <-  PurePursuitController(obs)
    a_x_target              <-  PIDSpeedController(obs, target_speed)
    state_gt                <-  VehicleSimulator.step(action)

Outputs per run:
    metrics/classical_<course>_mu<m>_v<v>.csv      -- raw trace
    metrics/classical_<course>_mu<m>_v<v>.json     -- 7-metric Phase 2 sanity report
    videos/classical_<course>_mu<m>_v<v>.mp4       -- ground-truth visualisation

Run from repository root (venv python):
    PY="/c/work/isaac/env_isaaclab/Scripts/python.exe"
    $PY scripts/sim/run_classical.py --course circle --mu 0.9 --target_speed 10 --duration 25 --headless
"""
import argparse
import csv
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 2 classical baseline driver")
parser.add_argument("--course", choices=["circle", "lemniscate", "s_curve", "dlc"],
                    default="circle")
parser.add_argument("--mu", type=float, default=0.9)
parser.add_argument("--target_speed", type=float, default=10.0)
parser.add_argument("--duration", type=float, default=25.0)
# Course parameters (only the relevant ones are used per course)
parser.add_argument("--radius", type=float, default=30.0,
                    help="circle: radius [m]; ignored otherwise")
parser.add_argument("--lemniscate_a", type=float, default=25.0,
                    help="lemniscate: scale parameter a [m]")
parser.add_argument("--s_length", type=float, default=100.0,
                    help="s_curve: total x length [m]")
parser.add_argument("--s_amplitude", type=float, default=5.0,
                    help="s_curve: lateral amplitude [m]")
# Controller gains
parser.add_argument("--pp_lookahead_min", type=float, default=2.0)
parser.add_argument("--pp_lookahead_gain", type=float, default=0.5,
                    help="L_d = max(min, gain * vx)")
parser.add_argument("--pid_kp", type=float, default=1.0)
parser.add_argument("--pid_ki", type=float, default=0.3)
# I/O
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

# --- imports below this line require the sim app ---
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.sensors import CameraCfg, Camera

from vehicle_rl.assets import (
    SEDAN_CFG, COG_Z_DEFAULT, WHEELBASE, STEERING_RATIO,
)
from vehicle_rl.controller import PurePursuitController, PIDSpeedController
from vehicle_rl.envs import VehicleAction, build_observation
from vehicle_rl.envs.simulator import VehicleSimulator
from vehicle_rl.planner import circle_path, lemniscate_path, s_curve_path, dlc_path
from vehicle_rl.utils import (
    ProgressAccumulator, summarize_trajectory, write_metrics_json,
)


GRAVITY = 9.81
LOOKAHEAD_DS = 1.0   # arc-length spacing of Plan window [m]
PLAN_K = 20          # number of lookahead points (covers ~ v_max * 1s preview)
OFF_TRACK_THRESHOLD_M = 1.0   # |lateral_error| above this counts as off-track
PROJ_SEARCH_RADIUS = 80       # local-window half-width for Path.project (samples)


def build_path(args, num_envs, device):
    """Construct the requested course as a Path."""
    if args.course == "circle":
        return circle_path(
            radius=args.radius, target_speed=args.target_speed,
            num_envs=num_envs, ds=0.2, device=device,
        )
    elif args.course == "lemniscate":
        return lemniscate_path(
            a=args.lemniscate_a, target_speed=args.target_speed,
            num_envs=num_envs, ds=0.2, device=device,
        )
    elif args.course == "s_curve":
        return s_curve_path(
            length=args.s_length, amplitude=args.s_amplitude,
            target_speed=args.target_speed, num_envs=num_envs, ds=0.2, device=device,
        )
    elif args.course == "dlc":
        return dlc_path(
            target_speed=args.target_speed, num_envs=num_envs, ds=0.2, device=device,
        )
    raise ValueError(f"unknown course {args.course}")


def camera_view_for_course(course):
    """Return (eye, target) world positions appropriate for the course."""
    if course == "circle":
        return (0.0, -60.0, 40.0), (0.0, 0.0, 0.0)
    elif course == "lemniscate":
        return (0.0, -55.0, 50.0), (0.0, 0.0, 0.0)
    elif course == "s_curve":
        return (50.0, -25.0, 30.0), (50.0, 0.0, 0.0)
    elif course == "dlc":
        return (30.0, -25.0, 25.0), (30.0, 1.5, 0.0)
    return (0.0, -50.0, 30.0), (0.0, 0.0, 0.0)


def design_scene(record_video):
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    sedan_cfg = SEDAN_CFG.copy()
    sedan_cfg.prim_path = "/World/Sedan"
    sedan = Articulation(cfg=sedan_cfg)

    cam = None
    if record_video:
        cam_cfg = CameraCfg(
            prim_path="/World/SideCam",
            update_period=1.0 / 30.0,
            height=540, width=960,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=10.0,
                horizontal_aperture=20.955, clipping_range=(0.1, 1000.0),
            ),
        )
        cam = Camera(cfg=cam_cfg)
    return sedan, cam


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


def yaw_to_quat_wxyz(yaw):
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def main():
    device = args_cli.device
    sim_cfg = sim_utils.SimulationCfg(
        device=device, dt=1.0 / 200.0, gravity=(0.0, 0.0, -GRAVITY),
    )
    sim = SimulationContext(sim_cfg)
    eye, target = camera_view_for_course(args_cli.course)
    sim.set_camera_view(list(eye), list(target))

    record_video = not args_cli.no_video
    sedan, cam = design_scene(record_video=record_video)
    sim.reset()

    if cam is not None:
        cam.set_world_poses_from_view(
            torch.tensor([list(eye)], device=device),
            torch.tensor([list(target)], device=device),
        )

    print(f"[INFO] course={args_cli.course} mu={args_cli.mu} "
          f"target_speed={args_cli.target_speed} duration={args_cli.duration}", flush=True)

    # Build Path
    path = build_path(args_cli, num_envs=1, device=device)
    print(f"[INFO] Path: M={path.num_samples} ds={path.ds:.4f} L={path.total_length:.2f} "
          f"is_loop={path.is_loop}", flush=True)

    # Initial pose from the path's start sample
    start_pos, start_yaw = path.start_pose
    qw, qx, qy, qz = yaw_to_quat_wxyz(float(start_yaw[0].item()))
    initial_pose = torch.tensor(
        [[float(start_pos[0, 0].item()), float(start_pos[0, 1].item()), COG_Z_DEFAULT,
          qw, qx, qy, qz]],
        device=device,
    )

    vsim = VehicleSimulator(sim, sedan, mu_default=args_cli.mu)
    state_gt = vsim.reset(initial_pose=initial_pose)

    # Controllers
    pp = PurePursuitController(
        wheelbase=WHEELBASE, steering_ratio=STEERING_RATIO, pinion_max=vsim.pinion_max,
        lookahead_min=args_cli.pp_lookahead_min,
        lookahead_gain=args_cli.pp_lookahead_gain,
        lookahead_ds=LOOKAHEAD_DS,
    )
    pid = PIDSpeedController(
        num_envs=1, dt=vsim.dt, kp=args_cli.pid_kp, ki=args_cli.pid_ki,
        device=device,
    )
    pid.reset()

    sim_dt = vsim.dt
    n_steps = int(args_cli.duration / sim_dt)
    cam_period = max(1, int(round(1.0 / 30.0 / sim_dt)))

    rows = [["t", "x", "y", "yaw_deg", "vx", "vy", "yaw_rate", "roll_deg",
             "lat_err", "hdg_err_deg", "s_proj",
             "delta_target", "pinion_target", "a_x_target",
             "delta_actual", "a_x_actual"]]
    frames = []

    # Course-progress accumulator (shared with Phase 3 RL eval). Owns the
    # wrap correction and per-step delta cap so completion metrics stay
    # honest under projection glitches / off-track detours.
    progress = ProgressAccumulator(
        num_envs=1,
        total_length=path.total_length,
        ds=path.ds,
        dt=sim_dt,
        is_loop=path.is_loop,
        device=device,
        off_track_threshold=OFF_TRACK_THRESHOLD_M,
    )

    # Local-window projection state (Path.project no longer scans the full
    # course; the caller seeds nearest_idx and updates it from the returned
    # closest_idx each step). Start at sample 0 since the vehicle was placed
    # at path.start_pose above.
    nearest_idx = torch.zeros(1, dtype=torch.long, device=device)

    for step in range(n_steps):
        t = step * sim_dt
        pos_xy = state_gt.pos_xyz[:, :2]
        yaw = state_gt.rpy[:, 2]

        # Pre-step projection feeds the controller (the controller observes
        # the current state, not the next one).
        plan, lat_err, hdg_err, _, nearest_idx = path.project(
            pos_xy, yaw, nearest_idx,
            search_radius_samples=PROJ_SEARCH_RADIUS,
            K=PLAN_K, lookahead_ds=LOOKAHEAD_DS,
        )
        obs = build_observation(state_gt, plan, lat_err, hdg_err)

        pinion_target = pp(obs)
        a_x_target = pid(obs, target_speed=args_cli.target_speed)
        action = VehicleAction(pinion_target=pinion_target, a_x_target=a_x_target)
        state_gt = vsim.step(action)

        # Re-project at post-step pose so the row's lateral / heading error
        # match the row's pose, vel, and actuator state. Without this the CSV
        # mixes pre-step error with post-step state, biasing rms_lat /
        # off_track_time at low-mu / DLC where the error changes quickly.
        # Reuse the just-updated nearest_idx (don't advance it twice in one
        # control step -- the post-step argmin window is centred where the
        # vehicle actually is, so the update is monotone).
        post_pos_xy = state_gt.pos_xyz[:, :2]
        post_yaw = state_gt.rpy[:, 2]
        _, lat_err_log, hdg_err_log, s_proj_log, nearest_idx = path.project(
            post_pos_xy, post_yaw, nearest_idx,
            search_radius_samples=PROJ_SEARCH_RADIUS,
            K=PLAN_K, lookahead_ds=LOOKAHEAD_DS,
        )
        progress.update(s_proj_log, state_gt.vel_body[:, 0], lat_err_log)

        rows.append([
            t,
            float(state_gt.pos_xyz[0, 0]), float(state_gt.pos_xyz[0, 1]),
            math.degrees(float(state_gt.rpy[0, 2])),
            float(state_gt.vel_body[0, 0]), float(state_gt.vel_body[0, 1]),
            float(state_gt.angvel_body[0, 2]),
            math.degrees(float(state_gt.rpy[0, 0])),
            float(lat_err_log[0]), math.degrees(float(hdg_err_log[0])),
            float(s_proj_log[0]),
            float(pinion_target[0] / STEERING_RATIO),
            float(pinion_target[0]),
            float(a_x_target[0]),
            float(state_gt.delta_actual[0]),
            float(state_gt.a_x_actual[0]),
        ])

        if (cam is not None) and (step % cam_period == 0):
            cam.update(sim_dt)
            rgb = cam.data.output["rgb"][0].cpu().numpy()
            if rgb.shape[2] == 4:
                rgb = rgb[..., :3]
            frames.append(rgb.astype(np.uint8))

    # ----- Outputs -----
    os.makedirs(args_cli.video_dir, exist_ok=True)
    os.makedirs(args_cli.metrics_dir, exist_ok=True)
    tag = f"classical_{args_cli.course}_mu{args_cli.mu:.2f}_v{args_cli.target_speed:.1f}"
    csv_path = os.path.join(args_cli.metrics_dir, f"{tag}.csv")
    json_path = os.path.join(args_cli.metrics_dir, f"{tag}.json")
    video_path = os.path.join(args_cli.video_dir, f"{tag}.mp4")

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[INFO] wrote {csv_path}", flush=True)

    if record_video:
        write_mp4(frames, video_path, fps=30)
        print(f"[INFO] wrote {video_path} ({len(frames)} frames)", flush=True)

    # ----- 7-metric Phase 2 sanity report (PLAN.md §2) -----
    # Uses the shared `summarize_trajectory` helper so Phase 3 RL eval
    # produces identical metric definitions per env.
    arr = np.array([row for row in rows[1:]], dtype=np.float64)
    metrics = summarize_trajectory(
        lat_err_m=arr[:, 8],
        vx_mps=arr[:, 4],
        yaw_rate_rad_s=arr[:, 6],
        roll_deg=arr[:, 7],
        dt=sim_dt,
        target_speed=args_cli.target_speed,
        traveled_arc=float(progress.traveled_arc[0]),
        on_track_arc=float(progress.on_track_arc[0]),
        course_length=path.total_length,
        off_track_threshold=OFF_TRACK_THRESHOLD_M,
    )

    write_metrics_json(
        metrics, json_path,
        course=args_cli.course,
        mu=args_cli.mu,
        target_speed_mps=args_cli.target_speed,
    )
    summary = {
        "course": args_cli.course,
        "mu": args_cli.mu,
        "target_speed_mps": args_cli.target_speed,
        **{k: v for k, v in metrics.__dict__.items()},
    }
    print(f"[INFO] wrote {json_path}", flush=True)

    print("=" * 60, flush=True)
    print(f"[RESULT] course                 = {summary['course']}", flush=True)
    print(f"[RESULT] mu                     = {summary['mu']}", flush=True)
    print(f"[RESULT] rms_lateral_error      = {summary['rms_lateral_error_m']:.3f} m", flush=True)
    print(f"[RESULT] max_lateral_error      = {summary['max_lateral_error_m']:.3f} m", flush=True)
    print(f"[RESULT] completion_rate        = {summary['completion_rate']:.3f} "
          f"({summary['traveled_arc_m']:.1f} / {summary['course_length_m']:.1f} m)", flush=True)
    print(f"[RESULT] on_track_progress_rate = {summary['on_track_progress_rate']:.3f}", flush=True)
    print(f"[RESULT] mean_speed_error       = {summary['mean_speed_error_mps']:+.3f} m/s", flush=True)
    print(f"[RESULT] max_yaw_rate           = {summary['max_yaw_rate_rad_s']:.3f} rad/s", flush=True)
    print(f"[RESULT] max_roll_angle         = {summary['max_roll_angle_deg']:.3f} deg", flush=True)
    print(f"[RESULT] off_track_time         = {summary['off_track_time_sec']:.3f} s "
          f"(|lat|>{OFF_TRACK_THRESHOLD_M}m)", flush=True)
    print("=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
