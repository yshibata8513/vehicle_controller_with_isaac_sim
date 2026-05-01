"""Phase 1 sanity check: spawn the case-B sedan and verify articulation health.

Phase 1 spawns SEDAN_CFG (visual-only wheels, no tire forces yet) under
**zero gravity** by default. Verifies that:
  - The Articulation loads without exploding.
  - Joints (4 spin + 2 steer) resolve correctly.
  - roll / pitch stay near zero in a short window after spawn.
  - All joint velocities stay below 0.1 rad/s.

Phase 1.5 will reuse this script with `--gravity` enabled once the dynamics
module injects Fz / tire forces to support the chassis under gravity.

Run from repository root:
    python scripts/sim/spawn_sedan.py --duration 1.0
    python scripts/sim/spawn_sedan.py --duration 2.0 --gravity   # for diagnostic
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 1 sedan spawn + articulation sanity check")
parser.add_argument("--duration", type=float, default=1.0,
                    help="Sim seconds to record (default: 1.0)")
parser.add_argument("--gravity", action="store_true",
                    help="Enable gravity (default: OFF for Phase 1)")
parser.add_argument("--video_dir", type=str,
                    default="C:/Users/user/vehicle_rl/videos")
parser.add_argument("--metrics_dir", type=str,
                    default="C:/Users/user/vehicle_rl/metrics")
parser.add_argument("--no_video", action="store_true",
                    help="Skip MP4 recording (dry run)")
parser.add_argument("--steer_target", type=float, default=0.0,
                    help="Constant steer angle target [rad] for diagnostics")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Cameras must be enabled for video recording.
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

from vehicle_rl.assets import SEDAN_CFG


# Phase 1 completion thresholds (PLAN.md):
ROLL_PITCH_LIMIT_DEG = 0.5
JOINT_VEL_LIMIT_RAD_S = 0.1


def design_scene(record_video: bool):
    # No physics material on the ground -- in case-B the wheels have no
    # collision so ground friction is irrelevant for Phase 1.
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    sim_utils.create_prim("/World/Origin0", "Xform", translation=[0.0, 0.0, 0.0])
    sedan_cfg = SEDAN_CFG.copy()
    sedan_cfg.prim_path = "/World/Origin0/Sedan"
    sedan = Articulation(cfg=sedan_cfg)

    if not record_video:
        return {"sedan": sedan, "cam": None}

    # Side-front camera so chassis + wheels are both visible.
    cam_cfg = CameraCfg(
        prim_path="/World/SideCam",
        update_period=1.0 / 30.0,
        height=540,
        width=960,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=10.0,
            horizontal_aperture=20.955, clipping_range=(0.1, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(7.0, -5.0, 2.5),
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="world",
        ),
    )
    cam = Camera(cfg=cam_cfg)
    return {"sedan": sedan, "cam": cam}


def write_mp4(frames_rgb, out_path, fps=30):
    """Write list of HxWx3 uint8 numpy arrays to MP4 via PyAV."""
    import av
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


def quat_wxyz_to_rpy(q):
    """Convert (w, x, y, z) quaternion to (roll, pitch, yaw) in radians."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def main():
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=1.0 / 200.0,
        gravity=(0.0, 0.0, -9.81) if args_cli.gravity else (0.0, 0.0, 0.0),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([7.0, -5.0, 2.5], [0.0, 0.0, 0.5])

    record_video = not args_cli.no_video
    entities = design_scene(record_video=record_video)
    sim.reset()

    sedan: Articulation = entities["sedan"]
    cam: Camera = entities["cam"]

    print(f"[INFO] gravity = {sim_cfg.gravity}", flush=True)
    print(f"[INFO] joint_names = {sedan.joint_names}", flush=True)
    print(f"[INFO] body_names  = {sedan.body_names}", flush=True)

    # Reset state
    root_state = sedan.data.default_root_state.clone()
    sedan.write_root_pose_to_sim(root_state[:, :7])
    sedan.write_root_velocity_to_sim(root_state[:, 7:])
    joint_pos = sedan.data.default_joint_pos.clone()
    joint_vel = sedan.data.default_joint_vel.clone()
    sedan.write_joint_state_to_sim(joint_pos, joint_vel)
    sedan.reset()

    # Resolve joint indices for diagnostics.
    steer_ids, steer_names = sedan.find_joints("front_(left|right)_steer_joint")
    wheel_ids, wheel_names = sedan.find_joints("(front|rear)_(left|right)_wheel_joint")
    print(f"[INFO] steer joints: {list(zip(steer_ids, steer_names))}", flush=True)
    print(f"[INFO] wheel joints: {list(zip(wheel_ids, wheel_names))}", flush=True)

    sim_dt = sim.get_physics_dt()
    n_steps = int(args_cli.duration / sim_dt)
    cam_period = max(1, int(round(1.0 / 30.0 / sim_dt)))
    print(f"[INFO] duration={args_cli.duration}s steps={n_steps} dt={sim_dt:.4f} "
          f"cam_every={cam_period}", flush=True)

    n_steer = len(steer_ids)
    steer_target = torch.full((1, n_steer), args_cli.steer_target, device=sim.device)

    frames = []
    rows = [["t", "x", "y", "z", "qw", "qx", "qy", "qz",
             "roll_deg", "pitch_deg", "yaw_deg",
             "vx", "vy", "vz", "wx", "wy", "wz",
             "steer_FL", "steer_FR",
             "wheelvel_FL", "wheelvel_FR", "wheelvel_RL", "wheelvel_RR"]]

    max_roll = 0.0
    max_pitch = 0.0
    max_jvel = 0.0

    for i in range(n_steps):
        t = i * sim_dt

        # Hold the steer target constant (default 0.0).
        sedan.set_joint_position_target(steer_target, joint_ids=steer_ids)
        sedan.write_data_to_sim()
        sim.step()
        sedan.update(sim_dt)
        if cam is not None:
            cam.update(sim_dt)

        rs = sedan.data.root_state_w[0].cpu().numpy()
        # rs = [px,py,pz, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]
        roll, pitch, yaw = quat_wxyz_to_rpy(rs[3:7])
        jpos = sedan.data.joint_pos[0].cpu().numpy()
        jvel = sedan.data.joint_vel[0].cpu().numpy()

        rows.append([t, rs[0], rs[1], rs[2], rs[3], rs[4], rs[5], rs[6],
                     math.degrees(roll), math.degrees(pitch), math.degrees(yaw),
                     rs[7], rs[8], rs[9], rs[10], rs[11], rs[12],
                     float(jpos[steer_ids[0]]), float(jpos[steer_ids[1]]),
                     float(jvel[wheel_ids[0]]), float(jvel[wheel_ids[1]]),
                     float(jvel[wheel_ids[2]]), float(jvel[wheel_ids[3]])])

        max_roll = max(max_roll, abs(math.degrees(roll)))
        max_pitch = max(max_pitch, abs(math.degrees(pitch)))
        max_jvel = max(max_jvel, float(np.max(np.abs(jvel))))

        if (cam is not None) and (i % cam_period == 0):
            rgb = cam.data.output["rgb"][0].cpu().numpy()
            if rgb.shape[2] == 4:
                rgb = rgb[..., :3]
            frames.append(rgb.astype(np.uint8))

    # Write outputs
    os.makedirs(args_cli.video_dir, exist_ok=True)
    os.makedirs(args_cli.metrics_dir, exist_ok=True)
    suffix = "gravity" if args_cli.gravity else "nograv"
    csv_path = os.path.join(args_cli.metrics_dir, f"phase1_spawn_{suffix}.csv")
    json_path = os.path.join(args_cli.metrics_dir, f"phase1_spawn_{suffix}.json")
    video_path = os.path.join(args_cli.video_dir, f"phase1_spawn_{suffix}.mp4")

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[INFO] wrote {csv_path}", flush=True)

    if not args_cli.no_video and len(frames) > 0:
        write_mp4(frames, video_path, fps=30)
        print(f"[INFO] wrote {video_path} ({len(frames)} frames)", flush=True)

    # Completion criteria (PLAN.md Phase 1):
    #   |roll| < 0.5 deg, |pitch| < 0.5 deg, max joint velocity < 0.1 rad/s
    pass_roll = max_roll < ROLL_PITCH_LIMIT_DEG
    pass_pitch = max_pitch < ROLL_PITCH_LIMIT_DEG
    pass_jvel = max_jvel < JOINT_VEL_LIMIT_RAD_S
    overall_pass = pass_roll and pass_pitch and pass_jvel

    summary = {
        "duration_sec": args_cli.duration,
        "gravity_enabled": bool(args_cli.gravity),
        "max_roll_deg": max_roll,
        "max_pitch_deg": max_pitch,
        "max_joint_velocity_rad_s": max_jvel,
        "thresholds": {
            "roll_pitch_deg": ROLL_PITCH_LIMIT_DEG,
            "joint_vel_rad_s": JOINT_VEL_LIMIT_RAD_S,
        },
        "pass": {
            "roll": pass_roll,
            "pitch": pass_pitch,
            "joint_velocity": pass_jvel,
            "overall": overall_pass,
        },
        "joint_names": list(sedan.joint_names),
        "body_names": list(sedan.body_names),
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] wrote {json_path}", flush=True)

    print("=" * 60, flush=True)
    print(f"[RESULT] max_roll  = {max_roll:.4f} deg  (< {ROLL_PITCH_LIMIT_DEG}: {pass_roll})", flush=True)
    print(f"[RESULT] max_pitch = {max_pitch:.4f} deg  (< {ROLL_PITCH_LIMIT_DEG}: {pass_pitch})", flush=True)
    print(f"[RESULT] max_jvel  = {max_jvel:.4f} rad/s (< {JOINT_VEL_LIMIT_RAD_S}: {pass_jvel})", flush=True)
    print(f"[RESULT] overall   = {'PASS' if overall_pass else 'FAIL'}", flush=True)
    print("=" * 60, flush=True)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
