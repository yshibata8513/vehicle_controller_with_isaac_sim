"""Phase 3 PPO eval / replay entry point.

Loads a trained policy checkpoint and runs N envs of the same gym task
deterministically, then dumps:

  - `metrics/play_<course>_<run>_<ckpt>.csv`     -- per-step trajectory log
  - `videos/play_<course>_<run>_<ckpt>.png`      -- top-down path-vs-vehicle
                                                    plot + lat_err / vx
                                                    panels (1 figure)

Use this to *see* what `train_ppo.py` actually learned -- TensorBoard
shows scalar trends, this shows the trajectory and speed profile.

Usage (from repo root):
    PY="/c/work/isaac/env_isaaclab/Scripts/python.exe"
    $PY scripts/rl/play.py --task Vehicle-Tracking-Direct-v0 \
        --course random_long --random_path_cfg configs/random_path.yaml \
        --num_envs 1 --duration 25 --headless

    # random_bank: each reset samples a fresh path from the configured bank.
    # `random_reset_along_path=False` (forced below) ensures spawn at sample 0
    # of the freshly-sampled path, so the recorded trajectory is reproducible.
    $PY scripts/rl/play.py --task Vehicle-Tracking-Direct-v0 \
        --course random_bank --random_path_cfg configs/random_path.yaml \
        --experiment_name phase3_random_bank \
        --num_envs 1 --duration 25 --headless

Defaults to the most recent run + last `model_*.pt` when --load_run /
--checkpoint are omitted.
"""
import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a trained vehicle_rl policy.")
parser.add_argument("--task", type=str, default="Vehicle-Tracking-Direct-v0")
parser.add_argument("--course", type=str, default=None)
parser.add_argument("--random_path_cfg", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration", type=float, default=25.0)
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--load_run", type=str, default=None,
                    help="Run dir under logs/rsl_rl/<experiment>/. "
                         "Defaults to the most recent.")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Checkpoint file (e.g. model_200.pt). "
                         "Defaults to the highest-numbered model_*.pt.")
parser.add_argument("--metrics_dir", type=str,
                    default="C:/Users/user/vehicle_rl/metrics")
parser.add_argument("--plot_dir", type=str,
                    default="C:/Users/user/vehicle_rl/videos")
parser.add_argument("--no_plot", action="store_true")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--episode_length_s", type=float, default=None,
                    help="Override TrackingEnvCfg.episode_length_s so the env "
                         "does not auto-reset mid-rollout. Set this to >= "
                         "--duration when measuring long-duration tracking.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- imports below this line require the sim app -------------------------------
import csv
import numpy as np
import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import vehicle_rl.tasks  # noqa: F401  -- registers gym ids


def _load_agent_cfg(task_name: str):
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    return load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")


def _save_plot(rows, path_xy, out_path):
    """Top-down trajectory + vx + lat_err panels in one figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.array(rows, dtype=np.float64)
    t = arr[:, 0]
    x = arr[:, 1]
    y = arr[:, 2]
    vx = arr[:, 4]
    lat_err = arr[:, 5]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_xy = axes[0, 0]
    ax_xy_zoom = axes[0, 1]
    ax_v = axes[1, 0]
    ax_le = axes[1, 1]

    ax_xy.plot(path_xy[:, 0], path_xy[:, 1], "k-", lw=0.8, alpha=0.4, label="path")
    ax_xy.plot(x, y, "r-", lw=1.2, label="vehicle")
    ax_xy.plot(x[0], y[0], "go", ms=6, label="start")
    ax_xy.plot(x[-1], y[-1], "rx", ms=8, label="end")
    ax_xy.set_aspect("equal")
    ax_xy.legend(loc="best", fontsize=8)
    ax_xy.set_title("trajectory (full path = grey, vehicle = red)")
    ax_xy.set_xlabel("x [m]"); ax_xy.set_ylabel("y [m]")

    # zoom around the vehicle's actual extent
    pad = 5.0
    ax_xy_zoom.plot(path_xy[:, 0], path_xy[:, 1], "k-", lw=0.8, alpha=0.4)
    ax_xy_zoom.plot(x, y, "r-", lw=1.2)
    ax_xy_zoom.plot(x[0], y[0], "go", ms=6)
    ax_xy_zoom.plot(x[-1], y[-1], "rx", ms=8)
    ax_xy_zoom.set_xlim(min(x) - pad, max(x) + pad)
    ax_xy_zoom.set_ylim(min(y) - pad, max(y) + pad)
    ax_xy_zoom.set_aspect("equal")
    ax_xy_zoom.set_title("trajectory zoom (vehicle extent only)")
    ax_xy_zoom.set_xlabel("x [m]"); ax_xy_zoom.set_ylabel("y [m]")

    ax_v.plot(t, vx, "b-")
    ax_v.set_xlabel("t [s]"); ax_v.set_ylabel("vx [m/s]")
    ax_v.set_title("body-frame longitudinal speed")
    ax_v.grid(True, alpha=0.3)

    ax_le.plot(t, lat_err, "m-")
    ax_le.axhline(0, color="k", lw=0.5)
    ax_le.set_xlabel("t [s]"); ax_le.set_ylabel("lateral_error [m]")
    ax_le.set_title("signed lateral error (>0 = vehicle is left of path)")
    ax_le.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs, use_fabric=True)
    if args_cli.course is not None:
        env_cfg.course = args_cli.course
    if args_cli.random_path_cfg is not None:
        env_cfg.random_path_cfg_path = args_cli.random_path_cfg
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    # Eval is deterministic: spawn at path[0] so the recorded run is
    # reproducible across replays.
    env_cfg.random_reset_along_path = False

    agent_cfg = _load_agent_cfg(args_cli.task)
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_root = os.path.join(repo_root, "logs", "rsl_rl", agent_cfg.experiment_name)
    # Restrict checkpoint regex to model_*.pt so the auto-latest pick doesn't
    # accidentally land on the TensorBoard event file.
    gck_kwargs = {"checkpoint": r"model_.*\.pt"}
    if args_cli.load_run is not None:
        gck_kwargs["run_dir"] = args_cli.load_run
    if args_cli.checkpoint is not None:
        gck_kwargs["checkpoint"] = args_cli.checkpoint
    ckpt_path = get_checkpoint_path(log_root, **gck_kwargs)
    print(f"[INFO] loading checkpoint: {ckpt_path}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    # Cache the path geometry (env-local frame) for plotting.
    env_unwrapped = env.unwrapped
    path = env_unwrapped.path
    path_xy = torch.stack([path.x[0], path.y[0]], dim=-1).cpu().numpy()

    obs = env.get_observations()
    sim_dt = env_unwrapped.cfg.sim.dt
    control_dt = sim_dt * env_unwrapped.cfg.decimation
    n_control_steps = int(round(args_cli.duration / control_dt))

    rows = []  # (t, x, y, yaw_deg, vx, lat_err, hdg_err_deg, action_pinion)
    reset_events = []  # list of (t, kind) where kind ∈ {"timeout","terminated"}

    for step in range(n_control_steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _reward, dones, _info = env.step(actions)

        # Detect mid-rollout resets so the long-duration test can report
        # "the env teleported the vehicle back to path[0] at t=X" instead of
        # silently swallowing them. RslRlVecEnvWrapper merges terminated +
        # time_out into `dones`; the unwrapped env still exposes the cause.
        if bool(dones[0].item()):
            t_reset = (step + 1) * control_dt
            terminated_buf = getattr(env_unwrapped, "reset_terminated", None)
            if terminated_buf is not None and bool(terminated_buf[0].item()):
                reset_events.append((t_reset, "terminated"))
            else:
                reset_events.append((t_reset, "timeout"))

        state_gt = env_unwrapped._last_state_gt
        # env-local position (the path is in env-local; subtract env_origin
        # so the trajectory plots overlay cleanly on path_xy).
        pos_world = state_gt.pos_xyz[0].cpu().numpy()
        env_origin = env_unwrapped.scene.env_origins[0, :2].cpu().numpy()
        x_local = float(pos_world[0] - env_origin[0])
        y_local = float(pos_world[1] - env_origin[1])
        yaw = float(state_gt.rpy[0, 2].cpu())
        vx = float(state_gt.vel_body[0, 0].cpu())
        lat_err = float(env_unwrapped._last_lat_err[0].cpu())
        hdg_err = float(env_unwrapped._last_hdg_err[0].cpu())
        action_pinion = float(actions[0, 0].cpu())
        t = (step + 1) * control_dt
        rows.append((
            t, x_local, y_local, math.degrees(yaw), vx,
            lat_err, math.degrees(hdg_err), action_pinion,
        ))

    # Tags for output filenames.
    run_tag = os.path.basename(os.path.dirname(ckpt_path))
    ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    course = env_cfg.course

    # CSV
    os.makedirs(args_cli.metrics_dir, exist_ok=True)
    csv_path = os.path.join(
        args_cli.metrics_dir,
        f"play_{course}_{run_tag}_{ckpt_tag}.csv",
    )
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "yaw_deg", "vx", "lat_err",
                    "hdg_err_deg", "action_pinion"])
        w.writerows(rows)
    print(f"[INFO] wrote {csv_path}", flush=True)

    # Plot
    if not args_cli.no_plot:
        os.makedirs(args_cli.plot_dir, exist_ok=True)
        plot_path = os.path.join(
            args_cli.plot_dir,
            f"play_{course}_{run_tag}_{ckpt_tag}.png",
        )
        _save_plot(rows, path_xy, plot_path)
        print(f"[INFO] wrote {plot_path}", flush=True)

    # Console summary.
    arr = np.array(rows, dtype=np.float64)
    if arr.shape[0] > 0:
        rms_lat = float(np.sqrt(np.mean(arr[:, 5] ** 2)))
        max_lat = float(np.max(np.abs(arr[:, 5])))
        mean_vx = float(np.mean(arr[:, 4]))
        print("=" * 60, flush=True)
        print(f"[RESULT] course           = {course}", flush=True)
        print(f"[RESULT] checkpoint       = {os.path.basename(ckpt_path)}", flush=True)
        print(f"[RESULT] duration         = {arr[-1, 0]:.2f} s ({arr.shape[0]} steps)", flush=True)
        print(f"[RESULT] rms_lateral_err  = {rms_lat:.3f} m", flush=True)
        print(f"[RESULT] max_lateral_err  = {max_lat:.3f} m", flush=True)
        print(f"[RESULT] mean_vx          = {mean_vx:.3f} m/s", flush=True)
        n_term = sum(1 for _, k in reset_events if k == "terminated")
        n_to = sum(1 for _, k in reset_events if k == "timeout")
        print(f"[RESULT] resets           = {len(reset_events)} "
              f"(terminated={n_term}, timeout={n_to})", flush=True)
        for t_r, kind in reset_events[:10]:
            print(f"  reset @ t={t_r:.2f}s  ({kind})", flush=True)
        if len(reset_events) > 10:
            print(f"  ... and {len(reset_events) - 10} more", flush=True)
        print("=" * 60, flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
