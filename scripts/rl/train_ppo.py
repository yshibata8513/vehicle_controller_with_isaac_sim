"""Phase 3 PPO training entry point (rsl_rl).

Adapted from `IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py`
(BSD-3, Isaac Lab project) with vehicle_rl-specific changes:

  - Imports `vehicle_rl.tasks` to register `Vehicle-Tracking-Direct-v0`.
  - Logs to `vehicle_rl/logs/rsl_rl/<experiment>/<timestamp>` instead of
    Isaac Lab's repo-relative `logs/`.
  - Drops Hydra (`hydra_task_config`) so the entry point works without
    Hydra config files; cfgs come from the gym task registry.
  - Drops Distillation / MARL paths -- not used in vehicle_rl.

Usage (from repo root):
    PY="/c/work/isaac/env_isaaclab/Scripts/python.exe"
    $PY scripts/rl/train_ppo.py --task Vehicle-Tracking-Direct-v0 \
        --num_envs 64 --max_iterations 100 --headless

Stage 0 sanity: --num_envs 64 --max_iterations 50 should print monotonically
increasing mean reward in TensorBoard within ~2 minutes on RTX 5090.
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

# --- arg parser (kept compatible with official rsl_rl CLI) -----------------
parser = argparse.ArgumentParser(description="Train a vehicle-tracking RL agent with rsl_rl.")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record videos during training (slow; eval only by default).")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None,
                    help="Override TrackingEnvCfg.scene.num_envs.")
parser.add_argument("--task", type=str, default="Vehicle-Tracking-Direct-v0")
parser.add_argument("--course", type=str, default=None,
                    help="Override TrackingEnvCfg.course "
                         "(circle | s_curve | dlc | lemniscate | random_long | random_bank).")
parser.add_argument("--random_path_cfg", type=str, default=None,
                    help="Override TrackingEnvCfg.random_path_cfg_path "
                         "(used when course is random_long or random_bank).")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--logger", type=str, default=None,
                    choices={"wandb", "tensorboard", "neptune"})
parser.add_argument("--log_project_name", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- check rsl-rl version (matches official train.py) -----------------------
import importlib.metadata as metadata
from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    print(f"[ERROR] rsl-rl-lib {installed_version} < required {RSL_RL_VERSION}", flush=True)
    sys.exit(1)
print(f"[INFO] rsl-rl-lib version: {installed_version}", flush=True)

# --- imports below this line require the sim app ---------------------------
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Register vehicle_rl task ids before gym.make. The import has the side
# effect of registering Vehicle-Tracking-Direct-v0; without it, gym.make
# raises NameNotFound.
import vehicle_rl.tasks  # noqa: F401


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _load_agent_cfg(task_name: str):
    """Load the rsl_rl runner cfg registered for `task_name`."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    return load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")


def _apply_cli_overrides(env_cfg, agent_cfg, args):
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        env_cfg.sim.device = args.device
    if args.course is not None:
        env_cfg.course = args.course
    if args.random_path_cfg is not None:
        env_cfg.random_path_cfg_path = args.random_path_cfg
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    if args.run_name is not None:
        agent_cfg.run_name = args.run_name
    if args.resume:
        agent_cfg.resume = True
    if args.load_run is not None:
        agent_cfg.load_run = args.load_run
    if args.checkpoint is not None:
        agent_cfg.load_checkpoint = args.checkpoint
    if args.logger is not None:
        agent_cfg.logger = args.logger
    if args.log_project_name and agent_cfg.logger in {"wandb", "neptune"}:
        agent_cfg.wandb_project = args.log_project_name
        agent_cfg.neptune_project = args.log_project_name
    env_cfg.seed = agent_cfg.seed
    return env_cfg, agent_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    agent_cfg = _load_agent_cfg(args_cli.task)
    env_cfg, agent_cfg = _apply_cli_overrides(env_cfg, agent_cfg, args_cli)

    # Log dir under repo root: vehicle_rl/logs/rsl_rl/<experiment>/<timestamp>.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_root_path = os.path.join(repo_root, "logs", "rsl_rl", agent_cfg.experiment_name)
    os.makedirs(log_root_path, exist_ok=True)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging to: {log_dir}", flush=True)

    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.", flush=True)
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume:
        print(f"[INFO] Loading model checkpoint: {resume_path}", flush=True)
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    start = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[INFO] Training time: {round(time.time() - start, 2)} s", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
