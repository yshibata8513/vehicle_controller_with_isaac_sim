"""Phase 0 sanity check: spawn a WheeledLab env (F1TenthDriftRL) and step it briefly with random actions, headless.

Goal: confirm wheeledlab_tasks/wheeledlab_assets import cleanly on Isaac Lab 2.3 / Sim 5.1
and that the F1Tenth USD + Articulation + Ackermann action wiring all run without crashing.

This does NOT use the full WheeledLab training stack (no wandb, no CustomRecordVideo, no Hydra).
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 0 WheeledLab sanity check")
parser.add_argument("--task", type=str, default="Isaac-F1TenthDriftRL-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import wheeledlab_tasks  # noqa: F401  -- registers Isaac-F1TenthDriftRL-v0 etc.

from isaaclab_tasks.utils import parse_env_cfg


def main():
    import sys
    print("[CHECK] entered main()", flush=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    print("[CHECK] parse_env_cfg done", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[CHECK] gym.make done. obs={env.observation_space} act={env.action_space}", flush=True)
    obs, _ = env.reset()
    print(f"[CHECK] reset done. obs keys: {list(obs.keys()) if hasattr(obs, 'keys') else type(obs)}", flush=True)
    for i in range(args_cli.steps):
        with torch.inference_mode():
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            obs, rew, term, trunc, info = env.step(actions)
        if i % 20 == 0:
            print(f"[step {i:4d}] reward sample: {rew[0].item():+.3f}", flush=True)
            sys.stdout.flush()
    env.close()
    print("[CHECK] WheeledLab F1Tenth sanity check OK.", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
    simulation_app.close()
