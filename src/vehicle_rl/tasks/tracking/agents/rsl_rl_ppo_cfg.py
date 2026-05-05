"""rsl_rl PPO config factory for `Vehicle-Tracking-Direct-v0`.

PR 3: replaced the class-level numeric defaults with a thin wrapper around
`vehicle_rl.config.isaac_adapter.make_ppo_runner_cfg`. The cfg is now built
from `configs/agents/rsl_rl/ppo_tracking.yaml`. The gym registry's
`rsl_rl_cfg_entry_point` calls
`vehicle_rl.tasks.tracking.entry_points:rsl_rl_runner_cfg_factory` directly;
this module remains importable for backward-compat callers that still want
a classical "default cfg" by name.
"""
from __future__ import annotations

import os

from vehicle_rl import VEHICLE_RL_ROOT
from vehicle_rl.config.isaac_adapter import make_ppo_runner_cfg
from vehicle_rl.config.loader import load_yaml_strict


def make_default_tracking_ppo_cfg():
    """Build the default rsl_rl PPO cfg from the committed YAML."""
    bundle = load_yaml_strict(
        os.path.join(VEHICLE_RL_ROOT, "configs", "agents", "rsl_rl", "ppo_tracking.yaml")
    )
    return make_ppo_runner_cfg(bundle)


__all__ = ["make_default_tracking_ppo_cfg"]
