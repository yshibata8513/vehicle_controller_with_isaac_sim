"""Path-tracking RL tasks (Stage 0+).

PR 3: env / agent cfg entry points are now 0-arg factories that load the
experiment YAML and build the Isaac cfg via the adapter (see
`vehicle_rl.tasks.tracking.entry_points`). The legacy class-based entry
points are retired here.
"""
import gymnasium as gym

from . import agents  # noqa: F401  (kept to preserve import side effects)

gym.register(
    id="Vehicle-Tracking-Direct-v0",
    entry_point="vehicle_rl.envs.tracking_env:TrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "vehicle_rl.tasks.tracking.entry_points:tracking_env_cfg_factory",
        "rsl_rl_cfg_entry_point": "vehicle_rl.tasks.tracking.entry_points:rsl_rl_runner_cfg_factory",
    },
)
