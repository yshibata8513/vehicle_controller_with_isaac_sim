"""Path-tracking RL tasks (Stage 0+).

Registers `Vehicle-Tracking-Direct-v0` (Stage 0 default: circle, μ=0.9,
64 envs). Future stages (μ randomization, multi-course) will register
additional ids here.
"""
import gymnasium as gym

from . import agents

gym.register(
    id="Vehicle-Tracking-Direct-v0",
    entry_point="vehicle_rl.envs.tracking_env:TrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "vehicle_rl.envs.tracking_env:TrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TrackingPPORunnerCfg",
    },
)
