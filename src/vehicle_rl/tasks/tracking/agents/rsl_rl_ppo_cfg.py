"""rsl_rl PPO config for `Vehicle-Tracking-Direct-v0`.

Stage 0 defaults follow docs/phase3_training_review.md item 5:
  - num_steps_per_env=64 (1.28 s rollout @ 50 Hz; circle lap is ~19 s, so
    PPO sees ~7% of a lap per rollout vs. <3% with the prior 24)
  - gamma=0.995 (effective horizon ~200 steps, matches lap-scale credit)
  - max_iterations=300 (sanity Stage 0a; raise for real training)
  - actor/critic [256, 256] elu, clip 0.2, lr 3e-4, entropy 0.005
  - clip_actions=1.0 to keep policy action in [-1, 1]
"""
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class TrackingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 300
    save_interval = 50
    experiment_name = "vehicle_tracking_direct"
    clip_actions = 1.0
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy = RslRlPpoActorCriticCfg(
        # Lower than the Cartpole-Direct default of 1.0: with 1.0 the
        # initial random pinion samples ±1 → ±9.78 rad steer command, and
        # while the actuator's tau_steer=50 ms filters fast oscillation,
        # the running mean still puts the vehicle into max-lock turns
        # before the policy can learn anything useful. 0.3 keeps initial
        # exploration in the linear-tire regime.
        init_noise_std=0.3,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
