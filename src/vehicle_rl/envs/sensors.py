"""Observation construction from VehicleStateGT, body-frame Plan, and the
planner-supplied path-following errors.

Single source of truth for the (state_gt, plan, errors) -> observation
mapping. Both Phase 2 classical controllers and Phase 3 RL policies see
the same observation.

Path errors (lateral_error, heading_error) are computed by the **planner**
during the projection from world-frame course to body-frame lookahead
window: the planner already finds the closest point on the full course
(needed to start the K-point window) so it can return errors in the same
pass. Recomputing them here from a body-frame lookahead window would be
both redundant and ill-defined (the lookahead window typically excludes
points behind the vehicle, so a window-local "closest" is not the true
projection).

Sign conventions (set by the planner):
  lateral_error > 0: vehicle is to the LEFT of the path
                     (closest path point is to the RIGHT in body frame)
  heading_error > 0: path tangent at the projection point heads LEFT of
                     the vehicle's forward direction
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from vehicle_rl.envs.types import VehicleObservation, VehicleStateGT
from vehicle_rl.planner.plan import Plan


@dataclass
class NoiseCfg:
    """Per-channel additive Gaussian noise (independent per step / per env).

    Each field is a scalar standard deviation; None disables that channel.
    Phase 2 default = no noise (controllers operate on clean GT). Phase 3
    domain-randomization will instantiate non-None values.
    """
    pos_xy_std: float | None = None      # m
    yaw_std: float | None = None         # rad
    vx_std: float | None = None          # m/s
    ax_std: float | None = None          # m/s^2
    ay_std: float | None = None          # m/s^2
    yaw_rate_std: float | None = None    # rad/s
    roll_std: float | None = None        # rad
    pitch_std: float | None = None       # rad
    pinion_std: float | None = None      # rad


def _add_noise(t: Tensor, std: float | None) -> Tensor:
    if std is None or std == 0.0:
        return t
    return t + torch.randn_like(t) * std


def build_observation(
    state_gt: VehicleStateGT,
    plan: Plan,
    lateral_error: Tensor,
    heading_error: Tensor,
    *,
    noise: NoiseCfg | None = None,
) -> VehicleObservation:
    """Assemble VehicleObservation from physics GT and planner outputs.

    `lateral_error` and `heading_error` are produced by the planner's
    projection (e.g., `Path.project(pos_xy, yaw, K)`), not recomputed here.

    Hidden from the policy: tire angle (`delta_actual`), lateral velocity
    (`vy = vel_body[:, 1]`), per-wheel quantities, mu, and actuator
    longitudinal internal (`a_x_actual`).
    """
    if noise is None:
        noise = NoiseCfg()

    return VehicleObservation(
        # IMU
        vx=_add_noise(state_gt.vel_body[:, 0], noise.vx_std),
        yaw_rate=_add_noise(state_gt.angvel_body[:, 2], noise.yaw_rate_std),
        ax=_add_noise(state_gt.ax_body, noise.ax_std),
        ay=_add_noise(state_gt.ay_body, noise.ay_std),
        roll=_add_noise(state_gt.rpy[:, 0], noise.roll_std),
        pitch=_add_noise(state_gt.rpy[:, 1], noise.pitch_std),
        # GPS
        pos_xy=_add_noise(state_gt.pos_xyz[:, :2], noise.pos_xy_std),
        yaw=_add_noise(state_gt.rpy[:, 2], noise.yaw_std),
        # Steering column encoder
        pinion_angle=_add_noise(state_gt.pinion_actual, noise.pinion_std),
        # Path-following errors (computed by the planner; passed through)
        lateral_error=lateral_error,
        heading_error=heading_error,
        # Reference path window
        plan=plan,
    )
