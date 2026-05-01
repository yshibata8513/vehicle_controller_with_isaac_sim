"""Pure Pursuit lateral controller.

Pure Pursuit picks a lookahead point at distance L_d on the reference path
ahead of the vehicle, and computes the steering angle that would produce
a circular arc through that point starting tangent to the current heading:

    delta = atan2(2 * L_wheel * y_la, x_la^2 + y_la^2)

where (x_la, y_la) is the lookahead point in the body frame and L_wheel
is the wheelbase. We then convert the front-tire angle to a pinion target
via the fixed steering ratio.

Lookahead distance is speed-dependent (`L_d = max(L_min, gain * v)`) for
stability across the speed range. Per-env values are gathered from the
body-frame Plan so this is fully batched (no Python loops).
"""
from __future__ import annotations

import torch
from torch import Tensor

from vehicle_rl.envs.types import VehicleObservation


class PurePursuitController:
    """Stateless lateral controller; one instance can serve any num_envs."""

    def __init__(
        self,
        *,
        wheelbase: float,
        steering_ratio: float,
        pinion_max: float,
        lookahead_min: float = 2.0,
        lookahead_gain: float = 0.5,
        lookahead_ds: float = 1.0,
    ):
        if wheelbase <= 0.0 or steering_ratio <= 0.0 or pinion_max <= 0.0:
            raise ValueError("wheelbase, steering_ratio, pinion_max must all be positive")
        if lookahead_min <= 0.0 or lookahead_gain <= 0.0:
            raise ValueError("lookahead parameters must all be positive")
        if lookahead_ds <= 0.0:
            raise ValueError("lookahead_ds must be positive")
        self.wheelbase = float(wheelbase)
        self.steering_ratio = float(steering_ratio)
        self.pinion_max = float(pinion_max)
        self.lookahead_min = float(lookahead_min)
        self.lookahead_gain = float(lookahead_gain)
        self.lookahead_ds = float(lookahead_ds)

    def __call__(self, obs: VehicleObservation) -> Tensor:
        """Returns pinion_target, shape (N,) in [-pinion_max, +pinion_max]."""
        plan = obs.plan
        N = plan.num_envs
        K = plan.num_points
        device = plan.device

        # Speed-dependent lookahead distance (clip negative v to 0 -- vehicle
        # going backward picks the same min-distance preview).
        v_pos = obs.vx.clamp(min=0.0)
        L_d = (self.lookahead_gain * v_pos).clamp(min=self.lookahead_min)   # (N,)
        # Map L_d to a Plan waypoint index. Always look at least one step
        # ahead (>= 1), and never beyond the last waypoint.
        idx = (L_d / self.lookahead_ds).round().long().clamp(min=1, max=K - 1)

        batch = torch.arange(N, device=device)
        x_la = plan.x[batch, idx]   # (N,)
        y_la = plan.y[batch, idx]

        L_sq = x_la * x_la + y_la * y_la + 1e-6
        delta = torch.atan2(2.0 * self.wheelbase * y_la, L_sq)
        pinion = (delta * self.steering_ratio).clamp(-self.pinion_max, self.pinion_max)
        return pinion
