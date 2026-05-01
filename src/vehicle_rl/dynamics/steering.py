"""Steering pinion-angle to tire-angle conversion.

A car driver / controller commands the steering column (pinion); the rack
& pinion gearbox + steering linkage translate that to the front-tire
steering angle. Phase 2 step 1 uses a fixed scalar ratio. Phase 4+ may
swap in a nonlinear / variable-ratio map (e.g., progressive rack, tabulated
lookup) without changing call sites.

Only the front-axle "center" steering angle is produced here; the per-wheel
Ackermann distribution (left/right) lives in
`tire_force.LinearFrictionCircleTire.per_wheel_steer`.

Wheel-order convention is unaffected (this module operates on a single
scalar per env).
"""
from __future__ import annotations

from typing import Protocol

from torch import Tensor


class SteeringModel(Protocol):
    """Bidirectional map between pinion angle and front-tire steering angle.

    Both directions are required so callers can convert physical limits
    (`delta_max` -> `pinion_max`) without assuming linearity.
    """

    def pinion_to_delta(self, pinion: Tensor) -> Tensor:
        """(N,) pinion angle [rad] -> (N,) front-tire angle [rad]."""
        ...

    def delta_to_pinion(self, delta: Tensor) -> Tensor:
        """(N,) front-tire angle [rad] -> (N,) pinion angle [rad]."""
        ...


class FixedRatioSteeringModel:
    """Linear constant-ratio steering: delta = pinion / ratio.

    Typical passenger-car overall steering ratio is 14--18 (one full hand-
    wheel turn produces ~20-26 deg of front-tire steer). Default 16.0.
    """

    def __init__(self, ratio: float = 16.0):
        if ratio <= 0.0:
            raise ValueError(f"steering ratio must be positive, got {ratio}")
        self.ratio = float(ratio)

    def pinion_to_delta(self, pinion: Tensor) -> Tensor:
        return pinion / self.ratio

    def delta_to_pinion(self, delta: Tensor) -> Tensor:
        return delta * self.ratio
