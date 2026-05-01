"""Virtual roll / pitch PD damper applied as body-frame torque on base_link.

The chassis-only rigid-body has no suspension, so its natural roll/pitch
mode is high-frequency and lightly damped (only the moment arm of the
horizontal tire forces about CoG resists tilt). This damper supplies the
"virtual suspension" required to keep roll / pitch in a sensible regime
during Phase 1.5 step 1.

PLAN.md guidance: start weak; tune stiffness up if oscillation is visible.
1.5-2 Hz / zeta ~= 0.3 is the Phase 3.5 target, not a Phase 1.5 step 1
requirement.

YAW IS NOT DAMPED -- yaw is the controlled DoF.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .state import VehicleState


class AttitudeDamper:
    def __init__(
        self,
        k_roll: float,
        c_roll: float,
        k_pitch: float,
        c_pitch: float,
    ):
        self.k_roll = float(k_roll)
        self.c_roll = float(c_roll)
        self.k_pitch = float(k_pitch)
        self.c_pitch = float(c_pitch)

    def compute(self, state: VehicleState) -> Tensor:
        roll = state.rpy[..., 0]
        pitch = state.rpy[..., 1]
        wx = state.angvel_body[..., 0]
        wy = state.angvel_body[..., 1]
        tau_x = -self.k_roll * roll - self.c_roll * wx
        tau_y = -self.k_pitch * pitch - self.c_pitch * wy
        tau_z = torch.zeros_like(tau_x)
        return torch.stack([tau_x, tau_y, tau_z], dim=-1)   # (N, 3) body frame
