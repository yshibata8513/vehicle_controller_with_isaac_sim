"""Aggregate per-tire forces into a single body-frame (force, torque) pair
to be applied at base_link via `Articulation.set_external_force_and_torque`.

Approximation in case-B Phase 1.5 step 1: Fz is applied in BODY-z direction
(not world-z). For small roll/pitch this matches the analytic flat-ground
StaticNormalLoadModel within ~1% per degree of tilt; the attitude damper
keeps tilt small enough that this is acceptable.

Forces and torques returned are both in the BODY frame, matching Isaac Lab's
`set_external_force_and_torque` convention (forces stored in `_external_force_b`).
"""
from __future__ import annotations

import torch
from torch import Tensor


def aggregate_tire_forces_to_base_link(
    Fx_tire: Tensor,           # (N, 4) tire-frame longitudinal forces
    Fy_tire: Tensor,           # (N, 4) tire-frame lateral forces
    Fz: Tensor,                # (N, 4) body-z magnitude (positive = upward in body)
    delta_per_wheel: Tensor,   # (N, 4) steering angle per wheel (radians)
    r_body: Tensor,            # (4, 3) tire positions in body frame [FL, FR, RL, RR]
) -> tuple[Tensor, Tensor]:
    """Returns (force_body, torque_body), both shape (N, 3)."""
    N = Fx_tire.shape[0]
    cos_d = torch.cos(delta_per_wheel)
    sin_d = torch.sin(delta_per_wheel)

    # Rotate tire-frame Fx, Fy into body frame (rotation about body-z by +delta).
    Fx_body = cos_d * Fx_tire - sin_d * Fy_tire
    Fy_body = sin_d * Fx_tire + cos_d * Fy_tire
    Fz_body = Fz
    F_per_tire = torch.stack([Fx_body, Fy_body, Fz_body], dim=-1)   # (N, 4, 3)

    F_total = F_per_tire.sum(dim=1)                                  # (N, 3)
    r = r_body.unsqueeze(0).expand(N, -1, -1)                        # (N, 4, 3)
    tau_total = torch.cross(r, F_per_tire, dim=-1).sum(dim=1)        # (N, 3)
    return F_total, tau_total
