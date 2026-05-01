"""Tire force models (case-B / Phase 1.5).

Phase 1.5 step 1: `LinearFrictionCircleTire` -- linear cornering stiffness
in the lateral direction, drive/brake force as a direct command, with the
combined |F| <= mu*Fz friction-circle clip applied at the end.

Phase 1.5 step 2 will add `FialaTire` using slip ratio kappa as well.

Wheel order convention: [FL, FR, RL, RR].
"""
from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from .state import VehicleState


class TireForceModel(Protocol):
    def compute(
        self,
        state: VehicleState,
        Fz: Tensor,           # (N, 4)
        mu: Tensor,           # (N, 4)
        fx_command: Tensor,   # (N, 4) tire-frame longitudinal command per wheel
    ) -> tuple[Tensor, Tensor]:
        """Returns (Fx_tire, Fy_tire) in TIRE frame, shape (N, 4) each."""
        ...

    def tire_positions_body(self) -> Tensor:
        """(4, 3) tire contact points in body frame, [FL, FR, RL, RR]."""
        ...

    def per_wheel_steer(self, delta_actual: Tensor) -> Tensor:
        """(N, 4) per-wheel steering angle (no Ackermann correction in step 1)."""
        ...


class LinearFrictionCircleTire:
    """Lateral force from linear cornering stiffness, longitudinal force from
    direct command, then both clipped to the friction circle |F| <= mu*Fz.

    Slip angle:
        alpha_i = atan(v_lat_i / max(|v_long_i|, eps))
    using |v_long| in the denominator so the sign of alpha is determined by
    v_lat alone (consistent with forward driving). Reverse driving is not
    a target regime for Phase 1.5 step 1.

    Lateral force convention (ISO):
        Fy_tire = -C_alpha * alpha
    so that positive alpha (lateral velocity in +y_tire direction) produces
    a force in -y_tire direction, opposing the slip.
    """

    def __init__(
        self,
        cornering_stiffness: float,   # C_alpha (N/rad), per wheel
        wheelbase: float,
        track: float,
        a_front: float,
        h_cg: float,
        eps_vlong: float = 1e-2,
    ):
        self.C_alpha = float(cornering_stiffness)
        self.L = float(wheelbase)
        self.T = float(track)
        self.a = float(a_front)
        self.b = float(wheelbase) - float(a_front)
        self.h_cg = float(h_cg)
        self.eps = float(eps_vlong)
        # Tire contact-point positions in body frame, order [FL, FR, RL, RR].
        # z = -h_cg places the contact at ground level (base_link is at +h_cg).
        self._r_body = torch.tensor([
            [+self.a, +self.T / 2.0, -self.h_cg],
            [+self.a, -self.T / 2.0, -self.h_cg],
            [-self.b, +self.T / 2.0, -self.h_cg],
            [-self.b, -self.T / 2.0, -self.h_cg],
        ], dtype=torch.float32)

    def tire_positions_body(self) -> Tensor:
        return self._r_body

    def per_wheel_steer(self, delta_actual: Tensor) -> Tensor:
        # Step 1: parallel steering (no Ackermann). Front wheels both get
        # delta_actual; rear wheels get 0.
        zero = torch.zeros_like(delta_actual)
        return torch.stack([delta_actual, delta_actual, zero, zero], dim=-1)

    def compute(
        self,
        state: VehicleState,
        Fz: Tensor,
        mu: Tensor,
        fx_command: Tensor,
    ) -> tuple[Tensor, Tensor]:
        device = Fz.device
        if self._r_body.device != device:
            self._r_body = self._r_body.to(device)

        N = state.vel_body.shape[0]

        # Tire-position velocity in body frame: v_tire_body = v_body + omega_body x r_body.
        v_body = state.vel_body.unsqueeze(1)              # (N, 1, 3)
        w_body = state.angvel_body.unsqueeze(1)           # (N, 1, 3)
        r = self._r_body.unsqueeze(0).expand(N, -1, -1)   # (N, 4, 3)
        v_tire_body = v_body.expand(-1, 4, -1) + torch.cross(
            w_body.expand(-1, 4, -1), r, dim=-1
        )   # (N, 4, 3)

        # Project body-frame velocity onto tire frame (rotate by -delta_per_wheel about z).
        delta_per_wheel = self.per_wheel_steer(state.delta_actual)   # (N, 4)
        cos_d = torch.cos(delta_per_wheel)
        sin_d = torch.sin(delta_per_wheel)
        vx_b = v_tire_body[..., 0]
        vy_b = v_tire_body[..., 1]
        v_long = cos_d * vx_b + sin_d * vy_b
        v_lat = -sin_d * vx_b + cos_d * vy_b

        # Slip angle, using |v_long| in denominator (forward-driving convention).
        denom = torch.clamp(torch.abs(v_long), min=self.eps)
        alpha = torch.atan(v_lat / denom)

        Fy_raw = -self.C_alpha * alpha            # (N, 4)
        Fx_raw = fx_command                        # (N, 4)

        F_norm = torch.sqrt(Fx_raw * Fx_raw + Fy_raw * Fy_raw + 1e-12)
        F_max = mu * Fz
        scale = torch.where(F_norm > F_max, F_max / F_norm, torch.ones_like(F_norm))
        Fx = Fx_raw * scale
        Fy = Fy_raw * scale
        return Fx, Fy
