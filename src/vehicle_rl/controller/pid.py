"""PI longitudinal speed controller.

Action: target longitudinal acceleration `a_x_target` based on speed error.
No D term -- vx is computed from PhysX velocity which is already smoothed,
and adding D is sensitive to numerical noise.

Anti-windup: integral is clamped to `[-integral_max, +integral_max]` so
the integral contribution `ki * I` stays well within the action box even
under prolonged saturation.

State: per-env integrator (Tensor, shape (num_envs,)). Reset before each
episode (or per-env reset on Phase 3 partial resets).
"""
from __future__ import annotations

import torch
from torch import Tensor

from vehicle_rl.envs.types import VehicleObservation


class PIDSpeedController:
    """Stateful PI speed controller (one integrator per env)."""

    def __init__(
        self,
        *,
        num_envs: int,
        dt: float,
        kp: float = 1.0,
        ki: float = 0.3,
        integral_max: float = 10.0,
        a_x_min: float = -5.0,
        a_x_max: float = 3.0,
        device: torch.device | str = "cpu",
    ):
        if num_envs <= 0 or dt <= 0.0:
            raise ValueError("num_envs and dt must be positive")
        if a_x_min >= a_x_max:
            raise ValueError(f"a_x_min ({a_x_min}) must be < a_x_max ({a_x_max})")
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.kp = float(kp)
        self.ki = float(ki)
        self.integral_max = float(integral_max)
        self.a_x_min = float(a_x_min)
        self.a_x_max = float(a_x_max)
        self._integral = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: Tensor | None = None) -> None:
        if env_ids is None:
            self._integral.zero_()
        else:
            self._integral[env_ids] = 0.0

    def __call__(
        self,
        obs: VehicleObservation,
        target_speed: float | Tensor,
    ) -> Tensor:
        """Returns a_x_target, shape (N,) in [a_x_min, a_x_max]."""
        if not isinstance(target_speed, Tensor):
            target_speed = torch.full_like(obs.vx, float(target_speed))
        e = target_speed - obs.vx                                       # (N,)
        self._integral = (self._integral + e * self.dt).clamp(
            -self.integral_max, self.integral_max
        )
        u = self.kp * e + self.ki * self._integral
        return u.clamp(self.a_x_min, self.a_x_max)

    @property
    def integral(self) -> Tensor:
        return self._integral
