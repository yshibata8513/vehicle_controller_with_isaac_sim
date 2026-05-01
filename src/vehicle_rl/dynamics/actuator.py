"""First-order lag actuator (case-B / Phase 1.5 step 1).

Discrete-time exponential integrator:
    y_next = y + (1 - exp(-dt / tau)) * (u - y)

Stable for any dt regardless of tau (no Euler stiffness).

Used for:
  - Steering: tau_steer ~ 50 ms (single tau)
  - Longitudinal: tau_drive ~ 200 ms (positive command), tau_brake ~ 70 ms (negative)
"""
from __future__ import annotations

import torch
from torch import Tensor


class FirstOrderLagActuator:
    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        tau_pos: float,
        tau_neg: float | None = None,
        initial_value: float = 0.0,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.tau_pos = float(tau_pos)
        self.tau_neg = float(tau_neg) if tau_neg is not None else float(tau_pos)
        self._y = torch.full((num_envs,), float(initial_value), device=self.device)

    def step(self, u: Tensor, dt: float) -> Tensor:
        """Advance one step. u shape (N,). Returns the new actuator value."""
        tau_pos_t = torch.tensor(self.tau_pos, device=self.device)
        tau_neg_t = torch.tensor(self.tau_neg, device=self.device)
        # Split tau by sign of TARGET command (drive vs brake intent), not by error sign.
        tau = torch.where(u >= 0, tau_pos_t, tau_neg_t)
        alpha = 1.0 - torch.exp(-dt / tau)
        self._y = self._y + alpha * (u - self._y)
        return self._y

    def reset(self, value: float = 0.0, env_ids: Tensor | None = None) -> None:
        if env_ids is None:
            self._y[:] = value
        else:
            self._y[env_ids] = value

    @property
    def value(self) -> Tensor:
        return self._y
