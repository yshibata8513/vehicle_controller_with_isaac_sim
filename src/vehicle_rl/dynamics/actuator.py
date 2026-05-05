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
        self._split_tau = self.tau_pos != self.tau_neg
        self._initial_value = float(initial_value)
        self._y = torch.full((num_envs,), self._initial_value, device=self.device)
        # dt-dependent alpha = 1 - exp(-dt/tau). Lazily computed and cached on
        # first step(); refreshed only if dt changes (it doesn't in current
        # usage, since SimulationContext's physics dt is fixed). Computing
        # on-device through `torch.tensor(tau)` + `torch.exp` mirrors the
        # original per-step expression exactly, so output is bit-identical
        # to the previous implementation.
        self._dt_cache: float = float("nan")
        self._alpha_pos_t: Tensor = torch.zeros((), device=self.device)
        self._alpha_neg_t: Tensor = torch.zeros((), device=self.device)

    def _refresh_alphas(self, dt: float) -> None:
        self._dt_cache = dt
        tau_pos_t = torch.tensor(self.tau_pos, device=self.device)
        self._alpha_pos_t = 1.0 - torch.exp(-dt / tau_pos_t)
        if self._split_tau:
            tau_neg_t = torch.tensor(self.tau_neg, device=self.device)
            self._alpha_neg_t = 1.0 - torch.exp(-dt / tau_neg_t)
        else:
            self._alpha_neg_t = self._alpha_pos_t

    def step(self, u: Tensor, dt: float) -> Tensor:
        """Advance one step. u shape (N,). Returns the new actuator value."""
        if dt != self._dt_cache:
            self._refresh_alphas(dt)
        if self._split_tau:
            # Split alpha by sign of TARGET command (drive vs brake intent).
            alpha = torch.where(u >= 0, self._alpha_pos_t, self._alpha_neg_t)
        else:
            alpha = self._alpha_pos_t   # 0-D tensor, broadcasts over (N,)
        self._y = self._y + alpha * (u - self._y)
        return self._y

    def reset(self, value: float | None = None, env_ids: Tensor | None = None) -> None:
        """Reset actuator state. If `value` is None (default), reverts to the
        `initial_value` saved at construction (so YAML
        `actuator_lag.initial_value` reaches both __init__ and reset).
        Callers may still pass an explicit numeric override.
        """
        target = self._initial_value if value is None else float(value)
        if env_ids is None:
            self._y[:] = target
        else:
            self._y[env_ids] = target

    @property
    def value(self) -> Tensor:
        return self._y
