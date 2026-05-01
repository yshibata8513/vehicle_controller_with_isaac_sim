"""Tire normal-load (Fz) models (case-B / Phase 1.5).

The Protocol allows swapping `StaticNormalLoadModel` (Phase 1.5 step 1, flat-ground
+ analytic weight transfer) with `RaycastNormalLoadModel` (Phase 4, raycast +
spring-damper) without changing the env or tire-force code.
"""
from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from .state import VehicleState


class NormalLoadModel(Protocol):
    def compute(self, state: VehicleState) -> Tensor:
        """Returns (N, 4) tire vertical loads Fz_i in newtons.

        Wheel order: [FL, FR, RL, RR]. Fz is non-negative; the model is
        responsible for clipping if a wheel would otherwise lift off.
        """
        ...


class StaticNormalLoadModel:
    """Static + simple weight-transfer Fz model.

    Assumes:
      - Flat ground; path normal = world z.
      - All four wheels in contact (no liftoff).
      - Weight transfer driven by `a_x_actual` (longitudinal) and
        `a_y_estimate` (lateral; supplied by caller to break the
        Fz <-> a_y feedback loop).

    Sum of Fz equals m*g exactly under the analytic formula; an optional
    z-drift PD adds a small correction distributed equally to all four wheels:
        F_z_correction_total = -kz * (z_base - z_ref) - cz * vz_world
    The PD is meant to suppress integration drift only; gains should be
    much weaker than the implicit "spring" formed by the load distribution.
    """

    def __init__(
        self,
        mass: float,
        wheelbase: float,
        track: float,
        h_cg: float,
        a_front: float,
        z_ref: float,
        gravity: float = 9.81,
        z_drift_kp: float = 0.0,
        z_drift_kd: float = 0.0,
    ):
        self.mass = float(mass)
        self.L = float(wheelbase)
        self.T = float(track)
        self.h_cg = float(h_cg)
        self.a = float(a_front)
        self.b = float(wheelbase) - float(a_front)
        self.z_ref = float(z_ref)
        self.g = float(gravity)
        self.z_drift_kp = float(z_drift_kp)
        self.z_drift_kd = float(z_drift_kd)

    def compute(self, state: VehicleState) -> Tensor:
        m, g = self.mass, self.g
        L, T, h, a, b = self.L, self.T, self.h_cg, self.a, self.b

        Fz_static_front = m * g * b / L / 2.0   # per front wheel
        Fz_static_rear  = m * g * a / L / 2.0   # per rear wheel

        ax = state.a_x_actual                   # (N,)
        ay = state.a_y_estimate                 # (N,)

        # Longitudinal weight transfer: total m*ax*h/L shifts front -> rear.
        # Per-wheel: front wheels lose half each, rear wheels gain half each.
        dFz_long = m * ax * h / L               # (N,)

        # Lateral weight transfer: per-wheel.
        dFz_lat_f = m * ay * h / T * (b / L)    # (N,)
        dFz_lat_r = m * ay * h / T * (a / L)    # (N,)

        # Wheel order: FL (+y), FR (-y), RL (+y), RR (-y).
        Fz_FL = Fz_static_front - 0.5 * dFz_long - dFz_lat_f
        Fz_FR = Fz_static_front - 0.5 * dFz_long + dFz_lat_f
        Fz_RL = Fz_static_rear  + 0.5 * dFz_long - dFz_lat_r
        Fz_RR = Fz_static_rear  + 0.5 * dFz_long + dFz_lat_r
        Fz = torch.stack([Fz_FL, Fz_FR, Fz_RL, Fz_RR], dim=-1)   # (N, 4)

        # z-drift correction distributed equally to all four wheels.
        if self.z_drift_kp > 0.0 or self.z_drift_kd > 0.0:
            z_err = state.pos_world[..., 2] - self.z_ref
            vz = state.vel_world[..., 2]
            F_corr_total = -self.z_drift_kp * z_err - self.z_drift_kd * vz   # (N,)
            Fz = Fz + (F_corr_total / 4.0).unsqueeze(-1)

        # Tires can't pull on the ground.
        Fz = torch.clamp(Fz, min=0.0)
        return Fz
