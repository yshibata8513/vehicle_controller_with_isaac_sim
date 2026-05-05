"""Observation / Action / GT-state schemas shared between Phase 2 (classical
controllers: Pure Pursuit + PID) and Phase 3 (RL).

Conventions:
- N is the number of parallel envs (single-env Phase 2 uses N=1).
- All wheel-indexed tensors are ordered [FL, FR, RL, RR].
- `VehicleObservation` contains only **path-relative** signals: IMU,
  steering-column encoder, lateral/heading error vs the path, and a
  body-frame lookahead window. World-frame absolute pose (`pos_xy`, `yaw`)
  is intentionally excluded -- a real instrumented car has GPS, but for
  controller / policy design we want translation/rotation-invariant
  inputs so the same policy generalizes across courses, env-origin
  offsets, and (Phase 3 DR) randomized course placements. The absolute
  pose remains available on `VehicleStateGT` for metrics, reset logic,
  and termination checks. See docs/phase3_training_review.md item 7.
- The front-tire steering angle (`delta_actual`) and lateral velocity
  (`vy`) are NOT observable on a real car -- they live in
  `VehicleStateGT` only.
- `VehicleStateGT` is the simulator-side ground truth, used only for
  evaluation metrics and as input to the dynamics modules. Never pass GT
  fields to a controller / policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path as _PyPath
from typing import TYPE_CHECKING

from torch import Tensor

from vehicle_rl import VEHICLE_RL_ROOT
from vehicle_rl.config.isaac_adapter import make_action_limits
from vehicle_rl.config.loader import load_yaml_strict

if TYPE_CHECKING:
    from vehicle_rl.planner.plan import Plan


# Action ranges (used by controllers and gym Box construction in Phase 3).
# `pinion_target` range is computed in the action term factory from
# DELTA_MAX_RAD * STEERING_RATIO; we don't hard-code it here.
#
# PR 2 (YAML refactor): the literal numeric bounds now live in
# configs/dynamics/linear_friction_circle_flat.yaml and are pulled in at
# import time via the adapter. The names are kept as module-level
# constants for backwards compatibility with tracking_env.py (PR 3 scope);
# new consumers should call `make_action_limits(dynamics_bundle)` from the
# adapter directly with their experiment's resolved dynamics bundle.
_DEFAULT_DYNAMICS_YAML = (
    _PyPath(VEHICLE_RL_ROOT) / "configs" / "dynamics" / "linear_friction_circle_flat.yaml"
)
_A_X_MIN, _A_X_MAX = make_action_limits(load_yaml_strict(_DEFAULT_DYNAMICS_YAML))
A_X_TARGET_MIN = _A_X_MIN
A_X_TARGET_MAX = _A_X_MAX


@dataclass
class VehicleObservation:
    """What a controller / RL policy is allowed to see (path-relative only)."""

    # IMU
    vx: Tensor          # (N,) body-frame longitudinal velocity [m/s]
    yaw_rate: Tensor    # (N,) [rad/s]
    ax: Tensor          # (N,) body-frame longitudinal accel [m/s^2]
    ay: Tensor          # (N,) body-frame lateral accel [m/s^2]
    roll: Tensor        # (N,) [rad]
    pitch: Tensor       # (N,) [rad]

    # Steering-column encoder. Post-actuator (first-order lag output),
    # before the SteeringModel; NOT the front-tire angle.
    pinion_angle: Tensor    # (N,) [rad]

    # Path-following errors, computed from `plan` + observed pose.
    lateral_error: Tensor   # (N,) signed cross-track distance [m]
    heading_error: Tensor   # (N,) tangent-frame yaw error [rad]

    # Reference path: K-point lookahead window in body frame (K=10 default;
    # subject to change). The planner handles the world->body projection so
    # the policy never needs the world-frame course.
    plan: "Plan"


@dataclass
class VehicleAction:
    """Controller / policy output. Goes through actuator first-order lag."""

    pinion_target: Tensor   # (N,) [rad]; physical limit = DELTA_MAX_RAD * STEERING_RATIO
    a_x_target: Tensor      # (N,) [m/s^2]; range [A_X_TARGET_MIN, A_X_TARGET_MAX]


@dataclass
class VehicleStateGT:
    """Simulator ground truth.

    Used for: (a) evaluation metrics, (b) input to dynamics modules
    (`StaticNormalLoadModel`, `LinearFrictionCircleTire`, `injector`),
    (c) generating `VehicleObservation` via `envs.sensors.build_observation`,
    (d) reset / termination logic that needs world-frame pose.

    Three groups of signals coexist here:
      - IMU-equivalent (vx, yaw_rate, ax, ay, roll, pitch): the policy-visible
        subset is mirrored into `VehicleObservation`.
      - GPS-equivalent absolute pose (`pos_xyz`, `quat_wxyz`, world `yaw`):
        kept here for metrics / reset / termination but **intentionally
        hidden from the policy** -- the observation uses path-relative
        signals only (`lateral_error`, `heading_error`, body-frame plan)
        so the policy generalizes across courses and randomized origins.
        See module docstring + docs/phase3_training_review.md item 7.
      - Not-observable-on-a-real-car (tire angle, vy, mu, per-wheel forces):
        physics internals exposed for diagnostics, never on observation.
    """

    # World-frame pose / motion
    pos_xyz: Tensor         # (N, 3) [m]
    quat_wxyz: Tensor       # (N, 4)
    vel_world: Tensor       # (N, 3) [m/s]
    angvel_world: Tensor    # (N, 3) [rad/s]

    # Body-frame derived quantities
    vel_body: Tensor        # (N, 3) [m/s] -- vy lives here, NOT in observation
    angvel_body: Tensor     # (N, 3) [rad/s]
    rpy: Tensor             # (N, 3) [rad]
    ax_body: Tensor         # (N,) longitudinal accel from sum of tire forces [m/s^2]
    ay_body: Tensor         # (N,) lateral accel from sum of tire forces [m/s^2]

    # Actuator internals (none of these are exposed to the policy)
    pinion_actual: Tensor   # (N,) [rad]; first-order lag output of pinion_target
    delta_actual: Tensor    # (N,) [rad]; tire angle = SteeringModel(pinion_actual)
    a_x_actual: Tensor      # (N,) [m/s^2]; first-order lag output of a_x_target

    # Per-wheel quantities (computed by dynamics modules each step)
    mu_per_wheel: Tensor    # (N, 4)
    Fz_per_wheel: Tensor    # (N, 4) [N]
    Fx_per_wheel: Tensor    # (N, 4) [N], tire-frame longitudinal
    Fy_per_wheel: Tensor    # (N, 4) [N], tire-frame lateral
    slip_angle: Tensor      # (N, 4) [rad]
    omega_wheel: Tensor     # (N, 4) [rad/s], wheel spin rates from PhysX joint state

    # NOTE: path-following errors (lateral_error, heading_error) are NOT
    # part of the physics-only GT. They are derived features of (state, plan)
    # and are computed by `envs.sensors.compute_path_errors`. The same
    # function is used by `build_observation` (with optional noise) and by
    # the metrics evaluator (clean) -- single source of truth.
