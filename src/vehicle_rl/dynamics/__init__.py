"""Case-B / Phase 1.5 vehicle dynamics modules.

Submodels are intentionally split so they can be swapped independently:

  StaticNormalLoadModel       -- Phase 1.5 step 1 (flat ground + analytic weight transfer)
  RaycastNormalLoadModel      -- Phase 4 (raycast + spring-damper)         [not implemented]
  LinearFrictionCircleTire    -- Phase 1.5 step 1 (linear C_alpha, |F| <= mu*Fz)
  FialaTire                   -- Phase 1.5 step 2 (slip-ratio + slip-angle) [not implemented]

All wheel-indexed tensors use order [FL, FR, RL, RR].
"""
from .state import VehicleState, quat_wxyz_to_rotmat, quat_wxyz_to_rpy
from .actuator import FirstOrderLagActuator
from .normal_load import NormalLoadModel, StaticNormalLoadModel
from .tire_force import TireForceModel, LinearFrictionCircleTire
from .attitude_damper import AttitudeDamper
from .injector import aggregate_tire_forces_to_base_link
from .steering import SteeringModel, FixedRatioSteeringModel

__all__ = [
    "VehicleState",
    "quat_wxyz_to_rotmat",
    "quat_wxyz_to_rpy",
    "FirstOrderLagActuator",
    "NormalLoadModel",
    "StaticNormalLoadModel",
    "TireForceModel",
    "LinearFrictionCircleTire",
    "AttitudeDamper",
    "aggregate_tire_forces_to_base_link",
    "SteeringModel",
    "FixedRatioSteeringModel",
]
