"""Sedan ArticulationCfg for vehicle_rl (case-B / Phase 1).

SEDAN_CFG is the real-vehicle-scale chassis used throughout the project:
  - PhysX rigid body for the chassis (base_link)
  - Visual-only wheels (no collision in URDF)
  - Steering joints kept for actuator integration (Phase 1.5 sends position
    targets after a first-order lag on delta_target)
  - Spin joints kept as state variables (used in Phase 1.5 step 2 for slip
    ratio computation; in step 1 they freewheel and are not load-bearing)

Tire forces are injected externally by `vehicle_rl.dynamics`. This module
only defines the articulation; it does not compute or apply forces.

PR 2 (YAML config externalization): all numeric constants and the
ArticulationCfg are now sourced from `configs/vehicles/sedan.yaml` via
`vehicle_rl.config.isaac_adapter.make_sedan_cfg`. The module-level names
below (`WHEELBASE`, `TOTAL_MASS`, `SEDAN_CFG`, ...) are populated at
import time from that YAML so every existing consumer keeps working
without restating the values. PR 3 will rewrite consumers
(`tracking_env.py`) to read the bundle directly and these compatibility
re-exports will go away.
"""
from __future__ import annotations

from pathlib import Path

from vehicle_rl import VEHICLE_RL_ROOT
from vehicle_rl.config.loader import load_yaml_strict
from vehicle_rl.config.isaac_adapter import (
    derived_pinion_max,
    make_sedan_cfg,
    make_vehicle_geometry,
)


# Repo-relative location of the canonical vehicle bundle. Single source of
# truth for SEDAN_CFG and the convenience module-level constants below.
_DEFAULT_VEHICLE_YAML = Path(VEHICLE_RL_ROOT) / "configs" / "vehicles" / "sedan.yaml"


def _load_default_vehicle_bundle() -> dict:
    return load_yaml_strict(_DEFAULT_VEHICLE_YAML)


# Joint name conventions (must match assets/urdf/sedan.urdf). Kept as module
# constants because string regexes are not "tunable defaults"; they are part
# of the asset's structural identity.
_BUNDLE = _load_default_vehicle_bundle()
STEER_JOINT_REGEX = _BUNDLE["joints"]["steer_regex"]
WHEEL_JOINT_REGEX = _BUNDLE["joints"]["wheel_regex"]

# Vehicle dimensions (kept in sync with the URDF via configs/vehicles/sedan.yaml).
# All numeric values below are sourced from the YAML at import time; no
# tunable literals live in this file. (PR 3 will rewrite tracking_env.py to
# read the bundle directly and these compatibility re-exports go away.)
_GEOM = make_vehicle_geometry(_BUNDLE)
WHEELBASE = _GEOM["wheelbase_m"]
TRACK = _GEOM["track_m"]
WHEEL_RADIUS = _GEOM["wheel_radius_m"]
WHEEL_WIDTH = _GEOM["wheel_width_m"]
TOTAL_MASS = _GEOM["total_kg"]
COG_Z_DEFAULT = _GEOM["cog_z_m"]

# Steering: front-tire angle limit and column-side overall gear ratio.
# pinion_max = DELTA_MAX_RAD * STEERING_RATIO is the derived limit (computed
# at runtime in VehicleSimulator.pinion_max).
DELTA_MAX_RAD = float(_BUNDLE["steering"]["delta_max_rad"])
STEERING_RATIO = float(_BUNDLE["steering"]["steering_ratio"])

# Convenience derived constant for the few callers that need pinion_max
# without constructing a VehicleSimulator (kept for backwards compatibility
# until PR 3 wires this through the adapter explicitly).
PINION_MAX = derived_pinion_max(_BUNDLE)


def _build_default_sedan_cfg():
    """Construct SEDAN_CFG from the default vehicle bundle.

    Wrapped in a function so the lazy isaaclab import inside `make_sedan_cfg`
    is deferred until first attribute access -- otherwise importing this
    module without Isaac Sim available would fail. The result is cached on
    first access via `__getattr__` below.
    """
    return make_sedan_cfg(_BUNDLE)


_SEDAN_CFG_CACHE = None


def __getattr__(name: str):
    """PEP 562: lazily build SEDAN_CFG on first attribute access.

    Allows `from vehicle_rl.assets import SEDAN_CFG` to work without paying
    the isaaclab import cost when only the geometry constants are needed
    (e.g. unit tests, classical pure-pursuit gain computation).
    """
    global _SEDAN_CFG_CACHE
    if name == "SEDAN_CFG":
        if _SEDAN_CFG_CACHE is None:
            _SEDAN_CFG_CACHE = _build_default_sedan_cfg()
        return _SEDAN_CFG_CACHE
    raise AttributeError(f"module 'vehicle_rl.assets.sedan' has no attribute {name!r}")
