"""Vehicle-asset re-exports.

PR 2 round-1 fix (review finding 4): `SEDAN_CFG` is **not** eagerly
imported at package import time, because resolving it triggers
`isaaclab.sim` import via `make_sedan_cfg`. Instead this package's
`__getattr__` (PEP 562) lazily fetches `SEDAN_CFG` from `.sedan` only on
first access, so plain `import vehicle_rl.assets` (e.g. from a unit test
or a non-sim consumer that only needs the geometry constants) does not
require Isaac Sim to be importable.

The geometry / steering / joint constants below ARE eagerly exposed
because they only require PyYAML to read `configs/vehicles/sedan.yaml` —
no Isaac dependency.
"""
from __future__ import annotations

from .sedan import (
    COG_Z_DEFAULT,
    DELTA_MAX_RAD,
    PINION_MAX,
    STEER_JOINT_REGEX,
    STEERING_RATIO,
    TOTAL_MASS,
    TRACK,
    WHEEL_JOINT_REGEX,
    WHEEL_RADIUS,
    WHEEL_WIDTH,
    WHEELBASE,
)

__all__ = [
    "COG_Z_DEFAULT",
    "DELTA_MAX_RAD",
    "PINION_MAX",
    "SEDAN_CFG",
    "STEER_JOINT_REGEX",
    "STEERING_RATIO",
    "TOTAL_MASS",
    "TRACK",
    "WHEEL_JOINT_REGEX",
    "WHEEL_RADIUS",
    "WHEEL_WIDTH",
    "WHEELBASE",
]


def __getattr__(name: str):
    """PEP 562: lazily fetch SEDAN_CFG to defer the Isaac Lab import.

    `from vehicle_rl.assets import SEDAN_CFG` triggers this, which then
    routes through `vehicle_rl.assets.sedan.__getattr__` to actually
    build the cfg via `make_sedan_cfg` (and therefore import isaaclab).
    """
    if name == "SEDAN_CFG":
        from . import sedan as _sedan
        return _sedan.SEDAN_CFG
    raise AttributeError(f"module 'vehicle_rl.assets' has no attribute {name!r}")
