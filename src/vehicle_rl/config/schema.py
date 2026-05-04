"""Schema dataclasses + validate_keys helper for vehicle_rl YAML configs.

PR 1 scope:
- Provide validate_keys() that compares dict key sets against dataclass fields.
- Define top-level dataclass shells for each YAML category as documentation.
  Detailed nested validation grows as each category is wired (PR 2-4).

Rules (per docs/yaml_config_structure_plan.md):
- Schemas have NO defaults. Every field must be present in the YAML.
- Optional fields must be present as `null` in YAML; the consumer decides
  what `null` means.
- validate_keys raises ValueError on missing or extra keys.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, get_type_hints


def validate_keys(
    data: dict[str, Any],
    schema_type: type,
    path: tuple[str, ...] = (),
) -> None:
    """Recursively check that `data` matches the dataclass schema's field set.

    Walks into nested dataclass-typed fields. Lists, dicts, and primitive
    types are not recursed (no per-element schema yet in PR 1).
    """
    if not is_dataclass(schema_type):
        raise TypeError(f"schema_type must be a dataclass: {schema_type!r}")
    loc = ".".join(path) if path else "<root>"
    if not isinstance(data, dict):
        raise ValueError(
            f"expected mapping at {loc}, got {type(data).__name__}"
        )

    type_hints = get_type_hints(schema_type)
    schema_keys = {f.name for f in fields(schema_type)}
    data_keys = set(data.keys())

    missing = schema_keys - data_keys
    if missing:
        raise ValueError(f"missing keys at {loc}: {sorted(missing)}")
    extra = data_keys - schema_keys
    if extra:
        raise ValueError(f"unknown keys at {loc}: {sorted(extra)}")

    for f in fields(schema_type):
        nested_type = type_hints.get(f.name, f.type)
        if is_dataclass(nested_type):
            validate_keys(data[f.name], nested_type, path=path + (f.name,))


# ---------------------------------------------------------------------------
# Documentation-only shells. Refined in PR 2-4 when each category is wired.
# Fields here mirror configs/*/*.yaml top-level keys so validate_keys() can
# at least catch typos at the section level.
# ---------------------------------------------------------------------------


@dataclass
class VehicleSchema:
    schema_version: int
    name: str
    asset: dict
    joints: dict
    geometry: dict
    mass: dict
    steering: dict
    physx: dict


@dataclass
class DynamicsSchema:
    schema_version: int
    model: str
    gravity_mps2: float
    friction: dict
    action_limits: dict
    actuator_lag: dict
    tire: dict
    normal_load: dict
    attitude_damper: dict


@dataclass
class EnvSchema:
    schema_version: int
    task_id: str
    timing: dict
    scene: dict
    spaces: dict
    planner: dict
    reset: dict
    action_scaling: dict
    speed_controller: dict
    reward: dict
    termination: dict
    diagnostics: dict


# Per-shape controller schemas. Each `type` discriminator selects exactly one.
# These match the YAMLs committed in PR 1 (configs/controllers/*.yaml) field
# for field; validate_keys() will raise on any drift.


@dataclass
class SpeedPIControllerSchema:
    schema_version: int
    type: str  # discriminator value: "speed_pi"
    kp: float
    ki: float
    integral_max: float


@dataclass
class PurePursuitControllerSchema:
    schema_version: int
    type: str  # discriminator value: "pure_pursuit"
    lookahead_min_m: float
    lookahead_gain_s: float
    lookahead_ds_m: float


_CONTROLLER_SCHEMAS_BY_TYPE: dict[str, type] = {
    "speed_pi": SpeedPIControllerSchema,
    "pure_pursuit": PurePursuitControllerSchema,
}


def select_controller_schema(controller_bundle: dict) -> type:
    """Pick the controller schema class based on the `type` discriminator.

    Raises ValueError if the bundle has no `type` field or an unknown value.
    The returned class can be passed to `validate_keys`.
    """
    if not isinstance(controller_bundle, dict):
        raise ValueError(
            f"controller bundle must be a mapping, got {type(controller_bundle).__name__}"
        )
    if "type" not in controller_bundle:
        raise ValueError("controller bundle is missing 'type' discriminator")
    type_value = controller_bundle["type"]
    if type_value not in _CONTROLLER_SCHEMAS_BY_TYPE:
        raise ValueError(
            f"unknown controller type: {type_value!r} "
            f"(known: {sorted(_CONTROLLER_SCHEMAS_BY_TYPE)})"
        )
    return _CONTROLLER_SCHEMAS_BY_TYPE[type_value]


@dataclass
class AgentSchema:
    schema_version: int
    runner: dict
    policy: dict
    algorithm: dict


# Per-shape experiment schemas. The PR 1 RL and classical experiment YAMLs
# differ in which category refs they carry (RL has env/agent; classical has
# controllers/run instead), so they need separate top-level shells.


@dataclass
class RLExperimentSchema:
    schema_version: int
    kind: str  # discriminator value: "rl_train"
    seed: Any
    run_name: str
    vehicle: dict
    dynamics: dict
    env: dict
    course: dict
    agent: dict
    runtime: dict
    # `overrides` is consumed by the loader before validation.


@dataclass
class ClassicalExperimentSchema:
    schema_version: int
    kind: str  # discriminator value: "classical"
    seed: Any
    run_name: str
    vehicle: dict
    dynamics: dict
    course: dict
    controllers: dict
    runtime: dict
    run: dict
    # `overrides` is consumed by the loader before validation.


_EXPERIMENT_SCHEMAS_BY_KIND: dict[str, type] = {
    "rl_train": RLExperimentSchema,
    "classical": ClassicalExperimentSchema,
}


def select_experiment_schema(experiment_bundle: dict) -> type:
    """Pick the experiment schema class based on the `kind` discriminator.

    Raises ValueError if the bundle has no `kind` field or an unknown value.
    """
    if not isinstance(experiment_bundle, dict):
        raise ValueError(
            f"experiment bundle must be a mapping, got {type(experiment_bundle).__name__}"
        )
    if "kind" not in experiment_bundle:
        raise ValueError("experiment bundle is missing 'kind' discriminator")
    kind_value = experiment_bundle["kind"]
    if kind_value not in _EXPERIMENT_SCHEMAS_BY_KIND:
        raise ValueError(
            f"unknown experiment kind: {kind_value!r} "
            f"(known: {sorted(_EXPERIMENT_SCHEMAS_BY_KIND)})"
        )
    return _EXPERIMENT_SCHEMAS_BY_KIND[kind_value]


__all__ = [
    "AgentSchema",
    "ClassicalExperimentSchema",
    "DynamicsSchema",
    "EnvSchema",
    "PurePursuitControllerSchema",
    "RLExperimentSchema",
    "SpeedPIControllerSchema",
    "VehicleSchema",
    "select_controller_schema",
    "select_experiment_schema",
    "validate_keys",
]
