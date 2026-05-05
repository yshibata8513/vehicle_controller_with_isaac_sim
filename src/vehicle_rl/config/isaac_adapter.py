"""YAML bundle -> Isaac Lab cfg / runtime kwargs translators.

Stage 2 (PR 2) scope:
  - make_sedan_cfg(vehicle_bundle)        -> ArticulationCfg
  - make_simulator_kwargs(dynamics_bundle) -> dict (VehicleSimulator kwargs)
  - build_path(course_bundle, num_envs, device) -> Path

Each factory validates the bundle's key set against a per-shape schema in
`vehicle_rl.config.schema` BEFORE touching the values, so missing or stray
keys raise `ValueError` with a clear path before any work happens.

Heavy Isaac Lab imports (`isaaclab.sim`, `isaaclab.actuators`, etc.) are
done lazily inside the corresponding factory so the rest of the module
(dispatch + course factory) stays unit-testable without a running sim.

Derived values (per the YAML refactor plan, "## 注意点"):
  - pinion_max = delta_max_rad * steering_ratio (computed in
    `make_sedan_cfg` if a caller needs it; the canonical access path is
    `VehicleSimulator.pinion_max` which derives the same value from the
    SteeringModel constructed with the YAML's `steering_ratio`).
"""
from __future__ import annotations

import os
from typing import Any

from vehicle_rl.config.schema import (
    DynamicsSchema,
    VehicleSchema,
    validate_keys,
)


# ---------------------------------------------------------------------------
# Vehicle factory
# ---------------------------------------------------------------------------


def make_sedan_cfg(vehicle_bundle: dict[str, Any]):
    """Build an Isaac Lab `ArticulationCfg` from a resolved vehicle bundle.

    The bundle shape is `configs/vehicles/sedan.yaml` (see VehicleSchema).
    The USD path in the bundle is repo-relative and is resolved against
    `vehicle_rl.USD_DIR` (which is `<repo>/assets/usd/`).

    Returns the cfg WITHOUT setting `prim_path`; callers (single-env
    classical / multi-env training) set their own prim path on the copy.
    """
    validate_keys(vehicle_bundle, VehicleSchema)

    # Lazy Isaac imports so this module can be imported in unit tests without
    # a running Isaac Sim.
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    from vehicle_rl import VEHICLE_RL_ROOT

    asset = vehicle_bundle["asset"]
    geometry = vehicle_bundle["geometry"]
    physx = vehicle_bundle["physx"]
    rigid = physx["rigid_body"]
    art = physx["articulation"]
    actuators = physx["actuators"]
    steer_act = actuators["steer"]
    wheels_act = actuators["wheels"]
    joints = vehicle_bundle["joints"]

    # USD path is repo-relative in YAML; resolve against the repo root so the
    # file actually exists at simulator-spawn time regardless of CWD.
    usd_path = os.path.join(VEHICLE_RL_ROOT, asset["usd_path"])

    cog_z = float(geometry["cog_z_m"])

    cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=float(rigid["max_linear_velocity"]),
                max_angular_velocity=float(rigid["max_angular_velocity"]),
                max_depenetration_velocity=float(rigid["max_depenetration_velocity"]),
                enable_gyroscopic_forces=bool(rigid["enable_gyroscopic_forces"]),
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=bool(art["enabled_self_collisions"]),
                solver_position_iteration_count=int(art["solver_position_iteration_count"]),
                solver_velocity_iteration_count=int(art["solver_velocity_iteration_count"]),
                sleep_threshold=float(art["sleep_threshold"]),
                stabilization_threshold=float(art["stabilization_threshold"]),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, cog_z),
            joint_pos={
                "front_left_steer_joint": 0.0,
                "front_right_steer_joint": 0.0,
                "front_left_wheel_joint": 0.0,
                "front_right_wheel_joint": 0.0,
                "rear_left_wheel_joint": 0.0,
                "rear_right_wheel_joint": 0.0,
            },
        ),
        actuators={
            "steer": ImplicitActuatorCfg(
                joint_names_expr=[joints["steer_regex"]],
                stiffness=float(steer_act["stiffness"]),
                damping=float(steer_act["damping"]),
                effort_limit_sim=float(steer_act["effort_limit_sim"]),
                velocity_limit_sim=float(steer_act["velocity_limit_sim"]),
                friction=float(steer_act["friction"]),
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[joints["wheel_regex"]],
                stiffness=float(wheels_act["stiffness"]),
                damping=float(wheels_act["damping"]),
                effort_limit_sim=float(wheels_act["effort_limit_sim"]),
                velocity_limit_sim=float(wheels_act["velocity_limit_sim"]),
                friction=float(wheels_act["friction"]),
            ),
        },
    )
    return cfg


def derived_pinion_max(vehicle_bundle: dict[str, Any]) -> float:
    """Pure derived value: pinion_max = delta_max_rad * steering_ratio.

    Used by tests + callers that need the limit before constructing a
    VehicleSimulator. The canonical runtime path is
    `VehicleSimulator.pinion_max` which derives the same value from the
    SteeringModel.
    """
    validate_keys(vehicle_bundle, VehicleSchema)
    steering = vehicle_bundle["steering"]
    return float(steering["delta_max_rad"]) * float(steering["steering_ratio"])


# ---------------------------------------------------------------------------
# Dynamics factory
# ---------------------------------------------------------------------------


# Mapping from dynamics-bundle leaves to VehicleSimulator kwargs.
# Defined here (not in VehicleSimulator) so the simulator stays a pure
# physics object: the YAML schema -> kwargs translation is a config concern.
_VALID_FX_SPLIT_VALUES = ("rear", "front", "four_wheel")


def make_simulator_kwargs(dynamics_bundle: dict[str, Any]) -> dict[str, Any]:
    """Translate a resolved dynamics bundle into VehicleSimulator kwargs.

    Returns the dynamics-side kwargs (tau_*, actuator_initial_value,
    cornering_stiffness, eps_vlong, longitudinal_force_split, z_drift_*,
    k_roll/pitch, c_roll/pitch, mu_default, gravity). The caller MUST also
    supply `steering_ratio` (lives in the vehicle bundle, not the dynamics
    bundle) before passing the dict into `VehicleSimulator(**kwargs)`:

        sim_kwargs = make_simulator_kwargs(dynamics_bundle)
        sim_kwargs["steering_ratio"] = float(vehicle_bundle["steering"]["steering_ratio"])
        VehicleSimulator(sim, sedan, device=..., **sim_kwargs)

    The split is intentional: `steering_ratio` is a vehicle property, not
    a tire / actuator coefficient, so it stays in the vehicle YAML.

    PR 2 round-1 fix: `actuator_initial_value`, `eps_vlong`, and
    `longitudinal_force_split` are now actually returned (they were
    validated but silently dropped before).
    """
    validate_keys(dynamics_bundle, DynamicsSchema)

    friction = dynamics_bundle["friction"]
    lag = dynamics_bundle["actuator_lag"]
    tire = dynamics_bundle["tire"]
    normal = dynamics_bundle["normal_load"]
    attitude = dynamics_bundle["attitude_damper"]

    # Validate the longitudinal_force_split values at adapter time so a typo
    # raises before any sim object is built. The simulator branches on these
    # two strings; only "rear" / "front" / "four_wheel" are honoured.
    fx_split = tire["longitudinal_force_split"]
    accel_split = fx_split["accel"]
    brake_split = fx_split["brake"]
    if accel_split not in _VALID_FX_SPLIT_VALUES:
        raise ValueError(
            f"tire.longitudinal_force_split.accel must be one of "
            f"{_VALID_FX_SPLIT_VALUES}, got {accel_split!r}"
        )
    if brake_split not in _VALID_FX_SPLIT_VALUES:
        raise ValueError(
            f"tire.longitudinal_force_split.brake must be one of "
            f"{_VALID_FX_SPLIT_VALUES}, got {brake_split!r}"
        )

    return {
        "tau_steer": float(lag["tau_steer_s"]),
        "tau_drive": float(lag["tau_drive_s"]),
        "tau_brake": float(lag["tau_brake_s"]),
        "actuator_initial_value": float(lag["initial_value"]),
        "cornering_stiffness": float(tire["cornering_stiffness_n_per_rad"]),
        "eps_vlong": float(tire["eps_vlong_mps"]),
        "fx_split_accel": str(accel_split),
        "fx_split_brake": str(brake_split),
        "z_drift_kp": float(normal["z_drift_kp"]),
        "z_drift_kd": float(normal["z_drift_kd"]),
        "k_roll": float(attitude["k_roll"]),
        "c_roll": float(attitude["c_roll"]),
        "k_pitch": float(attitude["k_pitch"]),
        "c_pitch": float(attitude["c_pitch"]),
        "mu_default": float(friction["mu_default"]),
        "gravity": float(dynamics_bundle["gravity_mps2"]),
    }


def make_action_limits(dynamics_bundle: dict[str, Any]) -> tuple[float, float]:
    """Return (a_x_min, a_x_max) from the dynamics bundle's `action_limits` block."""
    validate_keys(dynamics_bundle, DynamicsSchema)
    al = dynamics_bundle["action_limits"]
    return float(al["a_x_min_mps2"]), float(al["a_x_max_mps2"])


def make_vehicle_geometry(vehicle_bundle: dict[str, Any]) -> dict[str, float]:
    """Return geometry / mass values from the vehicle bundle as a flat dict.

    Consumers that need wheelbase / track / cog height / mass without
    constructing a full `ArticulationCfg` (e.g. classical Pure Pursuit
    controller, dynamics modules instantiated outside VehicleSimulator).
    """
    validate_keys(vehicle_bundle, VehicleSchema)
    g = vehicle_bundle["geometry"]
    m = vehicle_bundle["mass"]
    return {
        "wheelbase_m": float(g["wheelbase_m"]),
        "track_m": float(g["track_m"]),
        "wheel_radius_m": float(g["wheel_radius_m"]),
        "wheel_width_m": float(g["wheel_width_m"]),
        "cog_z_m": float(g["cog_z_m"]),
        "a_front_m": float(g["a_front_m"]),
        "total_kg": float(m["total_kg"]),
    }


# ---------------------------------------------------------------------------
# Course factory
# ---------------------------------------------------------------------------


# Sentinel: course types this adapter knows how to build. Keeping the list
# explicit (vs reflection on `waypoints.py` symbols) so a typo or accidental
# new course YAML can't silently dispatch to the wrong generator.
_BUILTIN_COURSE_TYPES = {"circle", "lemniscate", "s_curve", "dlc"}
_RANDOM_COURSE_TYPES = {"random_long", "random_bank"}


def build_path(
    course_bundle: dict[str, Any],
    *,
    num_envs: int,
    device: str,
):
    """Construct a `Path` (or `RandomPathBank` for random_bank) from a course bundle.

    Dispatches on `course_bundle["type"]`:
      - "circle"     -> waypoints.circle_path
      - "lemniscate" -> waypoints.lemniscate_path
      - "s_curve"    -> waypoints.s_curve_path
      - "dlc"        -> waypoints.dlc_path
      - "random_long"-> random_path.random_clothoid_path (single 20km open path)
      - "random_bank"-> random_path.random_clothoid_path_bank (P paths)

    For random courses the bundle must carry the resolved `generator:`
    sub-bundle (loaded via `<...>_ref` from random_path_generator.yaml) and
    a `phase` discriminator selecting which sub-section of the generator
    config to use.
    """
    if not isinstance(course_bundle, dict):
        raise ValueError(
            f"course bundle must be a mapping, got {type(course_bundle).__name__}"
        )
    if "type" not in course_bundle:
        raise ValueError("course bundle is missing 'type' discriminator")
    course_type = course_bundle["type"]

    if course_type in _BUILTIN_COURSE_TYPES:
        return _build_builtin_path(course_bundle, num_envs=num_envs, device=device)
    if course_type in _RANDOM_COURSE_TYPES:
        return _build_random_path(course_bundle, num_envs=num_envs, device=device)
    raise ValueError(
        f"unknown course type: {course_type!r} "
        f"(known: {sorted(_BUILTIN_COURSE_TYPES | _RANDOM_COURSE_TYPES)})"
    )


_COMMON_COURSE_KEYS = ["schema_version", "type", "ds_m", "target_speed_mps", "is_loop"]
_COURSE_KEYS_BY_TYPE: dict[str, list[str]] = {
    "circle": ["radius_m"],
    "lemniscate": ["a_m", "n_raw"],
    "s_curve": ["length_m", "amplitude_m", "n_cycles", "n_raw"],
    "dlc": ["lane_offset_m", "n_raw"],
}


def _build_builtin_path(course_bundle: dict[str, Any], *, num_envs: int, device: str):
    from vehicle_rl.planner import (
        circle_path,
        dlc_path,
        lemniscate_path,
        s_curve_path,
    )

    course_type = course_bundle["type"]
    # Strict whole-bundle key check: union of common + per-course keys.
    # Both missing and unknown keys raise here so a typo (e.g. `radious_m`
    # for `radius_m`) is caught with a useful message.
    expected = _COMMON_COURSE_KEYS + _COURSE_KEYS_BY_TYPE[course_type]
    _required_top(course_bundle, expected)

    ds = float(course_bundle["ds_m"])
    target_speed = float(course_bundle["target_speed_mps"])
    is_loop_yaml = bool(course_bundle["is_loop"])

    if course_type == "circle":
        if not is_loop_yaml:
            raise ValueError("circle course must have is_loop=true")
        return circle_path(
            radius=float(course_bundle["radius_m"]),
            target_speed=target_speed,
            num_envs=num_envs,
            ds=ds,
            device=device,
        )

    if course_type == "lemniscate":
        return lemniscate_path(
            a=float(course_bundle["a_m"]),
            target_speed=target_speed,
            num_envs=num_envs,
            ds=ds,
            n_raw=int(course_bundle["n_raw"]),
            device=device,
        )

    if course_type == "s_curve":
        return s_curve_path(
            length=float(course_bundle["length_m"]),
            amplitude=float(course_bundle["amplitude_m"]),
            n_cycles=float(course_bundle["n_cycles"]),
            target_speed=target_speed,
            num_envs=num_envs,
            ds=ds,
            n_raw=int(course_bundle["n_raw"]),
            device=device,
        )

    if course_type == "dlc":
        return dlc_path(
            target_speed=target_speed,
            num_envs=num_envs,
            ds=ds,
            n_raw=int(course_bundle["n_raw"]),
            lane_offset=float(course_bundle["lane_offset_m"]),
            device=device,
        )

    raise AssertionError(f"unreachable course_type {course_type!r}")


def _build_random_path(course_bundle: dict[str, Any], *, num_envs: int, device: str):
    """Build a random clothoid path (random_long) or path bank (random_bank).

    Reads the resolved `generator:` sub-bundle from the course bundle (the
    loader has already resolved `generator_ref:` -> `generator:` content)
    and the `phase` discriminator. The generator bundle uses the new
    `*_m` / `*_mps` unit suffixes (configs/courses/random_path_generator.yaml),
    NOT the legacy `configs/random_path.yaml` shape.
    """
    from vehicle_rl.planner.random_path import (
        RandomPathGeneratorCfg,
        _GeneratorCfg,
        _Phase1Cfg,
        _Phase2Cfg,
        _Phase3Cfg,
        _ProjectionCfg,
        _ResetCfg,
        _SegmentsCfg,
        _SpeedCfg,
        random_clothoid_path,
        random_clothoid_path_bank,
    )

    course_type = course_bundle["type"]
    _required_top(course_bundle, ["schema_version", "type", "phase", "generator"])

    gen_bundle = course_bundle["generator"]
    if not isinstance(gen_bundle, dict):
        raise ValueError(
            "course.generator must be a mapping (resolved from generator_ref); "
            f"got {type(gen_bundle).__name__}"
        )

    # Translate the new-shape generator bundle -> RandomPathGeneratorCfg.
    cfg = _make_random_path_cfg(gen_bundle)

    phase = course_bundle["phase"]
    if course_type == "random_long":
        if phase != "phase1_long_path":
            raise ValueError(
                f"random_long expects phase='phase1_long_path', got {phase!r}"
            )
        ph = cfg.phase1_long_path
        return random_clothoid_path(
            cfg=cfg,
            num_envs=num_envs,
            length_m=ph.length_m,
            is_loop=ph.is_loop,
            device=device,
        )

    if course_type == "random_bank":
        if phase != "phase2_bank":
            raise ValueError(
                f"random_bank expects phase='phase2_bank', got {phase!r}"
            )
        ph = cfg.phase2_bank
        return random_clothoid_path_bank(
            cfg=cfg,
            num_paths=ph.num_paths,
            length_m=ph.length_m,
            is_loop=ph.is_loop,
            device=device,
        )

    raise AssertionError(f"unreachable random course_type {course_type!r}")


def _make_random_path_cfg(gen: dict[str, Any]):
    """Translate a resolved random-path generator bundle into RandomPathGeneratorCfg.

    The new-shape bundle (from configs/courses/random_path_generator.yaml)
    uses unit-suffixed keys (`ds_m`, `target_speed_mps`, `v_min_mps`, ...).
    The legacy dataclass field names are kept as-is to avoid touching the
    planner internals in stage 2.
    """
    from vehicle_rl.planner.random_path import (
        RandomPathGeneratorCfg,
        _GeneratorCfg,
        _Phase1Cfg,
        _Phase2Cfg,
        _Phase3Cfg,
        _ProjectionCfg,
        _ResetCfg,
        _SegmentsCfg,
        _SpeedCfg,
    )

    _required_top(
        gen,
        [
            "schema_version", "generator", "speed", "segments",
            "projection", "reset",
            "phase1_long_path", "phase2_bank", "phase3_regeneration",
        ],
    )

    g = gen["generator"]
    _required(g, ["seed", "ds_m", "target_speed_mps"], "generator.generator")
    s = gen["speed"]
    _required(
        s,
        ["v_min_mps", "v_max_mps", "ay_limit_mps2", "segment_constant",
         "max_resample_attempts"],
        "generator.speed",
    )
    seg = gen["segments"]
    _required(
        seg,
        ["straight_length_m", "min_radius_m", "max_radius_m",
         "turn_heading_change_rad", "clothoid_heading_fraction",
         "straight_probability", "turn_probability",
         "reverse_turn_probability"],
        "generator.segments",
    )
    pj = gen["projection"]
    _required(
        pj,
        ["search_radius_samples", "recovery_radius_samples", "max_index_jump_samples"],
        "generator.projection",
    )
    rs = gen["reset"]
    _required(rs, ["end_margin_extra_m"], "generator.reset")
    p1 = gen["phase1_long_path"]
    _required(p1, ["length_m", "is_loop"], "generator.phase1_long_path")
    p2 = gen["phase2_bank"]
    _required(p2, ["num_paths", "length_m", "is_loop"], "generator.phase2_bank")
    p3 = gen["phase3_regeneration"]
    _required(
        p3,
        ["enabled", "interval_resets", "fraction", "min_unused_slots"],
        "generator.phase3_regeneration",
    )

    cfg = RandomPathGeneratorCfg(
        generator=_GeneratorCfg(
            seed=int(g["seed"]),
            ds=float(g["ds_m"]),
            target_speed=float(g["target_speed_mps"]),
        ),
        speed=_SpeedCfg(
            v_min=float(s["v_min_mps"]),
            v_max=float(s["v_max_mps"]),
            ay_limit=float(s["ay_limit_mps2"]),
            segment_constant=bool(s["segment_constant"]),
            max_resample_attempts=int(s["max_resample_attempts"]),
        ),
        segments=_SegmentsCfg(
            straight_length_m=_pair(seg["straight_length_m"]),
            min_radius_m=float(seg["min_radius_m"]),
            max_radius_m=float(seg["max_radius_m"]),
            turn_heading_change_rad=_pair(seg["turn_heading_change_rad"]),
            clothoid_heading_fraction=_pair(seg["clothoid_heading_fraction"]),
            straight_probability=float(seg["straight_probability"]),
            turn_probability=float(seg["turn_probability"]),
            reverse_turn_probability=float(seg["reverse_turn_probability"]),
        ),
        projection=_ProjectionCfg(
            search_radius_samples=int(pj["search_radius_samples"]),
            recovery_radius_samples=int(pj["recovery_radius_samples"]),
            max_index_jump_samples=int(pj["max_index_jump_samples"]),
        ),
        reset=_ResetCfg(
            end_margin_extra_m=float(rs["end_margin_extra_m"]),
        ),
        phase1_long_path=_Phase1Cfg(
            length_m=float(p1["length_m"]),
            is_loop=bool(p1["is_loop"]),
        ),
        phase2_bank=_Phase2Cfg(
            num_paths=int(p2["num_paths"]),
            length_m=float(p2["length_m"]),
            is_loop=bool(p2["is_loop"]),
        ),
        phase3_regeneration=_Phase3Cfg(
            enabled=bool(p3["enabled"]),
            interval_resets=int(p3["interval_resets"]),
            fraction=float(p3["fraction"]),
            min_unused_slots=int(p3["min_unused_slots"]),
        ),
        source_path="<from-yaml-bundle>",
    )
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _required(d: dict[str, Any], keys: list[str], where: str) -> None:
    """Strict: raise if any of `keys` is missing or any unknown key is present.

    PR 2 round-1 fix (review finding 5): the previous "allows extra keys"
    behavior bypassed the plan's "Strict Loader" rule. Both directions are
    now checked. Used for the random-path generator sub-bundle which lacks
    a top-level dataclass schema (its dispatch is course-discriminated).
    """
    if not isinstance(d, dict):
        raise ValueError(f"{where} must be a mapping, got {type(d).__name__}")
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"{where} is missing keys: {missing}")
    unknown = sorted(set(d.keys()) - set(keys))
    if unknown:
        raise ValueError(f"{where} has unknown keys: {unknown}")


def _required_top(
    d: dict[str, Any],
    keys: list[str],
    *,
    extra_allowed_top: bool = False,
) -> None:
    """Strict version of `_required` for the top level of a course bundle.

    By default, also checks that no UNKNOWN keys are present; this catches
    typos in the course YAML. Set `extra_allowed_top=True` for the
    discriminated builtin courses so per-course optional keys (`radius_m`,
    `a_m`, ...) don't all need to be listed at the top check.
    """
    if not isinstance(d, dict):
        raise ValueError(f"course bundle must be a mapping, got {type(d).__name__}")
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"course bundle is missing keys: {missing}")
    if not extra_allowed_top:
        unknown = sorted(set(d.keys()) - set(keys))
        if unknown:
            raise ValueError(
                f"course bundle has unknown keys: {unknown} "
                f"(allowed: {sorted(keys)})"
            )


def _pair(v: Any) -> tuple[float, float]:
    """Coerce a 2-element list/tuple to (float, float). Raise on wrong shape."""
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return float(v[0]), float(v[1])
    raise ValueError(f"expected 2-element list, got {v!r}")


__all__ = [
    "build_path",
    "derived_pinion_max",
    "make_action_limits",
    "make_sedan_cfg",
    "make_simulator_kwargs",
    "make_vehicle_geometry",
]
