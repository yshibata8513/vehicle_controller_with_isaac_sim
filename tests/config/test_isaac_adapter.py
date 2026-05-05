"""Unit tests for `vehicle_rl.config.isaac_adapter`.

These tests are deliberately Isaac-Sim-free: each factory either operates
on plain dicts (validation, key checks, derived values) or invokes the
pure-Python path generators in `vehicle_rl.planner.waypoints` /
`vehicle_rl.planner.random_path`. Isaac Lab is only needed for
`make_sedan_cfg`; that single test is gated behind a try/import so the
suite runs in any environment.

Run with:
    python -m unittest tests.config.test_isaac_adapter -v
"""
from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vehicle_rl.config.isaac_adapter import (  # noqa: E402
    build_path,
    derived_pinion_max,
    make_action_limits,
    make_simulator_kwargs,
    make_vehicle_geometry,
)
from vehicle_rl.config.isaac_adapter import (  # noqa: E402
    _derived_action_space,
    _derived_observation_space,
)
from vehicle_rl.config.loader import (  # noqa: E402
    load_experiment,
    load_yaml_strict,
    resolve_refs,
)


def _vehicle_bundle() -> dict:
    return load_yaml_strict(REPO_ROOT / "configs" / "vehicles" / "sedan.yaml")


def _dynamics_bundle() -> dict:
    return load_yaml_strict(
        REPO_ROOT / "configs" / "dynamics" / "linear_friction_circle_flat.yaml"
    )


def _course_bundle(name: str) -> dict:
    raw = load_yaml_strict(REPO_ROOT / "configs" / "courses" / f"{name}.yaml")
    # resolve_refs handles `generator_ref:` for random_long / random_bank.
    return resolve_refs(raw, repo_root=REPO_ROOT)


def _env_bundle() -> dict:
    raw = load_yaml_strict(REPO_ROOT / "configs" / "envs" / "tracking.yaml")
    return resolve_refs(raw, repo_root=REPO_ROOT)


def _agent_bundle() -> dict:
    raw = load_yaml_strict(
        REPO_ROOT / "configs" / "agents" / "rsl_rl" / "ppo_tracking.yaml"
    )
    return resolve_refs(raw, repo_root=REPO_ROOT)


def _controller_bundle() -> dict:
    return load_yaml_strict(REPO_ROOT / "configs" / "controllers" / "speed_pi.yaml")


# ---------------------------------------------------------------------------
# derived_pinion_max
# ---------------------------------------------------------------------------


class TestDerivedPinionMax(unittest.TestCase):
    def test_default_bundle(self):
        b = _vehicle_bundle()
        # 0.611 * 16.0 = 9.776
        self.assertAlmostEqual(derived_pinion_max(b), 0.611 * 16.0, places=6)

    def test_synthetic_steering_block(self):
        b = _vehicle_bundle()
        b = copy.deepcopy(b)
        b["steering"]["delta_max_rad"] = 0.5
        b["steering"]["steering_ratio"] = 14.0
        self.assertAlmostEqual(derived_pinion_max(b), 7.0, places=6)

    def test_missing_key_rejected(self):
        b = _vehicle_bundle()
        b = copy.deepcopy(b)
        del b["mass"]
        with self.assertRaises(ValueError) as ctx:
            derived_pinion_max(b)
        self.assertIn("mass", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# make_vehicle_geometry
# ---------------------------------------------------------------------------


class TestMakeVehicleGeometry(unittest.TestCase):
    def test_default_bundle_returns_expected_values(self):
        g = make_vehicle_geometry(_vehicle_bundle())
        self.assertEqual(g["wheelbase_m"], 2.7)
        self.assertEqual(g["track_m"], 1.55)
        self.assertEqual(g["wheel_radius_m"], 0.33)
        self.assertEqual(g["wheel_width_m"], 0.225)
        self.assertEqual(g["cog_z_m"], 0.55)
        self.assertEqual(g["a_front_m"], 1.35)
        self.assertEqual(g["total_kg"], 1500.0)

    def test_missing_total_kg_rejected(self):
        b = copy.deepcopy(_vehicle_bundle())
        del b["mass"]["total_kg"]
        with self.assertRaises(ValueError) as ctx:
            make_vehicle_geometry(b)
        # validate_keys against MassSchema (PR 2 round-1 strict refinement)
        # raises with the missing key name.
        self.assertIn("total_kg", str(ctx.exception))

    def test_unknown_nested_geometry_key_rejected(self):
        # PR 2 round-1 fix (review finding 5): vehicle nested blocks now
        # validate against per-shape schemas; unknown keys raise.
        b = copy.deepcopy(_vehicle_bundle())
        b["geometry"]["bogus"] = 1.0
        with self.assertRaises(ValueError) as ctx:
            make_vehicle_geometry(b)
        self.assertIn("bogus", str(ctx.exception))

    def test_a_front_m_surfaces_in_geometry(self):
        # PR 2 round-1 fix (review finding 3): asymmetric a_front_m in YAML
        # must reach make_vehicle_geometry verbatim (not silently
        # recomputed as wheelbase / 2.0).
        b = copy.deepcopy(_vehicle_bundle())
        b["geometry"]["a_front_m"] = 1.20
        g = make_vehicle_geometry(b)
        self.assertEqual(g["a_front_m"], 1.20)
        # And it differs from the default's wheelbase / 2.0 = 1.35.
        self.assertNotEqual(g["a_front_m"], g["wheelbase_m"] / 2.0)


# ---------------------------------------------------------------------------
# make_simulator_kwargs / make_action_limits
# ---------------------------------------------------------------------------


# The kwargs returned by `make_simulator_kwargs` must match the
# VehicleSimulator.__init__ signature exactly (minus `sim`, `sedan`,
# `device`, `steering_ratio`, `a_front` which live in the vehicle bundle).
_EXPECTED_SIM_KEYS = {
    "tau_steer", "tau_drive", "tau_brake", "actuator_initial_value",
    "cornering_stiffness", "eps_vlong",
    "fx_split_accel", "fx_split_brake",
    "z_drift_kp", "z_drift_kd",
    "k_roll", "c_roll", "k_pitch", "c_pitch",
    "mu_default", "gravity",
}


class TestMakeSimulatorKwargs(unittest.TestCase):
    def test_default_bundle_returns_expected_key_set(self):
        kw = make_simulator_kwargs(_dynamics_bundle())
        self.assertEqual(set(kw.keys()), _EXPECTED_SIM_KEYS)

    def test_default_bundle_values_match_yaml(self):
        kw = make_simulator_kwargs(_dynamics_bundle())
        # Spot-check a representative subset; full leaf coverage would just
        # restate the YAML.
        self.assertEqual(kw["tau_steer"], 0.05)
        self.assertEqual(kw["tau_drive"], 0.20)
        self.assertEqual(kw["tau_brake"], 0.07)
        self.assertEqual(kw["actuator_initial_value"], 0.0)
        self.assertEqual(kw["cornering_stiffness"], 60000.0)
        self.assertEqual(kw["eps_vlong"], 0.01)
        self.assertEqual(kw["fx_split_accel"], "rear")
        self.assertEqual(kw["fx_split_brake"], "four_wheel")
        self.assertEqual(kw["mu_default"], 0.9)
        self.assertEqual(kw["gravity"], 9.81)
        self.assertEqual(kw["k_roll"], 80000.0)

    def test_actuator_initial_value_flows_through(self):
        # PR 2 round-1 fix (review finding 2): tweaking the YAML must
        # surface in the returned kwargs, not be silently dropped.
        b = copy.deepcopy(_dynamics_bundle())
        b["actuator_lag"]["initial_value"] = 0.42
        kw = make_simulator_kwargs(b)
        self.assertEqual(kw["actuator_initial_value"], 0.42)

    def test_actuator_reset_uses_initial_value_from_yaml(self):
        # PR 2 round-2 fix: it's not enough that `actuator_initial_value`
        # flows into VehicleSimulator.__init__; VehicleSimulator.reset()
        # must also revert actuator state to that value (it previously
        # hard-coded 0.0). We test FirstOrderLagActuator directly because
        # VehicleSimulator construction needs Isaac Sim, and the simulator's
        # reset() now simply forwards to actuator.reset() with no `value=`,
        # so the actuator's behavior is the contract under test.
        import torch

        from vehicle_rl.dynamics import FirstOrderLagActuator

        act = FirstOrderLagActuator(
            num_envs=3, device="cpu",
            tau_pos=0.05, initial_value=0.3,
        )
        # Drive state away from the initial value.
        u = torch.full((3,), 1.0)
        for _ in range(5):
            act.step(u, dt=0.02)
        self.assertGreater(float(act.value[0].item()), 0.3)

        # reset() with no explicit value reverts to initial_value=0.3.
        act.reset()
        for v in act.value.tolist():
            self.assertAlmostEqual(v, 0.3, places=6)

        # Drive state away again, then verify partial reset (env_ids subset)
        # also targets initial_value, leaving other envs untouched.
        for _ in range(5):
            act.step(u, dt=0.02)
        moved = float(act.value[2].item())
        act.reset(env_ids=torch.tensor([0, 1]))
        self.assertAlmostEqual(float(act.value[0].item()), 0.3, places=6)
        self.assertAlmostEqual(float(act.value[1].item()), 0.3, places=6)
        # Env 2 was excluded -> still moved.
        self.assertAlmostEqual(float(act.value[2].item()), moved, places=6)

        # Explicit override still works (back-compat for any caller that
        # wants 0.0 specifically).
        act.reset(value=0.0)
        for v in act.value.tolist():
            self.assertAlmostEqual(v, 0.0, places=6)

    def test_eps_vlong_flows_through(self):
        b = copy.deepcopy(_dynamics_bundle())
        b["tire"]["eps_vlong_mps"] = 0.05
        kw = make_simulator_kwargs(b)
        self.assertEqual(kw["eps_vlong"], 0.05)

    def test_longitudinal_force_split_flows_through(self):
        b = copy.deepcopy(_dynamics_bundle())
        b["tire"]["longitudinal_force_split"]["accel"] = "front"
        b["tire"]["longitudinal_force_split"]["brake"] = "rear"
        kw = make_simulator_kwargs(b)
        self.assertEqual(kw["fx_split_accel"], "front")
        self.assertEqual(kw["fx_split_brake"], "rear")

    def test_unknown_fx_split_accel_rejected(self):
        b = copy.deepcopy(_dynamics_bundle())
        b["tire"]["longitudinal_force_split"]["accel"] = "all_wheel"
        with self.assertRaises(ValueError) as ctx:
            make_simulator_kwargs(b)
        self.assertIn("longitudinal_force_split.accel", str(ctx.exception))

    def test_unknown_fx_split_brake_rejected(self):
        b = copy.deepcopy(_dynamics_bundle())
        b["tire"]["longitudinal_force_split"]["brake"] = "diagonal"
        with self.assertRaises(ValueError) as ctx:
            make_simulator_kwargs(b)
        self.assertIn("longitudinal_force_split.brake", str(ctx.exception))

    def test_kwargs_signature_matches_simulator_init(self):
        # We don't import VehicleSimulator (it pulls isaaclab); instead we
        # statically read the simulator source and confirm every returned
        # key appears as a parameter name. This catches signature drift
        # without booting Isaac.
        src = (
            REPO_ROOT / "src" / "vehicle_rl" / "envs" / "simulator.py"
        ).read_text(encoding="utf-8")
        # fx_split_* are str-typed (discriminator strings); everything else
        # is float-typed. Match against either annotation form.
        str_typed = {"fx_split_accel", "fx_split_brake"}
        for k in _EXPECTED_SIM_KEYS:
            ann = "str" if k in str_typed else "float"
            self.assertIn(
                f"        {k}: {ann}",
                src,
                f"VehicleSimulator.__init__ has no parameter named {k!r} "
                f"with annotation {ann}",
            )

    def test_unknown_top_level_key_rejected(self):
        b = copy.deepcopy(_dynamics_bundle())
        b["bogus"] = 42
        with self.assertRaises(ValueError) as ctx:
            make_simulator_kwargs(b)
        self.assertIn("unknown", str(ctx.exception).lower())

    def test_missing_top_level_key_rejected(self):
        b = copy.deepcopy(_dynamics_bundle())
        del b["actuator_lag"]
        with self.assertRaises(ValueError) as ctx:
            make_simulator_kwargs(b)
        self.assertIn("actuator_lag", str(ctx.exception))

    def test_missing_leaf_key_rejected(self):
        b = copy.deepcopy(_dynamics_bundle())
        del b["actuator_lag"]["tau_steer_s"]
        with self.assertRaises(ValueError) as ctx:
            make_simulator_kwargs(b)
        self.assertIn("tau_steer_s", str(ctx.exception))

    def test_unknown_nested_key_rejected(self):
        # PR 2 round-1 fix (review finding 5): nested dicts are now
        # validated leaf-by-leaf; an unknown key under any nested block
        # must raise.
        b = copy.deepcopy(_dynamics_bundle())
        b["tire"]["bogus_field"] = 1.0
        with self.assertRaises(ValueError) as ctx:
            make_simulator_kwargs(b)
        self.assertIn("bogus_field", str(ctx.exception))

    def test_missing_longitudinal_force_split_rejected(self):
        b = copy.deepcopy(_dynamics_bundle())
        del b["tire"]["longitudinal_force_split"]
        with self.assertRaises(ValueError) as ctx:
            make_simulator_kwargs(b)
        self.assertIn("longitudinal_force_split", str(ctx.exception))


class TestMakeActionLimits(unittest.TestCase):
    def test_default_bundle(self):
        lo, hi = make_action_limits(_dynamics_bundle())
        self.assertEqual(lo, -5.0)
        self.assertEqual(hi, 3.0)

    def test_missing_action_limits_rejected(self):
        b = copy.deepcopy(_dynamics_bundle())
        del b["action_limits"]
        with self.assertRaises(ValueError):
            make_action_limits(b)


# ---------------------------------------------------------------------------
# build_path -- builtin courses
# ---------------------------------------------------------------------------


class TestBuildPathCircle(unittest.TestCase):
    def test_circle_default_yaml(self):
        bundle = _course_bundle("circle")
        path = build_path(bundle, num_envs=4, device="cpu")
        # Geometry: r=30, M ~= round(2*pi*30 / 0.2). total_length ~= 2*pi*r.
        self.assertTrue(path.is_loop)
        self.assertEqual(path.num_envs, 4)
        self.assertAlmostEqual(path.total_length, 2 * math.pi * 30.0, places=1)
        # All envs share the same broadcast course.
        self.assertEqual(path.x.shape[0], 4)
        # Target speed honoured.
        self.assertAlmostEqual(float(path.v[0, 0].item()), 10.0, places=4)

    def test_circle_unknown_key_rejected(self):
        b = copy.deepcopy(_course_bundle("circle"))
        b["bogus_field"] = 1.0
        with self.assertRaises(ValueError) as ctx:
            build_path(b, num_envs=1, device="cpu")
        self.assertIn("bogus_field", str(ctx.exception))

    def test_circle_missing_radius_rejected(self):
        b = copy.deepcopy(_course_bundle("circle"))
        del b["radius_m"]
        with self.assertRaises(ValueError) as ctx:
            build_path(b, num_envs=1, device="cpu")
        self.assertIn("radius_m", str(ctx.exception))

    def test_unknown_course_type_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            build_path(
                {"schema_version": 1, "type": "spiral", "ds_m": 0.2,
                 "target_speed_mps": 10.0, "is_loop": True},
                num_envs=1, device="cpu",
            )
        self.assertIn("spiral", str(ctx.exception))

    def test_missing_type_rejected(self):
        with self.assertRaises(ValueError):
            build_path({}, num_envs=1, device="cpu")


class TestBuildPathSCurve(unittest.TestCase):
    def test_s_curve_default_yaml(self):
        bundle = _course_bundle("s_curve")
        path = build_path(bundle, num_envs=2, device="cpu")
        self.assertFalse(path.is_loop)
        self.assertEqual(path.num_envs, 2)
        # length=100 m straight-line, sinusoid increases length slightly.
        self.assertGreaterEqual(path.total_length, 100.0)
        self.assertLess(path.total_length, 120.0)
        self.assertAlmostEqual(float(path.v[0, 0].item()), 12.0, places=4)


class TestBuildPathLemniscate(unittest.TestCase):
    def test_lemniscate_default_yaml(self):
        bundle = _course_bundle("lemniscate")
        path = build_path(bundle, num_envs=1, device="cpu")
        self.assertTrue(path.is_loop)
        self.assertGreater(path.total_length, 50.0)


class TestBuildPathDLC(unittest.TestCase):
    def test_dlc_default_yaml(self):
        bundle = _course_bundle("dlc")
        path = build_path(bundle, num_envs=1, device="cpu")
        self.assertFalse(path.is_loop)
        # ISO 3888-2 simplified is 61 m straight-line; the cosine smoothstep
        # adds a small amount of arc length.
        self.assertGreaterEqual(path.total_length, 61.0)
        self.assertLess(path.total_length, 65.0)


# ---------------------------------------------------------------------------
# build_path -- random courses
# ---------------------------------------------------------------------------


class TestBuildPathRandomBank(unittest.TestCase):
    def test_random_bank_resolves_via_generator_ref(self):
        # Resolve refs so the bundle has a `generator:` sub-bundle in place
        # of the `generator_ref:` from random_bank.yaml.
        bundle = _course_bundle("random_bank")
        # Override num_paths/length to keep the test fast (P=4, length=200 m).
        bundle = copy.deepcopy(bundle)
        bundle["generator"]["phase2_bank"]["num_paths"] = 4
        bundle["generator"]["phase2_bank"]["length_m"] = 200.0
        bank = build_path(bundle, num_envs=1, device="cpu")
        # random_bank returns a RandomPathBank, not a plain Path.
        self.assertEqual(bank.num_paths, 4)
        # is_loop honoured from generator's phase2_bank section.
        self.assertFalse(bank.is_loop)
        # ds matches generator.generator.ds_m (0.2).
        self.assertAlmostEqual(bank.ds, 0.2, places=6)

    def test_random_bank_wrong_phase_rejected(self):
        bundle = copy.deepcopy(_course_bundle("random_bank"))
        bundle["phase"] = "phase1_long_path"
        with self.assertRaises(ValueError) as ctx:
            build_path(bundle, num_envs=1, device="cpu")
        self.assertIn("phase2_bank", str(ctx.exception))

    def test_random_bank_missing_generator_subbundle_rejected(self):
        # If the generator_ref didn't resolve (caller forgot to call
        # resolve_refs), `generator` is absent and we raise.
        bundle = load_yaml_strict(
            REPO_ROOT / "configs" / "courses" / "random_bank.yaml"
        )
        with self.assertRaises(ValueError):
            build_path(bundle, num_envs=1, device="cpu")


class TestBuildPathRandomLong(unittest.TestCase):
    def test_random_long_resolves(self):
        bundle = _course_bundle("random_long")
        bundle = copy.deepcopy(bundle)
        # Shorten the path massively for test speed (default is 20 km).
        bundle["generator"]["phase1_long_path"]["length_m"] = 200.0
        path = build_path(bundle, num_envs=1, device="cpu")
        # random_long returns a single broadcast Path, not a bank.
        self.assertFalse(path.is_loop)
        self.assertEqual(path.num_envs, 1)
        # length_m=200 with ds=0.2 -> M=1000 samples.
        self.assertEqual(path.num_samples, 1000)


# ---------------------------------------------------------------------------
# make_sedan_cfg -- gated on isaaclab availability
# ---------------------------------------------------------------------------


def _isaaclab_available() -> bool:
    try:
        import isaaclab.sim  # noqa: F401
        from isaaclab.assets import ArticulationCfg  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_isaaclab_available(),
                     "isaaclab not importable; skipping ArticulationCfg test")
class TestMakeSedanCfg(unittest.TestCase):
    def test_default_yaml_returns_valid_cfg(self):
        from vehicle_rl.config.isaac_adapter import make_sedan_cfg
        cfg = make_sedan_cfg(_vehicle_bundle())
        # Spot check: the cfg has the expected actuator names and the
        # values match the YAML field-for-field.
        self.assertIn("steer", cfg.actuators)
        self.assertIn("wheels", cfg.actuators)
        self.assertEqual(cfg.actuators["steer"].stiffness, 8000.0)
        self.assertEqual(cfg.actuators["steer"].damping, 400.0)
        self.assertEqual(cfg.actuators["steer"].effort_limit_sim, 500.0)
        self.assertEqual(cfg.actuators["wheels"].effort_limit_sim, 400.0)
        self.assertEqual(cfg.actuators["wheels"].velocity_limit_sim, 200.0)
        self.assertAlmostEqual(cfg.init_state.pos[2], 0.55)

    def test_missing_mass_total_kg_rejected(self):
        from vehicle_rl.config.isaac_adapter import make_sedan_cfg
        b = copy.deepcopy(_vehicle_bundle())
        del b["mass"]
        with self.assertRaises(ValueError):
            make_sedan_cfg(b)


# ---------------------------------------------------------------------------
# Schema integration: factories accept any bundle that loads cleanly via
# load_experiment for the classical experiment YAMLs. No Isaac needed.
# ---------------------------------------------------------------------------


class TestAssetsLazyIsaacImport(unittest.TestCase):
    """Review finding 4: `import vehicle_rl.assets` must not import isaaclab.

    The package's `__getattr__` should defer the heavy import until
    `SEDAN_CFG` is actually accessed.
    """

    def test_import_assets_does_not_pull_isaaclab(self):
        # Run in a subprocess so we don't pollute this test process's
        # already-loaded sys.modules. The child reports back via exit code.
        import subprocess
        import textwrap
        code = textwrap.dedent(
            """
            import sys
            # Force ImportError for any attempt to import isaaclab; if
            # `import vehicle_rl.assets` triggers the import we'll see it.
            class _Blocker:
                def find_module(self, name, path=None):
                    if name == 'isaaclab' or name.startswith('isaaclab.'):
                        return self
                    return None
                def load_module(self, name):
                    raise ImportError(
                        f'eager isaaclab import: {name}'
                    )
            sys.meta_path.insert(0, _Blocker())
            import vehicle_rl.assets  # noqa: F401
            # Geometry constants must work without isaaclab.
            assert vehicle_rl.assets.WHEELBASE == 2.7, vehicle_rl.assets.WHEELBASE
            assert 'isaaclab' not in sys.modules
            """
        )
        env = {
            "PYTHONPATH": str(REPO_ROOT / "src"),
        }
        # Inherit the rest of the env so we keep PATH etc.
        import os
        full_env = {**os.environ, **env}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=full_env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            proc.returncode, 0,
            f"importing vehicle_rl.assets pulled isaaclab.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )


class TestExperimentBundleIntegration(unittest.TestCase):
    def test_classical_circle_baseline_has_required_subbundles(self):
        bundle = load_experiment(
            REPO_ROOT / "configs" / "experiments" / "classical" / "circle_baseline.yaml",
            repo_root=REPO_ROOT,
        )
        # All three category bundles should validate against their factories.
        kw = make_simulator_kwargs(bundle["dynamics"])
        self.assertIn("mu_default", kw)
        geom = make_vehicle_geometry(bundle["vehicle"])
        self.assertEqual(geom["wheelbase_m"], 2.7)
        # Course bundle accepts the dispatch.
        path = build_path(bundle["course"], num_envs=1, device="cpu")
        self.assertTrue(path.is_loop)


# ---------------------------------------------------------------------------
# PR 3: env / agent factories. The isaac-free tests cover derived values and
# schema validation; the gated tests construct the actual cfg objects.
# ---------------------------------------------------------------------------


class TestDerivedActionSpace(unittest.TestCase):
    def test_steering_only_is_one(self):
        self.assertEqual(_derived_action_space("steering_only"), 1)

    def test_steering_and_accel_is_two(self):
        self.assertEqual(_derived_action_space("steering_and_accel"), 2)

    def test_unknown_action_mode_rejected(self):
        with self.assertRaises(ValueError):
            _derived_action_space("yaw_torque")


class TestDerivedObservationSpace(unittest.TestCase):
    def test_default_yaml_is_39(self):
        # Default tracking.yaml: 6 imu fields + pinion + 2 path errors + 0 world
        # = 9 scalar; plan_K=10 * 3 channels = 30; total = 39.
        self.assertEqual(_derived_observation_space(_env_bundle()), 39)

    def test_world_pose_adds_three(self):
        b = copy.deepcopy(_env_bundle())
        b["spaces"]["observation"]["include_world_pose"] = True
        self.assertEqual(_derived_observation_space(b), 42)

    def test_no_plan_drops_plan_channels(self):
        b = copy.deepcopy(_env_bundle())
        b["spaces"]["observation"]["include_plan"] = False
        # 9 scalar (unchanged) + 0 plan = 9.
        self.assertEqual(_derived_observation_space(b), 9)

    def test_changing_plan_K_scales_obs(self):
        b = copy.deepcopy(_env_bundle())
        b["planner"]["plan_K"] = 5
        # 9 scalar + 5 * 3 = 24.
        self.assertEqual(_derived_observation_space(b), 24)


class TestEnvBundleSchema(unittest.TestCase):
    """The PR 3-refined EnvSchema validates every leaf of tracking.yaml."""

    def test_default_env_yaml_validates(self):
        from vehicle_rl.config.schema import EnvSchema, validate_keys
        validate_keys(_env_bundle(), EnvSchema)  # no raise

    def test_unknown_reward_key_rejected(self):
        from vehicle_rl.config.schema import EnvSchema, validate_keys
        b = copy.deepcopy(_env_bundle())
        b["reward"]["bogus_field"] = 1.0
        with self.assertRaises(ValueError) as ctx:
            validate_keys(b, EnvSchema)
        self.assertIn("bogus_field", str(ctx.exception))

    def test_missing_termination_max_lateral_error_rejected(self):
        from vehicle_rl.config.schema import EnvSchema, validate_keys
        b = copy.deepcopy(_env_bundle())
        del b["termination"]["max_lateral_error_m"]
        with self.assertRaises(ValueError) as ctx:
            validate_keys(b, EnvSchema)
        self.assertIn("max_lateral_error_m", str(ctx.exception))

    def test_missing_top_level_key_rejected(self):
        from vehicle_rl.config.schema import EnvSchema, validate_keys
        b = copy.deepcopy(_env_bundle())
        del b["reward"]
        with self.assertRaises(ValueError) as ctx:
            validate_keys(b, EnvSchema)
        self.assertIn("reward", str(ctx.exception))


class TestAgentBundleSchema(unittest.TestCase):
    def test_default_agent_yaml_validates(self):
        from vehicle_rl.config.schema import AgentSchema, validate_keys
        validate_keys(_agent_bundle(), AgentSchema)

    def test_unknown_algorithm_key_rejected(self):
        from vehicle_rl.config.schema import AgentSchema, validate_keys
        b = copy.deepcopy(_agent_bundle())
        b["algorithm"]["bogus_field"] = 0.1
        with self.assertRaises(ValueError) as ctx:
            validate_keys(b, AgentSchema)
        self.assertIn("bogus_field", str(ctx.exception))

    def test_missing_runner_max_iterations_rejected(self):
        from vehicle_rl.config.schema import AgentSchema, validate_keys
        b = copy.deepcopy(_agent_bundle())
        del b["runner"]["max_iterations"]
        with self.assertRaises(ValueError) as ctx:
            validate_keys(b, AgentSchema)
        self.assertIn("max_iterations", str(ctx.exception))


class TestEntryPointFactoryDispatch(unittest.TestCase):
    """`vehicle_rl.tasks.tracking.entry_points` has the legacy course->yaml table."""

    def test_legacy_circle_maps_to_phase3_circle_yaml(self):
        from vehicle_rl.tasks.tracking.entry_points import LEGACY_COURSE_TO_EXPERIMENT
        self.assertIn("circle", LEGACY_COURSE_TO_EXPERIMENT)
        self.assertTrue(
            LEGACY_COURSE_TO_EXPERIMENT["circle"].endswith("phase3_circle_stage0a.yaml")
        )

    def test_random_long_maps_to_phase3_random_long_yaml(self):
        from vehicle_rl.tasks.tracking.entry_points import LEGACY_COURSE_TO_EXPERIMENT
        self.assertTrue(
            LEGACY_COURSE_TO_EXPERIMENT["random_long"].endswith("phase3_random_long.yaml")
        )

    def test_random_bank_maps_to_phase3_random_bank_yaml(self):
        from vehicle_rl.tasks.tracking.entry_points import LEGACY_COURSE_TO_EXPERIMENT
        self.assertTrue(
            LEGACY_COURSE_TO_EXPERIMENT["random_bank"].endswith("phase3_random_bank.yaml")
        )


@unittest.skipUnless(_isaaclab_available(),
                     "isaaclab not importable; skipping make_tracking_env_cfg test")
class TestMakeTrackingEnvCfg(unittest.TestCase):
    def test_default_yaml_returns_valid_cfg(self):
        from vehicle_rl.config.isaac_adapter import make_tracking_env_cfg

        env_b = _env_bundle()
        course_b = _course_bundle("circle")
        veh_b = _vehicle_bundle()
        dyn_b = _dynamics_bundle()
        cfg = make_tracking_env_cfg(
            env_b, course_b,
            controller_bundle=_controller_bundle(),
            vehicle_bundle=veh_b, dynamics_bundle=dyn_b,
        )
        # Top-level fields match YAML.
        self.assertEqual(int(cfg.scene.num_envs), 128)
        self.assertEqual(float(cfg.episode_length_s), 25.0)
        self.assertEqual(int(cfg.action_space), 1)         # steering_only
        self.assertEqual(int(cfg.observation_space), 39)   # derived: 9 + 10*3
        # Reward / termination / planner.
        self.assertEqual(float(cfg.rew_progress), 1.0)
        self.assertEqual(float(cfg.rew_alive), 0.1)
        self.assertEqual(float(cfg.max_lateral_error), 4.0)
        self.assertAlmostEqual(float(cfg.max_roll_rad), 1.047, places=4)
        self.assertEqual(int(cfg.plan_K), 10)
        self.assertEqual(float(cfg.lookahead_ds), 1.0)
        # Course-derived.
        self.assertEqual(cfg.course, "circle")
        self.assertEqual(float(cfg.radius), 30.0)
        self.assertEqual(float(cfg.target_speed), 10.0)
        self.assertAlmostEqual(float(cfg.course_ds), 0.2, places=6)
        # Dynamics-derived.
        self.assertEqual(float(cfg.a_x_min), -5.0)
        self.assertEqual(float(cfg.a_x_max), 3.0)
        # PI gains from speed_pi controller bundle.
        self.assertEqual(float(cfg.pi_kp), 1.0)
        self.assertEqual(float(cfg.pi_ki), 0.3)
        # Derived pinion_max.
        self.assertAlmostEqual(float(cfg.pinion_max), 0.611 * 16.0, places=5)

    def test_action_mode_steering_and_accel_yields_two(self):
        from vehicle_rl.config.isaac_adapter import make_tracking_env_cfg
        b = copy.deepcopy(_env_bundle())
        b["spaces"]["action_mode"] = "steering_and_accel"
        cfg = make_tracking_env_cfg(
            b, _course_bundle("circle"),
            controller_bundle=_controller_bundle(),
            vehicle_bundle=_vehicle_bundle(), dynamics_bundle=_dynamics_bundle(),
        )
        self.assertEqual(int(cfg.action_space), 2)
        self.assertFalse(cfg.steering_only)

    def test_missing_reward_progress_rejected(self):
        from vehicle_rl.config.isaac_adapter import make_tracking_env_cfg
        b = copy.deepcopy(_env_bundle())
        del b["reward"]["progress"]
        with self.assertRaises(ValueError) as ctx:
            make_tracking_env_cfg(
                b, _course_bundle("circle"),
                controller_bundle=_controller_bundle(),
                vehicle_bundle=_vehicle_bundle(), dynamics_bundle=_dynamics_bundle(),
            )
        self.assertIn("progress", str(ctx.exception))

    pass  # gated cases above; static signature gate moved to module level.


class TestMakeTrackingEnvCfgSignature(unittest.TestCase):
    """Static gate: every TrackingEnvCfg field is set by make_tracking_env_cfg.

    Mirrors stage-2's `test_kwargs_signature_matches_simulator_init`. Reads
    source files only, so it runs without isaaclab.
    """

    def test_make_tracking_env_cfg_signature_matches_TrackingEnvCfg(self):
        env_src = (
            REPO_ROOT / "src" / "vehicle_rl" / "envs" / "tracking_env.py"
        ).read_text(encoding="utf-8")
        adapter_src = (
            REPO_ROOT / "src" / "vehicle_rl" / "config" / "isaac_adapter.py"
        ).read_text(encoding="utf-8")
        import re
        cfg_class_match = re.search(
            r"class TrackingEnvCfg\(DirectRLEnvCfg\):\n(.*?)(?=\nclass |\Z)",
            env_src, re.DOTALL,
        )
        self.assertIsNotNone(cfg_class_match)
        body = cfg_class_match.group(1)
        field_re = re.compile(r"^    ([a-zA-Z_]\w*):\s+\S", re.MULTILINE)
        fields = set(field_re.findall(body))
        # state_space is intentionally constant 0; everything else must be
        # assigned by the factory.
        fields_to_check = fields - {"state_space"}
        self.assertGreater(len(fields_to_check), 20,
                           f"unexpectedly few fields scanned: {fields_to_check}")
        for fname in fields_to_check:
            self.assertIn(
                f"cfg.{fname} =",
                adapter_src,
                f"make_tracking_env_cfg does not assign cfg.{fname}; "
                f"either fill it in the factory or remove it from "
                f"TrackingEnvCfg.",
            )


@unittest.skipUnless(_isaaclab_available(),
                     "isaaclab not importable; skipping make_ppo_runner_cfg test")
class TestMakePPORunnerCfg(unittest.TestCase):
    def test_default_yaml_returns_valid_cfg(self):
        from vehicle_rl.config.isaac_adapter import make_ppo_runner_cfg
        cfg = make_ppo_runner_cfg(_agent_bundle())
        # runner spot-checks
        self.assertEqual(int(cfg.num_steps_per_env), 64)
        self.assertEqual(int(cfg.max_iterations), 300)
        self.assertEqual(int(cfg.save_interval), 50)
        self.assertEqual(cfg.experiment_name, "vehicle_tracking_direct")
        self.assertEqual(float(cfg.clip_actions), 1.0)
        # policy spot-checks
        self.assertAlmostEqual(float(cfg.policy.init_noise_std), 0.3, places=6)
        self.assertEqual(list(cfg.policy.actor_hidden_dims), [256, 256])
        self.assertEqual(list(cfg.policy.critic_hidden_dims), [256, 256])
        self.assertEqual(cfg.policy.activation, "elu")
        # algorithm spot-checks
        self.assertAlmostEqual(float(cfg.algorithm.gamma), 0.995, places=6)
        self.assertAlmostEqual(float(cfg.algorithm.lam), 0.95, places=6)
        self.assertAlmostEqual(float(cfg.algorithm.learning_rate), 3.0e-4, places=8)
        self.assertEqual(cfg.algorithm.schedule, "adaptive")

    def test_unknown_nested_key_rejected(self):
        from vehicle_rl.config.isaac_adapter import make_ppo_runner_cfg
        b = copy.deepcopy(_agent_bundle())
        b["policy"]["bogus_field"] = 1.0
        with self.assertRaises(ValueError) as ctx:
            make_ppo_runner_cfg(b)
        self.assertIn("bogus_field", str(ctx.exception))

    def test_missing_algorithm_block_rejected(self):
        from vehicle_rl.config.isaac_adapter import make_ppo_runner_cfg
        b = copy.deepcopy(_agent_bundle())
        del b["algorithm"]
        with self.assertRaises(ValueError) as ctx:
            make_ppo_runner_cfg(b)
        self.assertIn("algorithm", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
