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
        # The leaf check fires before validate_keys would catch a missing
        # nested field on `mass`, so the error message names total_kg.
        self.assertIn("total_kg", str(ctx.exception))


# ---------------------------------------------------------------------------
# make_simulator_kwargs / make_action_limits
# ---------------------------------------------------------------------------


# The kwargs returned by `make_simulator_kwargs` must match the
# VehicleSimulator.__init__ signature exactly (minus `sim`, `sedan`,
# `device`, and `steering_ratio` which lives in the vehicle bundle).
_EXPECTED_SIM_KEYS = {
    "tau_steer", "tau_drive", "tau_brake",
    "cornering_stiffness",
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
        self.assertEqual(kw["cornering_stiffness"], 60000.0)
        self.assertEqual(kw["mu_default"], 0.9)
        self.assertEqual(kw["gravity"], 9.81)
        self.assertEqual(kw["k_roll"], 80000.0)

    def test_kwargs_signature_matches_simulator_init(self):
        # We don't import VehicleSimulator (it pulls isaaclab); instead we
        # statically read the simulator source and confirm every returned
        # key appears as a parameter name. This catches signature drift
        # without booting Isaac.
        src = (
            REPO_ROOT / "src" / "vehicle_rl" / "envs" / "simulator.py"
        ).read_text(encoding="utf-8")
        for k in _EXPECTED_SIM_KEYS:
            self.assertIn(
                f"        {k}: float",
                src,
                f"VehicleSimulator.__init__ has no parameter named {k!r}",
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


if __name__ == "__main__":
    unittest.main()
