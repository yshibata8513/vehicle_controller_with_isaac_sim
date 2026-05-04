"""Loader / schema unit tests.

Run with:
    python -m unittest tests.config.test_loader -v
or with pytest:
    pytest tests/config/test_loader.py -v
"""
from __future__ import annotations

import textwrap
import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

# Hardcode REPO_ROOT to the repo where this test file lives. Resolves the
# `src/` import without requiring an editable install.
REPO_ROOT = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from vehicle_rl.config import (
    deep_merge_overrides,
    dump_resolved_config,
    load_experiment,
    load_yaml_strict,
    resolve_refs,
    validate_keys,
)


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")


class TestStrictYamlLoad(unittest.TestCase):
    def test_loads_simple_mapping(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.yaml"
            _write(p, """
                schema_version: 1
                a: 1
                b: hello
            """)
            data = load_yaml_strict(p)
            self.assertEqual(data, {"schema_version": 1, "a": 1, "b": "hello"})

    def test_rejects_duplicate_key(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.yaml"
            _write(p, """
                a: 1
                a: 2
            """)
            with self.assertRaises(yaml.constructor.ConstructorError) as ctx:
                load_yaml_strict(p)
            self.assertIn("duplicate", str(ctx.exception).lower())

    def test_rejects_non_mapping_root(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.yaml"
            _write(p, "- 1\n- 2\n")
            with self.assertRaises(ValueError):
                load_yaml_strict(p)

    def test_rejects_empty_file(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.yaml"
            p.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_yaml_strict(p)


class TestResolveRefs(unittest.TestCase):
    def test_resolves_simple_ref(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "child.yaml", """
                kp: 1.0
                ki: 0.3
            """)
            _write(root / "parent.yaml", """
                schema_version: 1
                child_ref: child.yaml
            """)
            raw = load_yaml_strict(root / "parent.yaml")
            resolved = resolve_refs(raw, repo_root=root)
            self.assertEqual(resolved["child"], {"kp": 1.0, "ki": 0.3})
            self.assertNotIn("child_ref", resolved)

    def test_transitive_refs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "leaf.yaml", "value: 42\n")
            _write(root / "mid.yaml", """
                schema_version: 1
                leaf_ref: leaf.yaml
            """)
            _write(root / "root.yaml", """
                schema_version: 1
                mid_ref: mid.yaml
            """)
            raw = load_yaml_strict(root / "root.yaml")
            resolved = resolve_refs(raw, repo_root=root)
            self.assertEqual(resolved["mid"]["leaf"], {"value": 42})

    def test_ref_cache_loads_once(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "shared.yaml", "value: 1\n")
            _write(root / "parent.yaml", """
                schema_version: 1
                a_ref: shared.yaml
                b_ref: shared.yaml
            """)
            raw = load_yaml_strict(root / "parent.yaml")
            cache: dict = {}
            resolved = resolve_refs(raw, repo_root=root, cache=cache)
            self.assertEqual(len(cache), 1)
            # The same dict object should be reused (identity check).
            self.assertIs(resolved["a"], resolved["b"])

    def test_ref_cycle_raises(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "a.yaml", "b_ref: b.yaml\n")
            _write(root / "b.yaml", "a_ref: a.yaml\n")
            raw = load_yaml_strict(root / "a.yaml")
            with self.assertRaisesRegex(ValueError, "cycle"):
                resolve_refs(raw, repo_root=root)

    def test_absolute_ref_path_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "p.yaml", f"x_ref: {root.resolve() / 'leaf.yaml'}\n")
            _write(root / "leaf.yaml", "v: 1\n")
            raw = load_yaml_strict(root / "p.yaml")
            with self.assertRaisesRegex(ValueError, "absolute"):
                resolve_refs(raw, repo_root=root)

    def test_ref_escaping_repo_root_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            outside = Path(tmp) / "outside.yaml"
            _write(outside, "v: 1\n")
            _write(root / "p.yaml", "x_ref: ../outside.yaml\n")
            raw = load_yaml_strict(root / "p.yaml")
            with self.assertRaisesRegex(ValueError, "escape"):
                resolve_refs(raw, repo_root=root)

    def test_ref_target_missing_raises(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "p.yaml", "x_ref: nope.yaml\n")
            raw = load_yaml_strict(root / "p.yaml")
            with self.assertRaisesRegex(ValueError, "does not exist"):
                resolve_refs(raw, repo_root=root)

    def test_ref_overwrite_sibling_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "leaf.yaml", "x: 1\n")
            _write(root / "p.yaml", """
                child: existing
                child_ref: leaf.yaml
            """)
            raw = load_yaml_strict(root / "p.yaml")
            with self.assertRaisesRegex(ValueError, "overwrite sibling"):
                resolve_refs(raw, repo_root=root)


class TestDeepMergeOverrides(unittest.TestCase):
    def test_simple_scalar_replace(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        out = deep_merge_overrides(base, {"a": 10, "b": {"c": 20}})
        self.assertEqual(out, {"a": 10, "b": {"c": 20, "d": 3}})

    def test_unknown_key_rejected(self):
        base = {"a": 1}
        with self.assertRaisesRegex(ValueError, "unknown key.*banana"):
            deep_merge_overrides(base, {"banana": 5})

    def test_unknown_nested_key_rejected(self):
        base = {"a": {"b": 1}}
        with self.assertRaisesRegex(ValueError, "unknown key.*a\\.banana"):
            deep_merge_overrides(base, {"a": {"banana": 5}})

    def test_list_replaced_wholesale(self):
        base = {"hidden_dims": [256, 256]}
        out = deep_merge_overrides(base, {"hidden_dims": [512]})
        self.assertEqual(out, {"hidden_dims": [512]})

    def test_ref_in_overrides_rejected(self):
        base = {"vehicle": {"name": "sedan"}}
        with self.assertRaisesRegex(ValueError, "ref keys"):
            deep_merge_overrides(base, {"vehicle_ref": "configs/vehicles/other.yaml"})

    def test_does_not_mutate_inputs(self):
        base = {"a": {"b": 1}}
        overrides = {"a": {"b": 2}}
        _ = deep_merge_overrides(base, overrides)
        self.assertEqual(base, {"a": {"b": 1}})
        self.assertEqual(overrides, {"a": {"b": 2}})


class TestLoadExperiment(unittest.TestCase):
    def test_full_pipeline(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "configs/courses/circle.yaml", """
                schema_version: 1
                type: circle
                radius_m: 30.0
            """)
            _write(root / "configs/agents/ppo.yaml", """
                schema_version: 1
                runner:
                  max_iterations: 300
                  experiment_name: default
            """)
            _write(root / "configs/experiments/x.yaml", """
                schema_version: 1
                kind: rl_train
                course_ref: configs/courses/circle.yaml
                agent_ref: configs/agents/ppo.yaml
                overrides:
                  agent:
                    runner:
                      max_iterations: 500
                      experiment_name: x
            """)
            resolved = load_experiment(
                root / "configs/experiments/x.yaml", repo_root=root,
            )
            self.assertEqual(resolved["course"]["radius_m"], 30.0)
            self.assertEqual(resolved["agent"]["runner"]["max_iterations"], 500)
            self.assertEqual(resolved["agent"]["runner"]["experiment_name"], "x")
            self.assertEqual(resolved["kind"], "rl_train")
            self.assertNotIn("overrides", resolved)


class TestDumpResolvedConfig(unittest.TestCase):
    def test_roundtrip(self):
        with TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "log"
            data = {"a": 1, "b": {"c": [1, 2, 3]}}
            out = dump_resolved_config(data, log_dir)
            self.assertTrue(out.exists())
            with out.open("r", encoding="utf-8") as f:
                roundtrip = yaml.safe_load(f)
            self.assertEqual(roundtrip, data)


@dataclass
class _Inner:
    x: int
    y: str


@dataclass
class _Outer:
    schema_version: int
    name: str
    inner: _Inner


class TestValidateKeys(unittest.TestCase):
    def test_passes_on_exact_match(self):
        validate_keys(
            {"schema_version": 1, "name": "n", "inner": {"x": 1, "y": "s"}},
            _Outer,
        )

    def test_missing_top_level_key(self):
        with self.assertRaisesRegex(ValueError, "missing keys.*name"):
            validate_keys(
                {"schema_version": 1, "inner": {"x": 1, "y": "s"}}, _Outer,
            )

    def test_unknown_top_level_key(self):
        with self.assertRaisesRegex(ValueError, "unknown keys.*extra"):
            validate_keys(
                {
                    "schema_version": 1, "name": "n",
                    "inner": {"x": 1, "y": "s"}, "extra": True,
                },
                _Outer,
            )

    def test_recurses_into_nested_dataclass(self):
        with self.assertRaisesRegex(ValueError, "missing keys at inner.*y"):
            validate_keys(
                {"schema_version": 1, "name": "n", "inner": {"x": 1}},
                _Outer,
            )


class TestRepoExperimentYAMLs(unittest.TestCase):
    """Smoke test: every YAML under configs/experiments/ loads cleanly."""

    EXPERIMENTS = [
        "configs/experiments/rl/phase3_circle_stage0a.yaml",
        "configs/experiments/rl/phase3_random_long.yaml",
        "configs/experiments/rl/phase3_random_bank.yaml",
        "configs/experiments/rl/phase3_random_bank_play.yaml",
        "configs/experiments/classical/circle_baseline.yaml",
        "configs/experiments/classical/s_curve_baseline.yaml",
        "configs/experiments/classical/dlc_baseline.yaml",
        "configs/experiments/classical/circle_refactor_guard.yaml",
    ]

    def test_each_experiment_loads(self):
        for rel in self.EXPERIMENTS:
            with self.subTest(experiment=rel):
                resolved = load_experiment(REPO_ROOT / rel, repo_root=REPO_ROOT)
                self.assertIn("schema_version", resolved)
                self.assertIn("kind", resolved)


if __name__ == "__main__":
    unittest.main()
