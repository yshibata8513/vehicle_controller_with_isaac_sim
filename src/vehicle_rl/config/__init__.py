"""YAML config loader and schema for vehicle_rl experiments.

Public API:
    load_yaml_strict(path)         -- parse a YAML file with duplicate-key rejection.
    resolve_refs(data, repo_root)  -- replace `<name>_ref` keys with referenced YAML.
    deep_merge_overrides(base, ov) -- strict deep-merge (unknown key forbidden,
                                      list = replace, `*_ref` forbidden in overrides).
    load_experiment(path, repo_root) -- full pipeline (load -> resolve -> merge).
    dump_resolved_config(d, log_dir) -- write resolved bundle for reproducibility.
    validate_keys(data, schema)      -- check every dataclass field is present and
                                        no extra keys exist.

See `docs/yaml_config_structure_plan.md` for the design.
"""
from vehicle_rl.config.loader import (
    deep_merge_overrides,
    dump_resolved_config,
    load_experiment,
    load_yaml_strict,
    resolve_refs,
)
from vehicle_rl.config.schema import validate_keys

__all__ = [
    "deep_merge_overrides",
    "dump_resolved_config",
    "load_experiment",
    "load_yaml_strict",
    "resolve_refs",
    "validate_keys",
]
