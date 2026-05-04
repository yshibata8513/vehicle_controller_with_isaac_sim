"""Strict YAML loader with ref resolution and override deep-merge.

Design rules (see docs/yaml_config_structure_plan.md):
- missing key in YAML is the consumer's job to detect (use validate_keys).
- duplicate keys in a YAML mapping raise immediately at load time.
- `<name>_ref: path` resolves to `<name>: <referenced YAML content>`. Refs are
  repo-root relative, transitive, cached, and cycle-checked.
- Overrides are deep-merged onto the resolved bundle. Unknown keys raise.
  Lists are replaced wholesale. `*_ref` keys are not allowed inside overrides
  (changing the structure of the experiment must be done by writing a new
  experiment YAML).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REF_SUFFIX = "_ref"


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader subclass that rejects duplicate mapping keys."""


def _construct_mapping_strict(loader: _StrictLoader, node: yaml.MappingNode) -> dict:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                None, None,
                f"duplicate key in mapping: {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_strict,
)


def load_yaml_strict(path: Path) -> dict[str, Any]:
    """Load a YAML file and require the root to be a mapping.

    Duplicate keys raise yaml.constructor.ConstructorError.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_StrictLoader)
    if data is None:
        raise ValueError(f"YAML is empty: {path}")
    if not isinstance(data, dict):
        raise ValueError(
            f"YAML root must be a mapping, got {type(data).__name__}: {path}"
        )
    return data


def _resolve_ref_path(ref_value: str, repo_root: Path) -> Path:
    if not isinstance(ref_value, str):
        raise ValueError(f"ref value must be a string, got {type(ref_value).__name__}")
    candidate = Path(ref_value)
    if candidate.is_absolute():
        raise ValueError(
            f"ref must be repo-root relative, got absolute path: {ref_value!r}"
        )
    repo_root_resolved = Path(repo_root).resolve()
    resolved = (repo_root_resolved / candidate).resolve()
    try:
        resolved.relative_to(repo_root_resolved)
    except ValueError as exc:
        raise ValueError(f"ref escapes repo root: {ref_value!r}") from exc
    if not resolved.is_file():
        raise ValueError(f"ref target does not exist: {ref_value!r} -> {resolved}")
    return resolved


def resolve_refs(
    data: Any,
    *,
    repo_root: Path,
    cache: dict[Path, Any] | None = None,
    visiting: tuple[Path, ...] = (),
) -> Any:
    """Recursively replace `<name>_ref: path` keys with `<name>: <resolved>`.

    - cache: shared across the recursion; same ref path is loaded once.
    - visiting: chain of currently-resolving paths for cycle detection.
    """
    if cache is None:
        cache = {}

    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(key, str) and key.endswith(REF_SUFFIX) and key != REF_SUFFIX:
                target_key = key[: -len(REF_SUFFIX)]
                if target_key in data:
                    raise ValueError(
                        f"ref key {key!r} would overwrite sibling key {target_key!r}"
                    )
                if target_key in out:
                    raise ValueError(
                        f"ref key {key!r} resolved twice into {target_key!r}"
                    )
                ref_path = _resolve_ref_path(value, repo_root)
                if ref_path in visiting:
                    chain = " -> ".join(str(p) for p in visiting + (ref_path,))
                    raise ValueError(f"ref cycle detected: {chain}")
                if ref_path in cache:
                    resolved = cache[ref_path]
                else:
                    raw = load_yaml_strict(ref_path)
                    resolved = resolve_refs(
                        raw,
                        repo_root=repo_root,
                        cache=cache,
                        visiting=visiting + (ref_path,),
                    )
                    cache[ref_path] = resolved
                out[target_key] = resolved
            else:
                out[key] = resolve_refs(
                    value,
                    repo_root=repo_root,
                    cache=cache,
                    visiting=visiting,
                )
        return out

    if isinstance(data, list):
        return [
            resolve_refs(v, repo_root=repo_root, cache=cache, visiting=visiting)
            for v in data
        ]

    return data


def deep_merge_overrides(
    base: dict[str, Any],
    overrides: dict[str, Any],
    *,
    path: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Deep-merge `overrides` into `base`. Returns a new dict; does not mutate inputs.

    Strict rules:
    - Every key in `overrides` must exist in `base`. Unknown -> ValueError.
    - Lists are replaced wholesale (no element-wise merge).
    - `<name>_ref` keys in overrides -> ValueError. Refs are structural and
      must be set in the experiment composition root, not via override.
    """
    if not isinstance(overrides, dict):
        loc = ".".join(path) if path else "<root>"
        raise ValueError(
            f"override at {loc} must be a mapping, got {type(overrides).__name__}"
        )
    if not isinstance(base, dict):
        loc = ".".join(path) if path else "<root>"
        raise ValueError(
            f"override target at {loc} must be a mapping, got {type(base).__name__}"
        )

    out = dict(base)
    for key, value in overrides.items():
        full_path = path + (str(key),)
        if isinstance(key, str) and key.endswith(REF_SUFFIX) and key != REF_SUFFIX:
            raise ValueError(
                f"override forbids ref keys: {'.'.join(full_path)}"
            )
        if key not in out:
            raise ValueError(
                f"override unknown key: {'.'.join(full_path)} "
                f"(known keys at this level: {sorted(out.keys())})"
            )
        if isinstance(value, dict) and isinstance(out[key], dict):
            out[key] = deep_merge_overrides(out[key], value, path=full_path)
        else:
            out[key] = value
    return out


def load_experiment(path: Path, *, repo_root: Path) -> dict[str, Any]:
    """Load an experiment YAML: parse -> resolve refs -> merge overrides.

    Conventions:
    - The experiment YAML may contain an `overrides` mapping at the root.
      It is removed from the bundle, deep-merged onto the rest, and the
      result is returned.
    - `overrides` may not contain `*_ref` keys.
    - Every other key is preserved as-is in the resolved bundle.
    """
    raw = load_yaml_strict(Path(path))
    overrides = raw.pop("overrides", None)
    resolved = resolve_refs(raw, repo_root=Path(repo_root))
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError(
                f"experiment 'overrides' must be a mapping, got {type(overrides).__name__}"
            )
        resolved = deep_merge_overrides(resolved, overrides)
    return resolved


def dump_resolved_config(resolved: dict[str, Any], log_dir: Path) -> Path:
    """Dump a resolved bundle to `<log_dir>/resolved_config.yaml` for reproducibility."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "resolved_config.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False, allow_unicode=True)
    return out_path
