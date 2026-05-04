"""Compare two classical-rollout artifacts for refactor regression checks.

This is intentionally independent of how the rollout was launched. Before
the YAML refactor, the "before" artifacts can come from the old CLI; after
the refactor, the "after" artifacts can come from the new --config entry.
The test passes only if the CSV trajectory and JSON summary match within a
small numeric tolerance.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


def _resolve_artifacts(
    directory: Path | None,
    tag: str,
    csv_path: Path | None,
    json_path: Path | None,
) -> tuple[Path, Path]:
    if directory is not None:
        return directory / f"{tag}.csv", directory / f"{tag}.json"
    if csv_path is None or json_path is None:
        raise ValueError("provide either --dir or both --csv/--json")
    return csv_path, json_path


def _load_csv(path: Path) -> tuple[list[str], list[list[float]]]:
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    header = rows[0]
    data: list[list[float]] = []
    for row_i, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise ValueError(
                f"{path}:{row_i}: expected {len(header)} columns, got {len(row)}"
            )
        data.append([float(v) for v in row])
    return header, data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _close(a: float, b: float, *, atol: float, rtol: float) -> bool:
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def _compare_json(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> list[str]:
    errors: list[str] = []
    before_keys = set(before)
    after_keys = set(after)
    if before_keys != after_keys:
        missing = sorted(before_keys - after_keys)
        extra = sorted(after_keys - before_keys)
        if missing:
            errors.append(f"JSON missing keys: {missing}")
        if extra:
            errors.append(f"JSON extra keys: {extra}")
        return errors

    for key in sorted(before_keys):
        a = before[key]
        b = after[key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not _close(float(a), float(b), atol=atol, rtol=rtol):
                errors.append(f"JSON {key}: before={a!r} after={b!r}")
        elif a != b:
            errors.append(f"JSON {key}: before={a!r} after={b!r}")
    return errors


def _compare_csv(
    before_header: list[str],
    before_data: list[list[float]],
    after_header: list[str],
    after_data: list[list[float]],
    *,
    atol: float,
    rtol: float,
) -> tuple[list[str], float]:
    errors: list[str] = []
    max_abs = 0.0

    if before_header != after_header:
        return [f"CSV header mismatch: {before_header!r} != {after_header!r}"], max_abs
    if len(before_data) != len(after_data):
        return [f"CSV row count mismatch: {len(before_data)} != {len(after_data)}"], max_abs

    first_diff: str | None = None
    for row_i, (row_a, row_b) in enumerate(zip(before_data, after_data), start=2):
        for col_i, (a, b) in enumerate(zip(row_a, row_b)):
            diff = abs(a - b)
            if diff > max_abs:
                max_abs = diff
            if not _close(a, b, atol=atol, rtol=rtol) and first_diff is None:
                first_diff = (
                    f"CSV first mismatch at row={row_i} col={before_header[col_i]!r}: "
                    f"before={a!r} after={b!r} abs_diff={diff:.6g}"
                )
    if first_diff is not None:
        errors.append(first_diff)
    return errors, max_abs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare before/after run_classical rollout artifacts."
    )
    parser.add_argument(
        "--tag",
        default="classical_circle_mu0.90_v10.0",
        help="Artifact basename without .csv/.json when using --before-dir/--after-dir.",
    )
    parser.add_argument("--before-dir", type=Path)
    parser.add_argument("--after-dir", type=Path)
    parser.add_argument("--before-csv", type=Path)
    parser.add_argument("--before-json", type=Path)
    parser.add_argument("--after-csv", type=Path)
    parser.add_argument("--after-json", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-7)
    args = parser.parse_args()

    before_csv, before_json = _resolve_artifacts(
        args.before_dir, args.tag, args.before_csv, args.before_json
    )
    after_csv, after_json = _resolve_artifacts(
        args.after_dir, args.tag, args.after_csv, args.after_json
    )

    for path in (before_csv, before_json, after_csv, after_json):
        if not path.exists():
            print(f"[FAIL] missing artifact: {path}", file=sys.stderr)
            return 2

    bh, bd = _load_csv(before_csv)
    ah, ad = _load_csv(after_csv)
    csv_errors, csv_max_abs = _compare_csv(
        bh, bd, ah, ad, atol=args.atol, rtol=args.rtol
    )
    json_errors = _compare_json(
        _load_json(before_json), _load_json(after_json),
        atol=args.atol, rtol=args.rtol,
    )

    errors = csv_errors + json_errors
    if errors:
        print("[FAIL] rollout artifacts differ")
        for error in errors[:20]:
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  - ... {len(errors) - 20} more")
        print(f"[INFO] csv_max_abs_diff={csv_max_abs:.9g}")
        return 1

    print("[PASS] rollout artifacts match")
    print(f"[INFO] rows={len(bd)} columns={len(bh)} csv_max_abs_diff={csv_max_abs:.9g}")
    print(f"[INFO] atol={args.atol:g} rtol={args.rtol:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
