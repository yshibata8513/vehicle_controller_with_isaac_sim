"""Random clothoid+arc+straight path generator (Phase 3 random-path training).

A path is built by concatenating segments with one of three curvature
profiles:

  - straight   : kappa(s) = 0
  - clothoid   : kappa(s) varies linearly with arc length
  - arc        : kappa(s) = const

The integration is exact for the heading
(`psi[i+1] = wrap(psi[i] + kappa[i] * ds)`) and uses a midpoint-style
forward-Euler step for `(x, y)` so the path stays on the discretization
grid `s = i * ds`. Generation runs once at env construction (or at bank
build / regeneration) on CPU; the resulting tensors are uploaded to GPU
and never re-touched in the hot loop -- see PLAN §0.5.

This module is the single source of truth for random-path *geometry*.
Per-step projection lives in `vehicle_rl.planner.path:Path.project`
(local-window argmin) and is unaware of how the path was built.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path as _PyPath
from typing import Sequence

import torch
import yaml
from torch import Tensor

from .path import Path, _wrap_to_pi


# --------------------------------------------------------------------------
# Config dataclasses
# --------------------------------------------------------------------------


@dataclass
class _GeneratorCfg:
    seed: int
    ds: float
    target_speed: float


@dataclass
class _SpeedCfg:
    v_min: float
    v_max: float
    ay_limit: float
    segment_constant: bool
    # Per-segment retry budget when `v_curve_limit < v_min`. The geometry is
    # re-sampled (looser radius / shorter arc) until either a feasible
    # segment is found or this many attempts have been spent; on exhaustion
    # the last attempt is accepted with `v = v_min` so generation cannot
    # deadlock. Logged via the count printed by `random_clothoid_path`.
    max_resample_attempts: int = 50


@dataclass
class _SegmentsCfg:
    """Turn / straight composition.

    Turn segments are parameterized by the *whole-turn* heading change so
    `heading_total <= turn_heading_change_rad[1]` is structurally guaranteed
    (no reject-and-retry needed, item 3 better-design). The fraction of
    `heading_total` carried by the clothoid pair is sampled from
    `clothoid_heading_fraction`; the remainder goes to the arc. Per-side
    clothoid length and arc length are then derived as
    `L_clo = heading_clothoids / kappa_abs`,
    `L_arc = heading_arc / kappa_abs`.
    """
    straight_length_m: tuple[float, float]
    min_radius_m: float
    max_radius_m: float
    turn_heading_change_rad: tuple[float, float]
    clothoid_heading_fraction: tuple[float, float]
    straight_probability: float
    turn_probability: float
    reverse_turn_probability: float


@dataclass
class _ProjectionCfg:
    search_radius_samples: int
    recovery_radius_samples: int
    max_index_jump_samples: int


@dataclass
class _ResetCfg:
    random_reset_along_path: bool
    end_margin_extra_m: float


@dataclass
class _Phase1Cfg:
    length_m: float
    is_loop: bool


@dataclass
class _Phase2Cfg:
    num_paths: int
    length_m: float
    is_loop: bool


@dataclass
class _Phase3Cfg:
    enabled: bool
    interval_resets: int
    fraction: float
    min_unused_slots: int


@dataclass
class RandomPathGeneratorCfg:
    """Parsed `configs/random_path.yaml`. Subsections map 1:1 to YAML keys."""
    generator: _GeneratorCfg
    speed: _SpeedCfg
    segments: _SegmentsCfg
    projection: _ProjectionCfg
    reset: _ResetCfg
    phase1_long_path: _Phase1Cfg
    phase2_bank: _Phase2Cfg
    phase3_regeneration: _Phase3Cfg
    source_path: str = ""

    def validate(self) -> None:
        s = self.speed
        if s.v_min <= 0.0:
            raise ValueError(f"speed.v_min must be > 0, got {s.v_min}")
        if s.v_max < s.v_min:
            raise ValueError(f"speed.v_max ({s.v_max}) must be >= v_min ({s.v_min})")
        if s.ay_limit <= 0.0:
            raise ValueError(f"speed.ay_limit must be > 0, got {s.ay_limit}")
        if s.max_resample_attempts <= 0:
            raise ValueError(
                f"speed.max_resample_attempts must be > 0, got {s.max_resample_attempts}"
            )

        seg = self.segments
        if seg.min_radius_m <= 0.0:
            raise ValueError(f"segments.min_radius_m must be > 0, got {seg.min_radius_m}")
        if seg.max_radius_m < seg.min_radius_m:
            raise ValueError(
                f"segments.max_radius_m ({seg.max_radius_m}) must be >= min_radius_m ({seg.min_radius_m})"
            )
        if not (0.0 <= seg.straight_probability <= 1.0):
            raise ValueError(f"segments.straight_probability must be in [0,1]")
        if not (0.0 <= seg.turn_probability <= 1.0):
            raise ValueError(f"segments.turn_probability must be in [0,1]")
        # Allow them to not sum to 1 (renormalized at sample time), but warn-worthy.
        h_lo, h_hi = seg.turn_heading_change_rad
        if h_lo <= 0.0 or h_hi < h_lo:
            raise ValueError(
                f"segments.turn_heading_change_rad must be 0 < lo <= hi, got ({h_lo}, {h_hi})"
            )
        f_lo, f_hi = seg.clothoid_heading_fraction
        if not (0.0 <= f_lo <= f_hi <= 1.0):
            raise ValueError(
                f"segments.clothoid_heading_fraction must satisfy 0 <= lo <= hi <= 1, "
                f"got ({f_lo}, {f_hi})"
            )

        proj = self.projection
        if proj.search_radius_samples <= 0:
            raise ValueError(f"projection.search_radius_samples must be > 0")
        if proj.recovery_radius_samples < proj.search_radius_samples:
            raise ValueError(
                "projection.recovery_radius_samples must be >= search_radius_samples"
            )
        if proj.max_index_jump_samples <= 0:
            raise ValueError("projection.max_index_jump_samples must be > 0")

        gen = self.generator
        if gen.ds <= 0.0:
            raise ValueError(f"generator.ds must be > 0, got {gen.ds}")


def _as_tuple_pair(v) -> tuple[float, float]:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return float(v[0]), float(v[1])
    raise ValueError(f"expected 2-element list, got {v!r}")


def load_random_path_cfg(path: str | PathLike) -> RandomPathGeneratorCfg:
    """Parse `configs/random_path.yaml` (or any compatible file) into a cfg.

    Resolves the path as-is when absolute; relative paths are resolved
    against the repository root (the package's grandparent of grandparent
    directory) so `cfg=configs/random_path.yaml` works regardless of CWD.
    """
    p = _PyPath(path)
    if not p.is_absolute():
        # repo root = .../vehicle_rl/  (3 levels up from this file:
        # src/vehicle_rl/planner/random_path.py)
        repo_root = _PyPath(__file__).resolve().parents[3]
        p = (repo_root / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"random_path cfg not found: {p}")

    with open(p, "r") as f:
        raw = yaml.safe_load(f)

    g = raw["generator"]
    s = raw["speed"]
    seg = raw["segments"]
    pj = raw["projection"]
    rs = raw["reset"]
    p1 = raw["phase1_long_path"]
    p2 = raw["phase2_bank"]
    p3 = raw["phase3_regeneration"]

    cfg = RandomPathGeneratorCfg(
        generator=_GeneratorCfg(
            seed=int(g["seed"]),
            ds=float(g["ds"]),
            target_speed=float(g["target_speed"]),
        ),
        speed=_SpeedCfg(
            v_min=float(s["v_min"]),
            v_max=float(s["v_max"]),
            ay_limit=float(s["ay_limit"]),
            segment_constant=bool(s["segment_constant"]),
            # Optional: older configs without this key fall back to the
            # dataclass default (50). Defensive cast so YAML strings parse.
            max_resample_attempts=int(s.get("max_resample_attempts", 50)),
        ),
        segments=_SegmentsCfg(
            straight_length_m=_as_tuple_pair(seg["straight_length_m"]),
            min_radius_m=float(seg["min_radius_m"]),
            max_radius_m=float(seg["max_radius_m"]),
            turn_heading_change_rad=_as_tuple_pair(seg["turn_heading_change_rad"]),
            clothoid_heading_fraction=_as_tuple_pair(seg["clothoid_heading_fraction"]),
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
            random_reset_along_path=bool(rs["random_reset_along_path"]),
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
        source_path=str(p),
    )
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------
# Geometry generation
# --------------------------------------------------------------------------


def _sample_segment_kappa_length(
    cfg: RandomPathGeneratorCfg,
    rng: torch.Generator,
    *,
    last_turn_sign: int,
) -> tuple[Tensor, float, int]:
    """Sample one segment as (kappa_per_sample, length_m, new_turn_sign).

    Returned kappa is shape (n_samples,) sampled at `ds` spacing. The
    segment composition is one of:

      - straight: a single straight of [straight_length_m] m
      - turn    : clothoid-up + arc + clothoid-down (S-shape)

    Turn-segment procedure (review item 3 better-design):

        1. Sample R uniformly in [min_radius_m, max_radius_m].
        2. kappa_abs = 1 / R.
        3. Sample heading_total uniformly in turn_heading_change_rad.
        4. Sample fraction uniformly in clothoid_heading_fraction.
        5. heading_clothoids = heading_total * fraction
        6. heading_arc = heading_total - heading_clothoids
        7. L_clo = heading_clothoids / kappa_abs        (per-side clothoid length;
           up + down clothoids together contribute kappa_abs * L_clo of heading
           because each side averages 0.5 * kappa_abs over L_clo.)
        8. L_arc = heading_arc / kappa_abs

    The arc is allowed to be exactly zero (0-length). The clothoid pair
    minimum is one ds sample per side. Heading bound is structural: every
    accepted turn satisfies `heading_total <= turn_heading_change_rad[1]`
    by construction, no reject needed.

    The turn sign may flip with `reverse_turn_probability`.
    """
    seg = cfg.segments
    ds = cfg.generator.ds

    p_straight = seg.straight_probability
    p_turn = seg.turn_probability
    total_p = p_straight + p_turn
    if total_p <= 0.0:
        raise ValueError("straight_probability + turn_probability must be > 0")
    p_straight_norm = p_straight / total_p

    u = float(torch.rand((), generator=rng).item())
    if u < p_straight_norm:
        # Straight segment.
        L = _uniform(seg.straight_length_m, rng)
        n = max(1, int(round(L / ds)))
        return torch.zeros(n, dtype=torch.float32), n * ds, last_turn_sign

    # Turn: choose sign (possibly flip).
    if last_turn_sign == 0:
        sign = 1 if float(torch.rand((), generator=rng).item()) < 0.5 else -1
    else:
        flip = float(torch.rand((), generator=rng).item()) < seg.reverse_turn_probability
        sign = -last_turn_sign if flip else last_turn_sign

    # Steps 1-2: radius -> kappa magnitude.
    R = _uniform((seg.min_radius_m, seg.max_radius_m), rng)
    kappa_max = sign * (1.0 / R)
    kappa_abs = 1.0 / R

    # Steps 3-6: heading budget split between clothoid pair and arc.
    heading_total = _uniform(seg.turn_heading_change_rad, rng)
    fraction = _uniform(seg.clothoid_heading_fraction, rng)
    heading_clothoids = heading_total * fraction
    heading_arc = heading_total - heading_clothoids

    # Steps 7-8: derived lengths. Per-side clothoid length is
    # heading_clothoids / kappa_abs because the up + down clothoids together
    # average kappa_abs over L_clo. Floor (not round) the arc count so it
    # may be 0 when fraction == 1.0; require at least 1 ds sample per
    # clothoid side so the kappa profile has a non-degenerate ramp.
    L_clo = heading_clothoids / kappa_abs
    L_arc = heading_arc / kappa_abs
    n_clo = max(1, int(round(L_clo / ds)))
    n_arc = max(0, int(math.floor(L_arc / ds)))

    # Clothoid up: kappa goes 0 -> kappa_max linearly.
    k_up = torch.linspace(0.0, kappa_max, n_clo, dtype=torch.float32)
    # Arc: constant kappa_max (may be empty when fraction == 1.0).
    k_arc = torch.full((n_arc,), kappa_max, dtype=torch.float32)
    # Clothoid down: kappa_max -> 0 linearly.
    k_down = torch.linspace(kappa_max, 0.0, n_clo, dtype=torch.float32)

    kappa_seg = torch.cat([k_up, k_arc, k_down], dim=0)
    L_total = kappa_seg.shape[0] * ds
    return kappa_seg, L_total, sign


def _uniform(rng_pair: Sequence[float], rng: torch.Generator) -> float:
    a, b = float(rng_pair[0]), float(rng_pair[1])
    if b < a:
        a, b = b, a
    u = float(torch.rand((), generator=rng).item())
    return a + u * (b - a)


def _segment_speed(
    cfg: RandomPathGeneratorCfg,
    kappa_seg: Tensor,
    rng: torch.Generator,
) -> float | None:
    """Sample one constant speed for a segment given its kappa profile.

    `v_curve_limit = sqrt(ay_limit / max(|kappa|))`. Returns:
      - `None` if `v_curve_limit < v_min` (segment is infeasible at the
        configured `ay_limit`; caller is expected to reject + resample).
      - `v_min` exactly when `v_max == v_min` (degenerate constant-speed cfg).
      - a uniform sample in `[v_min, min(v_max, v_curve_limit)]` otherwise.

    Returning None lets the geometry sampler choose a looser radius / shorter
    arc instead of silently breaking the lateral-accel target. See
    `random_clothoid_path` for the retry logic.
    """
    s = cfg.speed
    kappa_max = float(kappa_seg.abs().max().item())
    if kappa_max < 1e-6:
        v_curve_limit = float("inf")
    else:
        v_curve_limit = math.sqrt(s.ay_limit / kappa_max)
    if v_curve_limit < s.v_min:
        return None
    v_sample_max = min(s.v_max, v_curve_limit)
    if v_sample_max <= s.v_min:
        # v_min == v_max degenerate cfg: deterministic constant speed.
        return s.v_min
    u = float(torch.rand((), generator=rng).item())
    return s.v_min + u * (v_sample_max - s.v_min)


def _integrate_path(
    kappa: Tensor,        # (M,) curvature at each sample [1/m]
    v: Tensor,            # (M,) target speed at each sample [m/s]
    ds: float,
    *,
    is_loop: bool,
    device: torch.device | str,
    num_envs: int,
) -> Path:
    """Integrate (kappa, v) into a Path (broadcast to num_envs)."""
    M = kappa.shape[0]
    # Forward integration on CPU for simplicity; this runs once at startup.
    psi = torch.zeros(M, dtype=torch.float32)
    x = torch.zeros(M, dtype=torch.float32)
    y = torch.zeros(M, dtype=torch.float32)
    # psi[0] = 0; x[0] = y[0] = 0.
    for i in range(M - 1):
        psi[i + 1] = psi[i] + kappa[i] * ds
        x[i + 1] = x[i] + math.cos(float(psi[i].item())) * ds
        y[i + 1] = y[i] + math.sin(float(psi[i].item())) * ds
    psi = _wrap_to_pi(psi)

    s = torch.arange(M, dtype=torch.float32) * ds

    # Move to device, broadcast to (num_envs, M).
    def expand(t: Tensor) -> Tensor:
        return t.to(device).unsqueeze(0).expand(num_envs, -1).contiguous()

    return Path(
        s=expand(s), x=expand(x), y=expand(y),
        v=expand(v), psi=expand(psi),
        is_loop=is_loop,
        ds_value=ds,
    )


def random_clothoid_path(
    *,
    cfg: RandomPathGeneratorCfg,
    num_envs: int,
    length_m: float,
    is_loop: bool,
    device: torch.device | str,
    seed_offset: int = 0,
) -> Path:
    """Generate one random clothoid+arc+straight path of length ~`length_m`.

    Generation runs on CPU; the resulting Path tensors are broadcast to
    `(num_envs, M)` on `device`. `seed_offset` is added to
    `cfg.generator.seed` so a single shared RNG schedule can produce a
    deterministic family of paths (used by the bank generator).

    `is_loop=True` is supported but does NOT close the loop geometrically
    -- the caller is responsible for picking parameters that bring x/y
    back near origin. Phase 1 uses `is_loop=False` so this is fine.

    The returned Path always has exactly `M = max(8, round(length_m / ds))`
    samples so a bank of paths sharing one cfg can be stacked into a
    `(P, M)` tensor.
    """
    ds = cfg.generator.ds
    M = max(8, int(round(length_m / ds)))

    rng = torch.Generator()
    rng.manual_seed(int(cfg.generator.seed) + int(seed_offset))

    kappa_buf: list[Tensor] = []
    v_buf: list[Tensor] = []
    last_turn_sign = 0
    total_n = 0
    n_v_min_fallback = 0
    n_speed_rejects = 0
    max_attempts = max(1, cfg.speed.max_resample_attempts)
    while total_n < M:
        # Per-segment speed-feasibility retry (item 4). Geometry is now
        # structurally bounded by `turn_heading_change_rad` (item 3
        # better-design) so `_sample_segment_kappa_length` never rejects;
        # only `_segment_speed` may reject when `v_curve_limit < v_min`.
        # Each retry redraws R / heading_total / fraction so a fresh kappa
        # profile is tried. The accepted `(k_seg, sign)` is committed only
        # after the loop so a rejected attempt does not bias
        # `last_turn_sign`. On retry exhaustion the last sampled geometry
        # is accepted at `v_min` as a last-resort fallback.
        k_seg: Tensor | None = None
        proposed_sign = last_turn_sign
        v_seg_val: float | None = None
        for _ in range(max_attempts):
            k_try, _L_seg, sign_try = _sample_segment_kappa_length(
                cfg, rng, last_turn_sign=last_turn_sign,
            )
            v_try = _segment_speed(cfg, k_try, rng)
            if v_try is None:
                n_speed_rejects += 1
                # Keep the latest geometry as a fallback candidate.
                k_seg = k_try
                proposed_sign = sign_try
                continue
            k_seg = k_try
            proposed_sign = sign_try
            v_seg_val = v_try
            break
        if v_seg_val is None:
            # All retries produced an infeasible curve speed; accept the
            # last sampled geometry at v_min so generation cannot deadlock.
            v_seg_val = cfg.speed.v_min
            n_v_min_fallback += 1
        v_seg = torch.full_like(k_seg, v_seg_val)
        kappa_buf.append(k_seg)
        v_buf.append(v_seg)
        last_turn_sign = proposed_sign
        total_n += k_seg.shape[0]
    if n_speed_rejects > 0 or n_v_min_fallback > 0:
        print(
            f"[INFO] random_clothoid_path: {n_speed_rejects} speed rejects, "
            f"{n_v_min_fallback} v_min fallbacks (total segments accepted = "
            f"{len(kappa_buf)})",
            flush=True,
        )

    kappa = torch.cat(kappa_buf, dim=0)[:M]
    v = torch.cat(v_buf, dim=0)[:M]

    return _integrate_path(
        kappa, v, ds, is_loop=is_loop, device=device, num_envs=num_envs,
    )
