"""Reference-path generators (Phase 2).

Each generator returns a `Path` discretized at uniform arc-length spacing
`ds` and broadcast across `num_envs`. For per-env-different parameters
(domain randomization in Phase 3), construct multiple Paths and stack
their tensors, or build the per-env arrays directly.

Pipeline:
    1. Sample raw (x_raw, y_raw) at non-uniform parameter (closed-form for
       circle; (t, x(t), y(t)) parametric for lemniscate / S / DLC).
    2. Resample to uniform arc-length `ds` via cumulative-length linear
       interpolation (`_resample_uniform_arclength`).
    3. Compute tangent heading `psi` from finite differences.
    4. Wrap into a `Path` and broadcast to `num_envs`.

Available shapes (PLAN.md §2):
    circle_path       -- closed CCW circle (closed-form, exact)
    lemniscate_path   -- Lissajous figure-eight (self-crossing at origin)
    s_curve_path      -- single sinusoidal cycle (open path)
    dlc_path          -- ISO 3888-2 double lane change (simplified)
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .path import Path, _wrap_to_pi


# ---------------------------------------------------------------------------
# Internal helpers


def _resample_uniform_arclength(
    x_raw: Tensor,
    y_raw: Tensor,
    ds_target: float,
    *,
    is_loop: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, float]:
    """Resample 1-D (x_raw, y_raw) to uniform arc-length spacing `ds_target`.

    Returns (s, x, y, psi, ds_actual) where the first four are 1-D Tensors of
    length M and `ds_actual` is the realized spacing (Python float). M is
    chosen so M * ds_actual ≈ total length and ds_actual ≈ ds_target.
    """
    device = x_raw.device
    dtype = x_raw.dtype

    if is_loop:
        # Append the closing segment from last raw point back to the first.
        x_ext = torch.cat([x_raw, x_raw[:1]])
        y_ext = torch.cat([y_raw, y_raw[:1]])
    else:
        x_ext = x_raw
        y_ext = y_raw

    dx = x_ext[1:] - x_ext[:-1]
    dy = y_ext[1:] - y_ext[:-1]
    seg_len = torch.sqrt(dx * dx + dy * dy)
    s_raw = torch.cat([torch.zeros(1, device=device, dtype=dtype), torch.cumsum(seg_len, dim=0)])
    L_total = float(s_raw[-1].item())
    if L_total <= 0.0:
        raise ValueError("Path has zero total length")

    M = max(8, int(round(L_total / ds_target)))
    ds_actual = L_total / M
    s_target = torch.arange(M, device=device, dtype=dtype) * ds_actual

    # Linear interp at s_target via searchsorted on s_raw.
    idx = torch.searchsorted(s_raw, s_target, right=False).clamp(1, s_raw.shape[0] - 1)
    s0 = s_raw[idx - 1]
    s1 = s_raw[idx]
    frac = (s_target - s0) / (s1 - s0).clamp(min=1e-12)
    x_out = x_ext[idx - 1] + frac * (x_ext[idx] - x_ext[idx - 1])
    y_out = y_ext[idx - 1] + frac * (y_ext[idx] - y_ext[idx - 1])

    # Tangent heading via forward finite-difference.
    if is_loop:
        x_next = torch.roll(x_out, -1)
        y_next = torch.roll(y_out, -1)
    else:
        # For the last sample, extrapolate using the previous segment's
        # direction so the tangent remains well-defined.
        x_next = torch.cat([x_out[1:], 2 * x_out[-1:] - x_out[-2:-1]])
        y_next = torch.cat([y_out[1:], 2 * y_out[-1:] - y_out[-2:-1]])
    psi = torch.atan2(y_next - y_out, x_next - x_out)

    return s_target, x_out, y_out, psi, ds_actual


def _broadcast_to_path(
    s: Tensor, x: Tensor, y: Tensor, psi: Tensor,
    target_speed: float, num_envs: int, is_loop: bool,
    ds_value: float,
) -> Path:
    M = s.shape[0]
    v = torch.full((M,), float(target_speed), device=s.device, dtype=s.dtype)

    def expand(t: Tensor) -> Tensor:
        return t.unsqueeze(0).expand(num_envs, -1).contiguous()

    return Path(
        s=expand(s), x=expand(x), y=expand(y),
        v=expand(v), psi=expand(psi),
        is_loop=is_loop,
        ds_value=ds_value,
    )


# ---------------------------------------------------------------------------
# Path generators


def circle_path(
    *,
    radius: float,
    target_speed: float,
    num_envs: int,
    ds: float,
    device: torch.device | str,
) -> Path:
    """Closed CCW circle of given radius, centered at origin.

    Vehicle naturally starts at sample[0] = (radius, 0) heading +y
    (yaw = pi/2). Closed-form generator (no resampling needed; spacing is
    exact at `2*pi*r / M` which differs from `ds` by sub-sample rounding).
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    if target_speed <= 0.0:
        raise ValueError(f"target_speed must be positive, got {target_speed}")

    L = 2.0 * math.pi * radius
    M = max(8, int(round(L / ds)))
    ds_actual = L / M

    theta = torch.arange(M, device=device, dtype=torch.float32) * (2.0 * math.pi / M)
    s = torch.arange(M, device=device, dtype=torch.float32) * ds_actual
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    psi = _wrap_to_pi(theta + math.pi / 2.0)

    return _broadcast_to_path(s, x, y, psi, target_speed, num_envs, is_loop=True, ds_value=ds_actual)


def lemniscate_path(
    *,
    a: float,
    target_speed: float,
    num_envs: int,
    ds: float,
    n_raw: int,
    device: torch.device | str,
) -> Path:
    """Lissajous figure-eight: x = a*sin(2t), y = a*sin(t), t in [0, 2*pi].

    Self-crosses at the origin. The planner uses a vectorized closest-sample
    argmin (chosen for GPU speed); branch disambiguation at the crossing
    would cost a per-step state lookup that breaks the pure-tensor
    projection. **Self-crossing is intentionally not handled** -- when the
    vehicle is near the crossing, projection can pick the wrong branch and
    Pure Pursuit may briefly steer toward the wrong lobe (observed on the
    Phase 2 baseline: rms_lateral_error ~3m, completion still =1.0). For
    Phase 3 RL evaluation, lemniscate is a stress test exposing this
    limitation; for Phase 2 baseline reporting, treat the figure-8
    numbers as "PP fails to follow" rather than as a Phase 2 sanity check.

    Default `a=25 m` gives a 50m-wide / 25m-tall figure-8. Vehicle starts
    at sample[0] = (0, 0) on the rising branch, tangent pointing into
    the upper-right lobe.
    """
    if a <= 0.0:
        raise ValueError(f"a must be positive, got {a}")
    t = torch.linspace(0, 2 * math.pi, n_raw + 1, device=device, dtype=torch.float32)[:-1]
    x_raw = a * torch.sin(2 * t)
    y_raw = a * torch.sin(t)
    s, x, y, psi, ds_actual = _resample_uniform_arclength(x_raw, y_raw, ds, is_loop=True)
    return _broadcast_to_path(s, x, y, psi, target_speed, num_envs, is_loop=True, ds_value=ds_actual)


def s_curve_path(
    *,
    length: float,
    amplitude: float,
    n_cycles: float,
    target_speed: float,
    num_envs: int,
    ds: float,
    n_raw: int,
    device: torch.device | str,
) -> Path:
    """Sinusoidal S-curve: y = amplitude * sin(2*pi*n_cycles*x/length),
    x in [0, length].

    Open path (`is_loop=False`). For an integer `n_cycles` the path ends at
    y=0 with the same tangent as the start. With `n_cycles=1` (default) the
    vehicle goes up, returns, dips, and returns -- one full S oscillation.

    Vehicle starts at sample[0] = (0, 0) heading +x (yaw ~ atan(slope at 0)).
    """
    if length <= 0.0 or amplitude < 0.0:
        raise ValueError("length must be positive and amplitude non-negative")
    x_raw = torch.linspace(0.0, length, n_raw, device=device, dtype=torch.float32)
    y_raw = amplitude * torch.sin(2 * math.pi * n_cycles * x_raw / length)
    s, x, y, psi, ds_actual = _resample_uniform_arclength(x_raw, y_raw, ds, is_loop=False)
    return _broadcast_to_path(s, x, y, psi, target_speed, num_envs, is_loop=False, ds_value=ds_actual)


def dlc_path(
    *,
    target_speed: float,
    num_envs: int,
    ds: float,
    n_raw: int,
    lane_offset: float,
    device: torch.device | str,
) -> Path:
    """ISO 3888-2 Double Lane Change (simplified geometry).

    Section breakdown (per ISO 3888-2):
        [   0,  12.0]: entry straight       (y = 0)
        [12.0,  25.5]: rising offset zone   (0 -> lane_offset, smoothstep)
        [25.5,  36.5]: offset plateau       (y = lane_offset)
        [36.5,  49.0]: returning zone       (lane_offset -> 0, smoothstep)
        [49.0,  61.0]: exit straight        (y = 0)
    Total length: 61 m. ISO target speed: 80 km/h = 22.2 m/s; we default to
    13 m/s (47 km/h) for an entry-level handling check. Open path.

    Lateral transitions use a cosine smoothstep -- continuous tangent at
    section boundaries, no overshoot, easy to differentiate.
    """
    L1, L2, L3, L4, L5 = 12.0, 13.5, 11.0, 12.5, 12.0
    L_total = L1 + L2 + L3 + L4 + L5   # 61 m
    x1, x2, x3, x4 = L1, L1 + L2, L1 + L2 + L3, L1 + L2 + L3 + L4

    x_raw = torch.linspace(0.0, L_total, n_raw, device=device, dtype=torch.float32)
    y_raw = torch.zeros_like(x_raw)

    # Cosine smoothstep helper: 0.5*(1-cos(pi*u)), u in [0, 1].
    def smoothstep(x_arr: Tensor, a: float, b: float) -> Tensor:
        u = ((x_arr - a) / (b - a)).clamp(0.0, 1.0)
        return 0.5 * (1.0 - torch.cos(math.pi * u))

    mask2 = (x_raw >= x1) & (x_raw < x2)
    mask3 = (x_raw >= x2) & (x_raw < x3)
    mask4 = (x_raw >= x3) & (x_raw < x4)
    y_raw = torch.where(mask2, lane_offset * smoothstep(x_raw, x1, x2), y_raw)
    y_raw = torch.where(mask3, torch.full_like(x_raw, lane_offset), y_raw)
    y_raw = torch.where(mask4, lane_offset * (1.0 - smoothstep(x_raw, x3, x4)), y_raw)

    s, x, y, psi, ds_actual = _resample_uniform_arclength(x_raw, y_raw, ds, is_loop=False)
    return _broadcast_to_path(s, x, y, psi, target_speed, num_envs, is_loop=False, ds_value=ds_actual)
