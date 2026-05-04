"""World-frame reference path with GPU-parallel local-window projection.

A `Path` discretizes a course into M uniformly-spaced (arc-length) samples
shared across N parallel envs. Projection cost is `O(N * W)` where
`W = 2 * search_radius_samples + 1` -- it does NOT scale with M, so very
long random paths (M = 100k) are as cheap to project as a 1 km loop.

Construction conventions (enforced by generators in `waypoints.py` and
`random_path.py`):
  - Uniform `ds` spacing along arc length (so lookahead is a pure gather)
  - `psi` is the path tangent heading at each sample, wrapped to [-pi, pi]
  - For closed loops (`is_loop=True`), s wraps modulo total length

Per-env-different courses are supported by stacking; share-one-course
across envs by broadcasting from (1, M).

Stateless contract: `Path` does NOT remember the previous-step nearest
sample. The caller (TrackingEnv, classical runner, ...) owns the
`nearest_idx` tensor and seeds it on reset.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .plan import Plan


def _wrap_to_pi(x: Tensor) -> Tensor:
    """Wrap radian angles to [-pi, pi]."""
    return torch.atan2(torch.sin(x), torch.cos(x))


@dataclass
class Path:
    """Discretized world-frame reference course."""

    s: Tensor       # (N, M) cumulative arc length [m], uniform spacing
    x: Tensor       # (N, M) world-frame x [m]
    y: Tensor       # (N, M) world-frame y [m]
    v: Tensor       # (N, M) target speed [m/s]
    psi: Tensor     # (N, M) tangent heading [rad], wrapped to [-pi, pi]
    is_loop: bool
    ds_value: float  # Arc-length spacing [m]; cached at construction so hot
                     # loops never need a GPU->CPU sync to read it.

    @property
    def num_envs(self) -> int:
        return self.x.shape[0]

    @property
    def num_samples(self) -> int:
        return self.x.shape[1]

    @property
    def device(self) -> torch.device:
        return self.x.device

    @property
    def ds(self) -> float:
        """Arc-length spacing between consecutive samples [m]."""
        return self.ds_value

    @property
    def total_length(self) -> float:
        """Course length [m]. For loops, equals M * ds; for open paths (M-1) * ds."""
        if self.is_loop:
            return self.num_samples * self.ds_value
        return (self.num_samples - 1) * self.ds_value

    @property
    def start_pose(self) -> tuple[Tensor, Tensor]:
        """Recommended initial vehicle pose: at sample[0] aligned to tangent.

        Returns (pos_xy: (N, 2), yaw: (N,)). Use this with
        `VehicleSimulator.reset(initial_pose=...)` to drop the vehicle on
        the start of the course pointing along the path.
        """
        pos_xy = torch.stack([self.x[:, 0], self.y[:, 0]], dim=-1)
        yaw = self.psi[:, 0]
        return pos_xy, yaw

    # ------------------------------------------------------------------
    # Projection (local-window argmin)

    def project(
        self,
        pos_xy: Tensor,             # (N, 2) world frame
        yaw: Tensor,                # (N,) world frame
        nearest_idx: Tensor,        # (N,) long; previous step's closest sample
        *,
        search_radius_samples: int,
        K: int = 10,
        lookahead_ds: float = 1.0,
    ) -> tuple[Plan, Tensor, Tensor, Tensor, Tensor]:
        """Project (pos_xy, yaw) onto the path within a local index window.

        Candidate samples are `nearest_idx[:, None] + arange(-W, W+1)` with
        `W = search_radius_samples`. Window is wrapped (mod M) for loops
        and clamped to `[0, M-1]` for open paths. Cost is O(N * (2W+1)).

        Branch-confusion safety: as long as
            per-step movement << W * ds << half-loop arc length
        the local argmin stays on the same branch even at self-crossings
        (e.g. lemniscate origin). The caller is responsible for seeding
        `nearest_idx` on the correct branch at reset time.

        Returns:
            plan: body-frame K-point lookahead window (Plan, shapes (N, K))
            lateral_error: (N,) signed perpendicular distance.
                           Positive = vehicle is LEFT of the path.
            heading_error: (N,) wrapped path-tangent yaw minus vehicle yaw.
                           Positive = path heads LEFT relative to vehicle.
            s_proj: (N,) cumulative arc-length at the closest sample.
            closest_idx: (N,) long; pass back as `nearest_idx` next step.

        `lookahead_ds` is the arc-length step between consecutive Plan
        waypoints. Snapped to the nearest integer multiple of `self.ds`
        (no interpolation -- generators ensure `ds` is small).
        """
        if pos_xy.shape != (self.num_envs, 2):
            raise ValueError(f"pos_xy shape {tuple(pos_xy.shape)} != ({self.num_envs}, 2)")
        if yaw.shape != (self.num_envs,):
            raise ValueError(f"yaw shape {tuple(yaw.shape)} != ({self.num_envs},)")
        if nearest_idx.shape != (self.num_envs,):
            raise ValueError(
                f"nearest_idx shape {tuple(nearest_idx.shape)} != ({self.num_envs},)"
            )
        if search_radius_samples <= 0:
            raise ValueError(
                f"search_radius_samples must be > 0, got {search_radius_samples}"
            )

        N = self.num_envs
        M = self.num_samples
        device = self.device
        batch = torch.arange(N, device=device)
        W = int(search_radius_samples)

        # 1) Candidate index window: (N, 2W+1).
        offsets = torch.arange(-W, W + 1, device=device, dtype=torch.long)
        cand = nearest_idx.unsqueeze(-1) + offsets.unsqueeze(0)   # (N, 2W+1)
        if self.is_loop:
            cand = cand % M
        else:
            cand = torch.clamp(cand, min=0, max=M - 1)

        # 2) Local argmin (cost is O(N * 2W+1), independent of M).
        x_cand = self.x.gather(1, cand)        # (N, 2W+1)
        y_cand = self.y.gather(1, cand)
        dx = x_cand - pos_xy[:, 0:1]
        dy = y_cand - pos_xy[:, 1:2]
        dist_sq = dx * dx + dy * dy             # (N, 2W+1)
        local_arg = dist_sq.argmin(dim=-1)      # (N,)
        closest = cand[batch, local_arg]        # (N,)

        x_proj = self.x[batch, closest]
        y_proj = self.y[batch, closest]
        psi_proj = self.psi[batch, closest]
        s_proj = self.s[batch, closest]

        # 3) Lateral / heading error using the exact tangent at the closest
        # sample. Perpendicular distance (left-positive):
        # lateral_error = -wx*sin(psi_proj) + wy*cos(psi_proj).
        cos_p = torch.cos(psi_proj)
        sin_p = torch.sin(psi_proj)
        wx = pos_xy[:, 0] - x_proj
        wy = pos_xy[:, 1] - y_proj
        lateral_error = -wx * sin_p + wy * cos_p
        heading_error = _wrap_to_pi(psi_proj - yaw)

        # 4) K-point lookahead window via integer-offset gather from `closest`.
        step_count = max(1, int(round(lookahead_ds / self.ds_value)))
        la_offsets = torch.arange(K, device=device, dtype=torch.long) * step_count
        gather_idx = closest.unsqueeze(-1) + la_offsets.unsqueeze(0)
        if self.is_loop:
            gather_idx = gather_idx % M
        else:
            gather_idx = torch.clamp(gather_idx, max=M - 1)

        wx_la = self.x.gather(1, gather_idx)
        wy_la = self.y.gather(1, gather_idx)
        wv_la = self.v.gather(1, gather_idx)

        # 5) World -> body transform (rotation by -yaw, then translate).
        cos_y = torch.cos(yaw).unsqueeze(-1)
        sin_y = torch.sin(yaw).unsqueeze(-1)
        dxw = wx_la - pos_xy[:, 0:1]
        dyw = wy_la - pos_xy[:, 1:2]
        plan_x = cos_y * dxw + sin_y * dyw       # body forward (+x)
        plan_y = -sin_y * dxw + cos_y * dyw      # body left (+y)

        return Plan(x=plan_x, y=plan_y, v=wv_la), lateral_error, heading_error, s_proj, closest
