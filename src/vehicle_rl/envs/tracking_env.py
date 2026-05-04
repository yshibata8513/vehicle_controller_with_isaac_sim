"""Phase 3 path-tracking RL env (DirectRLEnv).

Wraps `VehicleSimulator` for parallel-env RL training. The env owns:
  - the parallel `Articulation` (per-env Sedan via InteractiveScene)
  - one shared `Path` broadcast across envs (Stage 0: circle only)
  - the action [-1, 1] -> physical (pinion_target, a_x_target) rescale
  - reward / done computation from `VehicleStateGT` + path errors

Per-env env-origin offsets (from `InteractiveSceneCfg(env_spacing>0)`) are
handled by subtracting `scene.env_origins[:, :2]` from the world-frame
vehicle position before projecting onto the shared (env-local) path.

GPU contract (PLAN.md §0.5):
  - all hot-loop tensors stay on cuda
  - reward / dones use `torch.where` / arithmetic only
  - logging buffers (mean reward / lat-err) accumulate on-device, fetched
    only at episode boundaries by rsl_rl
"""
from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import torch
from torch import Tensor

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from vehicle_rl.assets import (
    COG_Z_DEFAULT,
    DELTA_MAX_RAD,
    SEDAN_CFG,
    STEERING_RATIO,
)
from vehicle_rl.controller import PIDSpeedController
from vehicle_rl.envs.sensors import build_observation
from vehicle_rl.envs.simulator import VehicleSimulator
from vehicle_rl.envs.types import A_X_TARGET_MAX, A_X_TARGET_MIN, VehicleAction
from vehicle_rl.planner import (
    circle_path,
    dlc_path,
    lemniscate_path,
    load_random_path_cfg,
    random_clothoid_path,
    s_curve_path,
)


PINION_MAX = DELTA_MAX_RAD * STEERING_RATIO   # 0.611 * 16 ≈ 9.776 rad


@configclass
class TrackingEnvCfg(DirectRLEnvCfg):
    """Stage 0 default: circle, μ=0.9, 64 envs, 50 Hz control."""

    # --- env timing ---
    decimation = 4                        # 200 Hz physics, 50 Hz control
    episode_length_s = 25.0               # ~1.3 laps of r=30, v=10 circle

    # --- Gym spaces (DirectRLEnv treats int as Box of that dim) ---
    # action_space = 1 for steering-only Stage 0a (default), 2 for full
    # [pinion, a_x] action. Set this consistently with `steering_only`.
    action_space = 1
    # 9 scalar fields + K=10 plan points × 3 channels = 39.
    # World-frame pos_xy and yaw are intentionally NOT included: the policy
    # only ever needs path-relative quantities (`lateral_error`,
    # `heading_error`, body-frame plan), and feeding absolute coordinates
    # encourages it to memorize "at this position do that turn" instead of
    # learning a translation/rotation-invariant tracking controller. See
    # docs/phase3_training_review.md item 7.
    observation_space = 9 + 3 * 10        # = 39
    state_space = 0

    # --- Simulation (matches Phase 1.5 / Phase 2: dt=1/200, gravity on) ---
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
    )

    # --- Scene ---
    # env_spacing must clear the course extent. circle r=30 -> diameter 60 m;
    # 200 m gives ~3x margin so neighbouring envs cannot interact through any
    # future ground / sensor effects.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=200.0, replicate_physics=True, clone_in_fabric=True
    )

    # --- Robot ---
    robot_cfg: ArticulationCfg = SEDAN_CFG.replace(prim_path="/World/envs/env_.*/Sedan")

    # --- Course (Stage 0 hardcodes circle) ---
    # Supported: "circle" (default Stage 0a), "s_curve", "dlc", "lemniscate",
    # and "random_long" (Phase 3 random-path). For "random_long" the geometry
    # comes from `random_path_cfg_path` (see configs/random_path.yaml).
    course: str = "circle"
    radius: float = 30.0
    target_speed: float = 10.0
    course_ds: float = 0.2
    plan_K: int = 10
    lookahead_ds: float = 1.0

    # --- Random-path config (only used when course == "random_long") ---
    # Path resolved relative to the repository root by `load_random_path_cfg`.
    random_path_cfg_path: str = "configs/random_path.yaml"

    # --- Projection (local-window argmin) ---
    # half-window for Path.project; full window = 2W+1. With ds=0.2 and W=80
    # the search covers ±16 m of arc length, which is >> per-step movement
    # (~0.2 m at 10 m/s) and << half-loop length on every supported course.
    # NOTE: only effective for built-in courses. For `course="random_long"`,
    # the YAML `projection.search_radius_samples` overrides this default
    # (cached on `self._projection_search_radius_samples` at __init__).
    projection_search_radius_samples: int = 80
    # Hard cap on per-step |delta_idx|; values above this count as a
    # diagnostic violation (logged, not corrected). Reset envs always
    # re-seed `_nearest_idx`, so a violation flags either branch confusion
    # or a numerically broken projection.
    # NOTE: only effective for built-in courses; YAML
    # `projection.max_index_jump_samples` overrides for random_long.
    projection_max_index_jump_samples: int = 120

    # --- Friction (Stage 0: fixed) ---
    mu_default: float = 0.9

    # --- Action limits ---
    # `pinion_max` is the URDF physical limit (≈9.78 rad). For training we
    # rescale [-1, 1] to a smaller `pinion_action_scale` so that the
    # policy's exploration noise (init_noise_std=0.3) maps to realistic
    # steering deltas. r=30 circle at 10 m/s needs only ±1.4 rad pinion
    # (≈5° steer); 3 rad gives 2× headroom for low-μ corrections without
    # letting noisy random actions saturate the steering.
    pinion_max: float = PINION_MAX                     # physical clip (safety)
    pinion_action_scale: float = 3.0                   # action=±1 → ±3 rad pinion
    a_x_max: float = A_X_TARGET_MAX                    # +3.0 m/s^2
    a_x_min: float = A_X_TARGET_MIN                    # -5.0 m/s^2

    # --- Stage 0a: steering-only training ---
    # When True, the policy outputs only the steering action; longitudinal
    # control is handled by an internal PI controller targeting `target_speed`.
    # `action_space` must be set to 1 in the cfg subclass (or via CLI) when
    # using steering_only=True. Per docs/phase3_training_review.md item 1:
    # learning to corner is the harder half; freezing speed first lets the
    # policy converge on steering before the action space doubles.
    steering_only: bool = True
    pi_kp: float = 1.0
    pi_ki: float = 0.3

    # --- Reset distribution ---
    # When True, each reset places the vehicle at a uniformly random
    # path index (yaw aligned with path tangent, velocity warm-started
    # along that tangent). Distributes experience across the whole course
    # rather than concentrating it near path[0] -- review item 2.
    random_reset_along_path: bool = True

    # --- Reward weights (sign baked in: progress/alive positive, others negative) ---
    # Per docs/phase3_training_review.md item 3, the dominant positive term
    # is `progress_norm = Δs / (target_speed × control_dt)`: at perfect
    # tracking it equals 1.0 per step, giving a clear directional gradient
    # ("move forward along the path"). Alive bonus is reduced to a small
    # baseline so the policy can't just camp -- it has to actually progress.
    rew_progress: float = 1.0              # × Δs / (target_speed × control_dt)
    rew_alive: float = 0.1                 # small baseline so reset envs don't dominate
    rew_lateral: float = -0.2              # × lateral_error^2
    rew_heading: float = -0.3              # × heading_error^2
    rew_speed: float = -0.01               # × (vx - v_target)^2
    rew_pinion_rate: float = -0.01         # × (Δaction[0])^2 in [-1,1]
    rew_jerk: float = -0.001               # × (Δaction[1])^2 in [-1,1]
    rew_termination: float = -10.0         # one-shot on early termination

    # --- Termination thresholds ---
    max_lateral_error: float = 4.0         # m
    max_roll_rad: float = 1.047            # 60 deg

    # --- Initial pose ---
    cog_z: float = COG_Z_DEFAULT


class TrackingEnv(DirectRLEnv):
    """Phase 3 path-tracking env. See `TrackingEnvCfg` for tunables."""

    cfg: TrackingEnvCfg

    def __init__(self, cfg: TrackingEnvCfg, render_mode: str | None = None, **kwargs):
        # Guard: steering_only must agree with action_space dim (review item E).
        # Mismatch silently breaks reward bookkeeping -- e.g. steering_only=True
        # with action_space=2 leaves the unused throttle channel collecting jerk
        # penalties from random init noise. Fail fast before super().__init__
        # builds gym spaces from cfg.action_space.
        if cfg.steering_only and int(cfg.action_space) != 1:
            raise ValueError(
                f"steering_only=True requires action_space=1, got {cfg.action_space}"
            )
        if (not cfg.steering_only) and int(cfg.action_space) != 2:
            raise ValueError(
                f"steering_only=False requires action_space=2, got {cfg.action_space}"
            )

        super().__init__(cfg, render_mode, **kwargs)

        # VehicleSimulator wraps the Articulation we registered in _setup_scene.
        # Constructing it after super().__init__() so SimulationContext + scene
        # are fully initialized.
        self.vsim = VehicleSimulator(
            self.sim, self.sedan,
            mu_default=self.cfg.mu_default,
        )

        # Course is shared across envs (Stage 0). Path tensors are broadcast
        # to (N, M); per-env vehicle pose is converted to env-local frame
        # via env_origins before projection. `_random_path_cfg` is set to
        # the parsed YAML for `course="random_long"`, otherwise None.
        self._random_path_cfg = None
        self.path = self._build_path()

        # Pre-compute initial pose (env-local) and yaw from path[0].
        # Stored on-device so reset never needs a CPU sync.
        pos_local, yaw_start = self.path.start_pose      # (N, 2), (N,)
        self._initial_pos_local = pos_local              # (N, 2)
        self._initial_yaw = yaw_start                    # (N,)

        # Last action (for rate / jerk penalties). Reset to 0 per env. Width
        # = `action_dim` so we can carry a 1-d (steering-only) or 2-d
        # (steering+throttle) policy uniformly through the rate computation.
        self._action_dim = int(self.cfg.action_space)
        self._last_action = torch.zeros(self.num_envs, self._action_dim, device=self.device)

        # Previous-step path-arc-length, for the progress reward. Initialized
        # to 0 here; `_reset_idx` overwrites with `path.s[idx]` of each env's
        # spawn pose so the first post-reset delta_s is one step of genuine
        # progress (item B). Wrap correction + per-step cap are applied in
        # `_update_progress` to handle loop boundaries and projection
        # glitches; logic mirrors `utils.metrics.ProgressAccumulator` so the
        # completion / progress reward stay consistent with Phase 2 metrics.
        self._s_prev = torch.zeros(self.num_envs, device=self.device)
        self._control_dt = self.cfg.sim.dt * self.cfg.decimation

        # Local-window projection state. Initialized to 0 here; `_reset_idx`
        # sets it to the chosen path sample index per env (item B's projection
        # analogue). The first `_compute_path_state` after a reset finds an
        # argmin centred on this seed, so the closest sample is at most one
        # step away -- no spurious idx-jump diagnostic alarm.
        self._nearest_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
        )

        # Effective projection hyperparameters. For random_long the YAML
        # `projection.*` block wins (the random-path generator is what knows
        # the geometric scale of the segments); for built-in courses the
        # `TrackingEnvCfg.projection_*` defaults apply. `_compute_path_state`
        # and the idx-jump diagnostic read these runtime members so a single
        # source of truth governs both paths.
        if self._random_path_cfg is not None:
            proj = self._random_path_cfg.projection
            self._projection_search_radius_samples = proj.search_radius_samples
            self._projection_max_index_jump_samples = proj.max_index_jump_samples
        else:
            self._projection_search_radius_samples = (
                self.cfg.projection_search_radius_samples
            )
            self._projection_max_index_jump_samples = (
                self.cfg.projection_max_index_jump_samples
            )

        # Per-env local target speed at the projected path sample. Always
        # populated from `path.v[env, closest_idx]` by `_compute_path_state`;
        # the PI controller, progress normalization, and speed-error reward
        # all read this single tensor so behaviour is uniform across courses
        # (built-in courses set path.v ≡ cfg.target_speed, random_long uses
        # per-segment speeds). Seeded here from path[0] so `_pre_physics_step`
        # can read it before the first `_compute_path_state`.
        self._last_target_speed = self.path.v[:, 0].clone()

        # Stage 0a: internal PI for the longitudinal channel when the
        # policy is steering-only. Its dt is the env step (control rate),
        # not the physics step, since the PI re-evaluates once per
        # `_pre_physics_step`, not once per physics substep.
        if self.cfg.steering_only:
            self._pi_speed = PIDSpeedController(
                num_envs=self.num_envs, dt=self._control_dt,
                kp=self.cfg.pi_kp, ki=self.cfg.pi_ki,
                a_x_min=self.cfg.a_x_min, a_x_max=self.cfg.a_x_max,
                device=self.device,
            )
        else:
            self._pi_speed = None

        # Per-term reward accumulators (one tensor per term; mean over reset
        # envs is reported via extras["log"] at episode end). Item 6.
        self._reward_term_keys = (
            "progress", "alive", "lateral", "heading",
            "speed", "action_rate", "termination",
        )
        self._episode_sums = {
            k: torch.zeros(self.num_envs, device=self.device)
            for k in self._reward_term_keys
        }

        # State / action diagnostic accumulators (review item F). Same flush
        # pattern as reward sums but keyed by full TensorBoard path so each
        # quantity lands in the right category. When reward stops climbing,
        # these tell you whether the cause is lateral drift, low vx, jittery
        # steering, or insufficient progress -- without re-running training.
        self._diag_term_keys = (
            "Episode_State/lat_err_abs",
            "Episode_State/heading_err_abs",
            "Episode_State/vx",
            "Episode_Action/pinion_abs",
            "Episode_Action/pinion_rate_abs",
            "Episode_Progress/progress_norm",
            # Local-window projection health. `idx_jump_abs` is the per-step
            # mean of `|closest_idx - prev_nearest_idx|` (wrap-aware for
            # loops); healthy projection sits near 1 (one sample per step at
            # vx=10, ds=0.2, control_dt=0.02). `idx_jump_violation_rate` is
            # the per-step mean of the indicator that the jump exceeded
            # `_projection_max_index_jump_samples` (cfg default for built-in
            # courses, YAML `projection.max_index_jump_samples` for
            # random_long) -- a non-zero value signals branch confusion (e.g.
            # lemniscate self-crossing) or window-too-small for the current
            # excursion.
            "Episode_PathProj/idx_jump_abs",
            "Episode_PathProj/idx_jump_violation_rate",
        )
        self._episode_diag_sums = {
            k: torch.zeros(self.num_envs, device=self.device)
            for k in self._diag_term_keys
        }
        # Per-env step counter for the logging denominator. We can't reuse
        # `self.episode_length_buf` because rsl_rl's `init_at_random_ep_len`
        # initializes that buffer to random values per env, while our
        # accumulators always start at 0; the resulting mismatch makes iter-0
        # per-step means look ~20× too small. This counter increments in
        # lock-step with the sums (in `_get_rewards`) and resets in
        # `_reset_idx`, so the ratio is always over the actual accumulated
        # step count.
        self._sum_step_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long,
        )

        # First reset: drop every env onto the path start pose.
        # `_reset_idx` is normally invoked by the framework after episode
        # termination, but we trigger it once here so the very first
        # `_get_observations` returns a valid state instead of whatever
        # default_root_state is.
        self._reset_idx(self.sedan._ALL_INDICES)

    # ------------------------------------------------------------------
    # Scene / Path setup

    def _setup_scene(self):
        # Sedan articulation, replicated per env by InteractiveScene.
        self.sedan = Articulation(self.cfg.robot_cfg)
        # Ground plane (shared across envs).
        sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
        # Light.
        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func(
            "/World/Light",
            sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
        )
        # Replicate envs.
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["sedan"] = self.sedan

    def _build_path(self):
        """Construct the requested course as a Path.

        For `course="random_long"`, also caches the parsed YAML cfg on
        `self._random_path_cfg` so `_reset_idx` can read the
        `reset.end_margin_extra_m` and `phase1_long_path` fields.
        """
        if self.cfg.course == "circle":
            return circle_path(
                radius=self.cfg.radius, target_speed=self.cfg.target_speed,
                num_envs=self.num_envs, ds=self.cfg.course_ds, device=self.device,
            )
        if self.cfg.course == "s_curve":
            return s_curve_path(
                target_speed=self.cfg.target_speed,
                num_envs=self.num_envs, ds=self.cfg.course_ds, device=self.device,
            )
        if self.cfg.course == "dlc":
            return dlc_path(
                target_speed=self.cfg.target_speed,
                num_envs=self.num_envs, ds=self.cfg.course_ds, device=self.device,
            )
        if self.cfg.course == "lemniscate":
            return lemniscate_path(
                target_speed=self.cfg.target_speed,
                num_envs=self.num_envs, ds=self.cfg.course_ds, device=self.device,
            )
        if self.cfg.course == "random_long":
            rp_cfg = load_random_path_cfg(self.cfg.random_path_cfg_path)
            if abs(rp_cfg.generator.ds - self.cfg.course_ds) > 1e-9:
                raise ValueError(
                    f"random_path generator.ds={rp_cfg.generator.ds} != "
                    f"TrackingEnvCfg.course_ds={self.cfg.course_ds}"
                )
            self._random_path_cfg = rp_cfg
            print(
                f"[INFO] random_long: generating {rp_cfg.phase1_long_path.length_m:.0f} m path "
                f"(seed={rp_cfg.generator.seed}, ds={rp_cfg.generator.ds}) ...",
                flush=True,
            )
            return random_clothoid_path(
                cfg=rp_cfg,
                num_envs=self.num_envs,
                length_m=rp_cfg.phase1_long_path.length_m,
                is_loop=rp_cfg.phase1_long_path.is_loop,
                device=self.device,
                seed_offset=0,
            )
        raise ValueError(f"unknown course {self.cfg.course}")

    # ------------------------------------------------------------------
    # DirectRLEnv API

    def _pre_physics_step(self, actions: Tensor) -> None:
        """Cache the policy action; rescale once per env step.

        Action[0] (pinion): symmetric linear scale to ±pinion_action_scale,
        then hard-clipped to ±pinion_max for safety.

        Action[1] (a_x), when present (`steering_only=False`): sign-aware
        scale -- action>=0 maps to [0, +a_x_max], action<0 maps to
        [a_x_min, 0). This keeps action=0 → 0 m/s² (no force), so a
        freshly-initialized N(0, σ) policy averages to a zero-force net
        command instead of a constant brake.

        When `steering_only=True`, longitudinal control is delegated to
        an internal PI controller targeting `target_speed`, and `actions`
        is expected to be (N, 1).

        The same VehicleAction is fed to `vsim.apply_action_to_physx` for
        every physics substep in the decimation loop; rescaling once here
        saves redundant ops while the actuator integrates per substep.
        """
        actions = actions.clamp(-1.0, 1.0)
        # Use the smaller training scale, then hard-clip to physical limit.
        pinion_target = (actions[:, 0] * self.cfg.pinion_action_scale).clamp(
            -self.cfg.pinion_max, self.cfg.pinion_max,
        )

        if self.cfg.steering_only:
            # Internal PI uses the most recent observation's vx (set by the
            # previous step's _get_observations / __init__ first reset). If
            # we haven't built an obs yet, fall back to the simulator's
            # post-reset vel_body[:, 0]. PIDSpeedController only reads
            # `obs.vx`, so a SimpleNamespace stub suffices.
            vx = self._last_obs.vx if hasattr(self, "_last_obs") else \
                self.vsim.get_state().vel_body[:, 0]
            # `_last_target_speed` follows the local path.v at the projected
            # sample so the PI tracks per-segment v on `random_long`. Seeded
            # in __init__ from path[0], refreshed every `_compute_path_state`.
            a_x_target = self._pi_speed(
                SimpleNamespace(vx=vx), target_speed=self._last_target_speed,
            )
        else:
            a_x_norm = actions[:, 1]
            a_x_target = torch.where(
                a_x_norm >= 0.0,
                a_x_norm * self.cfg.a_x_max,        # [0, +3]
                a_x_norm * (-self.cfg.a_x_min),     # [-5, 0)  (a_x_min is negative)
            )

        self._action_pinion = pinion_target
        self._action_a_x = a_x_target
        self._current_action = actions   # store [-1, 1] for rate penalty

    def _apply_action(self) -> None:
        action = VehicleAction(
            pinion_target=self._action_pinion,
            a_x_target=self._action_a_x,
        )
        self.vsim.apply_action_to_physx(action)

    # ------------------------------------------------------------------
    # Per-step computation. DirectRLEnv call order each step:
    #   _get_dones -> _get_rewards -> (reset terminated envs) -> _get_observations
    #
    # Split into two responsibilities so observation retrieval never mutates
    # progress state (review item A):
    #   _compute_path_state(): pure projection, refreshes _last_* caches.
    #     Called from both _get_dones (pre-reset) and _get_observations
    #     (post-reset, so reset envs return obs from their fresh pose).
    #   _update_progress(): consumes the cached s_proj to compute
    #     progress_norm AND advances _s_prev. Called exactly once per step,
    #     at the top of _get_rewards. _s_prev is otherwise touched only by
    #     _reset_idx (item B), which sets it to path.s[idx] for the new pose.

    def _compute_path_state(self) -> None:
        """Project current pose onto the path. No mutation of _s_prev.

        Updates `_nearest_idx` with the projection's `closest_idx` so the
        next call's local-window argmin is centred on the latest position.
        Also computes wrap-aware `|delta_idx|` against the previous
        `_nearest_idx` and stages it in `_last_idx_jump_abs` /
        `_last_idx_jump_violation` for the diag accumulators in
        `_get_rewards`.
        """
        state_gt = self.vsim.get_state()

        pos_world_xy = state_gt.pos_xyz[:, :2]
        pos_local_xy = pos_world_xy - self.scene.env_origins[:, :2]
        yaw = state_gt.rpy[:, 2]

        prev_idx = self._nearest_idx
        plan, lat_err, hdg_err, s_proj, closest_idx = self.path.project(
            pos_local_xy, yaw, prev_idx,
            search_radius_samples=self._projection_search_radius_samples,
            K=self.cfg.plan_K, lookahead_ds=self.cfg.lookahead_ds,
        )
        obs_struct = build_observation(state_gt, plan, lat_err, hdg_err)

        # Wrap-aware |delta_idx|. For loops, idx_jump from M-1 to 0 should
        # count as 1, not M-1; min(|d|, M-|d|) handles both directions.
        d_abs = (closest_idx - prev_idx).abs()
        if self.path.is_loop:
            M = self.path.num_samples
            d_abs = torch.minimum(d_abs, M - d_abs)
        self._last_idx_jump_abs = d_abs.to(torch.float32)
        self._last_idx_jump_violation = (
            d_abs > self._projection_max_index_jump_samples
        ).to(torch.float32)

        self._last_state_gt = state_gt
        self._last_obs = obs_struct
        self._last_lat_err = lat_err
        self._last_hdg_err = hdg_err
        self._last_plan = plan
        self._last_pos_local = pos_local_xy
        self._last_s_proj = s_proj
        self._nearest_idx = closest_idx
        # plan.v[:, 0] is path.v at the projected sample (`closest_idx`); this
        # is the single source of truth for the local target speed used by the
        # PI controller, progress normalization, and speed reward.
        self._last_target_speed = plan.v[:, 0]

    def _update_progress(self) -> None:
        """Compute progress_norm from cached s_proj and advance _s_prev.

        Wrap correction (loop courses) + per-step velocity-based cap (filters
        projection-foot glitches near self-intersections / off-track
        excursions). Mirrors `utils.metrics.ProgressAccumulator` so the reward
        and the Phase 2 completion metric agree on what a "step of progress"
        means. Normalized by the local target speed at the projected sample
        (`_last_target_speed`) so progress_norm ≈ 1.0 at perfect tracking
        regardless of whether the segment runs at v_min or v_max on
        random_long; built-in courses (uniform v) reduce to the historical
        `cfg.target_speed` denominator. Backward motion gives a negative
        value; standstill gives zero.

        Final clamp to [-1, 1] (review item D): the velocity cap above
        permits Δs up to `2 × |vx| × dt`, so a vehicle moving at 2× target
        speed -- or recovering from a projection-foot jump -- could otherwise
        score `progress_norm ≈ 2.0` for a single step and dominate the
        composite reward. Stage 0a aims for "ideal driving = +1/step"; the
        clamp keeps the reward shape readable. Loosen to e.g. [-1, 2] if
        a future stage explicitly rewards over-speed bursts.
        """
        s_proj = self._last_s_proj
        vx = self._last_state_gt.vel_body[:, 0]
        delta_s = s_proj - self._s_prev
        if self.path.is_loop:
            half = self.path.total_length / 2.0
            delta_s = torch.where(delta_s < -half, delta_s + self.path.total_length, delta_s)
            delta_s = torch.where(delta_s >  half, delta_s - self.path.total_length, delta_s)
        cap = torch.clamp(vx.abs() * self._control_dt * 2.0,
                          min=self.path.ds * 2.0)
        delta_s = torch.maximum(-cap, torch.minimum(cap, delta_s))
        target_v = self._last_target_speed.clamp(min=1e-3)
        progress_norm = delta_s / (target_v * self._control_dt)
        self._last_progress_norm = progress_norm.clamp(-1.0, 1.0)
        self._s_prev = s_proj.detach().clone()

    def _get_dones(self) -> tuple[Tensor, Tensor]:
        """Termination = off-track or rollover; truncation = episode timeout."""
        self._compute_path_state()
        lat_err = self._last_lat_err
        roll = self._last_obs.roll

        off_track = lat_err.abs() > self.cfg.max_lateral_error
        rollover = roll.abs() > self.cfg.max_roll_rad
        terminated = off_track | rollover

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Cached for the one-shot termination reward in _get_rewards.
        self._last_terminated = terminated
        return terminated, time_out

    def _get_rewards(self) -> Tensor:
        """Composite reward; weights are signed in cfg (progress/alive +, others -).

        Dominant positive term is `progress_norm = Δs / (target_speed × dt)`,
        which gives a clear directional gradient (forward along path = good)
        even before the policy can hold tight tracking.

        Per-term values accumulate in `self._episode_sums[key]` and are
        flushed to TensorBoard via `extras["log"]` from `_reset_idx`.
        Rate / jerk penalties use Δaction in unitless [-1, 1] space so
        the scale is decimation-independent and survives action_dim
        switches between Stage 0a (steering-only) and Stage 0b.
        """
        # Advance _s_prev exactly once per step, using the path projection
        # cached by _get_dones above. _get_observations only re-projects;
        # it never touches _s_prev (review item A).
        self._update_progress()

        obs = self._last_obs
        lat_err = self._last_lat_err
        hdg_err = self._last_hdg_err
        # Speed error against the local path target (uniform on built-in
        # courses, per-segment on random_long) so the policy is not penalized
        # for honoring path.v.
        speed_err = obs.vx - self._last_target_speed

        d_action = self._current_action - self._last_action
        rate_pinion = d_action[:, 0]
        # rate_a_x only meaningful when policy outputs a_x (Stage 0b). For
        # steering-only we still apply the steering rate penalty but skip
        # the jerk penalty by zeroing the channel.
        if self._action_dim >= 2:
            rate_a_x_sq = d_action[:, 1].square()
        else:
            rate_a_x_sq = torch.zeros_like(rate_pinion)

        r_progress = self.cfg.rew_progress * self._last_progress_norm
        r_alive = torch.full_like(r_progress, self.cfg.rew_alive)
        r_lateral = self.cfg.rew_lateral * lat_err.square()
        r_heading = self.cfg.rew_heading * hdg_err.square()
        r_speed = self.cfg.rew_speed * speed_err.square()
        r_rate = (
            self.cfg.rew_pinion_rate * rate_pinion.square()
            + self.cfg.rew_jerk * rate_a_x_sq
        )
        r_term = self.cfg.rew_termination * self._last_terminated.to(r_progress.dtype)

        # Accumulate per-env per-term sums (flushed at episode end).
        self._episode_sums["progress"] += r_progress
        self._episode_sums["alive"] += r_alive
        self._episode_sums["lateral"] += r_lateral
        self._episode_sums["heading"] += r_heading
        self._episode_sums["speed"] += r_speed
        self._episode_sums["action_rate"] += r_rate
        self._episode_sums["termination"] += r_term

        # Diagnostic accumulators (item F). lat/hdg as |·| so sign-cancellation
        # on a symmetric course doesn't hide jitter; vx kept signed so the
        # mean is the average forward speed; pinion in [-1, 1] action space
        # so the magnitude is decimation-independent.
        self._episode_diag_sums["Episode_State/lat_err_abs"] += lat_err.abs()
        self._episode_diag_sums["Episode_State/heading_err_abs"] += hdg_err.abs()
        self._episode_diag_sums["Episode_State/vx"] += obs.vx
        self._episode_diag_sums["Episode_Action/pinion_abs"] += self._current_action[:, 0].abs()
        self._episode_diag_sums["Episode_Action/pinion_rate_abs"] += rate_pinion.abs()
        self._episode_diag_sums["Episode_Progress/progress_norm"] += self._last_progress_norm
        self._episode_diag_sums["Episode_PathProj/idx_jump_abs"] += self._last_idx_jump_abs
        self._episode_diag_sums["Episode_PathProj/idx_jump_violation_rate"] += \
            self._last_idx_jump_violation

        # One step's worth of reward accumulated -- bump the per-env counter
        # used as the per-step normalization denominator in `_reset_idx`.
        self._sum_step_count += 1

        self._last_action = self._current_action.detach()
        return r_progress + r_alive + r_lateral + r_heading + r_speed + r_rate + r_term

    def _get_observations(self) -> dict:
        """39-dim path-relative observation.

        Layout (concatenated along dim=-1):
          [0:6]   IMU: vx, yaw_rate, ax, ay, roll, pitch
          [6]     pinion_angle (steering-column encoder)
          [7:9]   lateral_error, heading_error
          [9:39]  plan_xyv flattened: x[0..K-1], y[0..K-1], v[0..K-1]

        World-frame absolute pose (`pos_xyz`, `yaw`) is intentionally
        omitted from the policy view (see docs/phase3_training_review.md
        item 7 and types.py module docstring). It remains accessible via
        `self._last_state_gt` for metrics + termination checks.

        Recomputed here (post-reset) so envs that just reset return
        observations consistent with their fresh pose, not their pre-reset
        cached state. Pure projection -- does NOT advance `_s_prev`, so
        extra observation calls (debug / render / wrappers) never corrupt
        the progress reward (review item A).
        """
        self._compute_path_state()
        obs_struct = self._last_obs

        obs = torch.cat([
            obs_struct.vx.unsqueeze(-1),
            obs_struct.yaw_rate.unsqueeze(-1),
            obs_struct.ax.unsqueeze(-1),
            obs_struct.ay.unsqueeze(-1),
            obs_struct.roll.unsqueeze(-1),
            obs_struct.pitch.unsqueeze(-1),
            obs_struct.pinion_angle.unsqueeze(-1),
            self._last_lat_err.unsqueeze(-1),
            self._last_hdg_err.unsqueeze(-1),
            self._last_plan.x, self._last_plan.y, self._last_plan.v,
        ], dim=-1)
        return {"policy": obs}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == 0:
            env_ids = self.sedan._ALL_INDICES

        # Flush per-term episode rewards for the envs that just terminated.
        # `mean(... [env_ids])` averages over only the resetting envs so the
        # value reflects "what those envs accumulated this episode" rather
        # than a global mix.
        #
        # Normalize by each env's actual episode step count (review item C):
        # our reward terms are per-step quantities (e.g. progress_norm ≈ 1.0
        # per control step at perfect tracking), so dividing by step count
        # yields a per-step mean that is directly comparable across short
        # (early-terminated) and long (timeout) episodes. Dividing by
        # `max_episode_length_s` would overstate values for timeout episodes
        # by ~50× (the dt-to-seconds factor) and obscure that comparison.
        # `episode_length_buf` is incremented just before _get_dones, so it
        # holds the correct length of the just-finished episode here, before
        # `super()._reset_idx` zeros it.
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        log_extras: dict[str, Tensor] = {}
        # Use our own accumulated-step counter (not `episode_length_buf`)
        # because rsl_rl's `init_at_random_ep_len=True` desyncs the framework
        # counter from our sums on iter 0; see `_sum_step_count` setup.
        ep_len = self._sum_step_count[env_ids_t].clamp(min=1).float()
        for key, sums in self._episode_sums.items():
            log_extras[f"Episode_Reward/{key}"] = (
                sums[env_ids_t] / ep_len.to(sums.dtype)
            ).mean()
            sums[env_ids_t] = 0.0
        # Diagnostic state / action / progress means (item F). Same
        # per-step normalization as the reward terms above.
        for key, sums in self._episode_diag_sums.items():
            log_extras[key] = (sums[env_ids_t] / ep_len.to(sums.dtype)).mean()
            sums[env_ids_t] = 0.0
        self._sum_step_count[env_ids_t] = 0
        # Termination breakdown (off-track vs. timeout). reset_terminated /
        # reset_time_outs are set by the framework just before _reset_idx.
        log_extras["Episode_Termination/off_track_or_rollover"] = \
            self.reset_terminated[env_ids_t].float().mean()
        log_extras["Episode_Termination/time_out"] = \
            self.reset_time_outs[env_ids_t].float().mean()
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update(log_extras)

        super()._reset_idx(env_ids)

        # Per-env reset pose. Stage 0 default: random index along the path
        # (item 2). When `random_reset_along_path=False`, fall back to the
        # path[0] start pose for reproducibility (e.g., eval rollouts).
        n_reset = env_ids_t.shape[0]
        if self.cfg.random_reset_along_path:
            # Uniform random index ∈ [0, M); for open paths leave a margin
            # near the end so the vehicle can run for a full episode without
            # falling off the end. For built-in courses (lookahead is the
            # only horizon constraint) this is `plan_K * lookahead_step`. For
            # `random_long` we additionally need `v_max * episode_length_s`
            # of arc to ensure even the fastest sampled segment fits.
            M = self.path.num_samples
            if self.path.is_loop:
                idx = torch.randint(0, M, (n_reset,), device=self.device)
            else:
                step_count = max(1, int(round(self.cfg.lookahead_ds / self.path.ds)))
                lookahead_margin = self.cfg.plan_K * step_count
                if self._random_path_cfg is not None:
                    rp = self._random_path_cfg
                    end_margin_m = (
                        rp.speed.v_max * self.cfg.episode_length_s
                        + self.cfg.plan_K * self.cfg.lookahead_ds
                        + rp.reset.end_margin_extra_m
                    )
                    end_margin = max(
                        lookahead_margin,
                        int(round(end_margin_m / self.path.ds)),
                    )
                else:
                    end_margin = lookahead_margin
                idx_max = max(1, M - end_margin)
                idx = torch.randint(0, idx_max, (n_reset,), device=self.device)
            pos_local_xy = torch.stack(
                [self.path.x[env_ids_t, idx], self.path.y[env_ids_t, idx]], dim=-1
            )
            yaw = self.path.psi[env_ids_t, idx]
            s_reset = self.path.s[env_ids_t, idx]
            v_warmstart = self.path.v[env_ids_t, idx]
        else:
            pos_local_xy = self._initial_pos_local[env_ids_t]
            yaw = self._initial_yaw[env_ids_t]
            # path[0] sits at s=0 by construction (uniform arc-length).
            s_reset = torch.zeros(n_reset, device=self.device)
            idx = torch.zeros(n_reset, dtype=torch.long, device=self.device)
            # Read warm-start speed from the path itself (path.v[:, 0] equals
            # cfg.target_speed for built-in courses, the first segment's
            # sampled v on random_long). Same source as the random-reset
            # branch above so both paths route through path.v uniformly.
            v_warmstart = self.path.v[env_ids_t, idx]

        pos_world_xy = pos_local_xy + self.scene.env_origins[env_ids_t, :2]
        z = torch.full((n_reset, 1), self.cfg.cog_z, device=self.device)
        pos_world = torch.cat([pos_world_xy, z], dim=-1)                 # (n, 3)

        qw = torch.cos(yaw * 0.5)
        qz = torch.sin(yaw * 0.5)
        qzero = torch.zeros_like(qw)
        quat = torch.stack([qw, qzero, qzero, qz], dim=-1)               # (n, 4)
        initial_pose = torch.cat([pos_world, quat], dim=-1)              # (n, 7)

        self.vsim.reset(env_ids=env_ids_t, initial_pose=initial_pose)

        # Warm-start root velocity at the path's local target speed along the
        # tangent. Without this, vehicle starts at vx=0 vs (target ~10 m/s),
        # so speed_err^2 ≈ 100 from step 1 -- a penalty the actuator lag
        # (tau_drive=200ms) physically can't correct in <1s. For random_long
        # the per-segment v varies across the path, so the local v[idx]
        # gives a closer match to what the policy / internal PI will target.
        vel_world = torch.zeros((n_reset, 6), device=self.device)
        vel_world[:, 0] = torch.cos(yaw) * v_warmstart                   # vx_world
        vel_world[:, 1] = torch.sin(yaw) * v_warmstart                   # vy_world
        self.sedan.write_root_velocity_to_sim(vel_world, env_ids=env_ids_t)

        # Reset internal state for these envs.
        self._last_action[env_ids_t] = 0.0
        # Seed _s_prev to the new pose's arc-length so the first post-reset
        # `_update_progress` sees `delta_s ≈ vx * control_dt` (~one step of
        # genuine progress) instead of the random-index jump from the
        # pre-reset s. Required because _get_observations no longer mutates
        # _s_prev (review item B; pairs with item A).
        self._s_prev[env_ids_t] = s_reset
        # Seed local-window projection: the next `_compute_path_state` will
        # search ±W samples around `idx`, so the chosen branch (in particular
        # at self-crossings like lemniscate origin) is locked in deterministically.
        self._nearest_idx[env_ids_t] = idx
        if self._pi_speed is not None:
            self._pi_speed.reset(env_ids=env_ids_t)
