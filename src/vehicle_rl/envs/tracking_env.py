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
from dataclasses import MISSING
from types import SimpleNamespace

import torch
from torch import Tensor

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from vehicle_rl.controller import PIDSpeedController
from vehicle_rl.envs.sensors import build_observation
from vehicle_rl.envs.simulator import VehicleSimulator
from vehicle_rl.envs.types import VehicleAction
from vehicle_rl.planner import (
    Path,
    circle_path,
    dlc_path,
    lemniscate_path,
    load_random_path_cfg,
    random_clothoid_path,
    random_clothoid_path_bank,
    s_curve_path,
)


@configclass
class TrackingEnvCfg(DirectRLEnvCfg):
    """Tracking env cfg. PR 3: tunable defaults moved to configs/envs/tracking.yaml.

    Construct via `vehicle_rl.config.isaac_adapter.make_tracking_env_cfg(...)`
    which fills every MISSING field from the resolved YAML bundles. The
    gym registry's `env_cfg_entry_point` (vehicle_rl.tasks.tracking.entry_points
    :tracking_env_cfg_factory) calls the factory at task-make time so legacy
    `gym.make("Vehicle-Tracking-Direct-v0")` callers still work.
    """

    # --- env timing (filled by factory) ---
    decimation: int = MISSING               # 200 Hz physics, 50 Hz control
    episode_length_s: float = MISSING        # seconds

    # --- Gym spaces (DirectRLEnv treats int as Box of that dim) ---
    action_space: int = MISSING
    observation_space: int = MISSING
    state_space: int = 0

    # --- Simulation / Scene / Robot (filled by factory) ---
    sim: SimulationCfg = MISSING
    scene: InteractiveSceneCfg = MISSING
    robot_cfg: ArticulationCfg = MISSING

    # --- Course (still uses string discriminator at runtime; the in-place
    # YAML build path in `_build_path` is unchanged in PR 3 -- PR 4 will
    # collapse this onto adapter.build_path). ---
    course: str = MISSING
    radius: float = MISSING
    target_speed: float = MISSING
    course_ds: float = MISSING
    plan_K: int = MISSING
    lookahead_ds: float = MISSING

    # Legacy random-path config path; PR 4 deletes this and the runtime lookup.
    random_path_cfg_path: str = MISSING

    # --- Projection (local-window argmin) ---
    projection_search_radius_samples: int = MISSING
    projection_max_index_jump_samples: int = MISSING

    # --- Friction ---
    mu_default: float = MISSING

    # --- Action limits ---
    pinion_max: float = MISSING                       # physical clip (safety)
    pinion_action_scale: float = MISSING              # action=±1 → ±N rad pinion
    a_x_max: float = MISSING                           # m/s^2
    a_x_min: float = MISSING                           # m/s^2

    # --- Stage 0a: steering-only training ---
    steering_only: bool = MISSING
    pi_kp: float = MISSING
    pi_ki: float = MISSING
    pi_integral_max: float = MISSING

    # --- Reset distribution ---
    random_reset_along_path: bool = MISSING
    warm_start_velocity: bool = MISSING

    # --- Reward weights ---
    rew_progress: float = MISSING
    rew_alive: float = MISSING
    rew_lateral: float = MISSING
    rew_heading: float = MISSING
    rew_speed: float = MISSING
    rew_pinion_rate: float = MISSING
    rew_jerk: float = MISSING
    rew_termination: float = MISSING
    progress_clamp_low: float = MISSING
    progress_clamp_high: float = MISSING

    # --- Termination thresholds ---
    max_lateral_error: float = MISSING       # m
    max_roll_rad: float = MISSING            # rad

    # --- Diagnostics flags (gate logging accumulation/emission) ---
    log_reward_terms: bool = MISSING
    log_state_action_terms: bool = MISSING
    log_projection_health: bool = MISSING

    # --- Initial pose ---
    cog_z: float = MISSING

    # --- Observation layout (from configs/envs/tracking.yaml `spaces.observation`) ---
    obs_imu_fields: list = MISSING
    obs_include_pinion_angle: bool = MISSING
    obs_include_path_errors: bool = MISSING
    obs_include_plan: bool = MISSING
    obs_include_world_pose: bool = MISSING

    # --- Resolved YAML bundles (for dynamics-side simulator construction) ---
    # `dynamics_kwargs` is the full kwarg dict produced by
    # `make_simulator_kwargs(dynamics_bundle)` plus `steering_ratio` from the
    # vehicle bundle. Carrying it on the cfg lets `TrackingEnv.__init__`
    # construct `VehicleSimulator` directly from YAML values, eliminating the
    # previous hard-coded literals (a_front=2.7/2.0, lag=0.05/0.20/0.07,
    # cornering_stiffness=60000.0, etc.).
    dynamics_kwargs: dict = MISSING
    a_front: float = MISSING

    # --- vehicle-derived (steering kinematics) ---
    # PR 3 round-2 fix F1b: previously read from `vehicle_rl.assets.STEERING_RATIO`
    # at runtime; now sourced from the vehicle YAML via the factory so a
    # different `configs/vehicles/*.yaml` actually changes the simulator.
    steering_ratio: float = MISSING


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
        # PR 3 round-1 fix (review finding 1): the dynamics kwargs come from
        # the YAML via `make_simulator_kwargs(dynamics_bundle)` (with
        # `steering_ratio` and `a_front` taken from the vehicle bundle).
        # `TrackingEnvCfg.dynamics_kwargs` carries the full dict so changing
        # configs/dynamics/*.yaml actually changes the simulator.
        sim_kwargs = dict(self.cfg.dynamics_kwargs)
        self.vsim = VehicleSimulator(
            self.sim, self.sedan,
            steering_ratio=float(self.cfg.steering_ratio),
            a_front=float(self.cfg.a_front),
            **sim_kwargs,
        )

        # Course is shared across envs by default (broadcast Path(N, M)).
        # `course="random_bank"` instead caches a `(P, M)` bank of P
        # independent paths; each reset samples a fresh path index per env
        # and overwrites the corresponding row of `self.path`. The flags
        # below are populated by `_build_path`:
        #   `_random_path_cfg` -- parsed YAML, set for both random_long and
        #     random_bank (gates projection-cfg override and reset margin).
        #   `_is_bank` -- True only for random_bank; gates the per-reset
        #     bank-row gather in `_reset_idx`.
        #   `_path_bank` -- the (P, M) RandomPathBank; None for non-bank.
        #   `_env_path_idx` -- (N,) long, current path index per env;
        #     None for non-bank, otherwise overwritten on every reset.
        self._random_path_cfg = None
        self._is_bank = False
        self._path_bank = None
        self._env_path_idx = None
        self.path = self._build_path()

        # NOTE: spawn pose / yaw / arc-length are read directly from
        # `self.path[env_ids_t, 0]` in `_reset_idx`, so no `_initial_*`
        # cache is needed. For random_bank the bank-row gather happens
        # *before* the spawn read, so the spawn pose tracks the freshly
        # sampled path row -- not a stale `bank[0]` placeholder (Phase 2
        # review item 1).

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
                integral_max=float(self.cfg.pi_integral_max),
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
        if self.cfg.course == "random_bank":
            rp_cfg = load_random_path_cfg(self.cfg.random_path_cfg_path)
            if abs(rp_cfg.generator.ds - self.cfg.course_ds) > 1e-9:
                raise ValueError(
                    f"random_path generator.ds={rp_cfg.generator.ds} != "
                    f"TrackingEnvCfg.course_ds={self.cfg.course_ds}"
                )
            self._random_path_cfg = rp_cfg
            pb = rp_cfg.phase2_bank
            # `random_clothoid_path_bank` does not close the geometry, so
            # treating it as a loop would wrap projection / lookahead from
            # the open end back to the start across a discontinuity. Phase 2
            # supports open paths only; reject loop banks at construction
            # rather than producing silently-broken segments at runtime.
            # See docs/phase3_random_path_phase2_review.md item 4.
            if pb.is_loop:
                raise ValueError(
                    "phase2_bank.is_loop=true is not supported in Phase 2: "
                    "the bank generator does not close path geometry. "
                    "Set phase2_bank.is_loop=false in the random_path cfg."
                )
            # Sanity: each bank path must be long enough to fit one episode
            # plus the random-reset margin -- otherwise `idx_max <= 0` in
            # `_reset_idx` and the env will crash on first reset. We do the
            # same conservative computation here that `_reset_idx` does, so
            # misconfiguration fails fast at construction.
            min_required_m = (
                rp_cfg.speed.v_max * self.cfg.episode_length_s
                + self.cfg.plan_K * self.cfg.lookahead_ds
                + rp_cfg.reset.end_margin_extra_m
            )
            if pb.length_m <= min_required_m:
                raise ValueError(
                    f"phase2_bank.length_m={pb.length_m:.1f} m must exceed "
                    f"v_max*episode_length_s + plan_horizon + end_margin "
                    f"= {min_required_m:.1f} m (no room for random_reset_along_path)"
                )
            print(
                f"[INFO] random_bank: generating {pb.num_paths} paths of "
                f"{pb.length_m:.0f} m (seed={rp_cfg.generator.seed}, "
                f"ds={rp_cfg.generator.ds}) ...",
                flush=True,
            )
            bank = random_clothoid_path_bank(
                cfg=rp_cfg,
                num_paths=pb.num_paths,
                length_m=pb.length_m,
                is_loop=pb.is_loop,
                device=self.device,
            )
            self._is_bank = True
            self._path_bank = bank
            # Initial per-env path = bank[0] for all envs (placeholder; the
            # first `_reset_idx(_ALL_INDICES)` at the end of `__init__`
            # samples real per-env indices and overwrites these rows).
            self._env_path_idx = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device,
            )
            init_idx = self._env_path_idx
            return Path(
                s=bank.s[init_idx].clone(),
                x=bank.x[init_idx].clone(),
                y=bank.y[init_idx].clone(),
                v=bank.v[init_idx].clone(),
                psi=bank.psi[init_idx].clone(),
                is_loop=bank.is_loop,
                ds_value=bank.ds,
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
        # PR 3 round-1 fix (review finding 3): progress clamp range comes from
        # the YAML (`reward.progress_clamp`) instead of being hardcoded.
        self._last_progress_norm = progress_norm.clamp(
            float(self.cfg.progress_clamp_low),
            float(self.cfg.progress_clamp_high),
        )
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

        # PR 3 round-1 fix (review finding 3): each diagnostic family is gated
        # by its YAML flag. Accumulators stay zeroed when their flag is False
        # so the corresponding `Episode_*` log entries vanish from
        # `extras["log"]` (the emission path in `_reset_idx` only iterates
        # over `_episode_*_sums`, which we gate at construction).
        if self.cfg.log_reward_terms:
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
        if self.cfg.log_state_action_terms:
            self._episode_diag_sums["Episode_State/lat_err_abs"] += lat_err.abs()
            self._episode_diag_sums["Episode_State/heading_err_abs"] += hdg_err.abs()
            self._episode_diag_sums["Episode_State/vx"] += obs.vx
            self._episode_diag_sums["Episode_Action/pinion_abs"] += self._current_action[:, 0].abs()
            self._episode_diag_sums["Episode_Action/pinion_rate_abs"] += rate_pinion.abs()
            self._episode_diag_sums["Episode_Progress/progress_norm"] += self._last_progress_norm
        if self.cfg.log_projection_health:
            self._episode_diag_sums["Episode_PathProj/idx_jump_abs"] += self._last_idx_jump_abs
            self._episode_diag_sums["Episode_PathProj/idx_jump_violation_rate"] += \
                self._last_idx_jump_violation

        # One step's worth of reward accumulated -- bump the per-env counter
        # used as the per-step normalization denominator in `_reset_idx`.
        self._sum_step_count += 1

        self._last_action = self._current_action.detach()
        return r_progress + r_alive + r_lateral + r_heading + r_speed + r_rate + r_term

    def _get_observations(self) -> dict:
        """Path-relative observation. Layout driven by `spaces.observation` YAML.

        PR 3 round-1 fix (review finding 2): the tensor is built from the
        YAML's enabled fields (`imu_fields`, `include_pinion_angle`,
        `include_path_errors`, `include_plan`, `include_world_pose`) and the
        resulting width matches `_derived_observation_space(env_bundle)`. A
        runtime assertion at the end of this method guards against drift.

        Layout (when all flags True with default `imu_fields = [vx, yaw_rate,
        ax, ay, roll, pitch]`, `plan_K=10`, `include_world_pose=False`):
          [0:6]   IMU: vx, yaw_rate, ax, ay, roll, pitch
          [6]     pinion_angle
          [7:9]   lateral_error, heading_error
          [9:39]  plan_xyv flattened
        """
        self._compute_path_state()
        obs_struct = self._last_obs

        parts: list[Tensor] = []
        # IMU scalar fields (variable list driven by YAML).
        for fname in self.cfg.obs_imu_fields:
            v = getattr(obs_struct, fname)
            parts.append(v.unsqueeze(-1))
        if self.cfg.obs_include_pinion_angle:
            parts.append(obs_struct.pinion_angle.unsqueeze(-1))
        if self.cfg.obs_include_path_errors:
            parts.append(self._last_lat_err.unsqueeze(-1))
            parts.append(self._last_hdg_err.unsqueeze(-1))
        if self.cfg.obs_include_plan:
            parts.append(self._last_plan.x)
            parts.append(self._last_plan.y)
            parts.append(self._last_plan.v)
        if self.cfg.obs_include_world_pose:
            # x, y, yaw of the vehicle in env-local frame (env_origins removed
            # in `_compute_path_state`). Yaw comes from the most recent
            # ground-truth state.
            parts.append(self._last_pos_local[:, 0:1])
            parts.append(self._last_pos_local[:, 1:2])
            parts.append(self._last_state_gt.rpy[:, 2:3])

        obs = torch.cat(parts, dim=-1)
        # Runtime gate: if the YAML-derived `cfg.observation_space` and the
        # actual tensor width disagree, the gym Box and the policy's input
        # dim will desync silently. Raise here so the failure is visible.
        assert obs.shape[-1] == int(self.cfg.observation_space), (
            f"obs shape {obs.shape} != cfg.observation_space "
            f"{int(self.cfg.observation_space)}"
        )
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
        # PR 3 round-1 fix (review finding 3): emission gated by YAML flags.
        # When `log_reward_terms=False` the per-term sums never accumulated
        # (see `_get_rewards`); we additionally skip the emission here so the
        # `Episode_Reward/*` keys do not appear in `extras["log"]` at all.
        if self.cfg.log_reward_terms:
            for key, sums in self._episode_sums.items():
                log_extras[f"Episode_Reward/{key}"] = (
                    sums[env_ids_t] / ep_len.to(sums.dtype)
                ).mean()
                sums[env_ids_t] = 0.0
        # Diagnostic state / action / progress means (item F). Same
        # per-step normalization as the reward terms above.
        _state_action_keys = {
            "Episode_State/lat_err_abs",
            "Episode_State/heading_err_abs",
            "Episode_State/vx",
            "Episode_Action/pinion_abs",
            "Episode_Action/pinion_rate_abs",
            "Episode_Progress/progress_norm",
        }
        _projection_keys = {
            "Episode_PathProj/idx_jump_abs",
            "Episode_PathProj/idx_jump_violation_rate",
        }
        for key, sums in self._episode_diag_sums.items():
            if key in _state_action_keys and not self.cfg.log_state_action_terms:
                continue
            if key in _projection_keys and not self.cfg.log_projection_health:
                continue
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

        n_reset = env_ids_t.shape[0]

        # For random_bank: sample a fresh path index per resetting env and
        # overwrite the corresponding rows of self.path with the bank's
        # geometry. Done BEFORE the reset-pose logic below, which reads
        # `self.path.x[env_ids_t, idx]` etc. -- so the spawn pose / yaw /
        # arc-length / warm-start v all come from the newly assigned path.
        # Non-bank courses (built-in + random_long) leave self.path
        # untouched.
        #
        # Phase 2 design choice (review item 2): the original Phase 2 plan
        # called for path_id-gather projection -- per-env `_path_id` plus a
        # bank-aware `Path.project` that gathers `bank.x[path_id, idx]`
        # without ever copying. We instead copy the chosen bank row into
        # `self.path[env_ids_t]` at reset and keep the existing
        # `Path.project` unchanged. Cost is `n_reset * M * 5 * 4 B` per
        # reset; for the default 256 envs / M=5000 this is ~25 MB per
        # full-batch reset and is negligible vs the per-step physics work.
        # Trade-off accepted because (a) the projection / observation hot
        # path stays untouched, (b) the running env's path is independent
        # of the bank tensor (matters for Phase 3 partial regeneration --
        # we can overwrite bank slots without disturbing in-flight rollouts).
        if self._is_bank:
            new_path_idx = torch.randint(
                0, self._path_bank.num_paths, (n_reset,), device=self.device,
            )
            self._env_path_idx[env_ids_t] = new_path_idx
            self.path.s[env_ids_t] = self._path_bank.s[new_path_idx]
            self.path.x[env_ids_t] = self._path_bank.x[new_path_idx]
            self.path.y[env_ids_t] = self._path_bank.y[new_path_idx]
            self.path.v[env_ids_t] = self._path_bank.v[new_path_idx]
            self.path.psi[env_ids_t] = self._path_bank.psi[new_path_idx]

        # Per-env reset pose. Stage 0 default: random index along the path
        # (item 2). When `random_reset_along_path=False`, fall back to the
        # path[0] start pose for reproducibility (e.g., eval rollouts).
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
            # Spawn at sample 0 of each env's *current* path. For random_bank
            # this matters: by the time we reach this branch, the bank-row
            # gather above has overwritten `self.path[env_ids_t]` with the
            # freshly-sampled path, so `self.path.x[env_ids_t, 0]` etc. give
            # the start pose of the new path -- not the stale `bank[0]`
            # placeholder seeded at construction. For built-in courses /
            # random_long the per-env rows are identical to the broadcast
            # path[0], so behaviour is unchanged. play.py sets
            # `random_reset_along_path=False` for deterministic eval, which
            # makes this branch the hot path for random_bank rollouts. See
            # docs/phase3_random_path_phase2_review.md item 1.
            idx = torch.zeros(n_reset, dtype=torch.long, device=self.device)
            pos_local_xy = torch.stack(
                [self.path.x[env_ids_t, idx], self.path.y[env_ids_t, idx]],
                dim=-1,
            )
            yaw = self.path.psi[env_ids_t, idx]
            s_reset = self.path.s[env_ids_t, idx]
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
        # PR 3 round-1 fix (review finding 3): gated by `reset.warm_start_velocity`.
        # When False, the simulator's reset leaves vehicles at zero velocity
        # (path[0] start pose). Useful for ablations / classical eval.
        if self.cfg.warm_start_velocity:
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
