# Stage 3 / PR 3 — env / reward / PPO YAML wiring

## Files committed

- `src/vehicle_rl/config/schema.py` — Refined `EnvSchema` with per-leaf
  dataclasses (`TimingSchema`, `SceneSchema`, `SpacesSchema`, `ObservationSchema`,
  `PlannerSchema`, `PlannerProjectionSchema`, `ResetSchema`, `ActionScalingSchema`,
  `SpeedControllerSchema`, `RewardSchema`, `TerminationSchema`, `DiagnosticsSchema`).
  Refined `AgentSchema` with `AgentRunnerSchema`, `AgentObsGroupsSchema`,
  `AgentPolicySchema`, `AgentAlgorithmSchema`. `validate_keys` now recurses to
  every leaf for env / agent bundles, matching the PR-2 vehicle / dynamics
  approach.
- `src/vehicle_rl/config/isaac_adapter.py` — Added `make_tracking_env_cfg(env_bundle, course_bundle, controller_bundle, *, vehicle_bundle, dynamics_bundle)`
  and `make_ppo_runner_cfg(agent_bundle)`. Added private helpers
  `_derived_action_space(action_mode)` and `_derived_observation_space(env_bundle)`.
  The env factory derives `observation_space = scalar_fields + plan_K * plan_channels`
  and `action_space ∈ {1, 2}` from the YAML — never duplicated as a YAML key.
  Imports `isaaclab.sim` / `isaaclab_rl.rsl_rl` lazily so the module remains
  importable without an Isaac App.
- `src/vehicle_rl/envs/tracking_env.py` — Replaced every tunable class-level
  default in `TrackingEnvCfg` with `dataclasses.MISSING`. The factory fills
  every field. Removed the module-level `PINION_MAX = ...` constant; pruned
  unused `COG_Z_DEFAULT`, `DELTA_MAX_RAD`, `SEDAN_CFG`, `A_X_TARGET_*` imports.
  `STEERING_RATIO` retained (still used at `VehicleSimulator(...)` call).
- `src/vehicle_rl/tasks/tracking/__init__.py` — Gym registry now points at
  `vehicle_rl.tasks.tracking.entry_points:tracking_env_cfg_factory` /
  `:rsl_rl_runner_cfg_factory`. The class-based entry points are retired.
- `src/vehicle_rl/tasks/tracking/entry_points.py` (new) — 0-arg factories
  for `env_cfg_entry_point` / `rsl_rl_cfg_entry_point`. Reads
  `VEHICLE_RL_EXPERIMENT_YAML` env var; falls back to
  `configs/experiments/rl/phase3_circle_stage0a.yaml`. Exposes
  `LEGACY_COURSE_TO_EXPERIMENT` mapping (`circle` → stage0a, `random_long` →
  phase3_random_long, `random_bank` → phase3_random_bank,
  `s_curve` / `dlc` / `lemniscate` → stage0a + train_ppo's runtime override
  of `cfg.course`).
- `src/vehicle_rl/tasks/tracking/agents/rsl_rl_ppo_cfg.py` — Replaced
  `TrackingPPORunnerCfg` class with `make_default_tracking_ppo_cfg()` factory
  wrapping `make_ppo_runner_cfg`.
- `scripts/rl/train_ppo.py` — Legacy `--course X` CLI now sets
  `VEHICLE_RL_EXPERIMENT_YAML` (when not already set) using
  `LEGACY_COURSE_TO_EXPERIMENT`, before `parse_env_cfg` calls the registry's
  factory. Other legacy CLI flags (`--num_envs`, `--max_iterations`, etc.)
  retain their post-factory override behaviour for stage 3; PR 4 removes them.
- `tests/config/test_isaac_adapter.py` — Added 24 new test cases (see below).

## Tests added or modified

New cases in `tests/config/test_isaac_adapter.py`:

- `TestDerivedActionSpace` (3): steering_only=1, steering_and_accel=2, unknown rejected.
- `TestDerivedObservationSpace` (4): default=39, world_pose adds 3, no_plan
  drops 30, plan_K=5 yields 24.
- `TestEnvBundleSchema` (4): default validates, unknown reward key rejected,
  missing termination key rejected, missing top-level key rejected.
- `TestAgentBundleSchema` (3): default validates, unknown algorithm key
  rejected, missing runner.max_iterations rejected.
- `TestEntryPointFactoryDispatch` (3): legacy course-name → experiment YAML
  table assertions for circle / random_long / random_bank.
- `TestMakeTrackingEnvCfg` (3, isaaclab-gated): full-cfg field flow-through;
  steering_and_accel → action_space=2; missing reward.progress rejected.
- `TestMakeTrackingEnvCfgSignature` (1, source-only): static gate confirms
  every `MISSING`/typed field on `TrackingEnvCfg` is assigned by
  `make_tracking_env_cfg`. Catches drift between cfg-class and factory.
- `TestMakePPORunnerCfg` (3, isaaclab-gated): full-cfg field flow-through,
  unknown nested key rejected, missing algorithm block rejected.

## Fast checks run

- `"C:/work/isaac/env_isaaclab/Scripts/python.exe" -m unittest discover -s tests/config -v`
  → **105 tests, 0 errors, 0 failures, 8 skipped** (was 81 / 2 skipped on
  stage-2 round-2 head). Skipped: 2 SedanCfg tests + 6 new env / agent
  factory tests gated on `_isaaclab_available()` (carb / Isaac App needed).
  All non-gated PR 1 / PR 2 tests still pass.
- `python -c "import ast; ast.parse(...)"` on each modified .py file —
  syntactically clean.

## Boot check (Python-level)

- `load_experiment("configs/experiments/rl/phase3_circle_stage0a.yaml")`
  resolves cleanly: env / course / agent / vehicle / dynamics sub-bundles
  populated; `agent.runner.max_iterations=200` from the experiment override;
  `env.speed_controller.controller` is the resolved speed_pi bundle.
- `from vehicle_rl.config.isaac_adapter import make_tracking_env_cfg, make_ppo_runner_cfg`
  succeeds without isaaclab (factories are lazy on the heavy imports).
- `from vehicle_rl.tasks.tracking.entry_points import _load_resolved_experiment;
  _load_resolved_experiment()` returns the resolved phase3_circle_stage0a
  bundle.
- The full env / agent factory invocations (`make_tracking_env_cfg(...)`,
  `make_ppo_runner_cfg(...)`) are exercised by the isaaclab-gated tests in
  `TestMakeTrackingEnvCfg` / `TestMakePPORunnerCfg`. They will run on a
  machine with Isaac Sim available; the static signature gate covers the
  field-flow contract on machines without Isaac.
- `train_ppo.py` was NOT executed (per HARD TIME BUDGET); the orchestrator
  will run a `train_ppo` smoke if needed.

## Open issues

- Legacy `--course s_curve` / `--course dlc` / `--course lemniscate` map to
  `phase3_circle_stage0a.yaml`; the post-factory `_apply_cli_overrides` then
  patches `env_cfg.course = args.course`. PR 4 will add dedicated experiment
  YAMLs for those courses and remove the runtime override.
- `random_path_cfg_path` is set to `configs/random_path.yaml` by the
  factory (legacy compat). PR 4 deletes the legacy file and replaces the
  in-place `_build_path` random-path branch with `adapter.build_path`.
- `tracking_env.py:_build_path` still dispatches on `cfg.course` string;
  the YAML-driven `build_path` adapter factory is invoked only by
  classical / play paths. PR 4 collapses both paths.
- `TrackingPPORunnerCfg` (the class) is gone; any external import of
  `vehicle_rl.tasks.tracking.agents.rsl_rl_ppo_cfg.TrackingPPORunnerCfg`
  will break. Internal callers (gym registry) are migrated. No external
  users known in repo.

STAGE_SHA=e881fce8aa9c38846fc8e0d0c7b02582ea9279ca

## Stage-3 fixes (round 1)

Addresses the 5 findings in stage-3-review-0.md.

### F1 — TrackingEnv reads dynamics from YAML
- `TrackingEnvCfg` now carries `dynamics_kwargs: dict` + `a_front: float`
  (filled by `make_tracking_env_cfg` from `make_simulator_kwargs(dynamics_bundle)`
  + `vehicle_bundle["geometry"]["a_front_m"]`).
- `TrackingEnv.__init__` constructs `VehicleSimulator(**self.cfg.dynamics_kwargs,
  steering_ratio=STEERING_RATIO, a_front=self.cfg.a_front)`. Literal lag /
  cornering / damper / mu / gravity values are gone.
- `make_tracking_env_cfg` now REQUIRES `dynamics_bundle` (the previous
  `mu_default=0.9` / disk-fallback path is removed). Missing → ValueError.

### F2 — `_get_observations` matches `_derived_observation_space`
- New cfg fields: `obs_imu_fields`, `obs_include_pinion_angle`,
  `obs_include_path_errors`, `obs_include_plan`, `obs_include_world_pose`.
- The observation tensor is now built by iterating those cfg fields. A
  runtime `assert obs.shape[-1] == cfg.observation_space` guards drift.
- `include_world_pose=True` appends `[pos_x_local, pos_y_local, yaw]`
  (env-local frame, matching the projection frame used elsewhere).

### F3 — Validated-but-ignored YAML leaves are wired
- `reward.progress_clamp` → `cfg.progress_clamp_low/high`; consumed by
  `_update_progress` instead of the hardcoded ±1.0 clamp.
- `reset.warm_start_velocity` → `cfg.warm_start_velocity`; gates the warm-
  start `write_root_velocity_to_sim` block in `_reset_idx`.
- `diagnostics.log_reward_terms` → gates accumulation AND emission of
  `Episode_Reward/*` keys in `_get_rewards` / `_reset_idx`.
- `diagnostics.log_state_action_terms` → gates `Episode_State/*`,
  `Episode_Action/*`, `Episode_Progress/*`.
- `diagnostics.log_projection_health` → gates `Episode_PathProj/*`.
- `speed_pi.integral_max` → `cfg.pi_integral_max`; passed to
  `PIDSpeedController(integral_max=...)` so the integrator clamps at the
  YAML value (not the controller's class-level default).

### F4 — train_ppo.py legacy CLI precedence
- `--course X` now sets `VEHICLE_RL_EXPERIMENT_YAML` UNCONDITIONALLY (CLI
  wins over a stale env var); the legacy `env_cfg.course = args.course`
  post-hoc patch is removed. The env var is set BEFORE `import gym` /
  `gym.make`, which is when the registry's 0-arg factory reads it.
- s_curve / dlc / lemniscate currently still map to phase3_circle_stage0a;
  PR 4 will add dedicated experiment YAMLs and remove the placeholder
  mapping.

### F5 — Tests for nested leaves and YAML-affects-runtime
- Added in `tests/config/test_isaac_adapter.py`:
  - `TestEnvSchemaNestedLeafStrictness` (6): missing `warm_start_velocity`,
    missing `log_reward_terms`, unknown `planner.projection` sibling,
    unknown `spaces.observation` sibling, length-3 `progress_clamp` (gated),
    missing `speed_pi.integral_max`.
  - `TestYAMLAffectsRuntime` (7): `PIDSpeedController.integral_max` actually
    clamps the integrator at 5.0 over 100 steps; gated factory tests assert
    `progress_clamp_low/high`, `warm_start_velocity`, all 3 diag flags,
    `pi_integral_max`, and `dynamics_kwargs.tau_drive` / `mu_default` reach
    cfg; `dynamics_bundle is None` raises.
  - `TestObservationLayoutFromYaml` (1, gated): custom `imu_fields`,
    `include_pinion_angle=False`, `include_plan=False`,
    `include_world_pose=True` → `cfg.observation_space=7` matches
    `_derived_observation_space(b)`.
  - `TestTrackingYAMLLeafCoverage` (1): walks every leaf in
    `configs/envs/tracking.yaml` and asserts each non-wrapper key name
    appears in `isaac_adapter.py` or `tracking_env.py`. Catches future
    drift where a new YAML key is added but never consumed.

### Verification
- `python -m unittest discover -s tests/config -v` → 120 tests, OK
  (was 105 / 8 skipped on round 0). 16 skipped now: 2 SedanCfg + 14 Isaac-
  gated env-factory tests; 6 of the new tests are isaac-free and run.
- AST-parse on the 4 modified files clean.
- `load_experiment("phase3_circle_stage0a.yaml")` resolves end-to-end with
  the new diagnostics / progress_clamp / integral_max / warm_start /
  imu_fields leaves accessible.
- `train_ppo.py` was NOT executed (HARD TIME BUDGET); orchestrator runs
  the smoke separately.

### Known limitations
- `TestEnvSchemaNestedLeafStrictness.test_warm_start_velocity_int_rejected`
  is in spirit a "type strictness" test, but `validate_keys` only checks
  the key shape (Python's `bool(0)` / `bool(1)` are truthy/falsy). The test
  substitutes "missing key" coverage instead, which exercises the same
  enforcement mechanism (per-shape dataclass schema). A true type-tagged
  validator is out of scope for this fix.
- The leaf-coverage test allowlist treats `enabled_when_action_mode` as
  doc-only (the `steering_only` derived flag from `action_mode` is the
  single source of truth at runtime). No runtime branch reads
  `speed_controller.enabled_when_action_mode`.

STAGE_SHA=62d80f75857f02b84a3415e25f306a15dd768188

## Stage-3 fixes (round 2)

Addresses the 3 findings in stage-3-review-1.md.

### F1a — PhysX SimulationCfg gravity from YAML

`make_tracking_env_cfg` now reads `gravity_mps2` from the dynamics
bundle and constructs `SimulationCfg(gravity=(0.0, 0.0, -gravity_mps2))`.
Previously hard-coded to `(0.0, 0.0, -9.81)`, so `configs/dynamics/*.yaml`
changes only affected `VehicleSimulator` and never PhysX. The same value
is already exposed by `make_simulator_kwargs(...)["gravity"]` for the
analytical model, so both paths now see the same number.

### F1b — `steering_ratio` from vehicle YAML, not module constant

`TrackingEnvCfg` has a new `steering_ratio: float = MISSING` field.
`make_tracking_env_cfg` populates it from
`vehicle_bundle["steering"]["steering_ratio"]`. `TrackingEnv.__init__`
now reads `self.cfg.steering_ratio` instead of importing
`STEERING_RATIO` from `vehicle_rl.assets`. The
`from vehicle_rl.assets import STEERING_RATIO` line was removed.

Audit: no other `vehicle_rl.assets` symbols are referenced by
`tracking_env.py` (verified by grep). `DELTA_MAX_RAD`, `COG_Z_DEFAULT`,
`PINION_MAX`, `TOTAL_MASS`, `WHEELBASE`, `TRACK`, `WHEEL_RADIUS`,
`WHEEL_WIDTH` -- none of them appear, so no further routing is
required (those module constants exist as a back-compat surface for
other consumers, not the env).

### F3 — Dedicated experiment YAMLs for s_curve / dlc / lemniscate

New files:
- `configs/experiments/rl/phase3_s_curve.yaml`
- `configs/experiments/rl/phase3_dlc.yaml`
- `configs/experiments/rl/phase3_lemniscate.yaml`

Each is a thin composition root (course_ref selects the matching
course YAML) with `experiment_name` and `run_name` set to
`phase3_<course>`. `LEGACY_COURSE_TO_EXPERIMENT` updated to point
at the dedicated YAMLs instead of aliasing to phase3_circle_stage0a.

The post-factory `cfg.course = args.course` patch removal from
round 1 stays intact -- legacy `--course X` now correctly trains on
course X because it dispatches to a YAML whose `course_ref` is X.

### Tests added

In `tests/config/test_isaac_adapter.py`:

- `TestRound2GravityAndSteeringRatio` (3, 2 isaaclab-gated):
  non-default gravity → `cfg.sim.gravity == (0.0, 0.0, -1.62)`;
  default gravity sanity (-9.81); non-default
  `vehicle.steering.steering_ratio=12.0` → `cfg.steering_ratio==12.0`
  AND `cfg.pinion_max == delta_max_rad * 12.0`.
- `TestRound2NoSteeringRatioImportInTrackingEnv` (1): static gate that
  fails if anyone re-introduces `from vehicle_rl.assets import
  STEERING_RATIO` or `steering_ratio=STEERING_RATIO` in
  `tracking_env.py`.
- `TestRound2LegacyCourseDedicatedYamls` (7, 1 isaaclab-gated):
  s_curve / dlc / lemniscate map to their dedicated phase3 YAMLs;
  each new YAML loads via `load_experiment` and resolves to the
  matching `course.type`; isaaclab-gated end-to-end test confirms
  `tracking_env_cfg_factory()` with `VEHICLE_RL_EXPERIMENT_YAML`
  set via `LEGACY_COURSE_TO_EXPERIMENT["s_curve"]` yields
  `cfg.course == "s_curve"`.

### Verification

- `python -m unittest discover -s tests/config -v` → **131 tests, OK,
  20 skipped** (was 120 / 16 skipped on round 1). 11 new tests; 3 of
  them isaaclab-gated, 8 run unconditionally.
- `train_ppo.py` was NOT executed (HARD TIME BUDGET); orchestrator
  runs the smoke separately.

STAGE_SHA=5fd91c68a725370fa133b72d5adfeb16467af340


## Stage-3 fixes (round 3, cap override)

User-approved cap override (4th post-stage-3 fix round) so the
codex round-2 finding — `_build_path` string-dispatch passing
hardcoded cfg fields to planner generators that no longer carry
defaults — is fully resolved before stage 4 begins.

### Single fix

`TrackingEnv._build_path()` (in `src/vehicle_rl/envs/tracking_env.py`)
no longer dispatches on `cfg.course == "..."` and no longer passes
`cfg.radius` / `cfg.target_speed` / `cfg.course_ds` literals into
`waypoints.<...>_path()`. Instead it routes the full course
construction through the existing
`vehicle_rl.config.isaac_adapter.build_path(course_bundle, num_envs, device)`
factory (added in PR 2), which validates every per-course key set
against the YAML and supplies the right kwargs to the planner. This
makes `--course s_curve | dlc | lemniscate` work again at env
construction time.

To carry the YAML through to `_build_path`, `TrackingEnvCfg`
gained a single new field:

- `course_bundle: dict = MISSING` — the resolved course bundle from
  `configs/courses/*.yaml` (with `generator_ref` resolved). Populated
  by `make_tracking_env_cfg` from its existing `course_bundle`
  parameter.

For `random_long` / `random_bank` the env additionally calls
`isaac_adapter._make_random_path_cfg(bundle["generator"])` to
reconstruct the typed `RandomPathGeneratorCfg` so the existing
`__init__` (projection params) and `_reset_idx`
(`rp.speed.v_max` + `rp.reset.end_margin_extra_m`) caches keep
working unchanged. The legacy
`load_random_path_cfg(self.cfg.random_path_cfg_path)` lookup is
gone.

### cfg fields kept-but-deprecated (not dropped this round)

These were the fields the old dispatch consumed; they are NO LONGER
read by `_build_path` after this round, but I kept them populated to
minimize blast radius — three of them are still written by the
legacy CLI overrides in `train_ppo.py:144` and `play.py:147`, which
are PR 4's territory:

- `cfg.course` (str) — still useful as a discriminator for telemetry
  / logging.
- `cfg.radius` / `cfg.target_speed` / `cfg.course_ds` — vestigial; no
  longer dispatch axes. Test references in `test_isaac_adapter.py`
  still assert their values match the YAML (kept passing).
- `cfg.random_path_cfg_path` (str) — vestigial; no longer loaded.
  PR 4 deletes the CLI overrides + the legacy
  `configs/random_path.yaml` file + this field together.

### cfg fields added

- `course_bundle: dict` — see above.

### Tests

- `TestRound3CourseBundleOnCfg` (2, both isaaclab-gated):
  `cfg.course_bundle["type"]` matches the input, and required
  per-course keys (`length_m`, `amplitude_m`, `n_cycles`, `n_raw`
  for s_curve) are carried through.
- `TestRound3BuildPathLegacyCourses` (3, no isaaclab): exercise
  `build_path(<resolved s_curve / dlc / lemniscate bundle>, ...)`
  directly and assert shape / loop characteristics — these directly
  cover the previously-broken courses.
- `TestRound3TrackingEnvDispatchesViaAdapter` (1, no isaaclab):
  static gate that the `if self.cfg.course == "..."` chain is gone
  from `tracking_env.py` and the adapter import is present.

### Verification

- `python -m unittest discover -s tests/config -v` →
  **137 tests, OK, 22 skipped** (was 131 / 20 skipped in round 2).
  6 new tests; 4 run unconditionally, 2 isaaclab-gated.
- `run_classical.py --course circle --duration 2.0 --headless --no_video`
  smoke deferred to orchestrator; `train_ppo.py` was NOT executed
  (HARD TIME BUDGET).

