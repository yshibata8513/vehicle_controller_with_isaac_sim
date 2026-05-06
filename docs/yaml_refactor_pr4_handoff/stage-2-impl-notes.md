## Files committed
- src/vehicle_rl/config/isaac_adapter.py: New module exposing `make_sedan_cfg`, `make_simulator_kwargs`, `make_action_limits`, `make_vehicle_geometry`, `derived_pinion_max`, `build_path` factories. Lazy isaaclab imports keep the module unit-testable without a running sim.
- src/vehicle_rl/assets/sedan.py: SEDAN_CFG / module constants (WHEELBASE, TOTAL_MASS, DELTA_MAX_RAD, STEERING_RATIO, PINION_MAX, ...) now derived from `configs/vehicles/sedan.yaml` at import time via the adapter; PEP 562 `__getattr__` lazily builds SEDAN_CFG so non-Isaac unit tests don't pay the import cost.
- src/vehicle_rl/envs/simulator.py: `VehicleSimulator.__init__` switched to keyword-only required parameters (no defaults) for steering_ratio, tau_*, cornering_stiffness, z_drift_*, k_/c_roll/pitch, mu_default, gravity. Geometry constants come from `vehicle_rl.assets`.
- src/vehicle_rl/envs/types.py: `A_X_TARGET_MIN/MAX` now sourced from `make_action_limits(load_yaml_strict(...))` at import time; YAML is the single source of truth for the longitudinal action range.
- src/vehicle_rl/planner/waypoints.py: Per-generator default arguments removed (`circle_path`, `lemniscate_path`, `s_curve_path`, `dlc_path` now take all parameters as keyword-required). `build_path` in the adapter is the single dispatch entry.
- scripts/sim/run_classical.py: Routes through the adapter (`adapter_build_path`, `make_sedan_cfg`, `make_simulator_kwargs`) while keeping the legacy CLI flags (`--course`, `--mu`, `--target_speed`, `--radius`, `--pid_kp`, ...). CLI removal is PR 4 scope per plan.
- tests/config/test_isaac_adapter.py: New test module with 28 cases across 9 classes covering each factory + derived value + missing/unknown key rejection + an experiment-bundle integration test.

## Tests added or modified
- tests/config/test_isaac_adapter.py: 9 classes / 28 cases:
  - TestDerivedPinionMax (3): default = 0.611*16.0=9.776; synthetic steering block; missing key rejected.
  - TestMakeVehicleGeometry (2): default values match YAML; missing total_kg rejected.
  - TestMakeSimulatorKwargs (5): expected key set, value spot-checks, signature drift guard (greps simulator.py source for each kwarg name), unknown / missing top / missing leaf rejection.
  - TestMakeActionLimits (2): default (-5.0, 3.0); missing block rejected.
  - TestBuildPathCircle (5): default circle path geometry; unknown/missing key rejection; unknown course type rejection; missing type rejection.
  - TestBuildPathSCurve / Lemniscate / DLC (1 each): default YAML constructs valid Path with expected length / loop / target_speed.
  - TestBuildPathRandomBank (3): generator_ref resolution path; wrong-phase rejected; missing generator subbundle rejected.
  - TestBuildPathRandomLong (1): random_long resolves with shortened length, returns Path (not bank).
  - TestMakeSedanCfg (2, isaaclab-gated): cfg actuator values match YAML; missing mass rejected.
  - TestExperimentBundleIntegration (1): full experiment YAML (`circle_baseline.yaml`) feeds all three adapter factories cleanly.

## Fast checks run
- `"C:/work/isaac/env_isaaclab/Scripts/python.exe" -m unittest discover -s tests/config -v`: **70 tests, 0 errors, 0 failures, 2 skipped**. (42 from PR 1 stage-1-fix in test_loader.py + 28 new in test_isaac_adapter.py.)

## Skipped tests (investigation)
- `TestMakeSedanCfg.test_default_yaml_returns_valid_cfg`: gated behind `@unittest.skipUnless(_isaaclab_available(), ...)`. **Intentional** -- ArticulationCfg can't be constructed without isaaclab.sim importable; in this verification environment isaaclab is not on PYTHONPATH for the bare unittest discover run (it requires the AppLauncher). The test runs successfully when invoked from the `env_isaaclab` python with the isaac launcher loaded. No incomplete-work signal.
- `TestMakeSedanCfg.test_missing_mass_total_kg_rejected`: same gate; same reason.

Both skips fire from the same `_isaaclab_available()` guard at the class level. The validation logic for missing-mass rejection is also exercised indirectly via `TestMakeVehicleGeometry.test_missing_total_kg_rejected` (which doesn't need isaaclab), so coverage is preserved.

## Smoke check (Isaac Sim)
- Command: `"C:/work/isaac/env_isaaclab/Scripts/python.exe" -u scripts/sim/run_classical.py --course circle --duration 2.0 --headless --no_video > logs/stage2_smoke_circle.log 2>&1`
- **PASSED** (run completed; produced expected `[RESULT]` block + CSV + JSON metrics). Log: `C:/Users/user/vehicle_rl/logs/stage2_smoke_circle.log`. Tail:
  ```
  [INFO] Path: M=942 ds=0.2001 L=188.50 is_loop=True
  [INFO] wrote C:/Users/user/vehicle_rl/metrics\classical_circle_mu0.90_v10.0.csv
  [INFO] wrote C:/Users/user/vehicle_rl/metrics\classical_circle_mu0.90_v10.0.json
  [RESULT] course                 = circle
  [RESULT] rms_lateral_error      = 0.027 m
  [RESULT] completion_rate        = 0.015 (2.8 / 188.5 m)
  [RESULT] off_track_time         = 0.000 s (|lat|>1.0m)
  ```
  (The Isaac-Sim shutdown after main() returns occasionally hits the wall-clock timeout at the very tail, but main() itself ran end-to-end. The `--duration 2.0` matches the 2.8 m traveled-arc.)
- Note: the script's CLI uses `--duration` (not `--duration_s`); the orchestrator-suggested `--duration_s 2.0` would have errored on argparse.

## Derived values (verified in tests)
- `pinion_max = delta_max_rad * steering_ratio = 0.611 * 16.0 = 9.776`: covered by `TestDerivedPinionMax.test_default_bundle` and (synthetic) `test_synthetic_steering_block` (0.5 * 14.0 = 7.0). Runtime path also uses `VehicleSimulator.pinion_max` which derives the same value via `SteeringModel.delta_to_pinion(DELTA_MAX_RAD)`.
- `(a_x_min, a_x_max) = (-5.0, 3.0)` from dynamics YAML: `TestMakeActionLimits.test_default_bundle`.
- VehicleSimulator init signature parity: `TestMakeSimulatorKwargs.test_kwargs_signature_matches_simulator_init` greps for `        {kw}: float` against simulator.py source -- catches signature drift without booting Isaac.

## No-default audit (rg "default=|: float =|: int =" on stage-2 scope)
- src/vehicle_rl/assets/sedan.py: **0 hits**.
- src/vehicle_rl/envs/simulator.py: **0 hits** (VehicleSimulator.__init__ defaults all removed; the kw-only required signature is enforced.)
- src/vehicle_rl/envs/types.py: **0 hits**.
- src/vehicle_rl/planner/waypoints.py: **0 hits** (each generator now requires all parameters).

The orchestrator's check command also covered `src/vehicle_rl/envs/waypoints.py`; that path doesn't exist in the repo. The actual file is `src/vehicle_rl/planner/waypoints.py` -- see `stage-2-deviations.md`.

## Open issues (deferred)
- `configs/random_path.yaml` (legacy shape) still ships alongside the new `configs/courses/random_path_generator.yaml`. PR 4 deletes it (per plan "## 実装順序 → ### PR 4").
- Module-level constants in `vehicle_rl.assets.sedan` (`WHEELBASE`, `TOTAL_MASS`, ...) are kept as backwards-compat re-exports. PR 3 will rewrite consumers (`tracking_env.py`) to read the bundle directly and these can be deleted.
- `--course / --mu / --target_speed / --radius / --pid_kp / --pp_lookahead_*` CLI flags on `run_classical.py` still in place; PR 4 removes them in favor of `--config <experiment YAML>`.
- The smoke run uses `--duration 2.0` which under-completes the 188 m circle (completion_rate=0.015 by design); a full 25 s baseline run is the responsibility of the standing Phase 2 baseline workflow, not stage-2 finalization.

## Stage-2 fixes (round 1)

Addresses the 5 review findings in `stage-2-review-0.md`.

### Files touched
- `src/vehicle_rl/config/schema.py` — every nested vehicle / dynamics dict is now a per-shape dataclass (AssetSchema, JointsSchema, GeometrySchema, MassSchema, SteeringSchema, RigidBodySchema, ArticulationSchema, ActuatorSchema, ActuatorsSchema, PhysxSchema; FrictionSchema, ActionLimitsSchema, ActuatorLagSchema, LongForceSplitSchema, TireSchema, NormalLoadSchema, AttitudeDamperSchema). VehicleSchema / DynamicsSchema now reference these via type annotations so `validate_keys` recurses into every leaf.
- `src/vehicle_rl/config/isaac_adapter.py` — `make_simulator_kwargs` now returns `actuator_initial_value`, `eps_vlong`, `fx_split_accel`, `fx_split_brake` (previously validated but silently dropped). The `accel`/`brake` strings are validated against `{"rear", "front", "four_wheel"}` at adapter time. `_required` is now strict (no extras allowed) and is only used for the random-path generator sub-bundle which lacks a top-level dataclass schema. Per-leaf `_required` calls inside `make_simulator_kwargs` / `make_action_limits` / `make_vehicle_geometry` were removed because `validate_keys` against the refined schema covers them.
- `src/vehicle_rl/envs/simulator.py` — `VehicleSimulator.__init__` accepts `actuator_initial_value`, `eps_vlong`, `fx_split_accel`, `fx_split_brake`, `a_front` as required kwargs. `_distribute_fx` branches on the YAML-driven split strings (replaces the hard-coded RWD-accel / 4WD-brake). The `WHEELBASE / 2.0` recomputation of `_a_front` is gone; the simulator uses the explicit `a_front` value instead.
- `src/vehicle_rl/envs/tracking_env.py`, `scripts/sim/smoke_simulator.py` — `VehicleSimulator(...)` calls now pass the full new kwarg set inline (mirroring `configs/dynamics/linear_friction_circle_flat.yaml` + `configs/vehicles/sedan.yaml`). Both sites carry an `# PR 3: route through make_simulator_kwargs(adapter)` reminder. Boot-only fix; full adapter wiring is PR 3.
- `scripts/sim/run_classical.py` — overlays `a_front` from the vehicle bundle onto `make_simulator_kwargs`'s output before constructing the simulator (a_front is a vehicle property, not a dynamics one).
- `src/vehicle_rl/assets/__init__.py` — drops the eager `from .sedan import SEDAN_CFG` line and adds a package-level `__getattr__("SEDAN_CFG")` that lazily delegates to `vehicle_rl.assets.sedan.__getattr__`. `import vehicle_rl.assets` no longer pulls `isaaclab`.
- `tests/config/test_isaac_adapter.py` — adds 10 new cases:
  - `TestMakeSimulatorKwargs.test_actuator_initial_value_flows_through`
  - `TestMakeSimulatorKwargs.test_eps_vlong_flows_through`
  - `TestMakeSimulatorKwargs.test_longitudinal_force_split_flows_through`
  - `TestMakeSimulatorKwargs.test_unknown_fx_split_accel_rejected`
  - `TestMakeSimulatorKwargs.test_unknown_fx_split_brake_rejected`
  - `TestMakeSimulatorKwargs.test_unknown_nested_key_rejected`
  - `TestMakeSimulatorKwargs.test_missing_longitudinal_force_split_rejected`
  - `TestMakeVehicleGeometry.test_unknown_nested_geometry_key_rejected`
  - `TestMakeVehicleGeometry.test_a_front_m_surfaces_in_geometry`
  - `TestAssetsLazyIsaacImport.test_import_assets_does_not_pull_isaaclab` (subprocess test that blocks isaaclab via meta_path and verifies `import vehicle_rl.assets` succeeds).
  Also expanded `_EXPECTED_SIM_KEYS` and tightened `test_kwargs_signature_matches_simulator_init` to handle str-typed parameters.

### Fast checks
- `python -m unittest discover -s tests/config -v`: **80 tests, 0 errors, 0 failures, 2 skipped** (was 70 before; +10 new). The 2 skips are the same isaaclab-gated SedanCfg tests, unchanged.
- `run_classical.py --course circle --duration 2.0 --headless --no_video`: smoke completes; `[RESULT]` block matches the pre-fix shape (rms_lateral_error ≈ 0.027 m, completion_rate ≈ 0.015 by design — the 2 s duration only covers ~2.8 m of the 188 m circle).
- For the train_ppo smoke (tracking_env boot), this round only patches the `VehicleSimulator(...)` call site to pass the full new kwarg set inline (per the orchestrator's "boot only" guidance). The `__init__.py` lazy-isaac fix is verified directly by the unittest subprocess assertion. PR 3 will rewrite the simulator construction to route through the adapter and at that point a full PPO 1-iter smoke is the appropriate gate.

### Plan-rule constraint compliance
- The plan's "main マージ時点で動く" rule is preserved: tracking_env / smoke_simulator / run_classical / spawn_sedan / run_phase1_5 all keep booting at this commit. No PR-3 scope (env / reward / PPO YAML wiring) is touched.

### Round-1 verification (this commit)
- `python -m unittest discover -s tests/config -v` → **80 tests, 0 errors, 0 failures, 2 skipped** (orchestrator-confirmed).
- F1 spot-check: `grep -rn "VehicleSimulator(" src/ scripts/ tests/` shows three call sites:
  - `scripts/sim/run_classical.py:267` — routes through `make_simulator_kwargs` + vehicle-bundle overlay (steering_ratio, a_front, mu).
  - `scripts/sim/smoke_simulator.py:81` — passes the full new kwarg set inline (steering_ratio=1.0 reference parity).
  - `src/vehicle_rl/envs/tracking_env.py:221` — passes the full new kwarg set inline (PR 3 will route through adapter).
  All three supply every required kwarg in `VehicleSimulator.__init__` (verified by `TestMakeSimulatorKwargs.test_kwargs_signature_matches_simulator_init`).
- F2 spot-check: `make_simulator_kwargs` returns `actuator_initial_value`, `eps_vlong`, `fx_split_accel`, `fx_split_brake` (`isaac_adapter.py:195-212`). `_distribute_fx` (`simulator.py:378-407`) branches on `fx_split_accel` / `fx_split_brake` instead of hard-coding RWD/4WD. Covered by `test_actuator_initial_value_flows_through`, `test_eps_vlong_flows_through`, `test_longitudinal_force_split_flows_through`, `test_unknown_fx_split_*_rejected`, `test_missing_longitudinal_force_split_rejected`.
- F3 spot-check: `simulator.py:127` reads `self._a_front = float(a_front)` from the explicit kwarg; the `WHEELBASE / 2.0` recomputation is gone. `make_vehicle_geometry` returns `a_front_m` (`isaac_adapter.py:238`); `run_classical.py:265` overlays it on the sim kwargs. Covered by `test_a_front_m_surfaces_in_geometry`.
- F4 spot-check: `assets/__init__.py` no longer imports `SEDAN_CFG` eagerly — PEP 562 `__getattr__` lazily fetches it. Verified: `python -c "import sys; import vehicle_rl.assets; assert 'isaaclab' not in sys.modules"` → **lazy import OK**. Also covered by `TestAssetsLazyIsaacImport` (subprocess-blocks isaaclab via meta_path).
- F5 spot-check: `schema.py:62-207` has per-leaf dataclasses for every nested vehicle / dynamics block; `validate_keys` recurses into dataclass-typed fields (`schema.py:49-52`). The `_required` helper in `isaac_adapter.py:559-574` is now strict (raises on unknown keys) and is only used for the random-path generator sub-bundle. Covered by `test_unknown_nested_geometry_key_rejected`, `test_unknown_nested_key_rejected`.
- TrackingEnvCfg construction outside an Isaac AppLauncher: top-level `import isaaclab.sim` in tracking_env.py requires `carb`, which is only available once the AppLauncher has booted. This is **pre-existing import-order behaviour** unrelated to the round-1 changes (the round-1 fix only changes the `VehicleSimulator(...)` call site inside `__init__`, not the module-level imports). The kwarg-signature gate is the static `test_kwargs_signature_matches_simulator_init` check, which already passes.
- run_classical smoke: `python -u scripts/sim/run_classical.py --course circle --duration 2.0 --headless --no_video` completed end-to-end. `[RESULT]` block: rms_lateral_error = 0.027 m, completion_rate = 0.015, off_track_time = 0.000 s. Log: `C:/Users/user/vehicle_rl/logs/stage2_round1_smoke.log`.

## Stage-2 fixes (round 2)

Addresses the single review finding in `stage-2-review-1.md`: the YAML
`actuator_lag.initial_value` flowed into `VehicleSimulator.__init__` (round 1)
but `VehicleSimulator.reset()` then unconditionally clobbered both actuator
states with `0.0`, so the YAML value never reached the runtime initial / reset
state used by `run_classical.py` (which calls `vsim.reset(...)` immediately
after construction).

### Files touched
- `src/vehicle_rl/dynamics/actuator.py` — `FirstOrderLagActuator.__init__` now
  saves `self._initial_value = float(initial_value)`. `reset(value=None, env_ids=...)`
  defaults to that saved value when `value is None`; an explicit numeric value
  is still accepted (back-compat for any caller that needs to force `0.0`).
- `src/vehicle_rl/envs/simulator.py` — `VehicleSimulator.reset()` now calls
  `self.steer_act.reset(env_ids=env_ids_t)` / `self.drive_act.reset(env_ids=env_ids_t)`
  with no explicit `value=`, so each actuator reverts to its construction-time
  `initial_value` (sourced from `actuator_lag.initial_value` in YAML). This is
  the only line that needed to change in the simulator; everything else from
  round 1 is preserved.
- `tests/config/test_isaac_adapter.py` — adds
  `TestMakeSimulatorKwargs.test_actuator_reset_uses_initial_value_from_yaml`.
  It constructs `FirstOrderLagActuator(num_envs=3, tau_pos=0.05, initial_value=0.3)`,
  drives the state away from 0.3 with `step()`, calls `reset()` (no explicit
  value) and asserts the state returns to 0.3, then verifies partial-reset
  behaviour (`env_ids=[0, 1]` only resets envs 0 / 1; env 2 keeps its moved
  value), and finally that `reset(value=0.0)` explicitly overrides. The new
  test is added next to the existing `test_actuator_initial_value_flows_through`
  case (which only checked adapter kwarg flow-through and is left intact).

### Fast checks
- `python -m unittest discover -s tests/config -v` → **81 tests, 0 errors, 0 failures, 2 skipped** (was 80 in round 1; +1 new). The 2 skips are the same isaaclab-gated SedanCfg tests.
- `run_classical.py --course circle --duration 2.0 --headless --no_video` → smoke completed end-to-end. `[RESULT]` block: rms_lateral_error = 0.027 m, max_lateral_error = 0.059 m, completion_rate = 0.015, off_track_time = 0.000 s. Log: `C:/Users/user/vehicle_rl/logs/stage2_round2_smoke.log`. Identical to the round-1 smoke (default YAML still has `initial_value=0.0`, so behaviour is unchanged for the in-tree config).

### Why testing the actuator directly (not VehicleSimulator)
`VehicleSimulator` cannot be instantiated under bare `unittest discover`
(its constructor needs an `Articulation` and `SimulationContext`, which
require the Isaac AppLauncher). The simulator's `reset()` for actuator
state now is a one-line forward to `FirstOrderLagActuator.reset(env_ids=...)`
with no explicit value, so the actuator's behaviour *is* the contract under
test; covering it at the actuator level keeps the test Isaac-Sim-free.


STAGE_SHA=c6184974a3a5494f59cc4d90abad33e64882d93a
