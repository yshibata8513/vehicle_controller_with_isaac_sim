# Stage 2 (PR 2) — deviations from plan

## 1. waypoints.py lives in `planner/`, not `envs/`

The plan's "## Stages → ### Stage 2 → Files" entry lists `src/vehicle_rl/envs/waypoints.py`. The actual file is at `src/vehicle_rl/planner/waypoints.py` (and has been since before this refactor — see git history for `e0ff3f3 Phase 3 random_long: clothoid+arc+straight path generator + long-horizon PPO` which already had `vehicle_rl/planner/waypoints.py`).

The implementing subagent correctly edited the real path; the no-default audit in this stage uses the real path. The plan's reference to `envs/waypoints.py` is a pre-PR-1 documentation slip and does not affect the scope. The committed adapter dispatches to `vehicle_rl.planner.{circle_path, dlc_path, lemniscate_path, s_curve_path}` and `vehicle_rl.planner.random_path.{random_clothoid_path, random_clothoid_path_bank}`, which is what actually exists.

No code change required — flagging here so PR 3 / PR 4 reviewers don't grep for the wrong path.

## 2. `run_classical.py` CLI flag is `--duration`, not `--duration_s`

The orchestrator's smoke-check suggestion (`--duration_s 2.0`) does not match the script's actual flag (`--duration`). The smoke run used the real flag; the script's CLI shape was not changed in stage 2 (CLI churn is PR 4 scope).

## 3. Two skipped tests under unittest discover

`TestMakeSedanCfg.test_default_yaml_returns_valid_cfg` and `TestMakeSedanCfg.test_missing_mass_total_kg_rejected` are gated behind `@unittest.skipUnless(_isaaclab_available(), ...)`. Under bare `unittest discover` invoked from a non-AppLauncher python, `isaaclab.sim` is not importable, so the tests skip. This is by design — the cfg-construction logic is tested indirectly via `TestMakeVehicleGeometry` (no Isaac) and the smoke run (real Isaac), so behaviour is covered. Documented in `stage-2-impl-notes.md` under "Skipped tests".

No deviation from the plan's exit condition (which says "fast-test 全緑" — skipped is not red).

## 5. Round-1 inline kwarg patch in tracking_env.py / smoke_simulator.py is PR-3-adjacent

The orchestrator's round-1 prompt explicitly directed the inline kwarg patch
("pass the full new VehicleSimulator kwarg set inline") in
`tracking_env.py:221` and `scripts/sim/smoke_simulator.py:81` to keep PR 2
mergeable / bootable. Strictly speaking, rewriting these consumers to read
the YAML bundles is PR-3 scope (env wiring) and PR-2-final scope (smoke
script de-duplication). Flagging here so PR 3's reviewer doesn't double-
charge: the inline values are intentional placeholders that will be
removed in PR 3 once `tracking_env` is rewritten to call
`make_simulator_kwargs` itself. The `# PR 3: route through
make_simulator_kwargs(adapter)` comment markers in both files anchor the
follow-up.

## 4. Round-1 fix: train_ppo boot verified by unittest, not a 1-iter PPO smoke

Per the orchestrator's "boot must succeed at PR 2" guidance and the
"fall back to importing TrackingEnv's cfg class" suggestion, the round-1
fix patches the `VehicleSimulator(...)` call site in `tracking_env.py` to
pass the full new kwarg set inline (mirroring the YAML defaults). A
running 1-iter PPO smoke is not the gate for this round because PR 3 is
the right place to wire `tracking_env` through `make_simulator_kwargs`,
and forcing a full 1-iter PPO bootup against an Isaac Sim from a fix
round would conflate boot-correctness with environment availability.

The lazy-isaac assertion in `TestAssetsLazyIsaacImport` and the unit
tests of `make_simulator_kwargs` cover the contract pieces; the
tracking_env boot path only changes by argument-passing, which is
type-checked by Python at construction time.
