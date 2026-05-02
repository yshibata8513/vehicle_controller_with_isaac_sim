# Phase 3 env-style decision: DirectRLEnv

PLAN.md §3 specified "両方式（ManagerBasedRLEnv / DirectRLEnv）で『観測 1 つ・報酬 1 つだけの最小 env』を `prototypes/` に書いて 30 分以内に動かし、書き味と print デバッグの容易さで決定" as the env-style decision protocol. We are skipping that prototype-comparison step and going directly with **DirectRLEnv**.

## Decision

`src/vehicle_rl/envs/tracking_env.py` is implemented as a `DirectRLEnv` subclass.

## Rationale

The Manager pattern's value is composition: many small ActionTerm / ObservationTerm / RewardTerm pieces selected via cfg, decoupled from a hand-written env class. Our case-B physics architecture is fundamentally one monolithic step:

1. `VehicleAction` (pinion_target, a_x_target) →
2. `FirstOrderLagActuator` (steer + drive) →
3. `FixedRatioSteeringModel` →
4. `StaticNormalLoadModel` (Fz per wheel) →
5. `LinearFrictionCircleTire` (Fx, Fy with μ-clip) →
6. `injector.aggregate_tire_forces_to_base_link` →
7. `Articulation.set_external_force_and_torque(base_link)`

This whole chain is already encapsulated in `VehicleSimulator.step(action)` (Phase 2) as a **single function**. Splitting it into N ActionTerms gains nothing — the steps are not independently configurable, are tightly coupled by shared state (`a_y_estimate`, actuator outputs), and have to fire in a fixed order. ManagerBased would force an artificial decomposition of a function that is already correct.

Reward likewise: the 5 tracking-error terms (lateral², heading², speed_error², pinion_rate², jerk²) are computed from the same observation tensors in the same step. RewardTerms would require splitting that into 5 functions that each re-extract the same quantities.

DirectRLEnv lets `_apply_action()` call `VehicleSimulator.apply_action_to_physx(action)` directly (one line), and `_get_rewards()` is a single-pass arithmetic expression. Print-debugging is straightforward because every per-step quantity is a local variable. If we later need fine-grained term ablation, we can refactor a specific reward into a manager — but starting Direct keeps the surface area minimal.

## When to revisit

Switch to ManagerBased if any of the following happens:
- We start needing 3+ different reward configurations swapped via cfg files (e.g., circle reward vs DLC reward) and the if/else chain in `_get_rewards` becomes >50 lines.
- We add multiple action modalities (pinion-only / full / brake-only) that need clean cfg-level switching.
- We share env logic with another vehicle env and the duplication is non-trivial.

None of these are predicted for Phase 3, so we proceed with Direct.
