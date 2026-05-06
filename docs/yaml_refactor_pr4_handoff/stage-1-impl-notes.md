## Files committed (PR 1, fc0bd93)

ブートストラップ前にユーザーが直接コミット (claude subagent 経由ではない)。
agent-coop skill 起動時点で fc0bd93 がすでに main 上に存在していたため、本 impl-notes はコミットメッセージから合成した。

### コードベース新規ファイル

- `src/vehicle_rl/config/__init__.py`: パッケージ宣言。
- `src/vehicle_rl/config/loader.py`: 以下を実装。
  - `load_yaml_strict(path)`: PyYAML SafeLoader + duplicate-key 拒否。
  - `resolve_refs(d, root, visited=None, cache=None)`: `<name>_ref` キーを `<name>: <referenced YAML>` に再帰展開。repo-root 相対のみ・推移参照 OK・cache あり・cycle 検出・repo escape 拒否・sibling key 重複拒否。
  - `deep_merge_overrides(base, overrides)`: deep-merge。unknown key / overrides 内 `<name>_ref` / 非 mapping target すべて `ValueError`。list は丸ごと差し替え。
  - `load_experiment(path)`: `load_yaml_strict` → `resolve_refs` → `deep_merge_overrides` のオーケストレータ。
  - `dump_resolved_config(resolved, log_dir)`: resolved bundle を log_dir に YAML dump。
- `src/vehicle_rl/config/schema.py`: 以下を実装。
  - `validate_keys(data, dataclass_cls)`: 再帰 missing/unknown-key check。nested dataclass-typed field を walk。
  - top-level dataclass shells: vehicle / dynamics / env / controller / agent / experiment 各カテゴリ。PR 2-4 で refine する前提。

### 設定 YAML (現行コード default を転記)

- `configs/vehicles/sedan.yaml`
- `configs/dynamics/linear_friction_circle_flat.yaml`
- `configs/envs/tracking.yaml`
- `configs/courses/{circle,s_curve,lemniscate,dlc,random_long,random_bank,random_path_generator}.yaml`
- `configs/controllers/{pure_pursuit,speed_pi}.yaml`
- `configs/agents/rsl_rl/ppo_tracking.yaml`
- `configs/runtime/{local_paths,play,train_video,classical_video}.yaml`
- `configs/experiments/rl/phase3_{circle_stage0a,random_long,random_bank,random_bank_play}.yaml`
- `configs/experiments/classical/{circle,s_curve,dlc}_baseline.yaml` + `circle_refactor_guard.yaml`

### Tests

- `tests/config/__init__.py`
- `tests/config/test_loader.py` (25 unittest cases)
  - duplicate / non-mapping / empty YAML rejection
  - transitive ref resolution, cache reuse, cycle detection, escape rejection, missing target rejection, sibling-overwrite rejection, absolute-path rejection
  - override unknown / nested-unknown / list-replace / ref-rejection / input-immutability
  - end-to-end `load_experiment` + `dump_resolved_config` roundtrip
  - `validate_keys` missing / unknown / nested recursion
  - smoke: configs/experiments/ 配下の全 YAML が clean に load される

### E2E guard machinery (本 refactor ではスキップだが PR 1 に同梱)

- `docs/refactor_e2e_guard.md`
- `scripts/dev/compare_classical_rollout.py`
- `.gitignore`: `.agent-work/` を追加

## Fast checks run

PR 1 マージ前にユーザー側で実行済み:

- `python -m unittest tests.config.test_loader -v` → 25 tests, all PASS
- runtime imports に regression なし

## Open issues (PR 2-4 へ持ち越し)

- `SEDAN_CFG` / `VehicleSimulator` / `waypoints.py` の数値 default は PR 2 で削除。
- `TrackingEnvCfg` / `rsl_rl_ppo_cfg.py` の数値 default は PR 3 で削除。
- script の `--config` 化と旧 CLI 撤去は PR 4。

STAGE_SHA=fc0bd93f832805154c045886a8ba6d8843abd7fd

## Stage-1 fixes (round 1)

Codex review of `fc0bd93` raised 2 findings (`stage-1-review-0.md`); both addressed
in a new commit on top of `fc0bd93`.

### Finding 1 — override subtree ref-key check (loader.py)

- Bug: `deep_merge_overrides` only rejected `*_ref` keys at the immediate level
  being merged. When a dict override wholesale-replaced a non-mapping target,
  the inside of that dict was never scanned, so e.g.
  `overrides: { run_name: { bad_ref: configs/x.yaml } }` slipped through.
- Fix: added `_assert_no_ref_in_subtree(value, path)` walking dicts and lists
  recursively. Called once at the top of `deep_merge_overrides` (the
  `not path` guard prevents re-walking on recursive merge calls). Dropped the
  per-key `*_ref` check inside the merge loop — the upfront subtree scan
  catches every case the per-key check did, and more.
- Files touched: `src/vehicle_rl/config/loader.py`.

### Finding 2 — per-shape schemas (schema.py)

- Bug: `ControllerSchema` only allowed `schema_version` + `type`, so it would
  reject the actual fields in `configs/controllers/speed_pi.yaml` and
  `configs/controllers/pure_pursuit.yaml`. `ExperimentSchema` required `env`
  and `agent` and had no `controllers`/`run`, so it could not validate
  classical experiment YAMLs.
- Fix: dropped both broken shells and replaced with discriminated per-shape
  schemas:
  - `SpeedPIControllerSchema`, `PurePursuitControllerSchema` (match
    `configs/controllers/*.yaml` field-for-field)
  - `RLExperimentSchema` (kind=`rl_train`, env+agent), `ClassicalExperimentSchema`
    (kind=`classical`, controllers+run). Both include `seed` and `run_name`
    (present in both committed YAMLs at PR 1, contrary to the reviewer note).
  - `select_controller_schema(bundle)` and `select_experiment_schema(bundle)`
    discriminator helpers. Each raises `ValueError` on missing or unknown
    discriminator value.
- `__all__` updated. No backwards-compat re-export for the dropped names; this
  is a same-PR-cycle fix and there are no callers outside `tests/config/`.
- Files touched: `src/vehicle_rl/config/schema.py`.

### Tests

Added to `tests/config/test_loader.py` (no existing test referenced the dropped
schemas, so no test had to be updated):

- `TestDeepMergeOverrides`: 3 new cases — `test_ref_in_replaced_subtree_rejected`
  (the exact scenario from finding 1), `test_ref_deep_inside_mapping_override_rejected`
  (3-level nesting), `test_ref_inside_list_override_rejected` (ref in a list
  element of a wholesale-replaced list).
- `TestPerShapeSchemas` (new): 12 cases covering positive validation against
  real controller YAMLs, positive validation of synthetic RL/classical
  experiment bundles, cross-feed negative cases (RL bundle vs classical
  schema and vice-versa, speed_pi YAML vs pure_pursuit schema), and the four
  discriminator helper paths (correct pick + missing/unknown discriminator).

### Final test count

`python -m unittest discover -s tests/config -v` → **41 tests, all PASS**
(previous: 25 + 16 new = 41).

## Stage-1 fixes (round 2)

Codex round-2 review of `c7f88647` raised 1 finding (`stage-1-review-1.md`);
addressed in a new commit on top of `c7f88647`.

### Finding 1 — `kind: rl_play` not registered in discriminator (schema.py)

- Bug: `_EXPERIMENT_SCHEMAS_BY_KIND` only mapped `rl_train` and `classical`,
  so `select_experiment_schema(load_experiment(...))` would raise
  `ValueError("unknown experiment kind: 'rl_play'")` for the committed
  `configs/experiments/rl/phase3_random_bank_play.yaml`. Tests missed it
  because the smoke loader test bypasses the discriminator and the synthetic
  per-shape tests only built `kind: rl_train` bundles.
- Fix: per the plan ("play YAML has the same category refs as train, with
  overrides for num_envs=1, random_reset_along_path=false"), the resolved
  top-level shape for `rl_play` matches `rl_train` exactly, so both
  discriminator values map to the existing `RLExperimentSchema`. Added the
  `"rl_play": RLExperimentSchema` entry; extended the `RLExperimentSchema`
  docstring to note both kinds map here.
- No new schema class. No change to loader / overrides handling.
- Files touched: `src/vehicle_rl/config/schema.py`.

### Tests

Added to `tests/config/test_loader.py::TestPerShapeSchemas`:

- New `_rl_play_bundle()` helper — same shape as `_rl_bundle()` with
  `kind: rl_play`.
- New `test_rl_play_experiment_schema_passes` — `validate_keys` accepts the
  rl_play bundle against `RLExperimentSchema`.
- Extended `test_select_experiment_schema_picks_by_kind` to also assert
  `select_experiment_schema(rl_play_bundle) is RLExperimentSchema`.

### Final test count

`python -m unittest discover -s tests/config -v` → **42 tests, all PASS**
(previous: 41 + 1 new = 42; the `_picks_by_kind` extension reuses an
existing test case, so the case count grows by exactly one).

