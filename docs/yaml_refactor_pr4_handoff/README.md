# YAML refactor — PR 4 handoff documentation

このフォルダは、別環境で **PR 4 (stage 4)** を実装するために必要な仕様 + stage 1-3 の実装結果 + plan からの逸脱記録をまとめた snapshot。

## 起点

- branch: `agent-coop/task-20260504-221723` (origin に push 済み)
- HEAD: `27f60285` (stage 3 = PR 3 が codex review で `STAGE_APPROVED`)
- baseline: `af868a3` (PR 1 親 commit)
- main: `fc0bd93` (PR 1 だけ含む状態)

## ファイル構成

| ファイル | 役割 |
|---|---|
| [`plan.md`](plan.md) | 本タスク全体の仕様。`docs/yaml_config_structure_plan.md` の committed 版 + working tree で追記された「確定した設計判断」セクション + agent-coop skill が要求する `## Stages` 構造化 (各 PR の Scope/Files/Fast-test/Exit-condition) を統合した最も詳しい版 |
| [`stage-1-impl-notes.md`](stage-1-impl-notes.md) | **PR 1** (loader + schema + 雛形 + 25 unit test) の実装結果。codex review round 0/1/2 で出た 4 件 findings の対応 (override subtree ref 検査、per-shape schema、`rl_play` discriminator 登録) |
| [`stage-2-impl-notes.md`](stage-2-impl-notes.md) | **PR 2** (vehicle / dynamics / course を YAML 化) の実装結果。codex review round 0/1 で出た 6 件 findings の対応 (TrackingEnv boot、YAML フィールド配線、`a_front_m`、lazy Isaac import、schema 厳格化、`actuator_lag.initial_value` reset) |
| [`stage-2-deviations.md`](stage-2-deviations.md) | PR 2 の plan 逸脱記録 (`waypoints.py` の正 path = `planner/waypoints.py`、`run_classical.py` の `--duration` flag 名等) |
| [`stage-3-impl-notes.md`](stage-3-impl-notes.md) | **PR 3** (env / reward / PPO を YAML 化) の実装結果。codex review round 0/1/2/3 で出た 9 件 findings の対応 (TrackingEnv の dynamics literal 撤去、observation 整合、validated-but-ignored YAML leaves、CLI precedence、PhysX gravity / steering_ratio 配線、`_build_path` adapter ルーティング) |
| [`stage-3-deviations.md`](stage-3-deviations.md) | PR 3 の plan 逸脱記録 (per-stage cap 超過、deprecated cfg 5 フィールド存続、3 つの dedicated experiment YAML 追加、`_random_path_cfg` キャッシュ書き換え) |

## PR 4 でやること (plan + stage 3 deviations から導出)

### plan `### PR 4` 由来

1. `scripts/rl/train_ppo.py`, `scripts/rl/play.py`, `scripts/sim/run_classical.py` を `--config` 単一入口に変更
2. 旧 CLI 即時撤去 (`--course`, `--num_envs`, `--max_iterations`, `--seed`, `--experiment_name`, `--run_name`, `--random_path_cfg`, `--mu`, `--target_speed`, `--radius`, `--pid_kp`)
3. `dump_resolved_config(resolved, log_dir)` を train / classical 起動時に呼ぶ
4. `README.md` の旧 CLI 例を全 `--config` 例に書き換え
5. `configs/random_path.yaml` 旧ファイルを削除

### stage 3 deviations 由来 (PR 4 で完遂すべき積み残し)

6. `TrackingEnvCfg` から deprecated 5 フィールド削除: `cfg.course`, `cfg.radius`, `cfg.target_speed`, `cfg.course_ds`, `cfg.random_path_cfg_path`
7. `make_tracking_env_cfg` の上記フィールドへの populate ロジック削除
8. `LEGACY_COURSE_TO_EXPERIMENT` mapping table (`src/vehicle_rl/tasks/tracking/entry_points.py`) 削除
9. `tasks/tracking/entry_points.py` 内の `VEHICLE_RL_EXPERIMENT_YAML` env-var dispatch を `--config` 直接渡しに書き換え (CLI 撤去で env-var 経由が不要になる)
10. `rsl_rl_ppo_cfg.py` の `make_default_tracking_ppo_cfg()` 関数名見直し (任意、判断保留)

### 最終 audit

11. `rg "default=|= [0-9]+\.|: float =|: int =" src/vehicle_rl/` で残骸ゼロを確認
12. `phase3_circle_stage0a` / `phase3_random_long` / `phase3_random_bank` の各 experiment YAML で短い PPO 学習が起動することを目視確認 (E2E guard CSV/JSON 比較は使わない)

## 別環境での再現手順 (例)

```bash
git clone <repo> && cd <repo>
git fetch origin
git switch agent-coop/task-20260504-221723
# このフォルダ docs/yaml_refactor_pr4_handoff/ が見える
# stage 4 を実装 → commit → push → 同 PR にぶら下げる
```

## 注意

- **作業ツリーの uncommitted 変更**: 元の環境では PR 4 の WIP として 11 file 変更が working tree に残っていた (本 push には含まれていない)。それを引き継ぎたい場合は別途 stash / patch 経由で運搬。
- agent-coop skill 自体を別環境で使いたい場合: `.claude/skills/agent-coop/` (gitignore 済み) と `.claude/settings.json` も別途運搬必要。
