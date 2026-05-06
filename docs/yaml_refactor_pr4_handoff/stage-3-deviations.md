# Stage 3 (PR 3) deviations from plan

stage 3 はオリジナルの subagent が `stage-3-deviations.md` を作らなかったため、deviation 内容は `stage-3-impl-notes.md` に統合されている。本ファイルは引き継ぎ用に内容を抜粋。

## 1. Per-stage iteration cap (3) を意図的に超過した

- 経緯: round 0 (5 findings) → round 1 fix (5 件全部対応) → round 1 review (3 findings) → round 2 fix (3 件全部対応) → round 2 review (1 finding) → **round 3 fix (cap 超過、ユーザー明示承認)** → round 3 review **STAGE_APPROVED**。
- 理由: round 2 review が指摘した `_build_path()` の string-dispatch 問題は、本来 PR 4 のスコープだったが、ここで放置すると `--course s_curve|dlc|lemniscate` が env 起動で例外吐く regression が PR 3 マージ時点で残り、「main マージで動く」原則違反だったため。
- 対応: `_build_path()` を adapter の `build_path(course_bundle, num_envs, device)` 経由に書き換え、`TrackingEnvCfg.course_bundle: dict` 追加。これで本来 PR 4 でやる予定だった「string-dispatch 退役」を前倒しで完遂。
- 影響: PR 4 のスコープから「`_build_path` 書き換え」が消え、純粋に「CLI 撤去 + README + `random_path.yaml` 削除 + dump_resolved_config wiring + deprecated cfg 撤去」のみになる。

## 2. Deprecated だが PR 3 マージ時点で **意図的に残した** `TrackingEnvCfg` フィールド

stage 3 round 3 で `_build_path()` が adapter 経由になり、以下のフィールドは **runtime で読まれなくなった** が、`scripts/rl/{train_ppo,play}.py` の `_apply_cli_overrides` が依然として書き込むため、cfg からは消していない:

- `cfg.course` — runtime ロガー / メトリクス用に維持 (識別子として残す可能性あり)
- `cfg.radius`
- `cfg.target_speed`
- `cfg.course_ds`
- `cfg.random_path_cfg_path` — `random_long` / `random_bank` の旧読込パスとして残骸

**PR 4 でやること**:
1. 上記 5 フィールドを `TrackingEnvCfg` から削除
2. `make_tracking_env_cfg` の populate ロジックも削除
3. `_apply_cli_overrides` 自体が消える (CLI 撤去) ので参照点も無くなる
4. `cfg.course` だけは「ログ用識別子」として残したいなら `cfg.run_name` 経由 / 別 cfg field で別途保持するか、丸ごと消すか判断

## 3. Stage 3 round 1 で本来一発で消すべき `STEERING_RATIO` import 残存

- round 1 fix で `tracking_env.py` の `from vehicle_rl.assets import STEERING_RATIO` が消し忘れていた (codex round 1 review が F1b として指摘)
- round 2 fix で `cfg.steering_ratio` を populate して import を削除完了
- 教訓: stage 4 でも `vehicle_rl.assets` 由来の constant import が残っていないか grep する

## 4. `_random_path_cfg` キャッシュの書き換え (round 3)

- 旧: `tracking_env._build_path()` が `load_random_path_cfg(self.cfg.random_path_cfg_path)` で YAML を直接読んで `self._random_path_cfg` にキャッシュ → `_reset_idx` で `rp.speed.v_max` / `rp.reset.end_margin_extra_m` 参照
- 新 (round 3): `_make_random_path_cfg(self.cfg.course_bundle["generator"])` で resolved bundle から再構築 → 同じ shape (`RandomPathGeneratorCfg`) のオブジェクトを `self._random_path_cfg` にキャッシュ
- PR 4 で legacy `random_path.yaml` を削除しても、bundle 由来になっているので影響なし

## 5. Round 2 で追加した 3 つの dedicated 実験 YAML

stage 3 round 2 で legacy `--course s_curve|dlc|lemniscate` を正しい course YAML へルーティングするため、以下を新規追加:

- `configs/experiments/rl/phase3_s_curve.yaml`
- `configs/experiments/rl/phase3_dlc.yaml`
- `configs/experiments/rl/phase3_lemniscate.yaml`

`LEGACY_COURSE_TO_EXPERIMENT` (in `src/vehicle_rl/tasks/tracking/entry_points.py`) もこれらを指すよう更新。

PR 4 が `--course` CLI 自体を撤去するため、この legacy mapping table も削除されるが、**3 つの dedicated YAML は残す** (`--config configs/experiments/rl/phase3_s_curve.yaml --headless` で直接呼べる、PR 4 後の正式入口として有用)。

## 6. `make_default_tracking_ppo_cfg()` の名前

- `rsl_rl_ppo_cfg.py` の factory 関数名が `make_default_tracking_ppo_cfg()` だが、"default" は code default のニュアンスを残す。
- 中身は `make_ppo_runner_cfg(load_experiment(...))` の薄ラッパで、code default は持たない (load_experiment 経由)。
- PR 4 で名前を `make_ppo_runner_cfg_from_experiment()` 等に変えるかは判断保留。本タスクスコープ外として保持。
