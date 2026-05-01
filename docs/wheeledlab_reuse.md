# WheeledLab 流用方針メモ（Phase 0 結論）

**前提:** 本プロジェクト (vehicle_rl) は Isaac Sim 5.1 / Isaac Lab 2.3 / WheeledLab 0.1.0 を使用。WheeledLab は BSD-3-Clause、`C:\work\isaac\WheeledLab` に展開済み、pip からも import 可能 (`wheeledlab`, `wheeledlab_assets`, `wheeledlab_rl`, `wheeledlab_tasks`)。

**ライセンス遵守:** copy 改変する場合は元ファイル先頭のコピーライト保持 + `# Adapted from WheeledLab (BSD-3-Clause): https://github.com/UWRobotLearning/WheeledLab` の注記を付ける。

## 大方針の更新（Phase 0 で判明）

- WheeledLab の **train_rl.py / play_policy.py / startup.py / hydra.py** などのスクリプト系は、Isaac Lab 公式の `scripts/reinforcement_learning/rsl_rl/train.py` `play.py` がほぼ同等のことをしており、後者の方が簡潔で 2.3 互換性も保たれている。**本プロジェクトのエントリポイントは Isaac Lab 公式版をベースにし、WheeledLab の同等物は参考のみ**。
- WheeledLab の本当に「ありがたい」資産は `wheeledlab.envs.mdp.actions` の **Ackermann action 実装**。これだけは積極的に import する。
- 動画記録は **`gym.wrappers.RecordVideo`** （Isaac Lab 公式 train.py / WheeledLab play_policy.py が共に使用）で十分。`CustomRecordVideo` は libx264/CRF/wandb hook 付きの上位版だが、Phase 0–3 では不要。

## ファイル別判定

| WheeledLab パス | 結論 | 理由 |
|---|---|---|
| `wheeledlab.envs.mdp.actions.ackermann_actions:AckermannAction` | **現状未使用（参考）** | 正規の Ackermann 幾何 (`atan(L/(R±W/2))`)。Phase 1.5 step 1 では parallel steering（左右前輪に同一 δ）で十分なため未導入。`tire_force.py:per_wheel_steer()` の差し替え 4 行で Ackermann 化可能。Phase 2 で旋回半径が小さい操作を走らせ内外輪のスクラブが顕在化したら import を再評価 |
| `wheeledlab.envs.mdp.actions.actions_cfg:AckermannActionCfg` | **現状未使用（参考）** | 上記の cfg 側。同タイミングで再評価 |
| `wheeledlab.envs.mdp.actions.rc_car_actions:RCCar*Action` | **参考のみ** | tan-steering 近似で正確な Ackermann ではない。RC カー用簡略実装 |
| `wheeledlab_tasks/common/observations.py` | **参考のみ → 必要部分を copy** | ObsTerm の組み立てパターンは参考になるが、本プロジェクトは IMU/GPS + lookahead waypoints と obs が大きく異なる |
| `wheeledlab_tasks/common/actions.py` | **参考のみ** | 各車両の joint 名と寸法を `RCCar4WDActionCfg` に渡しているだけ。本プロジェクトでは `AckermannActionCfg` (基底) を直接使う |
| `wheeledlab_assets/f1tenth.py` | **参考のみ** | `ArticulationCfg` の書き方の参考。乗用車スケールでは数値が桁違いに変わるため、構造のみ流用 |
| `wheeledlab_rl/utils/custom_video_recorder.py` | **参考のみ（Phase 0-3 不要）** | libx264 + CRF 制御 + wandb hook。`gym.wrappers.RecordVideo` で十分。Phase 4 で動画品質をチューニングしたくなったら参照 |
| `wheeledlab_rl/utils/hydra.py` | **不要** | `isaaclab_tasks.utils.hydra.hydra_task_config` を Isaac Lab 公式 train.py が直接使っており、それで十分 |
| `wheeledlab_rl/utils/modified_rsl_rl_runner.py` | **不要** | rsl_rl 3.1.2 の公式 `OnPolicyRunner` をそのまま使う Isaac Lab 公式 train.py で動作確認済み (Phase 0 Cartpole)。WheeledLab の改造版は不要 |
| `wheeledlab_rl/utils/clip_action.py` | **未読・必要時に確認** | 観測でアクション履歴を返す際に使うかも。実装が必要になったら読む |
| `wheeledlab_rl/configs/{common,rl}_cfg.py` | **構造を参考** | 学習 cfg のテンプレート。Isaac Lab の `RslRlBaseRunnerCfg` を直接使うので機能としては不要 |
| `wheeledlab_rl/scripts/train_rl.py` | **不要** | Isaac Lab 公式 `scripts/reinforcement_learning/rsl_rl/train.py` をベースにする (動作確認済み) |
| `wheeledlab_rl/scripts/play_policy.py` | **構造を参考** | playback + video 記録の流れの参考に。Isaac Lab 公式 `play.py` も同様 |
| `wheeledlab_rl/startup.py` | **不要** | Isaac Lab 公式の `AppLauncher.add_app_launcher_args` を直接呼ぶ標準パターンで足りる |

## 結論

**Phase 1.5 step 1 完了時点で、実装は WheeledLab のコードを 1 行も import せずに動作している。** `AckermannAction` は将来 import 候補として残してあるが現状は parallel steering で代替。その他は構造・設計の参考。

これにより本プロジェクトは:
- 依存ライブラリは IsaacLab + IsaacSim + WheeledLab (Ackermann のみ) + rsl_rl + 標準 gym/torch
- エントリポイント (`scripts/rl/train_ppo.py`, `scripts/rl/play.py`) は Isaac Lab 公式 `train.py` `play.py` を改変して作る
- 動画記録は `gym.wrappers.RecordVideo` で統一

## Phase 1 以降の TODO

- Phase 1 で sedan の `AckermannActionCfg` 設定例を `src/vehicle_rl/envs/mdp/actions.py` に書く
- Phase 3 で Isaac Lab 公式 `train.py` をコピペし、import 部分に `import vehicle_rl.tasks  # noqa` を追加するのみ
