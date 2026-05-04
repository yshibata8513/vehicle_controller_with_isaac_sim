# vehicle_rl

Isaac Sim / Isaac Lab で**実車相当の乗用車モデル**を構築し、コントローラを強化学習するプロジェクト。タスクは路面摩擦 (μ) を主な外乱とする経路追従。

タイヤ力は **PhysX の円柱接触に依存させない**: PhysX は車体の 6DoF 剛体運動を積分するためだけに使い、タイヤ力 (Fx / Fy / Fz) は自前で計算して `set_external_force_and_torque` 経由で base_link に注入する（**案 B**）。詳細は [`docs/PLAN.md`](./docs/PLAN.md)。

## 進捗

| Phase | 内容 | 状態 |
|---|---|---|
| 0 | 環境動作確認 + WheeledLab コード読解 | 完了 (2026-04-30) |
| 1 | 実車スケール sedan アセット (visual wheel + 剛体シャシ) | 完了 (2026-04-30) |
| 1.5 step 1 | 自前タイヤ力注入: 線形 + 摩擦円 + 静的荷重 | 完了 (2026-04-30) |
| 1.5 step 2 | Fiala / slip ratio | 未着手 |
| 2 | Pure Pursuit + PID ベースライン | 完了 (2026-05-01) |
| 3 step 1 | DirectRLEnv + Stage 0a (steering-only PPO, circle) | 完了 (2026-05-02) |
| 3 random_long | clothoid+arc+straight ランダム経路 (fraction-based 生成) で PPO 追従 | 完了 (2026-05-04) |
| 3 random_bank | P=1024 path bank, reset 毎に env ごと独立サンプル | 進行中 |

## セットアップ

既存の Isaac Lab venv を使用（vehicle_rl 専用 venv は作らない）:

```
venv:        C:\work\isaac\env_isaaclab\Scripts\python.exe
Isaac Sim:   5.1.0
Isaac Lab:   2.3.0  (C:\work\isaac\IsaacLab)
WheeledLab:  0.1.0  (C:\work\isaac\WheeledLab)
PyTorch:     2.7.0+cu128
GPU:         RTX 5090 / Driver 591.86 / CUDA 13.1
```

`vehicle_rl` パッケージを editable install:

```bash
PY="/c/work/isaac/env_isaaclab/Scripts/python.exe"
$PY -m pip install -e .
```

## クイックスタート

リポジトリルート (`C:\Users\user\vehicle_rl`) から実行。`PY` は venv の python:

```bash
PY="/c/work/isaac/env_isaaclab/Scripts/python.exe"
```

### Phase 0 — 環境健全性確認

Isaac Lab Cartpole を 5 iter 学習して MP4 を出力（cold start ~3 分、2 回目以降 ~1 分）:

```bash
$PY "/c/work/isaac/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py" \
  --task Isaac-Cartpole-Direct-v0 \
  --headless --video --video_length 100 --video_interval 100 \
  --num_envs 16 --max_iterations 5
```

WheeledLab `Isaac-F1TenthDriftRL-v0` をランダム行動で 60 ステップ走らせ、Isaac Lab 2.3 / Sim 5.1 互換を確認:

```bash
$PY -u scripts/sim/phase0_wheeledlab_check.py --headless --num_envs 2 --steps 60
```

### Phase 1 — sedan アセット動作確認

URDF → USD 変換（`assets/usd/sedan.usd` を生成）:

```bash
$PY scripts/sim/convert_sedan_urdf.py --headless
```

重力 OFF で spawn → 1 秒姿勢確認:

```bash
$PY scripts/sim/spawn_sedan.py --duration 1.0 --headless
cat metrics/phase1_spawn_nograv.json   # pass.overall == true なら PASS
```

オプション: `--gravity` (重力 ON で診断), `--steer_target <rad>` (操舵テスト), `--no_video`。

### Phase 1.5 step 1 — 案 B タイヤ力注入の物理サニティ

3 シナリオ (straight / brake / circle) で動力学検証:

```bash
# 直進: Σ Fz = mg, z ドリフト, roll/pitch 確認
$PY scripts/sim/run_phase1_5.py --scenario straight --mu 0.9 --duration 8 --headless

# 制動: 制動距離 ≈ v² / (2μg) を ±20% 以内で再現
$PY scripts/sim/run_phase1_5.py --scenario brake --mu 0.9 --target_speed 15 --duration 12 --headless
$PY scripts/sim/run_phase1_5.py --scenario brake --mu 0.6 --target_speed 15 --duration 14 --headless
$PY scripts/sim/run_phase1_5.py --scenario brake --mu 0.3 --target_speed 12 --duration 18 --headless

# 円旋回: 旋回半径 ≈ L/tan(δ), 限界速度 √(μgR) と比較
$PY scripts/sim/run_phase1_5.py --scenario circle --mu 0.9 --target_speed 8 --steer 0.1 --duration 15 --headless
```

出力先:
- `videos/phase1_5_<scenario>_mu<μ>.mp4`
- `metrics/phase1_5_<scenario>_mu<μ>.{csv,json}`

#### 達成済みの数値結果（2026-04-30）

| シナリオ | パラメタ | 結果 | 判定 |
|---|---|---|---|
| straight | μ=0.9, 8s | max_roll 0.01°, max_pitch 1.29°, z_drift 0.85 mm, Σ Fz 誤差 **0.31%** | PASS |
| brake | μ=0.9, v₀=13.92 m/s | actual 12.88 m, theory 10.98 m, **err +17.4%** | PASS (±20%) |
| brake | μ=0.6, v₀=13.83 m/s | actual 15.35 m, theory 16.24 m, **err −5.5%** | PASS |
| brake | μ=0.3, v₀=8.60 m/s | actual 10.34 m, theory 12.57 m, **err −17.8%** | PASS |
| circle | μ=0.9, v=7.97 m/s, δ=0.1 rad | R=27.09 m vs kinematic 26.92 m (誤差 0.6%) | PASS |

詳細は [`docs/PLAN.md` §1.5](./docs/PLAN.md) 参照。

### Phase 2 — 古典制御 (Pure Pursuit + PID) ベースライン

`circle` / `s_curve` / `dlc` / `lemniscate` 各コースを Pure Pursuit (横) + PID (速度) で追従:

```bash
$PY scripts/sim/run_classical.py --course circle     --mu 0.9 --target_speed 10 --duration 25 --headless
$PY scripts/sim/run_classical.py --course s_curve    --mu 0.9 --target_speed 10 --duration 25 --headless
$PY scripts/sim/run_classical.py --course dlc        --mu 0.9 --target_speed 10 --duration 20 --headless
$PY scripts/sim/run_classical.py --course lemniscate --mu 0.9 --target_speed 10 --duration 35 --headless
```

出力:
- `videos/classical_<course>_mu<μ>.mp4`
- `metrics/classical_<course>_mu<μ>.{csv,json}`

### Phase 3 — 強化学習 (PPO) 訓練 + 可視化

#### Stage 0a: circle で steering-only PPO (sanity)

```bash
# 訓練 (~2 分 / 50 iter, RTX 5090): cfg.target_speed の固定速 circle を steering だけで追従。
$PY scripts/rl/train_ppo.py --task Vehicle-Tracking-Direct-v0 \
    --course circle --num_envs 256 --max_iterations 50 --headless \
    --experiment_name phase3_circle --run_name stage0a_smoke
```

#### Phase 3 random_long: clothoid+arc+straight ランダム経路

`configs/random_path.yaml` の `turn_heading_change_rad` / `clothoid_heading_fraction` で turn 形状を制御。`speed.v_min/v_max/ay_limit` から segment ごとに目標速を引く。

```bash
# 訓練 (~10 分 / 300 iter, 256 envs, RTX 5090; ~32GB VRAM)
$PY scripts/rl/train_ppo.py --task Vehicle-Tracking-Direct-v0 \
    --course random_long --random_path_cfg configs/random_path.yaml \
    --num_envs 256 --max_iterations 300 --headless \
    --experiment_name phase3_random_long --run_name fraction_schema_300iter
```

#### Phase 3 random_bank: P 本の path bank からリセット毎にサンプル

`phase2_bank.num_paths` 本の独立な random path を起動時に生成（vectorized cumsum で 1024 paths × 1 km は ~0.4 s）。各 env はリセットの度に bank から path を一様サンプル。`random_long` の単一 path より経路多様性が大きい一方、bank は固定なので Phase 3 (再生成) は別途。

```bash
# 訓練 (~10 分 / 300 iter, 256 envs, RTX 5090)。bank 1024 paths × 1 km × 5 fields ≈ 100 MB on GPU。
$PY scripts/rl/train_ppo.py --task Vehicle-Tracking-Direct-v0 \
    --course random_bank --random_path_cfg configs/random_path.yaml \
    --num_envs 256 --max_iterations 300 --headless \
    --experiment_name phase3_random_bank --run_name fixed_bank_300iter
```

訓練ログは `logs/rsl_rl/<experiment>/<timestamp>_<run_name>/` に出力。TensorBoard で曲線確認:

```bash
$PY -m tensorboard.main --logdir logs/rsl_rl --port 6006
# → http://127.0.0.1:6006/
```

#### 結果可視化 (`play.py`)

学習済み policy を deterministic に走らせ、軌跡 PNG + per-step CSV をダンプ:

```bash
# 最新 run の最新 model_*.pt を自動選択
$PY scripts/rl/play.py --task Vehicle-Tracking-Direct-v0 \
    --course random_long --random_path_cfg configs/random_path.yaml \
    --experiment_name phase3_random_long \
    --num_envs 1 --duration 25 --headless

# 特定 ckpt を指定する場合
$PY scripts/rl/play.py --task Vehicle-Tracking-Direct-v0 \
    --course random_long --random_path_cfg configs/random_path.yaml \
    --experiment_name phase3_random_long \
    --load_run 2026-05-04_XX-XX-XX_fraction_schema_300iter \
    --checkpoint model_299.pt \
    --num_envs 1 --duration 25 --headless
```

出力:
- `metrics/play_<course>_<run>_<ckpt>.csv` — t, x, y, yaw_deg, vx, lat_err, hdg_err_deg, action_pinion
- `videos/play_<course>_<run>_<ckpt>.png` — 軌跡 (full + zoom) / vx / lat_err 4 パネル

末尾に `[RESULT] rms_lateral_err`, `max_lateral_err`, `mean_vx` をコンソールに表示。

## 設計

### 案 B（PhysX 接触に依存しない）

- 車体: PhysX 6DoF 剛体（`base_link`）。Isaac Sim / PhysX が運動方程式を積分
- 車輪: visual のみ。**collision 無効**（URDF→USD 変換時に `collision_from_visuals=False`）
- タイヤ力: 自前モデルで Fx / Fy / Fz / μ を計算 → body 座標系に集約して `set_external_force_and_torque` で base_link に注入
- 操舵: Python 側で 1 次遅れ積分 → `set_joint_position_target` で revolute joint に書く（ビジュアル + 観測用、力の効果は注入側に焼き込み済み）

### `src/vehicle_rl/dynamics/` — 差し替え可能インターフェース

| モジュール | 役割 | 差し替え対象 |
|---|---|---|
| `state.py` | `VehicleState` dataclass、quat ↔ rotmat / rpy | — |
| `actuator.py` | `FirstOrderLagActuator`（操舵 / 駆動 / 制動） | — |
| `normal_load.py` | `NormalLoadModel` Protocol + `StaticNormalLoadModel` | Phase 4 で `RaycastNormalLoadModel` に差し替え |
| `tire_force.py` | `TireForceModel` Protocol + `LinearFrictionCircleTire` | step 2 で `FialaTire`、Phase 4 で Pacejka |
| `attitude_damper.py` | 仮想 roll/pitch ダンパ（ヨーは未ダンプ） | — |
| `injector.py` | 4 タイヤ力 → base_link 1 点 force/torque（body frame） | — |

ハイパーパラメタ採用値: τ_steer=50ms, τ_drive=200ms, τ_brake=70ms, Cα=60,000 N/rad/輪, z-drift PD (50,000 N/m, 5,000 Ns/m), 姿勢ダンパ (80,000 Nm/rad, 8,000 Nms/rad)。

## ディレクトリ構成

```
vehicle_rl/
├── README.md                     # このファイル
├── pyproject.toml
├── .gitignore
├── docs/
│   ├── PLAN.md                   # マスタープラン
│   ├── wheeledlab_reuse.md       # WheeledLab 流用方針
│   ├── phase3_random_path_plan.md          # Phase 3 ランダム経路設計
│   └── phase3_random_path_phase1_review.md # 同レビュー
├── configs/
│   └── random_path.yaml          # Phase 3 random_long / random_bank 生成パラメタ
├── assets/
│   └── urdf/sedan.urdf           # 実車スケール sedan URDF（USD は変換時生成）
├── src/vehicle_rl/
│   ├── assets/sedan.py           # SEDAN_CFG (ArticulationCfg)
│   ├── dynamics/                 # 案 B のタイヤ力・荷重・アクチュエータ
│   ├── envs/                     # DirectRLEnv (TrackingEnv)
│   ├── controller/               # Pure Pursuit + PID
│   ├── planner/                  # 経路 (circle/s_curve/dlc/lemniscate/random)
│   ├── tasks/                    # gym.register
│   └── utils/
├── scripts/sim/
│   ├── convert_sedan_urdf.py     # Phase 1: URDF → USD
│   ├── spawn_sedan.py            # Phase 1: spawn 確認
│   ├── run_phase1_5.py           # Phase 1.5: 動力学サニティ
│   ├── run_classical.py          # Phase 2: PP+PID ベースライン
│   └── phase0_wheeledlab_check.py
└── scripts/rl/
    ├── train_ppo.py              # Phase 3: PPO 訓練
    └── play.py                   # Phase 3: 学習済 policy の評価 / 可視化
```

実行結果（`videos/`, `metrics/`, `logs/`, `outputs/`）と生成物（`assets/usd/`）、旧 case-A スナップショット（`legacy/`）はリポジトリ管理対象外（`.gitignore`）。

## メモ

- WheeledLab 本来の `train_rl.py` は IsaacLab 2.3 / Sim 5.1 で互換問題があり、ビデオ録画が一時無効化されている (`docs/wheeledlab_reuse.md` 参照)。本プロジェクトは Isaac Lab 公式 `train.py` をベースにすることでこの問題を回避。
- Python の標準出力がバッファされて Isaac Sim 終了時に消えることがある。動作確認スクリプトでは `python -u` または `print(..., flush=True)` を使う。
- 旧 Phase 1a (case-A: PhysX 接触摩擦) の実装は `legacy/phase1a/` に self-contained に退避。回帰確認用、本体からは参照しない。
