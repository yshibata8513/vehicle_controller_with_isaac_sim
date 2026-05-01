# Isaac Sim / Isaac Lab を用いた車両運動制御 強化学習プロジェクト計画

## 0. ゴール

Isaac Sim + Isaac Lab 上で、**実車相当（乗用車スケール）の車両モデル**を構築し、その**コントローラを強化学習**する。タスクは **路面摩擦 (μ) 変化を主な外乱とする経路追従**。

- **プランナの出力（waypoint 列）は観測として与える。** RL の対象はコントローラのみ。
- 実行は **CLI 完結**、ヘッドレスで **MP4 動画出力**を主、GUI ポップアップ再生を従とする。
- **タイヤ力は PhysX の円柱接触に依存させない。** PhysX シャシ剛体 + 自前タイヤ力注入（**案 B**）で進める:
  - 車体: PhysX 6DoF 剛体（`base_link`）
  - 車輪: visual のみ。collision は無効
  - タイヤ・路面: 自前モデルで `Fx, Fy, Fz, μ` を計算
  - 力の反映: 各タイヤ位置の力を `base_link` 重心まわりに合成し、external force/torque として注入
  - 利点: μ をスカラ操作にできる / 円柱⇔平面接触の暗黙挙動から解放 / 並列環境での step が軽い
- 忠実度の段階:
  - Phase 1〜3.5: **案 B**（線形 + 摩擦円 → Fiala へ拡張）
  - Phase 4: PhysX Vehicle SDK + Pacejka との比較（後回し）

## 1. 環境（インストール済み）

| 項目 | バージョン | 場所 |
|---|---|---|
| GPU / Driver | RTX 5090 / 591.86 / CUDA 13.1 | — |
| Isaac Sim | 5.1.0（pip 版） | venv |
| Isaac Lab | 2.3.0 | `C:\work\isaac\IsaacLab` |
| WheeledLab | 0.1.0 | `C:\work\isaac\WheeledLab` |
| PyTorch | 2.7.0 + cu128 | venv |
| RL ライブラリ | rsl_rl 3.1.2, skrl 1.4.3, SB3 2.7.1 | venv |
| Python venv | `C:\work\isaac\env_isaaclab\Scripts\python.exe` | — |

**互換性の注意:** WheeledLab の README は IsaacLab 2.0.2 / IsaacSim 4.5 を想定。本環境は 2.3 / 5.1 のため `C:\work\isaac\WheeledLab\docs\isaaclab_compatibility_notes.md` に記載の調整が必要。

**既存の車両アセット**
- WheeledLab: F1Tenth / MuSHR / Hound（いずれも 1/10 ラジコンスケール — 「実車相当」には小さすぎる）
- Isaac Lab 標準: 車両アセット無し（humanoid / arm / quadruped / cartpole のみ）
- → 乗用車サイズの URDF を新規作成する必要がある。

## 1.5. 開発方針（案 C: ハイブリッド）

**車両アセット・タスク env・タイヤ力モデル・報酬は自前。WheeledLab の足場コードは積極的に流用**する方針。

理由:
- 1/10 ラジコンスケール (WheeledLab 既存) と実車スケール (本プロジェクト) は物理パラメタが桁違い → アセットと env は最初から書き直すのが綺麗。
- 案 B で進めるためタイヤ力モデルは WheeledLab に対応物がなく自前必須。
- 一方、Ackermann マッピング（操舵ジョイントへの位置コマンド作法）・Hydra 統合・PPO ランナー・ビデオ録画ユーティリティ等は WheeledLab のコードがそのまま使える。

**ライセンス:** WheeledLab は **BSD-3-Clause**（`C:\work\isaac\WheeledLab\LICENSE`）。流用する各ファイルの先頭にコピーライト保持と "based on / derived from WheeledLab" の注記を入れて取り込む。

### 流用候補ファイル

各ファイルの扱い（**import / copy / 参考のみ** のいずれか）は Phase 0 のコード読解＆互換性確認後に確定し、`docs/wheeledlab_reuse.md` に記録する。下表は **Phase 0 着手前の当たり**。**Phase 0 完了時の見直しで多くが「参考のみ」または「不要」に格下げされた**（最終判定は `docs/wheeledlab_reuse.md` を参照）。実装上は Phase 1.5 step 1 まで WheeledLab のコードを 1 行も import せずに完了している。

| WheeledLab パス | 当たりとしての扱い | 本プロジェクトでの対応先 |
|---|---|---|
| `wheeledlab_tasks/common/actions.py` | 参考 or copy 改変 | `src/vehicle_rl/envs/mdp/actions.py`（Ackermann 操舵マッピング） |
| `wheeledlab_tasks/common/observations.py` | 参考 or copy 改変 | `src/vehicle_rl/envs/mdp/observations.py` |
| `wheeledlab/envs/mdp/` | 構造を参考 | `src/vehicle_rl/envs/mdp/`（rewards / events / terminations） |
| `wheeledlab_rl/utils/custom_video_recorder.py` | import 第一候補（NG なら copy） | `src/vehicle_rl/utils/video_recorder.py` |
| `wheeledlab_rl/utils/hydra.py` | import 第一候補（NG なら copy） | `src/vehicle_rl/utils/hydra.py` |
| `wheeledlab_rl/utils/modified_rsl_rl_runner.py` | import 第一候補（NG なら copy） | `src/vehicle_rl/utils/rsl_rl_runner.py` |
| `wheeledlab_rl/utils/clip_action.py` | import 第一候補（NG なら copy） | `src/vehicle_rl/utils/clip_action.py` |
| `wheeledlab_rl/configs/common_cfg.py`, `rl_cfg.py` | テンプレートとして参考 | `configs/` 以下の Hydra YAML/Python の雛形 |
| `wheeledlab_rl/scripts/train_rl.py` | エントリポイント構造を参考 | `scripts/rl/train_ppo.py` |
| `wheeledlab_rl/scripts/play_policy.py` | 同上 | `scripts/rl/play.py` |
| `wheeledlab_assets/f1tenth.py` | `ArticulationCfg` 書き方の参考 | `src/vehicle_rl/assets/sedan.py` |

判断基準:
- **import**: WheeledLab 本体を依存に残しても良く、その関数/クラスが IsaacLab 2.3 / IsaacSim 5.1 で問題なく動く場合
- **copy 改変**: 動くが本プロジェクト固有の改変が必要、または WheeledLab 依存を減らしたい場合（コピー元のコピーライトと "based on / derived from WheeledLab" 注記を保持）
- **参考のみ**: 設計・API パターンを真似るが、新規実装する場合

## 2. Docker か venv か

**Phase 0〜3 は venv 継続**を推奨。

- NVIDIA 公式の Docker イメージ（`nvcr.io/nvidia/isaac-lab` 等）は存在するが、Windows では Docker Desktop の WSL2 バックエンド経由でしか動作せず、実質 WSL2 上で動くことになる → 「Windows ネイティブ希望」方針と矛盾。
- Windows + Docker での GUI ポップアップ再生は X サーバ経由で実用的でない。
- 既存の venv に Sim 5.1 + Lab 2.3 + WheeledLab + cu128 が完動しており、再構築する利点が現状少ない。
- **Docker 化を再検討するタイミング**: 共有・リモート実行・CI 化が必要になった時点。

## 3. ディレクトリ構成

```
C:\Users\user\vehicle_rl\
├── PLAN.md                       # 本ファイル
├── README.md                     # クイックスタート用コマンド
├── pyproject.toml                # vehicle_rl パッケージの editable install
├── assets/
│   ├── urdf/sedan.urdf           # 自作の乗用車 URDF（visual wheel only）
│   └── usd/sedan.usd             # URDF→USD 変換で自動生成
├── src/vehicle_rl/
│   ├── assets/sedan.py           # ArticulationCfg
│   ├── dynamics/                 # 案 B のタイヤ力・荷重・アクチュエータモデル
│   │   ├── state.py              # VehicleState dataclass
│   │   ├── normal_load.py        # NormalLoadModel (Static / Raycast Protocol)
│   │   ├── tire_force.py         # TireForceModel (LinearFrictionCircle / Fiala)
│   │   ├── actuator.py           # FirstOrderLagActuator (操舵 / 駆動 / 制動)
│   │   ├── injector.py           # base_link への合力・合トルク注入
│   │   └── disturbance.py        # 経路座標系の外乱マップ (Phase 3.5)
│   ├── planner/waypoints.py      # 円 / リサジュー / S 字 / ISO-DLC
│   ├── controller/
│   │   ├── pure_pursuit.py
│   │   └── pid.py
│   ├── envs/
│   │   ├── tracking_env_cfg.py   # ManagerBasedRLEnv
│   │   └── mdp/                  # observations / rewards / events / terminations
│   └── tasks/__init__.py         # gym.register による ID 登録
├── scripts/
│   ├── sim/spawn_sedan.py        # Phase 1 の spawn 確認
│   ├── sim/run_classical.py      # Phase 2 の駆動スクリプト（PP + PID）
│   ├── rl/train_ppo.py           # Phase 3 学習
│   └── rl/play.py                # Phase 3 再生 + MP4 出力
├── configs/                      # Hydra/YAML 上書き設定
├── videos/                       # MP4 出力（gitignore）
└── logs/                         # tensorboard / wandb（gitignore）
```

## 4. フェーズ別計画

### Phase 0 — 環境動作確認 + WheeledLab コード読解（完了 2026-04-30）

- [x] `C:\Users\user\vehicle_rl\` の雛形を作成。
- [x] Isaac Lab `train.py` で Cartpole-Direct 5 iter 学習 → MP4 出力 (`videos/phase0_cartpole_sanity.mp4`, 374KB)。RTX 5090 / Vulkan / cu128 健全。
- [x] WheeledLab `Isaac-F1TenthDriftRL-v0` をランダム行動で 60 ステップ動作 → IsaacLab 2.3 / Sim 5.1 でクラッシュなし。
- [x] WheeledLab コード読解 → `docs/wheeledlab_reuse.md`。**結論: 実質 import するのは `wheeledlab.envs.mdp.actions` の Ackermann module 1 つだけ。** その他は構造の参考のみ、エントリポイントは Isaac Lab 公式 `train.py` / `play.py` をベースにする。
- [x] CustomRecordVideo 問題は標準 `gym.wrappers.RecordVideo` を使うことで回避（Phase 0–3 では libx264/CRF 制御は不要）。
- [x] `README.md` に動作確認済みコマンド記録。

**完了条件達成:** (a) `videos/phase0_cartpole_sanity.mp4` 存在、(b) `docs/wheeledlab_reuse.md` 記載完了。

### Phase 1 — 実車スケール sedan アセット（visual wheel + 剛体シャシ、完了 2026-04-30）

**旧 Phase 1a / 1b は方針変更（案 B 採用）により破棄。** 案 B では車輪は物理に関与しない（collision off）ため、「最小モデルで物理パイプラインを通す」必要が消滅。最初から実車スケール + visual wheel only で作る。

旧 Phase 1a 一式（`sedan_minimal.urdf` / USD / `spawn_sedan.py` / `SEDAN_MIN_CFG`）は **`legacy/phase1a/`** に self-contained な実行可能スナップショットとして退避済み（`legacy/phase1a/README.md` 参照）。本体プロジェクトからは参照しない。

- **参考にする WheeledLab コード:** `wheeledlab_assets/f1tenth.py`（`ArticulationCfg` 書き方）、`wheeledlab_tasks/common/actions.py`（操舵ジョイントへの位置コマンド作法）。
- [x] `assets/urdf/sedan.urdf`:
  - `base_link`: 質量 1500 kg、ボックスシャシ (4.5 x 1.8 x 0.7 m)、CoG 高さ 0.55 m
  - 慣性テンソル: 実車相当（Ixx = 500、Iyy = 2500、Izz = 2700 kg·m²）
  - 4 輪: visual のみ（cylinder r=0.33, L=0.225, mass=0.1 kg ダミー）。**collision 要素は付けない**
  - 操舵ジョイント (`revolute`): 前 2 輪、Z 軸、limit ±0.611 rad、effort 500、velocity 3
  - 車輪回転ジョイント (`continuous`): 全 4 輪、Y 軸、damping 0、friction 0
  - ホイールベース 2.7 m、トレッド 1.55 m
  - サスペンション・ジョイントは**作らない**（案 B では Fz 計算で代替する）
- [x] URDF→USD 変換 → `assets/usd/sedan.usd`
- [x] `SEDAN_CFG: ArticulationCfg`:
  - 操舵アクチュエータ: `ImplicitActuatorCfg`（PhysX-side PD、`stiffness=8000, damping=400`）で位置制御。Phase 1.5 で Python 側の δ_actual 一次遅れを位置目標として書き込む
  - 車輪回転ジョイント: stiffness=0、damping=0（free spin。Phase 1.5 step 2 でトルクを直接印加）
- [x] `scripts/sim/spawn_sedan.py`:
  - **重力 OFF** で spawn → 1 秒 short window で姿勢確認
  - 完了条件: `|roll| < 0.5°`、`|pitch| < 0.5°`、すべての joint velocity max < 0.1 rad/s
  - ※ Phase 1.5 で重力 ON にして再利用（`--gravity` フラグ）

**完了条件達成 (2026-04-30):**
- (a) `videos/phase1_spawn_nograv.mp4` 出力
- (b) `metrics/phase1_spawn_nograv.json` の `pass.overall = true`（max_roll = 0.0°, max_pitch = 0.0°, max_joint_vel = 0.0 rad/s）

#### 設計判断（採択）

| 項目 | 判断 | 理由 |
|---|---|---|
| **重力 OFF/ON** | Phase 1 は OFF | 案 B では車輪 collision がないため重力 ON では自由落下する。Phase 1 は Articulation 健全性の純粋確認に集中。Phase 1.5 で Fz 注入と同時に ON |
| **慣性テンソル** | PLAN 値（500/2500/2700）を採用、ボックス均質値ではない | 実車のセダンは質量分布が中央寄りで `Izz` が低い。「実車相当」優先 |
| **base_link collision** | 残す | 重力 ON で Fz 注入が万一不調でも床抜けせず腹で受ける fall-through guard |
| **wheel mass** | 0.1 kg ダミー | 全車両質量を base_link に集中。視覚のみで物理的役割を持たない。Phase 1.5 step 2 で slip ratio が必要なら revisit |
| **steer 中間リンク** | visual なし、`mass=0.01` の kinematic-only リンク | 操舵ジョイントと車輪回転ジョイントを分離する目的だけ。USD 変換時に `Unresolved reference` 警告が出るが無害 |
| **steer actuator** | `ImplicitActuatorCfg`（PhysX-side PD） | legacy で動作確認済み。高 stiffness で 1 次遅れ近似が成立 |
| **URDF→USD 変換オプション** | `collision_from_visuals=False` を明示 | 案 B 死守ポイント: visual cylinder が自動的に collider に昇格しないことを保証 |

#### 実装ファイル

| ファイル | 役割 |
|---|---|
| `assets/urdf/sedan.urdf` | 実車スケール sedan URDF。base_link に visual + collision、4 輪は visual only |
| `assets/usd/sedan.usd` | URDF から自動生成（`+ configuration/` 4 ファイル + `config.yaml`） |
| `src/vehicle_rl/assets/sedan.py` | `SEDAN_CFG: ArticulationCfg` 定義。`STEER_JOINT_REGEX` 等の定数も export |
| `src/vehicle_rl/assets/__init__.py` | `SEDAN_CFG` と関連定数を re-export |
| `scripts/sim/convert_sedan_urdf.py` | URDF→USD 変換。`UrdfConverterCfg(collision_from_visuals=False)` で案 B を死守 |
| `scripts/sim/spawn_sedan.py` | spawn 確認。`--gravity` フラグ（default OFF）で重力切替。CSV / JSON / MP4 出力。完了条件を JSON に判定し return code に反映 |

#### 実行方法

リポジトリルート (`C:\Users\user\vehicle_rl`) から:

```bash
# venv の python を直接呼ぶ
PY="/c/work/isaac/env_isaaclab/Scripts/python.exe"

# 1. URDF → USD 変換 (assets/usd/sedan.usd を生成)
$PY scripts/sim/convert_sedan_urdf.py --headless

# 2. Phase 1 spawn 確認 (重力 OFF・1 秒)
$PY scripts/sim/spawn_sedan.py --duration 1.0 --headless

# 3. 結果確認
cat metrics/phase1_spawn_nograv.json   # pass.overall == true なら PASS
```

オプション:
- `--duration <sec>`: 計測時間（default: 1.0）
- `--gravity`: 重力 ON（Phase 1.5 で動作診断に使用）
- `--steer_target <rad>`: 操舵角を一定値で印加（Phase 1.5 着手前のステア動作確認用）
- `--no_video`: MP4 録画スキップ（dry run）

出力先:
- `videos/phase1_spawn_<nograv|gravity>.mp4`
- `metrics/phase1_spawn_<nograv|gravity>.csv`（200 step × 23 列）
- `metrics/phase1_spawn_<nograv|gravity>.json`（max 値と PASS/FAIL 判定）

### Phase 1.5 — 自前タイヤ力注入モデル（**step 1 完了 2026-04-30**）

**今回の中核**。案 B のタイヤ力・荷重・アクチュエータ動力学を実装する。

#### 実装ファイル

| ファイル | 役割 |
|---|---|
| `src/vehicle_rl/dynamics/state.py` | `VehicleState` dataclass、quat→rotmat / quat→rpy ヘルパ |
| `src/vehicle_rl/dynamics/actuator.py` | `FirstOrderLagActuator`（指数積分。drive/brake で τ 切替） |
| `src/vehicle_rl/dynamics/normal_load.py` | `NormalLoadModel` Protocol + `StaticNormalLoadModel`（静的 + 縦/横荷重移動 + z-drift PD） |
| `src/vehicle_rl/dynamics/tire_force.py` | `TireForceModel` Protocol + `LinearFrictionCircleTire`（線形 Cα + 摩擦円クリップ、parallel steering） |
| `src/vehicle_rl/dynamics/attitude_damper.py` | `AttitudeDamper`（roll/pitch PD トルク、ヨーは未ダンプ） |
| `src/vehicle_rl/dynamics/injector.py` | `aggregate_tire_forces_to_base_link`（4 タイヤ力 → base_link 1 点 force/torque、body frame） |
| `scripts/sim/run_phase1_5.py` | 検証スクリプト。`--scenario {straight,brake,circle} --mu --duration --target_speed --steer` |

#### Step 1 完了条件達成（2026-04-30）

| シナリオ | パラメタ | 結果 | 完了条件 |
|---|---|---|---|
| **straight** | μ=0.9, 8 s | max_roll 0.01° / max_pitch 1.29° / z_drift 0.85 mm / Σ Fz 誤差 **0.31%** | (a)(c)(d)(e) PASS |
| **brake** | μ=0.9, v₀=13.92 m/s | actual **12.88 m**, theory v²/(2μg)=10.98 m, **err +17.4%** | (b) PASS (±20% 以内) |
| **brake** | μ=0.6, v₀=13.83 m/s | actual **15.35 m**, theory 16.24 m, **err −5.5%** | (b) PASS |
| **brake** | μ=0.3, v₀= 8.60 m/s | actual **10.34 m**, theory 12.57 m, **err −17.8%** | (b) PASS |
| **circle** | μ=0.9, v=7.97 m/s, δ=0.1 rad | 半径 **27.09 m**, kinematic L/tan(δ)=26.92 m (誤差 0.6%), 限界速度 √(μgR)=15.47 m/s（安全領域）, max_roll 0.07°, Σ Fz 誤差 0.14% | (a)(c)(d) PASS |

**ハイパーパラメタ採用値**:
- 一次遅れ: τ_steer=50 ms, τ_drive=200 ms, τ_brake=70 ms
- コーナリング剛性: Cα = 60,000 N/rad/輪
- z-drift PD: kz=50,000 N/m, cz=5,000 Ns/m
- 仮想姿勢ダンパ: k_roll=k_pitch=80,000 Nm/rad, c_roll=c_pitch=8,000 Nms/rad

**注意点**:
- ハードブレーキ中の `Σ Fz` は **3.5〜4.9% pulse** で 1% 閾値を超えます。これは pitch dive (chassis z が 1 cm 沈む) を z-drift PD が補償するためで、定常状態の (d) は 0.31% で合格。動的パルスは仕様内挙動として許容。
- 直進加速時の **velocity loss ~12%** (5s で v=8.46 m/s vs 理論 9.6 m/s) は body-z 方向に Fz を注入する近似によるもの (chassis が pitch で僅かに傾くと Fz の世界 z 成分が cos(pitch) で減る)。Phase 3.5 で raycast に切り替えれば解消するが、Phase 1.5 step 1 完了条件には影響しないので保留。
- 初回ブレーキ実装で「停止後も制動力が効き続けて逆走する」バグを発見・修正（`v_long > 0.1` のときのみ a_x_target=-10 を許可）。

#### 実行方法

リポジトリルートから:

```bash
PY="/c/work/isaac/env_isaaclab/Scripts/python.exe"

# 直進
$PY scripts/sim/run_phase1_5.py --scenario straight --mu 0.9 --duration 8  --headless

# 制動 (μ スイープ)
$PY scripts/sim/run_phase1_5.py --scenario brake --mu 0.9 --target_speed 15 --duration 12 --headless
$PY scripts/sim/run_phase1_5.py --scenario brake --mu 0.6 --target_speed 15 --duration 14 --headless
$PY scripts/sim/run_phase1_5.py --scenario brake --mu 0.3 --target_speed 12 --duration 18 --headless

# 円旋回
$PY scripts/sim/run_phase1_5.py --scenario circle --mu 0.9 --target_speed 8 --steer 0.1 --duration 15 --headless
```

出力先: `videos/phase1_5_<scenario>_mu<μ>.mp4`, `metrics/phase1_5_<scenario>_mu<μ>.{csv,json}`



#### 1.5.0 設計原則

`src/vehicle_rl/dynamics/` 配下に**差し替え可能インターフェース**として実装する。Protocol で `NormalLoadModel` と `TireForceModel` を分離し、Phase 4 で raycast / Pacejka に置き換えるときに env 側を書き換えずに差し替えられる構成にする。

```python
# state.py
@dataclass
class VehicleState:
    pos_world: Tensor       # (N, 3) base_link world position
    rot_world: Tensor       # (N, 3, 3) base_link rotation
    vel_body: Tensor        # (N, 3) linear velocity in body frame
    angvel_body: Tensor     # (N, 3) angular velocity in body frame
    delta_actual: Tensor    # (N,)  操舵実値（Ackermann 配分前のセンタ角）
    omega_wheel: Tensor     # (N, 4) 車輪回転角速度（step 2 で使用）
    a_x_actual: Tensor      # (N,)  縦加速度コマンド実値（一次遅れ後）

# normal_load.py
class NormalLoadModel(Protocol):
    def compute(self, state: VehicleState) -> Tensor: ...   # (N, 4) Fz_i

class StaticNormalLoadModel:    # Phase 1.5 step 1 — 静的 + 簡易荷重移動
class RaycastNormalLoadModel:   # Phase 4 想定 — raycast + spring-damper

# tire_force.py
class TireForceModel(Protocol):
    def compute(self, state, Fz, mu) -> tuple[Tensor, Tensor]: ...  # Fx, Fy

class LinearFrictionCircleTire:  # Phase 1.5 step 1
class FialaTire:                 # Phase 1.5 step 2

# actuator.py
class FirstOrderLagActuator:
    """τ · dy/dt = u - y を陽に積分。操舵・駆動・制動で別インスタンス。"""
```

#### 1.5.1 Step 1: 線形 + 摩擦円 + 静的荷重（μ 変化を扱える最小構成）

**重要**: μ 変化は step 1 で**完全に**扱える（摩擦円クリップ経由）。step 2 で追加されるのはスリップ率動力学であって μ 効果ではない。

- **参考にする WheeledLab コード**: `wheeledlab_tasks/common/actions.py`（Ackermann 操舵マッピング）

##### a) アクチュエータ一次遅れ

```
操舵:  τ_steer · dδ_actual/dt = δ_target - δ_actual          τ_steer ≈ 50 ms
駆動:  τ_drive · da_x_actual/dt = a_x_target - a_x_actual    τ_drive ≈ 200 ms (a_x_target ≥ 0)
制動:  τ_brake · da_x_actual/dt = a_x_target - a_x_actual    τ_brake ≈ 70 ms (a_x_target < 0)
```

- 操舵: Python 側で `δ_actual` を陽に積分し、操舵 revolute joint の位置目標として渡す（`IdealPDActuatorCfg` に位置を投げる）。**step 1 では parallel steering（左右前輪に同一 δ_actual）** を採用する。Ackermann 配分（δ_left = atan(L/(R−W/2)), δ_right = atan(L/(R+W/2))）は導入しない。理由: Phase 1.5 の検証シナリオは δ ≤ 0.1 rad で δ_left/δ_right の差が 3% 以下、parallel 近似で十分。`tire_force.py:per_wheel_steer()` は `(N, 4)` の per-wheel 角度を返す API になっているので、Ackermann 化は当該関数 4 行の差し替えで完結する。step 2（Fiala）または Phase 2 で旋回半径が小さい操作を走らせ、内外輪のスクラブが顕在化したら導入する
- 縦方向: `a_x_actual` から `Fx_total = m · a_x_actual` を計算し、駆動配分（**RWD: 後輪 2 輪等分** / AWD: 4 輪等分）。制動時は 4 輪等分（実車のブレーキ配分は前後 6:4 程度だが Phase 1.5 では等分で十分）
- 観測には `δ_actual`, `a_x_actual` を含める（旧計画の「行動履歴」を置き換える）

##### b) 法線荷重 — `StaticNormalLoadModel`

平坦路・常時接地仮定のもとで、各輪 Fz を解析的に計算:

```
a = CoG → 前軸距離,  b = CoG → 後軸距離,  L = a + b = wheelbase

Fz_static_front = m·g · b / L / 2     # 前 1 輪あたり
Fz_static_rear  = m·g · a / L / 2     # 後 1 輪あたり

縦荷重移動: ΔFz_long = m · a_x_actual · h_cg / L
横荷重移動 (前): ΔFz_lat_f = m · a_y_estimate · h_cg / track · (b/L)
横荷重移動 (後): ΔFz_lat_r = m · a_y_estimate · h_cg / track · (a/L)

Fz_FL = Fz_static_front - ΔFz_long/2 - ΔFz_lat_f
Fz_FR = Fz_static_front - ΔFz_long/2 + ΔFz_lat_f
Fz_RL = Fz_static_rear  + ΔFz_long/2 - ΔFz_lat_r
Fz_RR = Fz_static_rear  + ΔFz_long/2 + ΔFz_lat_r
```

- `a_y_estimate` は前ステップの `vy, yaw_rate` から推定（フィードバックループを切る）
- `Σ Fz_i = m·g` が解析的に成立。各タイヤ位置に **world z 上向き**で適用 → ロール/ピッチモーメントが Fz 分布から自動的に発生
- **z ドリフト抑制**: 弱い PD `F_z_correction = -kz·(z_base - z_ref) - cz·vz_z` を 4 輪に等分加算。`z_ref = 0.55 m`、ゲインは roll/pitch ダンパよりさらに弱く（数値ドリフトだけ消す目的）

##### c) タイヤ力 — `LinearFrictionCircleTire`

各タイヤで:

```
タイヤ位置 (body): r_i = (±a, ±track/2, -h_cg)
タイヤ局所速度: v_i_body = v_body + ω_body × r_i_body
操舵角 δ_i を考慮しタイヤ座標系へ: v_long_i, v_lat_i
スリップ角: α_i = atan2(v_lat_i, max(|v_long_i|, ε))

Fy_raw_i = -C_α · α_i
Fx_raw_i = (Fx_total / N_drive) for drive wheels else 0   (制動時は 4 輪等分)

# 摩擦円クリップ
F_norm = sqrt(Fx_raw² + Fy_raw²)
F_max  = μ_i · Fz_i
if F_norm > F_max:
    scale = F_max / F_norm
    Fx_i = Fx_raw · scale
    Fy_i = Fy_raw · scale
```

`C_α`（コーナリング剛性）は実車 1 輪あたり 50,000〜80,000 N/rad 程度から始める。

##### d) 力・トルク注入 — `injector.py`

```
F_total_world = Σ R_body→world · (Fx_i, Fy_i, 0) + Σ (0, 0, Fz_i)
τ_total_body  = Σ r_i_body × (R_world→body · F_i_world)
```

`Articulation.set_external_force_and_torque(F_total_world, τ_total_body, body_ids=[base_link_id])` で base_link に注入。

##### e) 仮想姿勢ダンパ (roll / pitch)

body frame で:

```
τ_roll  = -k_roll  · roll  - c_roll  · ω_x_body
τ_pitch = -k_pitch · pitch - c_pitch · ω_y_body
τ_yaw   = 0   ← ヨーには絶対にダンパを入れない
```

最初は弱めから入れて、発散・高周波振動が出たら剛性を足す方針。固有振動数を 1.5〜2 Hz に厳密合わせ込む必要は Phase 1.5 ではない（Phase 3.5 の外乱マップ導入時に再調整）。

##### f) 路面 μ

`μ_i = 0.9`（全輪共通の単一スカラ）から開始。env reset 時の per-env スカラ randomization は Phase 3 で導入。

##### Step 1 完了条件

- (a) 直進加速 / 制動 / 一定舵角円旋回の 3 シナリオで `videos/phase1_5_step1_*.mp4` 出力
- (b) 動力学サニティ（数値検証）:
  - μ ∈ {1.0, 0.6, 0.3} で **制動距離 ≈ v² / (2μg)** が ±20% 以内
  - μ ∈ {1.0, 0.6, 0.3} で **定常円旋回限界速度 ≈ √(μgR)** が ±20% 以内
- (c) ロール/ピッチが発散せず、高周波振動が見えない（roll/pitch RMS < 5°、卓越周波数 < 20 Hz）
- (d) `Σ Fz_i = m·g` が常に成立（CSV ログで確認、誤差 < 1%）
- (e) 直進 5 分連続走行で `|z_base - z_ref| < 0.05 m`（z ドリフト抑制が機能）

#### 1.5.2 Step 2: Fiala / slip ratio（必要になったら）

step 1 のままで Phase 3 まで進む。step 2 着手は **ホイールスピン・ロック等の挙動が問題になり始めたら**。Phase 3.5 直前か Phase 4 で十分。

- 駆動・制動コマンドを spin joint へのトルク `T_i` として印加
- spin joint の `ω_wheel_i` を読んで `κ_i = (ω_i·r - v_long_i) / max(|v_long_i|, |ω_i·r|)`
- Fiala 式で `Fx_i = f(κ_i, μ_i, Fz_i)`、`Fy_i = g(α_i, κ_i, μ_i, Fz_i)`（縦横の連成は Fiala の摩擦楕円で扱う）
- 反作用トルク `-r · Fx_i` を spin joint へフィードバック

##### Step 2 完了条件

- step 1 のサニティ（制動距離・限界速度）を引き続き満たす
- ホイールスピン（低 μ + 高スロットルで `ω·r >> v_x`）が観測できる
- ロック制動（低 μ + フルブレーキで `ω → 0`、`v_x > 0`）が観測できる
- step 1 と step 2 で同一コース・同一 μ の挙動を CSV / MP4 で定量比較

### Phase 2 — 古典制御によるベースライン（約 1 日）

Phase 1.5 step 1 のタイヤ力モデルの上で Pure Pursuit + PID を走らせる。

- **参考にする WheeledLab コード:** `wheeledlab_tasks/common/observations.py`（センサ観測の組み立てパターン）、`wheeledlab_rl/utils/custom_video_recorder.py`（録画ユーティリティ）。
- [ ] センサ: シャシに `ImuCfg`、「GPS」は root pose（`x, y, ψ`）を直接取得し任意でガウシアンノイズを付加。
- [ ] プランナ（`planner/waypoints.py`）: **円 / リサジュー（8 字）/ S 字 / ISO 3888-2 ダブルレーンチェンジ**の生成器。出力は `(N, 3)` の (x, y, target_speed) 配列。
- [ ] コントローラ:
  - 横方向: **Pure Pursuit**（look-ahead 距離は速度比例）→ `δ_target` を出力
  - 前後方向: **PID**（速度追従）→ `a_x_target` を出力
  - 出力は Phase 1.5 のアクチュエータ一次遅れを通って `δ_actual, a_x_actual` になり、タイヤ力モデルへ渡る
- [ ] 駆動スクリプト `scripts/sim/run_classical.py`:
  - CLI: `--course {circle,lemniscate,s,dlc} --mu 0.9 --speed 15 --duration 30 --video`
  - ヘッドレス実行 → `videos/classical_<course>_mu<μ>_v<v>.mp4`
  - CSV ログ: `t, x, y, ψ, vx, vy, yaw_rate, ax, ay, roll, pitch, δ_target, δ_actual, a_x_target, a_x_actual, lateral_error, heading_error, μ, Fx_i, Fy_i, Fz_i, slip_angle_i (i=1..4)`
- [ ] μ を数値スイープ（例: 1.0, 0.6, 0.3）して Pure Pursuit + PID が破綻する境界を把握。
- [ ] **Vehicle dynamics sanity 指標** を CSV から計算して `metrics/classical_<course>_mu<μ>_v<v>.json` に保存:
  - `rms_lateral_error` [m]
  - `max_lateral_error` [m]
  - `completion_rate` [0/1]（目標経路の終端まで到達したか）
  - `mean_speed_error` [m/s]
  - `max_yaw_rate` [rad/s]
  - `max_roll_angle` [rad]
  - `off_track_time` [s]（`|lateral_error| > 1m` の累積時間）
- これらの指標は Phase 3 の RL 評価でも **同じ実装で再利用** し、ベースラインとの apples-to-apples 比較を担保する。

**完了条件:**
- (a) μ = 1.0 で円 / S 字を綺麗に追従、μ = 0.3 で目に見えて崩れる様子が MP4 で確認できる。
- (b) 上記 7 指標が全コース × μ スイープ分について JSON で出力されている。
- (c) μ = 1.0 / 中速での `rms_lateral_error < 0.3 m`、`completion_rate = 1.0`、`max_roll_angle < 5°`（妥当な制御が成立している証跡）。

### Phase 3 — RL コントローラ（数日）

- **参考にする WheeledLab コード:** `wheeledlab/envs/mdp/`（rewards/events/terminations の構造）、`wheeledlab_rl/configs/{common_cfg,rl_cfg}.py`（学習 cfg テンプレート）、`wheeledlab_rl/scripts/{train_rl,play_policy}.py`（エントリポイント）、`wheeledlab_rl/utils/modified_rsl_rl_runner.py`（PPO ランナー）。

- [ ] **env 実装方式の決定（Phase 3 の最初のタスク）:** `ManagerBasedRLEnv` か `DirectRLEnv` か。
  - `ManagerBasedRLEnv`: Manager 群（Action / Observation / Reward / Event / Termination）を cfg で組み立てる構成。コンポーネント再利用に強い。WheeledLab はこちらを採用。
  - `DirectRLEnv`: `_get_observations()`, `_get_rewards()`, `_apply_action()` 等を直接 override する構成。**案 B のように毎ステップ自前タイヤ力注入を行う場合はデバッグしやすい**。
  - **判定方法:** WheeledLab を読み終えた後、両方式で「観測 1 つ・報酬 1 つだけの最小 env」を `prototypes/` に書いて 30 分以内に動かし、書き味と print デバッグの容易さで決定。結果を `docs/env_choice.md` に記録。
- [ ] 決定した方式で `envs/tracking_env_cfg.py`（または `tracking_env.py`）を構成。Phase 1.5 の `dynamics/` モジュール群を組み込む。

  **観測**（プランナ出力は所与）:
  - 車両状態: `vx, vy, yaw_rate, ax, ay`（IMU 由来）
  - 自己位置: `x, y, ψ`（GPS 由来、ノイズあり）
  - 姿勢: `roll, pitch`（オプション、頑健性評価用）
  - Lookahead waypoints: 直前の N=10 点を車両座標系で `(Δx, Δy, Δv_target)`
  - **アクチュエータ実値**: `δ_actual, a_x_actual`（旧計画の「行動履歴」を置換）
  - μ は観測に**入れない**（頑健性訓練）。ただしオラクル比較として「μ 真値を観測に入れた policy」を 1 本だけ別途学習し、性能上限を測る

  **行動**: 目標操舵角 `δ_target ∈ [-0.6, 0.6] rad` と目標前後加速度 `a_x_target ∈ [-5, 3] m/s²`。アクチュエータ一次遅れを通って実値になる。

  **報酬**:
  - `−w₁ · 横偏差²`
  - `−w₂ · ヘディング誤差²`
  - `−w₃ · (v − v_target)²`
  - `−w₄ · δ̇²`（操舵レート罰則）
  - `−w₅ · jerk²`
  - 小さな `+w₀` 生存ボーナス

  **ドメインランダマイゼーション (events)**:
  - **`μ ∈ [0.3, 1.0]`** をタイヤモデルに per-env スカラで注入（`LinearFrictionCircleTire.mu` を env reset 時に書き換え）
  - 車重 ±15%、CoG 高さ ±0.1 m
  - IMU ノイズ σ、GPS ノイズ σ
  - **アクチュエータ時定数** `τ_steer ±30%`、`τ_drive ±30%`
  - エピソード毎にコース種類をサンプリング

  **終了条件**: `|横偏差| > 4 m`、ロールオーバー（`|roll| > 60°`）、コース完走、シミュレーション発散検知。

- [ ] 学習スクリプト `scripts/rl/train_ppo.py`（**rsl_rl** 使用）:
  - **並列環境数は段階的に上げる**（最初から 4096 にするとリセット・接触バグがログに埋もれる）:
    1. **デバッグ: 1〜16 env** — 1 step ごとに観測・報酬・終了を print して妥当性確認、env リセットが破綻しないこと、自前タイヤ力注入が per-env で正しく機能すること
    2. **報酬チューニング: 64〜256 env** — 短時間学習で報酬曲線の形状確認、報酬重みのバランス
    3. **初回学習: 512〜1024 env** — 1 コースのみで学習が収束することを確認
    4. **本番: 2048〜4096 env** — 全コース + DR フル投入
  - 各段階で問題が出たら 1 段戻る
  - PPO デフォルトから開始、初回後にエントロピー・学習率を調整
  - `--headless`、TensorBoard ログを `logs/` に出力
- [ ] 再生スクリプト `scripts/rl/play.py --checkpoint <ckpt> --course dlc --mu 0.5 --video` → テレメトリオーバーレイ付き MP4 出力。
- [ ] **Phase 2 と同じ 7 指標**（rms / max lateral error, completion rate, mean speed error, max yaw rate, max roll, off-track time）を同じスクリプトで計算し、ベースラインと数値比較。

**完了条件:**
- (a) RL ポリシーが μ ≤ 0.5 において 4 コース全てで Pure Pursuit + PID を `rms_lateral_error` で上回る。
- (b) `completion_rate` がベースライン以上。
- (c) `max_roll_angle`・`max_yaw_rate` がベースラインから極端に悪化していない（aggressive すぎる挙動になっていない）。

### Phase 3.5 — 経路座標系外乱マップ（疑似悪路）

3D 悪路メッシュを物理的に走るのではなく、**悪路が車両に与える主要な外乱を経路座標 `(s, d)` 上で再現**する。

#### 設計

車両位置をコース中心線に射影して `s`（経路に沿った進行距離）、`d`（中心からの横偏差）を求める。閉形式が書ける曲線（円・クロソイド・スプライン）を優先、複雑形状は KD-tree。

`disturbance.py` で以下の関数を `(s, d, env_id)` から計算:

```
μ_i(s, d, env_id)            # 各輪別 μ。split-μ や低 μ パッチ
Fz_scale_i(s, d, env_id)     # 各輪 Fz の倍率。ペア相関で Σ Fz ≈ mg を維持
roll_disturbance(s, d)
pitch_disturbance(s, d)
yaw_moment_disturbance(s, d)
```

#### 段階的導入

1. **各輪 μ マップ**（最初に入れる）:
   - 例: 0〜20 m 全輪 μ=0.9 / 20〜40 m 左 μ=0.4 右 μ=0.9（split-μ）/ 40〜60 m 全輪 μ=0.5 / 60〜80 m ランダム低 μ パッチ
   - タイヤ力モデルの μ をこのマップから読む形に変更（`LinearFrictionCircleTire.mu` を per-env per-tire テンソル化）

2. **Fz 揺れ**:
   - `Fz_i = Fz_static_i · Fz_scale_i(s, d)`
   - **ペア相関で揺らす**（左右ペアまたは前後ペアを逆相に）: 「片輪段差」「うねり」に近い挙動になる
   - 単純に 4 輪独立で揺らすと `Σ Fz` が大きく振れて非物理的

3. **roll / pitch 外乱トルク**:
   ```
   τ_roll_dist  = k_roll  · roll_disturbance(s)  - c_roll  · ω_x_body
   τ_pitch_dist = k_pitch · pitch_disturbance(s) - c_pitch · ω_y_body
   ```
   - Phase 1.5 step 1 で入れた仮想姿勢ダンパに**重畳**する形（姿勢を直接書き換えない）
   - この段階で姿勢ダンパの剛性・減衰を 1.5〜2 Hz・ζ≈0.3 相当に再チューニング

#### 完了条件

- (a) split-μ 区間で車両が自然にヨー方向へ乱される MP4 が確認できる
- (b) 低 μ パッチで横偏差が増えることが指標として出る
- (c) 外乱マップあり vs なしで RL ポリシーと古典制御の性能差を比較し、RL の頑健性優位を定量化（例: μ パッチ通過時の `rms_lateral_error` で RL がベースラインより 30% 以上小さい）
- (d) Phase 3 で学習したポリシーが、外乱マップ追加で再学習しなくてもある程度走れる（zero-shot 頑健性確認）。再学習したらさらに伸びる

### Phase 4 — PhysX Vehicle SDK との比較（後回し）

悪路メッシュ・段差・片輪接地・サスペンション底付きが本題になったときに、PhysX Vehicle SDK（Pacejka タイヤ + サスペンション + 駆動系）と比較する。

- [ ] `RaycastNormalLoadModel` を実装（案 B 内で raycast + spring-damper サス。Phase 3.5 までの平坦路前提を緩和）
- [ ] PhysX Vehicle SDK 化:
  - sedan USD に `PhysxVehicleAPI` を追加: タイヤモデル = Pacejka、サスストローク、アンチロールバー等
  - RL の行動 `(δ_target, a_x_target)` を PhysX Vehicle の駆動入力（`accelerator, brake, steer`）にマッピングする薄いアダプタを実装
  - Isaac Lab の `Articulation` は PhysX Vehicle SDK を直接ラップしていないため、アダプタは自前実装。ここに摩擦が出ることを見込む
- [ ] Phase 3 の学習パイプラインを (i) 案 B step 1 / (ii) 案 B step 2 (Fiala) / (iii) PhysX Vehicle (Pacejka) で再走、転移挙動を比較

**完了条件:** 3 つのタイヤ忠実度レベルで学習したポリシーの円 + DLC 走行を比較し、転移性能を `docs/fidelity_comparison.md` に記録。

## 5. 後回しの決定事項

- PPO の実装は rsl_rl / skrl どちらか（まずは rsl_rl で開始、両方インストール済み）
- W&B 併用するか TensorBoard のみか（W&B は `C:\work\isaac\WheeledLab\wandb` に既に存在）
- 路面バリエーション（平地 → バンク → 起伏） — Phase 3.5 までは経路座標上の外乱マップで代替、Phase 4 で raycast 化
- Sim-to-Real / 実車展開 — 現状スコープ外

## 6. ペース

「やりながら考える」方針。各 Phase を MP4 エビデンス + 数値指標付きで一区切りずつ完了させ、その都度プランを更新する。
