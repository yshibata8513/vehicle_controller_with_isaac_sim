# YAML Config 外部化 設計案

目的:

- 車両諸元、力学パラメータ、環境、コース、報酬、PPO、classical controller、実験選択を YAML に集約する。
- コード内の tunable な定量値と course selection をなくし、YAML 指定を必須にする。
- コード側は schema / validation / formula / wiring だけを持つ。
- `course=circle` か `random_bank` か、reward weight、PPO horizon、車両 mass などはすべて実験 YAML から辿れるようにする。

## 確定した設計判断

PR 1 着手前の打ち合わせで合意した、loader / 構成 / 移行戦略の具体仕様。本文の例より優先する。

### PR 戦略

段階 PR で進める。big-bang は壊れた時に切り戻しが重く、1 step 1 PR は中間状態の管理が増えるため避ける。

1. **PR 1**: loader + schema 骨組み + 全 YAML 雛形 + 単体テスト。runtime コードには触らない。**完了** (`fc0bd93`)。
2. **PR 2**: vehicle / dynamics / course を YAML 化。`SEDAN_CFG`, `VehicleSimulator`, `waypoints.py` の default 引数を削る。
3. **PR 3**: env / reward / PPO を YAML 化。`TrackingEnvCfg`, `rsl_rl_ppo_cfg.py` の数値 default を削る。
4. **PR 4**: train / play / classical scripts を `--config` 化。README 更新、旧 CLI 即時撤去。

各 PR の main マージ時点で「動く」状態は維持する。loader だけ入って scripts は旧式、は許容する。

### Loader / schema 実装

`PyYAML` + 自作 `@dataclass` schema + 自作 strict validator。Hydra / OmegaConf は使わない (暗黙挙動と Isaac Lab の Hydra 依存との衝突リスクを避ける)。Isaac Lab cfg object への変換は明示的な adapter 層で行う。

### Ref 解決の仕様

`<name>_ref: <repo-root-relative-path>` 形式の key を `<name>: <referenced-yaml-content>` に置換する。

- パス: **repo-root 相対のみ**。絶対パス・repo 外参照は `ValueError`。
- 推移参照: 許可 (experiment → course → random_path_generator のように 2 段以上ネスト OK)。
- cache: あり。同じ ref パスは 1 回だけ load し、結果を共有する。
- cycle: visited set で検出し `ValueError`。
- `${...}` 展開・環境変数展開は持たない。
- `<name>` が同階層に既に存在する場合 (`child:` と `child_ref:` が並ぶ等) は `ValueError`。

### Override 仕様

experiment YAML の `overrides:` ブロックは、refs を全部解決した後に deep-merge される。

- 未知 key (base 側に存在しない key) は `ValueError`。
- list は丸ごと差し替え (要素単位の merge はしない)。例: `actor_hidden_dims: [256, 256]` を `[512]` で上書き。
- `<name>_ref` を overrides 内に書くのは禁止 (`ValueError`)。**実験構造そのものを差し替えたい場合は別 experiment YAML を作る**。
- `dict` ↔ `dict` は再帰 deep-merge、`dict` ↔ scalar / list は丸ごと差し替え。

### CLI 撤去のタイミング

PR 4 で **即時撤去** (deprecation 期間なし)。

- 削る: `--course`, `--num_envs`, `--max_iterations`, `--seed`, `--experiment_name`, `--run_name`, `--random_path_cfg`, `--mu`, `--target_speed`, `--radius`, `--pid_kp` 等の実験条件 override。
- 残す: `--config`, `--headless`, `--resume`, `--load_run`, `--checkpoint`, Isaac Lab AppLauncher 系。
- README の旧 CLI 例も同 PR 内で全部更新する。

### Stage 0a 再現

学習インフラの基準点として `configs/experiments/rl/phase3_circle_stage0a.yaml` を作成し、circle steering-only PPO を YAML だけで再走できるようにする。`max_iterations` は 200 を default とする (履歴に明示値が残っていないため `phase3_random_long` と揃えた値)。

### Play の YAML 入口

`scripts/rl/play.py --config <play 専用 experiment YAML>` の単一入口。`--play_config` のような second YAML 入口は持たない。play 専用 experiment YAML は train 用と同じ category refs を持ち、`overrides:` で `num_envs=1`, `random_reset_along_path=false` などを上書きする。

`configs/runtime/play.yaml` は `paths_ref: configs/runtime/local_paths.yaml` で `local_paths` を内包する。これにより experiment YAML の `runtime_ref` は 1 本に保てる (paths と play 固有設定が両方乗る)。

### 触らないもの (本 refactor のスコープ外)

- `assets/usd/sedan.usd` / URDF の YAML からの逆生成 (geometry YAML は URDF と同期させるが factor of truth は URDF のまま)。
- `configs/numerics.yaml` (`1e-6` 等の eps 類)。物理・学習挙動に影響する eps は将来必要になったら別途。
- `configs/runtime/local_paths.yaml` は repo-relative の committable default (`logs/rsl_rl`, `metrics`, `videos`) を 1 本入れる。個人パス上書きが必要になったら `local_paths.local.yaml` を `.gitignore` で別管理にする (今は不要)。

### E2E guard の運用

`docs/refactor_e2e_guard.md` の before/after CSV/JSON 比較は **スキップ**。各 PR 後に `train_ppo.py` で短い PPO 学習が成立することだけを最終確認する。

理由: rollout の bit-equivalence を維持するために refactor の自由度を縛りたくない (PR 2-3 で derived 値の計算順序が変わる可能性がある)。

### 実装メモ (loader の癖が出た箇所)

- **`speed_controller` の controller 参照**: 本文例では `speed_controller: { ref: configs/controllers/speed_pi.yaml }` と書いていたが、loader は `<name>_ref` 一律で `<name>` に解決するため `speed_controller` 階層下では `controller_ref:` に統一する (`ref` 単体は予約していない)。
  ```yaml
  speed_controller:
    enabled_when_action_mode: steering_only
    controller_ref: configs/controllers/speed_pi.yaml
  ```

## 基本方針

### No Code Defaults

コードには「実験・物理・学習に影響する default 値」を置かない。

- OK: 数式上の `0`, `1`, `2`, tensor index、単位変換、`math.pi`、shape 展開など。
- 原則 YAML: mass, wheelbase, tire stiffness, tau, reward weight, dt, num_envs, course, radius, target speed, PPO gamma, learning rate, controller gain, output dir など。
- derived 値は YAML に重複させず、YAML の source 値からコードで計算する。
  - 例: `pinion_max = delta_max_rad * steering_ratio`
  - 例: `observation_space = scalar_fields + plan_K * plan_channels`

### Strict Loader

`src/vehicle_rl/config/` に YAML loader と schema を追加する。

想定:

```text
src/vehicle_rl/config/
  __init__.py
  loader.py          # load_yaml, resolve refs, strict validation
  schema.py          # dataclass / configclass schema, no defaults
  isaac_adapter.py   # YAML bundle -> Isaac Lab cfg objects
```

loader のルール:

- missing key は即 `ValueError`
- unknown key も即 `ValueError`
- `dict.get("key", default)` は使わない
- optional field も YAML 側に `null` として明示する
- schema version を持つ
- config load 後に「解決済み full config」を log dir に dump する

CLI は原則 `--config` だけに寄せる。`--headless`, `--resume`, `--checkpoint` のような実行制御は残してよいが、course / reward / PPO / vehicle などの実験条件は YAML に寄せる。

## 推奨ファイル構成

```text
configs/
  vehicles/
    sedan.yaml

  dynamics/
    linear_friction_circle_flat.yaml

  envs/
    tracking.yaml

  courses/
    circle.yaml
    s_curve.yaml
    lemniscate.yaml
    dlc.yaml
    random_path_generator.yaml
    random_long.yaml
    random_bank.yaml

  controllers/
    pure_pursuit.yaml
    speed_pi.yaml

  agents/
    rsl_rl/
      ppo_tracking.yaml

  runtime/
    local_paths.yaml
    train_video.yaml
    play.yaml
    classical_video.yaml

  experiments/
    rl/
      phase3_circle_stage0a.yaml
      phase3_random_long.yaml
      phase3_random_bank.yaml
    classical/
      circle_baseline.yaml
      s_curve_baseline.yaml
      dlc_baseline.yaml
```

`experiments/**.yaml` は composition root とする。各カテゴリ YAML を `ref` で参照し、最終的な train / play / classical はこの root だけを読む。

## 各 YAML の責務

### `configs/vehicles/sedan.yaml`

車両固有で、学習条件に依存しない値を置く。

移動元の代表:

- `src/vehicle_rl/assets/sedan.py`
- `src/vehicle_rl/envs/types.py`
- `src/vehicle_rl/envs/simulator.py`
- URDF と同期すべき寸法値

含めるもの:

```yaml
schema_version: 1
name: sedan

asset:
  usd_path: assets/usd/sedan.usd
  prim_path_train: /World/envs/env_.*/Sedan
  prim_path_single: /World/Sedan

joints:
  steer_regex: front_(left|right)_steer_joint
  wheel_regex: (front|rear)_(left|right)_wheel_joint
  base_body: base_link
  wheel_order: [FL, FR, RL, RR]

geometry:
  wheelbase_m: 2.7
  track_m: 1.55
  wheel_radius_m: 0.33
  wheel_width_m: 0.225
  cog_z_m: 0.55
  a_front_m: 1.35

mass:
  total_kg: 1500.0

steering:
  delta_max_rad: 0.611
  steering_ratio: 16.0

physx:
  rigid_body:
    max_linear_velocity: 100.0
    max_angular_velocity: 100.0
    max_depenetration_velocity: 10.0
    enable_gyroscopic_forces: true
  articulation:
    enabled_self_collisions: false
    solver_position_iteration_count: 8
    solver_velocity_iteration_count: 1
    sleep_threshold: 0.005
    stabilization_threshold: 0.001
  actuators:
    steer:
      stiffness: 8000.0
      damping: 400.0
      effort_limit_sim: 500.0
      velocity_limit_sim: 3.0
      friction: 0.1
    wheels:
      stiffness: 0.0
      damping: 0.0
      effort_limit_sim: 400.0
      velocity_limit_sim: 200.0
      friction: 0.0
```

コード側の `SEDAN_CFG` は、この YAML から生成する factory に置き換える。

### `configs/dynamics/linear_friction_circle_flat.yaml`

車両上で使う力学モデルと係数を置く。

移動元の代表:

- `VehicleSimulator.__init__`
- `LinearFrictionCircleTire`
- `StaticNormalLoadModel`
- `FirstOrderLagActuator`
- `AttitudeDamper`
- `A_X_TARGET_MIN/MAX`

含めるもの:

```yaml
schema_version: 1
model: linear_friction_circle_flat

gravity_mps2: 9.81

friction:
  mu_default: 0.9

action_limits:
  a_x_min_mps2: -5.0
  a_x_max_mps2: 3.0

actuator_lag:
  tau_steer_s: 0.05
  tau_drive_s: 0.20
  tau_brake_s: 0.07
  initial_value: 0.0

tire:
  type: linear_friction_circle
  cornering_stiffness_n_per_rad: 60000.0
  eps_vlong_mps: 0.01
  longitudinal_force_split:
    accel: rear
    brake: four_wheel

normal_load:
  type: static_weight_transfer
  z_drift_kp: 50000.0
  z_drift_kd: 5000.0

attitude_damper:
  k_roll: 80000.0
  c_roll: 8000.0
  k_pitch: 80000.0
  c_pitch: 8000.0
```

### `configs/envs/tracking.yaml`

DirectRLEnv と task 固有の設定を置く。course はここに直書きせず、experiment root から `course.ref` として選ぶ。

移動元の代表:

- `TrackingEnvCfg`
- `_get_rewards`
- `_get_dones`
- `_get_observations`

含めるもの:

```yaml
schema_version: 1
task_id: Vehicle-Tracking-Direct-v0

timing:
  physics_dt_s: 0.005
  decimation: 4
  episode_length_s: 25.0

scene:
  num_envs: 128
  env_spacing_m: 200.0
  replicate_physics: true
  clone_in_fabric: true

spaces:
  action_mode: steering_only        # steering_only | steering_and_accel
  observation:
    imu_fields: [vx, yaw_rate, ax, ay, roll, pitch]
    include_pinion_angle: true
    include_path_errors: true
    include_plan: true
    include_world_pose: false

planner:
  plan_K: 10
  lookahead_ds_m: 1.0
  projection:
    search_radius_samples: 80
    max_index_jump_samples: 120

reset:
  random_reset_along_path: true
  warm_start_velocity: true

action_scaling:
  pinion_action_scale_rad: 3.0

speed_controller:
  enabled_when_action_mode: steering_only
  controller_ref: configs/controllers/speed_pi.yaml   # NOTE: <name>_ref 形式に統一

reward:
  progress: 1.0
  alive: 0.1
  lateral: -0.2
  heading: -0.3
  speed: -0.01
  pinion_rate: -0.01
  jerk: -0.001
  termination: -10.0
  progress_clamp: [-1.0, 1.0]

termination:
  max_lateral_error_m: 4.0
  max_roll_rad: 1.047

diagnostics:
  log_reward_terms: true
  log_state_action_terms: true
  log_projection_health: true
```

`action_space` と `observation_space` は YAML に数値として重複させず、`action_mode` と observation spec から計算する。

### `configs/courses/*.yaml`

コース形状と target speed を置く。実験 root がどの course YAML を使うかを選ぶ。

例: `configs/courses/circle.yaml`

```yaml
schema_version: 1
type: circle
ds_m: 0.2
target_speed_mps: 10.0
radius_m: 30.0
is_loop: true
```

例: `configs/courses/s_curve.yaml`

```yaml
schema_version: 1
type: s_curve
ds_m: 0.2
target_speed_mps: 12.0
length_m: 100.0
amplitude_m: 5.0
n_cycles: 1.0
n_raw: 4096
is_loop: false
```

例: `configs/courses/random_bank.yaml`

```yaml
schema_version: 1
type: random_bank
generator_ref: configs/courses/random_path_generator.yaml
phase: phase2_bank
```

既存の `configs/random_path.yaml` は、`configs/courses/random_path_generator.yaml` へ移動するのが自然。`random_long.yaml` / `random_bank.yaml` は generator YAML のどの section を使うかだけを選ぶ薄い YAML にする。

### `configs/controllers/*.yaml`

classical baseline と RL 内部 PI の controller gain を置く。

例: `configs/controllers/speed_pi.yaml`

```yaml
schema_version: 1
type: speed_pi
kp: 1.0
ki: 0.3
integral_max: 10.0
```

例: `configs/controllers/pure_pursuit.yaml`

```yaml
schema_version: 1
type: pure_pursuit
lookahead_min_m: 2.0
lookahead_gain_s: 0.5
lookahead_ds_m: 1.0
```

### `configs/agents/rsl_rl/ppo_tracking.yaml`

PPO / policy / runner の値を置く。

移動元:

- `src/vehicle_rl/tasks/tracking/agents/rsl_rl_ppo_cfg.py`

含めるもの:

```yaml
schema_version: 1
runner:
  num_steps_per_env: 64
  max_iterations: 300
  save_interval: 50
  experiment_name: vehicle_tracking_direct
  clip_actions: 1.0
  obs_groups:
    policy: [policy]
    critic: [policy]

policy:
  init_noise_std: 0.3
  actor_obs_normalization: true
  critic_obs_normalization: true
  actor_hidden_dims: [256, 256]
  critic_hidden_dims: [256, 256]
  activation: elu

algorithm:
  value_loss_coef: 1.0
  use_clipped_value_loss: true
  clip_param: 0.2
  entropy_coef: 0.005
  num_learning_epochs: 5
  num_mini_batches: 4
  learning_rate: 0.0003
  schedule: adaptive
  gamma: 0.995
  lam: 0.95
  desired_kl: 0.01
  max_grad_norm: 1.0
```

### `configs/runtime/*.yaml`

machine-local な出力先、video、camera、play duration などを置く。

例: `configs/runtime/local_paths.yaml`

```yaml
schema_version: 1
logs_root: logs/rsl_rl
metrics_dir: metrics
video_dir: videos
```

例: `configs/runtime/play.yaml`

```yaml
schema_version: 1
num_envs: 1
duration_s: 25.0
force_random_reset_along_path: false
plot:
  enabled: true
  dpi: 120
  trajectory_pad_m: 5.0
```

絶対パスは避け、repo root 相対にする。個人環境だけの設定は `configs/runtime/local*.yaml` として `.gitignore` 対象にしてもよい。

### `configs/experiments/**/*.yaml`

実験で何を組み合わせるかだけを書く。course 選択もここ。

例: `configs/experiments/rl/phase3_random_bank.yaml`

```yaml
schema_version: 1
kind: rl_train

seed: 12345
run_name: fixed_bank_300iter

vehicle_ref: configs/vehicles/sedan.yaml
dynamics_ref: configs/dynamics/linear_friction_circle_flat.yaml
env_ref: configs/envs/tracking.yaml
course_ref: configs/courses/random_bank.yaml
agent_ref: configs/agents/rsl_rl/ppo_tracking.yaml
runtime_ref: configs/runtime/local_paths.yaml

overrides:
  env:
    scene:
      num_envs: 256
  agent:
    runner:
      max_iterations: 300
      experiment_name: phase3_random_bank
```

例: `configs/experiments/classical/circle_baseline.yaml`

```yaml
schema_version: 1
kind: classical

vehicle_ref: configs/vehicles/sedan.yaml
dynamics_ref: configs/dynamics/linear_friction_circle_flat.yaml
course_ref: configs/courses/circle.yaml
controller_refs:
  lateral: configs/controllers/pure_pursuit.yaml
  longitudinal: configs/controllers/speed_pi.yaml
runtime_ref: configs/runtime/local_paths.yaml

run:
  duration_s: 25.0
  off_track_threshold_m: 1.0
  plan_K: 20
  projection_search_radius_samples: 80
```

`overrides` を許す場合も、override 先の key は schema に存在するものだけ許可する。unknown override は失敗させる。

## Entry Point の変更案

### `scripts/rl/train_ppo.py`

現在:

```powershell
python scripts/rl/train_ppo.py --course random_bank --num_envs 256 --max_iterations 300
```

変更後:

```powershell
python scripts/rl/train_ppo.py --config configs/experiments/rl/phase3_random_bank.yaml --headless
```

残す CLI:

- `--config`
- `--headless` / Isaac Lab app args
- `--resume`
- `--load_run`
- `--checkpoint`

削る CLI:

- `--course`
- `--random_path_cfg`
- `--num_envs`
- `--max_iterations`
- `--seed`
- `--experiment_name`
- `--run_name`

これらは YAML に移す。

### `scripts/rl/play.py`

変更後:

```powershell
python scripts/rl/play.py --config configs/experiments/rl/phase3_random_bank.yaml --play_config configs/runtime/play.yaml --headless
```

policy checkpoint 指定だけ CLI に残すか、`runtime/play.yaml` に寄せる。

### `scripts/sim/run_classical.py`

変更後:

```powershell
python scripts/sim/run_classical.py --config configs/experiments/classical/circle_baseline.yaml --headless
```

`--course`, `--mu`, `--target_speed`, `--radius`, `--pid_kp` などは削る。

## 実装順序

「確定した設計判断 / PR 戦略」で 4 段階に分割。各 PR 内の細粒ステップ:

### PR 1 — loader + schema + 雛形 (完了, `fc0bd93`)

- [x] `src/vehicle_rl/config/{loader.py, schema.py, __init__.py}` を追加。
- [x] `configs/{vehicles,dynamics,envs,courses,controllers,agents,runtime,experiments}/**.yaml` の雛形を全部置く (現行コードの default 値を転記)。
- [x] `tests/config/test_loader.py` (unittest, 25 ケース)。
- [x] runtime コードは未変更。既存 smoke は不変。

### PR 2 — vehicle / dynamics / course を wire

- `src/vehicle_rl/config/isaac_adapter.py` を追加: `make_sedan_cfg(vehicle_bundle)`, `make_simulator_kwargs(dynamics_bundle)`, `build_path(course_bundle, num_envs, device)`。
- `SEDAN_CFG` を factory 経由に置き換え (`assets/sedan.py` の数値定数を YAML から取る)。
- `VehicleSimulator.__init__` の数値 default 引数を削除し、adapter から必須引数として渡す。
- `waypoints.py` の各 generator default 引数を削除し、`build_path` 経由のみで使う。
- `random_path.yaml` → `configs/courses/random_path_generator.yaml` への内部移行 (古い path も当面残す)。
- `run_classical.py` を adapter 経由に更新するが、CLI は据え置き (PR 4 で `--config` 化)。
- 短い `train_ppo` smoke で挙動を確認。

### PR 3 — env / reward / PPO を wire

- `make_tracking_env_cfg(env_bundle, course_bundle, controller_bundle)` を adapter に追加。
- `TrackingEnvCfg` の class-level numeric default を全部削り、factory が埋める形にする。
- `observation_space` などの derived 値は YAML から計算する (`scalar_fields + plan_K * plan_channels`)。
- `tasks/tracking/agents/rsl_rl_ppo_cfg.py` を factory 化 (`make_ppo_runner_cfg(agent_bundle)`)。
- gym registry エントリポイントを「YAML を読んで factory を回す関数」に切り替える。
- 短い PPO 学習 (10 iter, 64 envs) で iteration が回ることを確認。

### PR 4 — scripts を `--config` 化、CLI 撤去、README 更新

- `scripts/rl/train_ppo.py`, `scripts/rl/play.py`, `scripts/sim/run_classical.py` を `--config` 単一入口に変更。
- 旧 CLI (`--course`, `--num_envs`, `--max_iterations`, `--seed`, `--experiment_name`, `--run_name`, `--random_path_cfg`, `--mu`, `--target_speed`, `--radius`, `--pid_kp` 等) を即時撤去。
- `dump_resolved_config(resolved, log_dir)` を train / classical の起動時に呼ぶ。
- `README.md` の旧 CLI 例を全部 `--config` 例に書き換え。
- `configs/random_path.yaml` 旧ファイルを削除 (PR 2 で導入した `configs/courses/random_path_generator.yaml` が置き換え済み)。
- 旧 cfg class や default 引数の残骸を `rg "default=|= [0-9]+\\.|: float =|: int ="` で棚卸し。
- 最終確認: `phase3_circle_stage0a` / `phase3_random_long` / `phase3_random_bank` の各 experiment YAML で短い学習を起動し、エラーなく iteration が進むこと。

## 注意点

- Isaac Lab の `@configclass` / task registry は cfg object を要求するため、YAML を直接渡すのではなく `make_tracking_env_cfg(bundle)` のような adapter で Isaac cfg を生成する。
- `TrackingEnvCfg` に class-level numeric defaults を残すと目的に反する。必要なら registry entry point を「cfg class」から「YAML で埋めた cfg を返す factory」に寄せる。
- `observation_space=39` のような derived dimension は YAML 直書きにしない。observation spec と `plan_K` から計算する。
- `PINION_MAX` は YAML に直接持たせず、`delta_max_rad` と `steering_ratio` から導出する。
- `random_path_generator.yaml` の `reset.end_margin_extra_m` は geometry というより reset policy に近い。将来的には `env.reset.open_path_end_margin_extra_m` に移して、random path generator から切り離す方が分かりやすい。
- tiny eps (`1e-6`, `1e-12`) も完全外部化するなら `configs/numerics.yaml` を追加する。ただし最初は物理・学習挙動に影響する eps だけ外に出し、純粋なゼロ割防止 eps はコード内の数式定数として残す方が実装負荷は小さい。

## 完了条件

- train / play / classical の主要 CLI から実験条件 override が消えている。
- `TrackingEnvCfg`, `VehicleSimulator`, `SEDAN_CFG`, PPO cfg に tunable numeric default が残っていない。
- missing YAML key と unknown YAML key がテストで落ちる。
- log dir に resolved config 一式が保存される。
- 既存の `phase3_circle_stage0a`, `random_long`, `random_bank`, classical baseline が YAML だけで再現できる。
- 最終 PR で `train_ppo.py --config <experiment YAML>` の短い PPO 学習が成立することを目視確認 (E2E guard CSV/JSON 比較は使わない)。

## Stages

各 PR を 1 stage として扱う。各 stage は単独で main にマージ可能 (loader だけ入って scripts は旧式、を許容する) 設計。詳細は本文「## 実装順序」を参照。本セクションは agent-coop skill が要求する形式に合わせた要約であり、矛盾があれば「## 確定した設計判断」 + 「## 実装順序」を優先する。

### Stage 1: PR 1 — loader + schema + 雛形 (完了, fc0bd93)

- Scope: `src/vehicle_rl/config/{loader.py, schema.py, __init__.py}` を新設し、PyYAML SafeLoader + duplicate-key rejection + 自作 dataclass schema + 自作 strict validator を実装。`<name>_ref` 解決 (repo-root 相対のみ、推移参照、cache、cycle 検出、escape 拒否)。`overrides:` deep-merge (unknown key / ref-in-overrides / 非 mapping target すべて raise、list 丸ごと差し替え)。`load_experiment` = load → resolve_refs → deep_merge_overrides。`dump_resolved_config` で resolved bundle を log_dir に dump。すべての category / experiment YAML 雛形を新設 (現行コード default を転記)。runtime コードは未変更。
- Files:
  - `src/vehicle_rl/config/__init__.py`
  - `src/vehicle_rl/config/loader.py`
  - `src/vehicle_rl/config/schema.py`
  - `configs/vehicles/sedan.yaml`
  - `configs/dynamics/linear_friction_circle_flat.yaml`
  - `configs/envs/tracking.yaml`
  - `configs/courses/{circle,s_curve,lemniscate,dlc,random_long,random_bank,random_path_generator}.yaml`
  - `configs/controllers/{pure_pursuit,speed_pi}.yaml`
  - `configs/agents/rsl_rl/ppo_tracking.yaml`
  - `configs/runtime/{local_paths,play,train_video,classical_video}.yaml`
  - `configs/experiments/rl/phase3_{circle_stage0a,random_long,random_bank,random_bank_play}.yaml`
  - `configs/experiments/classical/{circle,s_curve,dlc}_baseline.yaml` および `circle_refactor_guard.yaml`
  - `tests/config/test_loader.py` (25 unittest cases)
- Fast-test subset: `python -m unittest tests.config.test_loader -v` (25 ケース全 PASS)。runtime imports に regression なし。
- Exit condition: 上記 unit test 25 ケース緑 + main の既存 smoke (`scripts/sim/run_classical.py --course circle`, `scripts/rl/train_ppo.py --course circle --num_envs 32 --max_iterations 1`) が PR 1 マージ後も同じ挙動 (旧 CLI のまま) で動く。

### Stage 2: PR 2 — vehicle / dynamics / course を YAML 化

- Scope: `src/vehicle_rl/config/isaac_adapter.py` を新設し `make_sedan_cfg(vehicle_bundle)`, `make_simulator_kwargs(dynamics_bundle)`, `build_path(course_bundle, num_envs, device)` を実装。`SEDAN_CFG` を YAML factory 化、`VehicleSimulator.__init__` の数値 default 引数を削除し adapter から必須引数として渡す、`waypoints.py` 各 generator の default 引数も削除。`scripts/sim/run_classical.py` を adapter 経由に切替 (CLI は据え置き、PR 4 で `--config` 化)。`configs/random_path.yaml` → `configs/courses/random_path_generator.yaml` への内部移行 (旧 path も当面は併存)。derived 値 (`pinion_max = delta_max_rad * steering_ratio` 等) は YAML に重複させずコードで計算。
- Files:
  - 新規: `src/vehicle_rl/config/isaac_adapter.py`
  - 修正: `src/vehicle_rl/assets/sedan.py` (SEDAN_CFG factory 化、module-level numeric default 撤去)
  - 修正: `src/vehicle_rl/envs/simulator.py` (`VehicleSimulator.__init__` default 引数撤去)
  - 修正: `src/vehicle_rl/envs/types.py` (`A_X_TARGET_MIN/MAX` 等の dynamics 定数を adapter 経由に)
  - 修正: `src/vehicle_rl/envs/waypoints.py` (各 generator の default 引数撤去)
  - 修正: `scripts/sim/run_classical.py` (adapter 経由化、CLI 据え置き)
  - テスト: `tests/config/test_isaac_adapter.py` (新設、vehicle/dynamics/course bundle → factory 結果の値検証)
- Fast-test subset: `python -m unittest tests.config.test_loader tests.config.test_isaac_adapter -v` 全 PASS。`python scripts/sim/run_classical.py --course circle --duration_s 2.0 --headless` smoke (短時間完走、stdout が "off-track threshold" 等の expected メトリクスを出力)。
- Exit condition: 上記 fast-test 全緑 + smoke 完走 + `rg "default=|: float =|: int =" src/vehicle_rl/{assets/sedan.py,envs/simulator.py,envs/waypoints.py,envs/types.py}` で PR 2 のスコープ内ファイルから tunable numeric default が消えている (`math.pi`, tensor index, 単位変換 0/1/2 等は許容)。

### Stage 3: PR 3 — env / reward / PPO を YAML 化

- Scope: `make_tracking_env_cfg(env_bundle, course_bundle, controller_bundle)` を adapter に追加。`TrackingEnvCfg` の class-level numeric default を全削除、factory が埋める形にする。`observation_space` 等の derived 値は YAML から計算 (`scalar_fields + plan_K * plan_channels`)。`tasks/tracking/agents/rsl_rl_ppo_cfg.py` を factory 化 (`make_ppo_runner_cfg(agent_bundle)`)。gym registry エントリポイントを「YAML を読んで factory を回す関数」に切替。reward / termination / planner / scene / spaces のすべて (`tracking.yaml` 配下) が YAML 由来になる。
- Files:
  - 修正: `src/vehicle_rl/config/isaac_adapter.py` (env / agent factory 追加)
  - 修正: `src/vehicle_rl/tasks/tracking/cfg.py` (or 該当 cfg ファイル) — `TrackingEnvCfg` の numeric default 撤去、factory に分離
  - 修正: `src/vehicle_rl/tasks/tracking/agents/rsl_rl_ppo_cfg.py` (factory 化)
  - 修正: gym registry 関連 (`src/vehicle_rl/tasks/__init__.py` 等)
  - テスト: `tests/config/test_isaac_adapter.py` 拡充 (env / agent factory のフィールド検証 + override 適用検証)
- Fast-test subset: 上記 unit test 全緑。短い PPO 学習 smoke (10 iter, 64 envs, headless) で iteration が回り NaN が出ない。
- Exit condition: fast-test 全緑 + 短 PPO smoke 完走 + `rg "default=|: float =|: int =" src/vehicle_rl/tasks/tracking/{cfg.py,agents/rsl_rl_ppo_cfg.py}` で残骸ゼロ。

### Stage 4: PR 4 — scripts を `--config` 化、CLI 撤去、README 更新

- Scope: `scripts/rl/train_ppo.py`, `scripts/rl/play.py`, `scripts/sim/run_classical.py` を `--config` 単一入口に変更。旧 CLI (`--course`, `--num_envs`, `--max_iterations`, `--seed`, `--experiment_name`, `--run_name`, `--random_path_cfg`, `--mu`, `--target_speed`, `--radius`, `--pid_kp` 等の実験条件 override) を即時撤去 (deprecation 期間なし)。残す CLI は `--config`, `--headless`, `--resume`, `--load_run`, `--checkpoint`, Isaac Lab AppLauncher 系のみ。`dump_resolved_config(resolved, log_dir)` を train / classical の起動時に呼ぶ。`README.md` の旧 CLI 例を全 `--config` 例に書き換え。`configs/random_path.yaml` 旧ファイルを削除 (PR 2 で `configs/courses/random_path_generator.yaml` が置き換え済み)。
- Files:
  - 修正: `scripts/rl/train_ppo.py`
  - 修正: `scripts/rl/play.py`
  - 修正: `scripts/sim/run_classical.py`
  - 修正: `README.md`
  - 削除: `configs/random_path.yaml`
  - テスト: `tests/scripts/test_train_cli.py`, `tests/scripts/test_play_cli.py`, `tests/scripts/test_run_classical_cli.py` (新設、argparse 構造 + 旧 CLI が消えたこと + `--config` のみで起動できることを smoke 確認)
- Fast-test subset: 全 unit test + 各 script の `--help` smoke + `python scripts/rl/train_ppo.py --config configs/experiments/rl/phase3_circle_stage0a.yaml --headless` を 1-2 iter で停止 (smoke)。
- Exit condition: fast-test 全緑 + 旧 CLI が argparse から消えている (`rg "\\b--course\\b|\\b--num_envs\\b|\\b--max_iterations\\b|\\b--mu\\b|\\b--target_speed\\b|\\b--radius\\b|\\b--pid_kp\\b" scripts/` で残骸ゼロ) + README が `--config` 例に更新済み + Phase 6 で `phase3_circle_stage0a` / `phase3_random_long` / `phase3_random_bank` の各 experiment YAML で短い PPO 学習が成立 (E2E guard CSV/JSON 比較は使わない、目視 + iteration が回ることだけを確認)。
