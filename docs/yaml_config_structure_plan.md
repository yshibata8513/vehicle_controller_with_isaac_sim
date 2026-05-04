# YAML Config 外部化 設計案

目的:

- 車両諸元、力学パラメータ、環境、コース、報酬、PPO、classical controller、実験選択を YAML に集約する。
- コード内の tunable な定量値と course selection をなくし、YAML 指定を必須にする。
- コード側は schema / validation / formula / wiring だけを持つ。
- `course=circle` か `random_bank` か、reward weight、PPO horizon、車両 mass などはすべて実験 YAML から辿れるようにする。

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
  ref: configs/controllers/speed_pi.yaml

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

1. `docs/yaml_config_structure_plan.md` を合意する。
2. `src/vehicle_rl/config/loader.py` と `schema.py` を追加する。
3. `configs/vehicles/sedan.yaml` と asset factory を作る。
4. `configs/dynamics/linear_friction_circle_flat.yaml` を作り、`VehicleSimulator` の default 引数を削る。
5. `configs/courses/*.yaml` を作り、`waypoints.py` の default 引数を削る。
6. `configs/envs/tracking.yaml` を作り、`TrackingEnvCfg` の数値 default を削る。
7. `configs/agents/rsl_rl/ppo_tracking.yaml` を作り、PPO cfg class を YAML factory 化する。
8. `configs/experiments/**/*.yaml` を作り、train / play / classical scripts を `--config` 起点に変更する。
9. `rg "default=|= [0-9]|: float =|: int ="` で残った default / 数字を棚卸しする。
10. smoke:
    - random path generator
    - `run_classical.py --config ...`
    - `train_ppo.py --config ... --max smoke equivalent` ではなく、smoke 用 experiment YAML を別途作る。

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
