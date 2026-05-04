# Phase 3 呼び出し階層 (main → 学習ループ → 物理 1 ステップ)

`scripts/rl/train_ppo.py` の `main()` を起点に、PPO 学習が PhysX を 1 ステップ進めるところまで、どのファイルのどの関数が何のために呼ばれるかを順に追う。Stage 0a (`Vehicle-Tracking-Direct-v0`, 円コース, μ=0.9, steering-only) を前提とする。

呼び出し階層 (要約)

```
main()                                   scripts/rl/train_ppo.py
├─ AppLauncher(args_cli).app             # Isaac Sim 起動 (Kit)
├─ import vehicle_rl.tasks               # gym.register("Vehicle-Tracking-Direct-v0")
├─ parse_env_cfg(...)                    # TrackingEnvCfg を組み立てる
├─ _load_agent_cfg(...)                  # TrackingPPORunnerCfg を取得
├─ _apply_cli_overrides(...)             # CLI で num_envs/seed/iters 等を上書き
├─ gym.make("Vehicle-Tracking-Direct-v0")
│   └─ TrackingEnv.__init__              src/vehicle_rl/envs/tracking_env.py
│       ├─ DirectRLEnv.__init__          # シーン構築・gym spaces・SimulationContext
│       │   └─ TrackingEnv._setup_scene  # Sedan Articulation + GroundPlane + Light
│       ├─ VehicleSimulator(...)         src/vehicle_rl/envs/simulator.py
│       │   ├─ FirstOrderLagActuator     # pinion / a_x の 1 次遅れ
│       │   ├─ FixedRatioSteeringModel   # pinion → tire 角
│       │   ├─ StaticNormalLoadModel     # 軸/輪荷重 (ロール・ピッチ寄与)
│       │   ├─ LinearFrictionCircleTire  # Fy/Fx (摩擦円飽和)
│       │   └─ AttitudeDamper            # ロール/ピッチ仮想ダンパ
│       ├─ self._build_path()            # Stage 0: circle_path() 共有 Path
│       ├─ PIDSpeedController            # steering_only 用の縦速度 PI
│       ├─ _episode_sums / _episode_diag_sums / _sum_step_count を確保
│       └─ self._reset_idx(_ALL_INDICES) # 初期姿勢 (random_reset_along_path)
├─ RslRlVecEnvWrapper(env, ...)          isaaclab_rl.rsl_rl
└─ OnPolicyRunner.learn(num_iterations, init_at_random_ep_len=True)
    └─ for iter in range(max_iterations):
        ├─ rollout (num_steps_per_env=64 step):
        │   ├─ actor.act(obs)                              # rsl_rl 内
        │   └─ env.step(action)                            # ↓ 下に展開
        └─ ppo update (num_learning_epochs=5, mini_batches=4)
```

---

## 1. プロセス起動と Sim Kit の初期化

| 場所 | 関数 / 行 | 役割 |
|---|---|---|
| `scripts/rl/train_ppo.py:25` | `from isaaclab.app import AppLauncher` | Isaac Sim を埋め込み起動するためのモジュール |
| `scripts/rl/train_ppo.py:46-47` | `AppLauncher.add_app_launcher_args(parser)` / `args_cli = parser.parse_args()` | `--headless`, `--device`, `--enable_cameras` などの公式 CLI を共有 |
| `scripts/rl/train_ppo.py:52-53` | `app_launcher = AppLauncher(args_cli)` / `simulation_app = app_launcher.app` | Kit を起動。**この行より上では Isaac Lab 由来の重い import を絶対に書かない** (起動順序ロック) |
| `scripts/rl/train_ppo.py:56-64` | `rsl-rl-lib` バージョン検証 | 3.0.1 未満なら exit (Cartpole リファレンスと挙動を揃える) |
| `scripts/rl/train_ppo.py:82` | `import vehicle_rl.tasks  # noqa: F401` | これだけで `gym.register("Vehicle-Tracking-Direct-v0", ...)` が走る (`src/vehicle_rl/tasks/tracking/__init__.py`) |

## 2. 設定の組み立て

| 場所 | 関数 | 役割 |
|---|---|---|
| `scripts/rl/train_ppo.py:126-131` | `parse_env_cfg(task, device, num_envs, use_fabric=True)` | `TrackingEnvCfg` を生成 (環境側 cfg)。`num_envs` を CLI で上書き、`use_fabric=True` で Fabric クローン経路を使う |
| `scripts/rl/train_ppo.py:132 / 91-94` | `_load_agent_cfg(task)` → `load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")` | `Vehicle-Tracking-Direct-v0` に紐付いた `TrackingPPORunnerCfg` (PPO ハイパーパラ) を取得 |
| `scripts/rl/train_ppo.py:97-122` | `_apply_cli_overrides(env_cfg, agent_cfg, args)` | `--num_envs --device --max_iterations --seed --resume` 等を両 cfg に反映。最後に `env_cfg.seed = agent_cfg.seed` で seed を一意化 |
| `src/vehicle_rl/tasks/tracking/__init__.py` | `gym.register(id="Vehicle-Tracking-Direct-v0", entry_point="vehicle_rl.envs.tracking_env:TrackingEnv", kwargs=...)` | gym ID と (a) 環境クラス、(b) `TrackingEnvCfg` のフルパス、(c) `TrackingPPORunnerCfg` のフルパスを束ねる |
| `src/vehicle_rl/tasks/tracking/agents/rsl_rl_ppo_cfg.py:20-56` | `TrackingPPORunnerCfg` | `num_steps_per_env=64`, `max_iterations=300`, actor/critic [256,256] ELU, gamma=0.995, init_noise_std=0.3 など Stage 0a 既定 |

## 3. ログ出力先と env 構築

| 場所 | 行 | 役割 |
|---|---|---|
| `scripts/rl/train_ppo.py:135-145` | `log_root_path = .../logs/rsl_rl/<experiment_name>` → 日付 + `run_name` で `log_dir` 生成 | TensorBoard / params dump 出力先。`env_cfg.log_dir` にも同じ値を書き戻し、env から後で参照可能にする |
| `scripts/rl/train_ppo.py:147` | `gym.make(task, cfg=env_cfg, render_mode=...)` | gym 経由で `TrackingEnv(cfg)` を構築 |
| `scripts/rl/train_ppo.py:152-161` | `gym.wrappers.RecordVideo(env, ...)` (`--video` 時のみ) | `step_trigger` で `video_interval` step 毎に MP4 録画 |
| `scripts/rl/train_ppo.py:163` | `RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)` | DirectRLEnv → rsl_rl が期待する `VecEnv` I/F に変換。観測を `obs_dict["policy"]` から取り出し、行動を `[-1, 1]` にクリップ |

## 4. 環境コンストラクタの中身 (`TrackingEnv.__init__`)

`src/vehicle_rl/envs/tracking_env.py:159-270`

| ステップ | 役割 |
|---|---|
| 165-172 行 | **`steering_only` ↔ `action_space` 整合チェック** (review item E)。不一致なら `ValueError` で即停止 (使われない throttle channel に jerk penalty が積もる事故を防ぐ) |
| 174 行 | `super().__init__(cfg, render_mode, **kwargs)` — `DirectRLEnv` 側で `SimulationContext` 構築・`_setup_scene()` 呼び出し・gym spaces 構築・`episode_length_buf` 等のフレームワーク buffer 確保 |
| `_setup_scene` (275-289 行) | Sedan `Articulation`、GroundPlane、DomeLight を作成し、`InteractiveScene` で num_envs 分クローン (`replicate_physics=True`, `clone_in_fabric=True`) |
| 179-182 行 | `VehicleSimulator(self.sim, self.sedan, mu_default=0.9)` — case-B 物理コア (アクチュエータ + 法線荷重 + タイヤ + 姿勢ダンパ) を生成 |
| 187 行 | `self.path = self._build_path()` — Stage 0 は `circle_path(radius=30, target_speed=10, num_envs=N)` で全 env 共有の Path テンソルを生成 |
| 189-193 行 | `path.start_pose` から初期 (env-local) 位置 / yaw を取得しキャッシュ |
| 198-199 行 | `_last_action = zeros(N, action_dim)` — rate / jerk penalty 用の前ステップ行動 |
| 208-209 行 | `_s_prev = zeros(N)` (後段 `_reset_idx` で `path.s[idx]` に上書き)、`_control_dt = sim.dt * decimation = 0.02s` |
| 215-223 行 | `steering_only=True` なら `PIDSpeedController(num_envs, dt=control_dt, kp=1.0, ki=0.3, ...)` を生成 (Stage 0a の縦速度内製 PI) |
| 227-234 行 | `_episode_sums` (7 個の reward 項) を per-env で確保 |
| 241-252 行 | `_episode_diag_sums` (lat_err, heading_err, vx, pinion, pinion_rate, progress_norm の 6 診断キー) を確保 (review item F) |
| 261-263 行 | `_sum_step_count` (per-env step カウンタ)。`init_at_random_ep_len=True` 時に `episode_length_buf` が iter 0 で乱数化されるのに対し、こちらは sums と必ずロックステップで進む — 平均値の分母として使う |
| 270 行 | `self._reset_idx(self.sedan._ALL_INDICES)` — 全 env を初期姿勢に置き、最初の `_get_observations` が有効な観測を返せるようにする |

## 5. PPO 学習ループ (`OnPolicyRunner.learn`)

`scripts/rl/train_ppo.py:165-176`

```
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=...)
runner.add_git_repo_to_log(__file__)
dump_yaml("params/env.yaml", env_cfg);  dump_yaml("params/agent.yaml", agent_cfg)
runner.learn(num_learning_iterations=agent_cfg.max_iterations,
             init_at_random_ep_len=True)
```

`OnPolicyRunner.learn` の中では `max_iterations` 回、以下を繰り返す (rsl-rl-lib 側):

1. **rollout**: `num_steps_per_env=64` 回、actor から行動を引き、`env.step(action)` を呼ぶ。
2. **PPO update**: 集めた `(obs, action, value, logp, reward, done)` で `num_learning_epochs=5 × num_mini_batches=4 = 20` 回の SGD。
3. **logging**: scalar (mean_reward, std_action, KL, lr) と `extras["log"]` をまとめて TensorBoard へ書き込み、必要なら `save_interval=50` で `model_*.pt` を保存。

`init_at_random_ep_len=True` は最初の rollout 開始時に `episode_length_buf` を per-env 乱数で初期化し、全 env が同時にタイムアウトしないように位相をずらす。

## 6. `env.step(action)` の中身 (1 制御ステップ = 4 物理サブステップ)

`isaaclab.envs.DirectRLEnv.step` の処理順 (要約):

```
DirectRLEnv.step(action):
  1) self._pre_physics_step(action)            # Python 側で行動を解釈
  2) for _ in range(decimation):               # = 4 回 (200 Hz 物理 / 50 Hz 制御)
       self._apply_action()                    # PhysX に外力・関節ターゲットを書く
       self.sim.step()                         # PhysX 1 step (dt = 1/200 s)
       self.scene.update(sim_dt)               # Articulation buffers を更新
  3) self.episode_length_buf += 1
  4) self.reset_terminated, self.reset_time_outs = self._get_dones()
  5) self.reward_buf = self._get_rewards()
  6) reset_env_ids = (self.reset_terminated | self.reset_time_outs).nonzero()
     if len(reset_env_ids) > 0: self._reset_idx(reset_env_ids)
  7) self.obs_buf = self._get_observations()
  return obs_buf, reward_buf, terminated, time_outs, extras
```

各 hook で何をしているかを `tracking_env.py` の実装から追う。

### 6.1 `_pre_physics_step(action)` — `tracking_env.py:318-365`

- `action.clamp(-1, 1)`
- pinion: `action[:,0] * pinion_action_scale=3.0 → ±3 rad`、その後 `±pinion_max≈9.78 rad` で安全クリップ。
- a_x:
  - `steering_only=True` のとき: `_last_obs.vx` を `PIDSpeedController` に渡して `target_speed=10 m/s` を追従する `a_x` を計算。
  - `False` のとき: `action[:,1]` を sign-aware にスケール (`>=0 → [0,+3]`、`<0 → [-5,0)`) — 0 入力 = 力ゼロ。
- 結果を `_action_pinion / _action_a_x / _current_action` にキャッシュ。**1 step あたり 1 回しか実行しない** (decimation サブステップでは再利用)。

### 6.2 `_apply_action()` — `tracking_env.py:367-372`

`VehicleAction(pinion_target, a_x_target)` を組み、`vsim.apply_action_to_physx(action)` を呼ぶ。

#### `VehicleSimulator.apply_action_to_physx` — `simulator.py:255-315`

1. `steer_act.step(pinion_target, dt)` / `drive_act.step(a_x_target, dt)` で **Python 側** に 1 次遅れアクチュエータを進める (PhysX は使わない)。
2. `steering.pinion_to_delta(pinion_actual)` でタイヤ角に変換。
3. `_read_vehicle_state(...)` で PhysX root_state_w → body 系へ変換。
4. `normal_load.compute(state)` で 4 輪の Fz、`tire.compute(state, Fz, mu, fx_cmd)` で Fx/Fy。
5. `aggregate_tire_forces_to_base_link(...)` で base_link への合力 / トルク、`attitude.compute(state)` でロール/ピッチダンパを加算。
6. `sedan.set_external_force_and_torque(...)` で外力を、`sedan.set_joint_position_target(...)` でステア関節ターゲットを **PhysX バッファに書き込む** (このメソッドは `sim.step()` を呼ばない)。
7. `_a_y_estimate = wz * v_long` を更新 (次ステップの normal_load 入力)。
8. キャッシュ `_Fz/_Fx/_Fy/_slip_angle/_ax_body/_ay_body` を更新 (後で `get_state()` が読む)。

これを `decimation=4` 回繰り返した後 `DirectRLEnv` 側が `sim.step()` と `scene.update(sim_dt)` を呼ぶ。

### 6.3 `_get_dones()` — `tracking_env.py:443-457`

1. `self._compute_path_state()` を呼んで現在姿勢をパスに射影。
   - `vsim.get_state()` で `VehicleStateGT` 取得 (PhysX root state + キャッシュ済み力/加速度)。
   - 世界座標位置から `scene.env_origins[:, :2]` を引いて env-local 化。
   - `path.project(pos_local, yaw, K=10, lookahead_ds=1.0)` で `(plan, lat_err, hdg_err, s_proj)` を取得。
   - `build_observation(state_gt, plan, lat_err, hdg_err)` で `VehicleObservation` を組み立て。
   - 結果を `_last_state_gt / _last_obs / _last_lat_err / _last_hdg_err / _last_plan / _last_pos_local / _last_s_proj` にキャッシュ。**`_s_prev` は触らない** (review item A)。
2. `terminated = (|lat_err| > 4 m) | (|roll| > 60°)`、`time_out = (episode_length_buf >= max_len - 1)`。
3. `_last_terminated` を `_get_rewards` 用にキャッシュして返す。

### 6.4 `_get_rewards()` — `tracking_env.py:459-528`

1. **`self._update_progress()`** — `_compute_path_state` がキャッシュした `_last_s_proj` から `delta_s = s_proj - _s_prev` を計算。
   - `path.is_loop` ならコース全長で wrap 補正 (周回境界を跨いだ符号反転を吸収)。
   - `cap = max(2|vx| dt, 2 ds)` で前後にクリップ (射影フット glitch を抑制)。
   - `progress_norm = clamp(delta_s / (target_speed * control_dt), -1, 1)` (review item D の clamp)。
   - `_s_prev = s_proj.detach()` と更新。**1 step につきここ 1 回だけ更新**。
2. 7 reward 項を計算: `progress / alive / lateral / heading / speed / action_rate (=pinion_rate + jerk) / termination`。
3. `_episode_sums[key] += r_*` で per-env per-項に積算。
4. 6 診断キー (`Episode_State/lat_err_abs`, `Episode_State/heading_err_abs`, `Episode_State/vx`, `Episode_Action/pinion_abs`, `Episode_Action/pinion_rate_abs`, `Episode_Progress/progress_norm`) を `_episode_diag_sums` に積算。
5. `_sum_step_count += 1`、`_last_action = _current_action.detach()`。
6. 合計 reward を返す。

### 6.5 `_reset_idx(env_ids)` — `tracking_env.py:567-675`

タイムアウト or 終了した env 集合を受け取り:

1. `ep_len = _sum_step_count[env_ids].clamp(min=1).float()` を分母に、`_episode_sums` / `_episode_diag_sums` を **per-step 平均** に正規化して `extras["log"]["Episode_Reward/<key>"]` / `extras["log"]["<diag_key>"]` に積み、当該 env の sums を 0 化 (review item C)。
2. `Episode_Termination/off_track_or_rollover` と `time_out` を `reset_terminated / reset_time_outs` から計算。
3. `super()._reset_idx(env_ids)` を呼んで framework 側 (`episode_length_buf` 等) をリセット。
4. `random_reset_along_path=True` なら `idx = randint(0, M)` でパス上の点を選び、`(x, y, psi, s)` を `path.x/y/psi/s[env_ids, idx]` から取得。`False` なら `path[0]` start pose。
5. ワールド系姿勢を組み立て (yaw → quat) て `vsim.reset(env_ids, initial_pose)` でアクチュエータと PhysX root pose を初期化。
6. `write_root_velocity_to_sim` で **目標速度方向に warm-start** (`v_start = target_speed`)。これを忘れると vx=0 から開始して `speed_err^2 = 100` のペナルティが actuator lag (200 ms) で取り戻せない。
7. `_last_action[env_ids] = 0`、`_s_prev[env_ids] = s_reset` (review item B; これがないと再 spawn 直後の `delta_s` が前 episode の s からの飛びになる)、`_pi_speed.reset(env_ids)` で内製 PI の積分項をクリア。

### 6.6 `_get_observations()` — `tracking_env.py:530-565`

1. `_compute_path_state()` を**もう 1 回**呼ぶ。reset 直後の env が「pre-reset 姿勢から作った観測」を返さないようにするため、ここでも **純粋な射影**として再投影する (`_s_prev` は触らない — item A の核)。
2. キャッシュした `obs_struct` から 39 次元の policy 観測 ([0:6] IMU、[6] pinion、[7:9] lat/hdg err、[9:39] plan_xyv) を `torch.cat` で組み、`{"policy": obs}` を返す。**ワールド系絶対姿勢は意図的に含めない** (review item 7; types.py 冒頭参照)。

---

## 7. 設定とテンソルの所在 (どこを触れば何が変わるか)

| 変えたいもの | 触る場所 |
|---|---|
| 並列 env 数 / コース半径 / 目標速度 / 制御周波数 | `TrackingEnvCfg` (`tracking_env.py:50-152`) |
| reward 重み・終了閾値 | 同上 (`rew_*`, `max_lateral_error`, `max_roll_rad`) |
| PPO ハイパーパラ (lr, clip, entropy, gamma, hidden dims) | `TrackingPPORunnerCfg` (`tasks/tracking/agents/rsl_rl_ppo_cfg.py`) |
| 観測内容 (ノイズ、追加チャネル) | `envs/sensors.py:build_observation`、`envs/types.py:VehicleObservation` |
| 物理 (タイヤ係数、actuator 時定数、ステア比) | `envs/simulator.py:VehicleSimulator.__init__`、`assets/sedan.py` |
| コース形状 | `envs/tracking_env.py:_build_path` + `planner/circle_path` 等 |
| Stage 0a → Stage 0b 切替 | `cfg.steering_only=False` & `cfg.action_space=2` (init の guard で整合チェック) |

---

## 8. ログとアーティファクトの出口

- TensorBoard scalar: `logs/rsl_rl/vehicle_tracking_direct/<timestamp>_<run_name>/`
  - `Train/mean_reward`, `Train/mean_episode_length`, `Loss/value_function`, `Loss/surrogate`, `Loss/learning_rate`, `Policy/mean_noise_std`
  - `Episode_Reward/{progress, alive, lateral, heading, speed, action_rate, termination}` (per-step 平均)
  - `Episode_State/{lat_err_abs, heading_err_abs, vx}` / `Episode_Action/{pinion_abs, pinion_rate_abs}` / `Episode_Progress/progress_norm`
  - `Episode_Termination/{off_track_or_rollover, time_out}`
- params dump: `<log_dir>/params/{env.yaml, agent.yaml}` (再現用)
- checkpoint: `<log_dir>/model_{0,50,100,...,max}.pt` (`save_interval=50`)
- video (`--video` 指定時): `<log_dir>/videos/train/rl-video-step-*.mp4`
- git snapshot: `<log_dir>/git/` (`runner.add_git_repo_to_log(__file__)` が記録)
