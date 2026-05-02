# Phase 3 PPO 学習レビューと改善方針

作成日: 2026-05-02

対象:
- `scripts/rl/train_ppo.py`
- `src/vehicle_rl/envs/tracking_env.py`
- `src/vehicle_rl/envs/simulator.py`
- `src/vehicle_rl/tasks/tracking/agents/rsl_rl_ppo_cfg.py`
- Stage 0 学習ログ v1 / v2 / v3

## 総評

Phase 3 の infrastructure はかなり良いところまで来ている。`DirectRLEnv`、`VehicleSimulator.apply_action_to_physx()` 分解、task registry、rsl_rl entry point は大筋で成立している。

報酬が伸びない主因は、実装が完全に壊れているというより、現在の Stage 0 が PPO の初期探索タスクとして難しすぎることにある。特に「操舵と加減速を同時に学ぶ」「毎回同じコース始点から reset」「報酬が alive + error penalty 中心」「rollout horizon が短い」の組み合わせで、policy が正しい円旋回にたどり着く前に off-track 終了しやすい。

## 観測された状況

- v1: reward が単調悪化。`a_x` の action mapping が `action=0 -> -1 m/s^2` になっており、初期 policy が常時ブレーキ気味になっていた。
- v2: sign-aware `a_x` mapping、warm-start `v=10 m/s`、alive reward 追加で episode length は伸びたが、reward は負で停滞。
- v3: `init_noise_std` を 1.0 から 0.3 へ下げ、100 iter 実行。episode length は 47 -> 341 step まで伸びたが、reward は正にならず。

これは「生存時間は学び始めているが、良い軌道追従の報酬勾配が弱い」状態に見える。

## 主要な問題

### 1. Stage 0 で操舵と加減速を同時に学ばせている

`tracking_env.py` では action が `[pinion_norm, a_x_norm]` の 2 次元になっている。

円コース・固定 target speed・warm-start なら、最初は加減速を policy から外した方がよい。現在は速度維持と操舵を同時に探索しており、速度ノイズが lateral error を悪化させ、操舵学習の信号を濁している。

推奨:
- Stage 0a は steering-only にする。
- `action_space = 1`
- `a_x_target = 0` 固定、または Phase 2 の PI controller を使う。
- 円追従が伸びた後、Stage 0b で `a_x` action を戻す。

### 2. reset 位置が毎回コース始点固定

`_reset_idx()` は常に `path[0]` へ車両を置いている。

早期に脱線する policy は円のごく短い区間しか経験できず、「円全体での追従」ではなく「始点から数秒生き延びる」学習になりやすい。

推奨:
- env ごとにランダムな `s` / waypoint index へ reset する。
- yaw は path tangent に合わせる。
- velocity はその yaw 方向に `target_speed` で warm-start する。
- 64 env でも円周上に経験分布が広がり、学習信号がかなり改善する。

### 3. 報酬に progress 成分がない

現在の reward は alive + lateral / heading / speed / action penalty が中心。

この形だと、悪い状態で長く生きると累積 reward が負になりやすく、「どちらへ動くと良いか」の方向づけも弱い。車両タスクでは、経路方向へ進んだこと自体を positive reward にした方が PPO に優しい。

推奨:

```text
reward =
    + 1.0 * progress_norm
    - 0.2 * lateral_error^2
    - 0.3 * heading_error^2
    - 0.01 * speed_error^2
    - 0.01 * action_rate^2
```

`alive=2.0` を大きくするより、`progress_norm = delta_s / (target_speed * control_dt)` を主報酬にする方が望ましい。

### 4. PPO rollout horizon が短い

`num_steps_per_env=24` は 50 Hz 制御で 0.48 秒分しかない。車両の円追従は数秒スケールなので、credit assignment が短すぎる可能性が高い。

推奨:
- `num_steps_per_env = 64` または `96`
- `gamma = 0.995` または `0.997`
- `num_envs = 128` 以上、可能なら 256
- `max_iterations = 300` 以上で傾向を見る

### 5. termination penalty が軽い

`rew_termination = -10` は、episode 全体の累積 reward と比べると軽め。悪い状態で数十 step 生きる負報酬と比較して、早期終了の悪さが十分に伝わっていない可能性がある。

推奨:
- `rew_termination = -50` から試す。
- もしくは Stage 0 では `max_lateral_error = 8.0` に緩め、timeout まで走らせて dense reward から形を覚えさせる。

### 6. action scale / exploration noise はまだ大きい可能性がある

`pinion_action_scale=3.0` は v1 より大幅に良い。ただし Stage 0 の円 r=30, v=10 では必要 pinion は約 1.4 rad なので、最初はさらに小さくしてよい。

推奨:
- Stage 0a: `pinion_action_scale = 2.0`
- `init_noise_std = 0.15`
- 学習が成立した後に `pinion_action_scale = 3.0`, `init_noise_std = 0.3` へ戻す

### 7. policy 観測に world / env-local の絶対 pose を入れない方がよい

現在の観測には `pos_xy_local` と `yaw` が含まれている。これは Stage 0 の固定 circle では一見便利に見えるが、policy が「この絶対位置ではこの操舵」という暗記をしやすくなる。

RL controller が本当に必要としているのは、世界座標のどこにいるかではなく、経路に対する相対状態である。

必要な情報:
- 車両運動: `vx`, `yaw_rate`, `ax`, `ay`, `roll`, `pitch`
- 操舵状態: `pinion_angle`
- 経路相対誤差: `lateral_error`, `heading_error`
- preview: body-frame `Plan.x[:K]`, `Plan.y[:K]`, `Plan.v[:K]`

不要または policy から隠したい情報:
- `pos_xy_local`
- world / env-local の絶対 `yaw`

`heading_error` があれば world yaw は不要であり、body-frame lookahead と lateral error があれば absolute position も不要。GT / logging / metrics 用には保持してよいが、policy observation からは外す方が、random reset、multi-course、DLC、将来の domain randomization へ汎化しやすい。

推奨 observation layout:

```text
[vx, yaw_rate, ax, ay, roll, pitch,
 pinion_angle,
 lateral_error, heading_error,
 plan.x[0:K], plan.y[0:K], plan.v[0:K]]
```

K=10 なら次元は `6 + 1 + 2 + 30 = 39`。現在の 42 dim から `pos_xy_local` 2 dim と `yaw` 1 dim を抜く。

## 推奨実験順

### Stage 0a: steering-only circle

目的: まず「円を曲がる」だけを PPO に覚えさせる。

設定:
- course: circle
- reset: 円周上ランダム
- observation: absolute `pos_xy` / `yaw` を含めず、経路相対量 + body-frame plan のみにする
- action: pinion のみ
- `a_x_target = 0` 固定、または PI controller
- warm-start velocity: target speed along path tangent
- reward: progress 主体
- `pinion_action_scale = 2.0`
- `init_noise_std = 0.15`
- `num_steps_per_env = 64`
- `gamma = 0.995`
- `num_envs = 128` 以上

成功目安:
- episode length が timeout 近くまで伸びる
- mean reward が上昇傾向
- deterministic play で circle を 1 周できる

### Stage 0b: acceleration action を戻す

目的: 操舵が成立した policy に、速度維持の自由度を戻す。

設定:
- action: `[pinion, a_x]`
- `a_x` sign-aware mapping は維持
- speed penalty は弱め
- `a_x` action rate penalty を少し強める

成功目安:
- Stage 0a と同程度の lateral error
- target speed 近傍を保つ

### Stage 1: course variation

目的: circle で成立した後に s_curve / dlc を追加する。

設定:
- s_curve 追加
- dlc はさらに後
- lemniscate は projection 問題があるため stress test 扱い

## 追加したい診断ログ

現状は mean reward / episode length だけでは、何が報酬を食っているかが見えにくい。`_get_rewards()` で per-term logging を入れると、改善が速くなる。

推奨ログ:
- `Episode_Reward/alive`
- `Episode_Reward/progress`
- `Episode_Reward/lateral`
- `Episode_Reward/heading`
- `Episode_Reward/speed`
- `Episode_Reward/action_rate`
- `Episode_Termination/off_track_rate`
- `Episode_State/lat_err_mean`
- `Episode_State/heading_err_mean`
- `Episode_State/vx_mean`
- `Episode_Action/pinion_abs_mean`
- `Episode_Action/ax_abs_mean`

## 優先順位

1. steering-only Stage 0a を作る。
2. reset を円周上ランダムにする。
3. policy observation から absolute `pos_xy` / `yaw` を外す。
4. progress reward を入れる。
5. PPO horizon を `num_steps_per_env=64`, `gamma=0.995` にする。
6. per-term reward logging を入れる。
7. その後に `a_x` action を戻す。

## 補足

`python -m compileall src scripts` は成功。現時点では「コードが根本的に壊れている」というより、PPO に渡している最初のタスク設計が少し厳しい、という評価。

## 再レビュー追記（2026-05-02）

Stage 0a 対応後の `tracking_env.py` / observation schema / PPO cfg を再確認した結果、主要な改善項目はかなり反映されている。

反映確認:
- policy observation から absolute `pos_xy` / `yaw` は削除済み。
- observation は 39 dim (`6 + pinion + 2 errors + 3K`) に変更済み。
- Stage 0a として `steering_only=True`, `action_space=1` に変更済み。
- longitudinal は内部 PI controller に委譲済み。
- reset は path 上ランダム index 化済み。
- progress reward が追加済み。
- PPO cfg は `num_steps_per_env=64`, `gamma=0.995`, `num_envs=128`, `max_iterations=300` に変更済み。
- per-term reward logging が追加済み。
- `python -m compileall src scripts` は成功。

残る注意点と推奨修正:

### A. progress 更新の副作用を observation 取得から分離する

現在の `_post_physics_compute()` は path projection だけでなく、`_s_prev` 更新と `progress_norm` 計算も行う。さらに `_post_physics_compute()` は `_get_dones()` と `_get_observations()` の両方から呼ばれている。

DirectRLEnv の通常 step 順では大きく壊れにくいが、observation 取得が reward 用 state を更新する構造は fragile。debug / render / wrapper 側から observation が余分に呼ばれた時に、progress state が意図せず進む可能性がある。

推奨:
- `_compute_path_state()` は副作用なしで `state_gt`, `plan`, `lat_err`, `hdg_err`, `s_proj` を作るだけにする。
- `_update_progress(s_proj, vx)` を別関数にし、reward / done 用の一箇所だけで呼ぶ。
- `_get_observations()` は `_s_prev` を更新しない。

### B. random reset 後に `_s_prev` を reset 位置の `s` に明示セットする

random reset で `idx` を選んで reset しているが、`self._s_prev[env_ids]` を同じ `idx` の `path.s` に明示セットしていない。

現状は reset 後の `_get_observations()` が `_post_physics_compute()` を呼ぶことで結果的に揃いやすいが、呼び出し順依存になっている。reset 時に明示する方が堅い。

推奨:

```python
self._s_prev[env_ids_t] = self.path.s[env_ids_t, idx]
```

`random_reset_along_path=False` の場合も、`path[0]` の `s=0` を明示しておく。

### C. reward term logging の正規化を episode step 平均にする

現在の per-term logging は `episode_sum / max_episode_length_s` に近い形になっている。Isaac Lab の一部サンプルでは reward が `dt` スケール込みなので読めるが、現状の reward は `progress_norm ≈ 1.0 per control step` のような step 報酬である。

そのため、秒で割ると TensorBoard 上の値が直感より大きく見える可能性がある。学習自体には影響しないが、ログ解釈が難しくなる。

推奨:

```python
episode_len = self.episode_length_buf[env_ids_t].clamp(min=1).to(sums.dtype)
log_extras[f"Episode_Reward/{key}"] = (sums[env_ids_t] / episode_len).mean()
```

### D. `progress_norm` を reward 投入前に clamp する

progress cap は `abs(vx) * control_dt * 2.0` を許しているため、target speed 付近では `progress_norm` が最大で約 2 まで入り得る。projection jump や off-track からの復帰で progress reward が過大になる可能性がある。

Stage 0a ではまず「理想走行で +1」を保った方が読みやすい。

推奨:

```python
progress_norm = progress_norm.clamp(-1.0, 1.0)
```

必要なら後で `[-1.0, 2.0]` に緩める。

### E. `steering_only` と `action_space` の整合 guard を入れる

現在の default は `steering_only=True`, `action_space=1` で整合している。ただし後で cfg override した時に、ズレると静かに変な学習になる。

例: `steering_only=True` のまま `action_space=2` にすると、使っていない throttle action に jerk penalty だけがかかる可能性がある。

推奨:

```python
if self.cfg.steering_only and int(self.cfg.action_space) != 1:
    raise ValueError("steering_only=True requires action_space=1")
if (not self.cfg.steering_only) and int(self.cfg.action_space) != 2:
    raise ValueError("steering_only=False requires action_space=2")
```

### F. state / action 診断ログを追加する

per-term reward logging は追加済みで良い。ただし reward が伸びない時に、原因が lateral error、速度低下、操舵暴れ、progress 不足のどれかを即座に見分けるには state / action の episode 平均も欲しい。

追加候補:
- `Episode_State/lat_err_abs`
- `Episode_State/heading_err_abs`
- `Episode_State/vx`
- `Episode_Action/pinion_abs`
- `Episode_Action/pinion_rate_abs`
- `Episode_Progress/progress_norm`

### G. 古いコメントを整理する

挙動には影響しないが、`VehicleStateGT` まわりの docstring に「IMU/GPS-equivalent signals also exposed via Observation」のような、absolute pose を observation から外す前のニュアンスが残っている。

GPS / absolute pose は `VehicleStateGT` / metrics / reset 用には残すが、policy observation には入れない、という現在の方針に合わせてコメントを整理すると混乱が減る。

再レビュー後の優先順位:

1. progress 更新と observation 取得の副作用分離。
2. reset 時に `_s_prev` を `path.s[idx]` へ明示セット。
3. `steering_only` / `action_space` guard。
4. `progress_norm` clamp。
5. reward logging を episode step 平均に変更。
6. state / action 診断ログ追加。
7. 古いコメント整理。
