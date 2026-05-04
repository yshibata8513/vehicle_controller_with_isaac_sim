# Phase 3 Random Path Phase 1 実装レビュー

対象:

- `src/vehicle_rl/planner/path.py`
- `src/vehicle_rl/planner/random_path.py`
- `src/vehicle_rl/envs/tracking_env.py`
- `scripts/rl/train_ppo.py`
- `scripts/sim/run_classical.py`
- `configs/random_path.yaml`

確認:

```powershell
python -m compileall src\vehicle_rl\planner src\vehicle_rl\envs scripts\rl\train_ppo.py scripts\sim\run_classical.py
```

結果: 構文チェックは通過。

## 指摘事項

### 1. random path の可変 `v` が reward / progress に反映されていない

該当:

- `src/vehicle_rl/envs/tracking_env.py`
- `_update_progress()`
- `_get_rewards()`

現状:

- `random_long` では path の `v` が segment ごとに変わる。
- PI controller と reset warm-start は local path speed を使っている。
- しかし progress 正規化と speed reward はまだ `cfg.target_speed` 固定。

問題:

正しく local target speed に追従しても、`cfg.target_speed` と違う区間では speed penalty が入る。progress reward も区間速度に対して正規化されない。

修正案:

- `_compute_path_state()` で現在の target speed を保持する。

```python
self._last_target_speed = self._last_plan.v[:, 0]
```

- `_update_progress()` を以下に変更する。

```python
target_v = self._last_target_speed.clamp(min=1e-3)
progress_norm = delta_s / (target_v * self._control_dt)
```

- `_get_rewards()` を以下に変更する。

```python
speed_err = obs.vx - self._last_target_speed
```

- built-in course では `path.v` が一様なので挙動は従来と同等。

### 2. YAML の projection 設定が実際には使われていない

該当:

- `configs/random_path.yaml`
- `src/vehicle_rl/envs/tracking_env.py`

現状:

- `random_path.yaml` に以下がある。

```yaml
projection:
  search_radius_samples: 80
  recovery_radius_samples: 400
  max_index_jump_samples: 120
```

- しかし `TrackingEnv` は `TrackingEnvCfg.projection_search_radius_samples` と `TrackingEnvCfg.projection_max_index_jump_samples` の default 値を使っている。
- YAML 値を変更しても training 挙動に反映されない。

問題:

計画書では random path の projection hyperparameter を YAML で管理する想定。現状は設定が二重化しており、YAML を調整しても効かない。

修正案:

- `random_long` 時に YAML の値を runtime member に反映する。

```python
self._projection_search_radius_samples = rp_cfg.projection.search_radius_samples
self._projection_max_index_jump_samples = rp_cfg.projection.max_index_jump_samples
```

- built-in course では `TrackingEnvCfg` の値を使う。

```python
self._projection_search_radius_samples = self.cfg.projection_search_radius_samples
self._projection_max_index_jump_samples = self.cfg.projection_max_index_jump_samples
```

- `_compute_path_state()` と idx jump diagnostic はこの runtime member を参照する。

### 3. `max_heading_change_rad` が turn 全体ではなく arc 部分にしか効いていない

該当:

- `src/vehicle_rl/planner/random_path.py`
- `_sample_segment_kappa_length()`

現状:

```python
L_arc_cap = seg.max_heading_change_rad / max(abs(kappa_max), 1e-6)
L_arc = min(L_arc, L_arc_cap)
```

問題:

clothoid up/down も heading change を発生させる。現在の実装では arc 部分だけを cap しているため、turn 全体の heading change は `max_heading_change_rad` を超える。

例:

- `R = 30 m`
- `L_clo = 40 m`
- `kappa_max = 1 / 30`

clothoid up/down だけでおおよそ以下の heading change が入る。

```text
0.5 * kappa_max * L_clo * 2 = kappa_max * L_clo ≈ 1.33 rad
```

この時点で `max_heading_change_rad = 1.2` を超える。

修正案:

turn segment の主制約は、arc 部分だけではなく、以下の 3 区間を合計した **turn 全体の方位角変化** に置く。

```text
clothoid up -> arc -> clothoid down
```

ここで `L_clo` は片側クロソイド長、`L_arc` は円弧長、`kappa_max` は最大曲率とする。クロソイドは曲率が `0 -> kappa_max`、または `kappa_max -> 0` に線形変化するため、片側クロソイドの平均曲率は `0.5 * kappa_max`。

したがって heading change は以下になる。

```text
heading_up        = 0.5 * abs(kappa_max) * L_clo
heading_arc       =       abs(kappa_max) * L_arc
heading_down      = 0.5 * abs(kappa_max) * L_clo

heading_clothoids = heading_up + heading_down
                  =       abs(kappa_max) * L_clo

heading_total     = heading_clothoids + heading_arc
                  =       abs(kappa_max) * (L_clo + L_arc)
```

注意:

- `L_clo` は片側クロソイド長。
- 実際の turn 全体の heading change は `abs(kappa_max) * (2 * L_clo + L_arc)` ではない。
- up/down の 2 本があるが、それぞれ平均曲率が `0.5 * kappa_max` なので、両側合計で `abs(kappa_max) * L_clo` になる。

最小修正案:

現状の `clothoid_length_m` / `arc_length_m` sample 方針をなるべく維持するなら、sample 後に turn 全体で cap / reject する。

```text
heading_clothoids = abs(kappa_max) * L_clo
heading_budget_for_arc = max_heading_change_rad - heading_clothoids
```

- `heading_budget_for_arc <= 0` の場合は、その turn segment を reject / resample する。
- `L_arc_cap = heading_budget_for_arc / abs(kappa_max)` とする。
- `L_arc = min(L_arc, L_arc_cap)` とする。
- `L_arc_cap` が `ds` 未満など極端に短い場合も reject / resample する。

より良い設計案:

最初から `heading_total` を sample し、それを clothoid と arc に配分する方が安定する。

YAML 例:

```yaml
segments:
  turn_heading_change_rad: [0.2, 1.2]
  clothoid_heading_fraction: [0.25, 0.55]
  min_radius_m: 30.0
  max_radius_m: 150.0
```

生成手順:

```text
1. R を sample
2. kappa_abs = 1 / R
3. heading_total を [min, max] から sample
4. clothoid_fraction を sample
5. heading_clothoids = heading_total * clothoid_fraction
6. heading_arc = heading_total - heading_clothoids
7. L_clo = heading_clothoids / kappa_abs
8. L_arc = heading_arc / kappa_abs
```

この設計なら `heading_total <= max_heading_change_rad` が構造的に保証される。クロソイド長を直接 sample してから後で削るより、道路分布が読みやすい。

補助制約として、クロソイドの滑らかさを `max_dkappa_ds` で制御してもよい。

```text
dkappa_ds = abs(kappa_max) / L_clo
dkappa_ds <= max_dkappa_ds
```

クロソイドは「方位角を変える」ためだけではなく「曲率を滑らかに立ち上げる」ための区間なので、個別の heading change 閾値より `dkappa_ds` 制約の方が自然。

### 4. `v_curve_limit < v_min` の segment を reject していない

該当:

- `src/vehicle_rl/planner/random_path.py`
- `_segment_speed()`

現状:

```python
if v_sample_max <= s.v_min:
    return s.v_min
```

問題:

曲率から求めた制限速度が `v_min` より低い場合でも `v_min` で走らせるため、`ay_limit` を破る segment が生成される。

計画書では、原則として segment を reject して再 sample し、retry 上限時のみ `v_min` とする方針。

修正案:

- `_segment_speed()` は `v_curve_limit < v_min` を呼び出し側へ知らせる。
- `_sample_segment...` または `random_clothoid_path()` 側で segment を reject / retry する。
- retry 上限を YAML に追加してもよい。

例:

```yaml
speed:
  max_resample_attempts: 50
```

## 優先度

1. `target_speed` 固定 reward / progress の修正
2. YAML projection 設定の反映
3. turn 全体の heading change cap
4. `v_curve_limit < v_min` の reject / retry

## 追加で確認したいこと

- `random_long` の short smoke training が起動すること。
- `course=circle` で既存 Stage 0a の reward 曲線が大きく変わらないこと。
- `scripts/sim/run_classical.py --course circle` が `Path.project()` API 変更後も従来相当の metrics を出すこと。
- `Episode_PathProj/idx_jump_violation_rate` が通常コースで 0 に近いこと。
