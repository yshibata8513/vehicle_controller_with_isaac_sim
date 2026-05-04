# Phase 3 ランダムパス実装計画

## 目的

Phase 3 の path tracking RL を、固定コースだけでなく、ランダム生成した長い道路、およびランダム道路 bank 上で学習できるようにする。

計算時間を最優先し、training hot loop では path を生成しない。path 生成は初期化時、または明示的な再生成タイミングだけで行う。projection は第一段階から「全点最近傍探索」ではなく「前回 nearest index 周辺の local window 探索」を使う。

## 対象外

- reset ごとに Python / NumPy で path を生成しない。
- reset ごとに CPU から GPU へ path 全体を転送しない。
- Phase 1 から path bank 再生成は入れない。
- 最初から self-intersection の完全解決はしない。生成側で過度な折り返しを避け、projection は nearest index window で安定化する。

## 共通設計

ランダム道路は曲率列 `kappa(s)` から生成する。

- 直線: `kappa = 0`
- クロソイド: `kappa` を arc length に対して線形変化
- 円弧: `kappa = const`

座標化は生成時だけ行う。

```text
psi[i+1] = wrap_to_pi(psi[i] + kappa[i] * ds)
x[i+1]   = x[i] + cos(psi[i]) * ds
y[i+1]   = y[i] + sin(psi[i]) * ds
```

生成後の tensor は GPU 上に保持する。RL step 中は `x/y/psi/s/v` の gather と、小さい window 内 argmin のみを行う。

## 設定ファイル

新規ファイル:

- `configs/random_path.yaml`

この YAML は Phase 1 から Phase 3 まで同じファイルを使う。各 phase で未使用の section は無視する。

必須 schema:

```yaml
generator:
  seed: 12345
  ds: 0.2
  target_speed: 10.0            # legacy / fallback 用。random path では speed.v_* を優先する。

speed:
  v_min: 5.0
  v_max: 12.0
  ay_limit: 4.0                 # 曲率から安全上限速度を決めるための横加速度上限
  segment_constant: true        # true: segment ごとに一定速度を sample する

segments:
  straight_length_m: [10.0, 80.0]
  min_radius_m: 30.0
  max_radius_m: 150.0
  turn_heading_change_rad: [0.2, 1.2]      # whole-turn heading change [rad]
  clothoid_heading_fraction: [0.25, 0.55]  # share of heading carried by clothoid pair
  straight_probability: 0.25
  turn_probability: 0.75
  reverse_turn_probability: 0.35

projection:
  search_radius_samples: 80
  recovery_radius_samples: 400
  max_index_jump_samples: 120

reset:
  random_reset_along_path: true
  end_margin_extra_m: 20.0

phase1_long_path:
  length_m: 20000.0
  is_loop: false

phase2_bank:
  num_paths: 1024
  length_m: 1000.0
  is_loop: false

phase3_regeneration:
  enabled: false
  interval_resets: 5000
  fraction: 0.05
  min_unused_slots: 16
```

検証ルール:

- `generator.ds` は `TrackingEnvCfg.course_ds` と一致させるか、明示的に override する。
- `speed.v_min > 0` かつ `speed.v_max >= speed.v_min`。
- `speed.ay_limit > 0`。
- `segments.min_radius_m > 0`。
- `phase1_long_path.length_m` は `speed.v_max * episode_length_s + lookahead horizon` より長いこと。
- `phase2_bank.num_paths >= scene.num_envs` が望ましい。ただし必須ではない。
- `projection.search_radius_samples > 0`。
- `projection.recovery_radius_samples >= projection.search_radius_samples`。

## Phase 1: 単一の長いランダムパス

`course="random_long"` オプションを追加する。env 初期化時に 1 本の長い open path を生成し、全 env に broadcast する。reset 時はその長い path 上の start index をランダムに選ぶ。

この段階で `Path.project()` は全点探索を廃止し、全コース共通で local window 探索に置き換える。`Path` class 自体は `nearest_idx` を状態として持たない。`TrackingEnv` や classical runner など、`project()` を呼ぶ側が `nearest_idx` を保持し、`project()` が返す `closest_idx` で次 step 用に更新する。

### 変更ファイル

#### `configs/random_path.yaml`

上記 schema の YAML ファイルを作成する。

Phase 1 で使う section:

- `generator`
- `speed`
- `segments`
- `projection`
- `reset`
- `phase1_long_path`

#### `src/vehicle_rl/planner/random_path.py`

追加するもの:

```python
@dataclass
class RandomPathGeneratorCfg:
    ...

def load_random_path_cfg(path: str | PathLike) -> RandomPathGeneratorCfg:
    ...

def random_clothoid_path(
    *,
    cfg: RandomPathGeneratorCfg,
    num_envs: int,
    length_m: float,
    is_loop: bool,
    device: torch.device | str,
) -> Path:
    ...
```

責務:

- YAML 由来の値を parse / validate する。
- `length_m` に達するまで segment 列を生成する。
- `kappa`, `s`, `x`, `y`, `psi`, `v` を構築する。
- shape `(num_envs, M)` の通常の `Path` を返す。
- `v` は segment ごとに random sample する。直線 / クロソイド / 円弧の各 segment に対して、まず曲率から許容上限速度を決め、その範囲内で segment 目標速度を sample する。
- segment の速度 sample 範囲は以下。

```text
kappa_segment_max = max(abs(kappa) over the segment)
v_curve_limit = sqrt(ay_limit / max(kappa_segment_max, eps))
v_sample_max = min(v_max, v_curve_limit)
v_segment = uniform(v_min, v_sample_max)
```

- `v_curve_limit < v_min` となる segment は、生成時点で曲率が厳しすぎる。原則としてその segment を reject して再 sample する。retry 上限に達した場合のみ `v_segment = v_min` とし、warning 用 counter を増やす。
- `speed.segment_constant=true` の場合、segment 内の `v` は `v_segment` で一定にする。クロソイド内で曲率が変化しても、その segment の最大曲率に基づく上限を使う。

#### `src/vehicle_rl/planner/path.py`

既存の全点探索版 `Path.project()` は廃止し、local window 版の `Path.project()` に置き換える。

新しい `project()`:

```python
def project(
    self,
    pos_xy: Tensor,
    yaw: Tensor,
    nearest_idx: Tensor,
    *,
    search_radius_samples: int,
    K: int = 10,
    lookahead_ds: float = 1.0,
) -> tuple[Plan, Tensor, Tensor, Tensor, Tensor]:
    ...
```

返り値:

```text
plan, lateral_error, heading_error, s_proj, closest_idx
```

実装要件:

- `Path` は `nearest_idx` を内部状態として保持しない。
- 呼び出し側が前回の `closest_idx` を `nearest_idx` として次 step に渡す。
- candidate index は `nearest_idx[:, None] + offsets[None, :]`。
- `offsets = [-W, ..., +W]`。
- loop path では modulo を使う。
- open path では `[0, M - 1]` に clamp する。
- argmin 対象は `(N, 2W + 1)` のみ。
- lookahead gather は `closest_idx` から開始する。
- hot path では `.item()`, `.cpu()`, `.numpy()`, env ごとの Python loop を使わない。
- 全点最近傍探索は production code から削除する。必要なら test / debug 用の slow reference helper としてのみ別名で持つ。

必要なら内部共通処理を切り出す。

```python
def _project_from_candidate_idx(...):
    ...
```

#### `src/vehicle_rl/planner/__init__.py`

export を追加する。

```python
from .random_path import RandomPathGeneratorCfg, load_random_path_cfg, random_clothoid_path
```

#### `src/vehicle_rl/envs/tracking_env.py`

`TrackingEnvCfg` を拡張する。

```python
course: str = "circle"  # "random_long" を追加
random_path_cfg_path: str = "configs/random_path.yaml"
projection_search_radius_samples: int = 80
projection_recovery_radius_samples: int = 400
```

実装内容:

- `_build_path()` で `course == "random_long"` を扱う。
- `course == "random_long"` のときは `random_path_cfg_path` を読み込み、1 本の `Path` を生成し、projection の既定値を YAML から設定する。
- 以下を追加する。

```python
self._nearest_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
```

- `_compute_path_state()` は全コースで local window 版 `Path.project()` を使う。

```python
plan, lat_err, hdg_err, s_proj, nearest_idx = self.path.project(
    pos_local_xy,
    yaw,
    self._nearest_idx,
    search_radius_samples=self.cfg.projection_search_radius_samples,
    K=self.cfg.plan_K,
    lookahead_ds=self.cfg.lookahead_ds,
)
self._nearest_idx = nearest_idx
```

- `_reset_idx()` では path sample から reset するときに `self._nearest_idx[env_ids_t] = idx` を設定する。
- open random path では、reset index の上限に episode 全体を走れるだけの margin を確保する。

```text
end_margin_m =
  speed.v_max * episode_length_s
  + plan_K * lookahead_ds
  + end_margin_extra_m
```

これを sample 数に変換し、`course == "random_long"` では現在の lookahead-only margin の代わりに使う。

#### `scripts/rl/train_ppo.py`

CLI を追加する。

```text
--course
--random_path_cfg
```

override 処理を追加する。

```python
if args.course is not None:
    env_cfg.course = args.course
if args.random_path_cfg is not None:
    env_cfg.random_path_cfg_path = args.random_path_cfg
```

#### `scripts/sim/run_classical.py`

`Path.project()` の返り値と引数変更に追従する。

実装内容:

- single-env 用の `nearest_idx` tensor を追加する。

```python
nearest_idx = torch.zeros(1, dtype=torch.long, device=device)
```

- control 用 projection では `closest_idx` を受け取り、次 step 用に更新する。

```python
plan, lat_err, hdg_err, s_proj, closest_idx = path.project(
    pos_xy,
    yaw,
    nearest_idx,
    search_radius_samples=projection_search_radius_samples,
    K=PLAN_K,
    lookahead_ds=LOOKAHEAD_DS,
)
nearest_idx = closest_idx
```

- logging 用に同じ step で再 projection する場合は、更新済み `nearest_idx` を渡す。logging のためだけに `nearest_idx` を別方向へ進めない。
- circle / s_curve / dlc / lemniscate の既存 smoke が動くことを確認する。

#### `src/vehicle_rl/envs/sensors.py`

docstring 内の `Path.project(pos_xy, yaw, K)` という古い説明を、新しい `nearest_idx` 付き API に合わせて更新する。挙動変更は不要。

#### `scripts/rl/play.py`

学習済み policy を random path で再生できるように、`--course` と `--random_path_cfg` の override を train 側と同様に追加する。

### Phase 1 完了条件

- default の `course=circle` の挙動が変わらない。
- `course=random_long` で、reset 中に CPU path 生成を行わず学習できる。
- circle / s_curve / dlc / random_long のすべてで `_compute_path_state()` が local projection を使う。
- TensorBoard logging が従来通り動く。
- open path の終端に近すぎる位置へ reset しない。
- smoke command:

```powershell
c:\work\isaac\env_isaaclab\Scripts\python.exe scripts\rl\train_ppo.py --task Vehicle-Tracking-Direct-v0 --course random_long --random_path_cfg configs\random_path.yaml --num_envs 128 --max_iterations 5 --headless
```

## Phase 2: ランダム Path Bank、再生成なし

`course="random_bank"` を追加する。env 初期化時に多数の random path を一度だけ生成する。各 env は現在割り当てられている bank index と `nearest_idx` を持つ。reset 時に bank index と start index の両方をランダムに選ぶ。Phase 2 では path slot の再生成は行わない。

Phase 2 の実装方針は **reset-copy 方式** とする。当初案では `path_id` による bank-aware gather を hot loop に入れて `(n_reset, M)` の active path copy を避ける想定だったが、`Path.project()` / lookahead / observation / plotting がすべて bank-aware になり実装が複雑化するため採用しない。代わりに reset 時だけ選択した bank row を env ごとの `self.path` row にコピーし、毎 step の projection は既存の `Path.project()` をそのまま使う。

### 変更ファイル

#### `src/vehicle_rl/planner/random_path.py`

追加するもの:

```python
@dataclass
class RandomPathBank:
    s: Tensor       # (P, M)
    x: Tensor       # (P, M)
    y: Tensor       # (P, M)
    v: Tensor       # (P, M)
    psi: Tensor     # (P, M)
    is_loop: bool
    ds: float
```

property:

```python
num_paths
num_samples
```

Phase 2 では bank-aware projection API は追加しない。

追加するもの:

```python
def random_clothoid_path_bank(
    *,
    cfg: RandomPathGeneratorCfg,
    num_paths: int,
    length_m: float,
    is_loop: bool,
    device: torch.device | str,
) -> RandomPathBank:
    ...
```

実装内容:

- 初期化時に `P` 本の path を生成する。
- tensor は `(P, M)` として保持する。
- path ごとに deterministic な seed offset を使う。

```text
seed_i = cfg.seed + i
```

- すべての path が同じ `M` になるようにする。`M = round(length_m / ds)` とし、正確に `M` samples を生成する。
- `phase2_bank.is_loop=true` は Phase 2 では未対応として `TrackingEnv` 側で拒否する。bank generator は幾何的に閉じた loop を生成しないため、open geometry を loop として扱うと終端 wrap が不連続になる。

#### `src/vehicle_rl/planner/__init__.py`

export を追加する。

```python
from .random_path import RandomPathBank, random_clothoid_path_bank
```

#### `src/vehicle_rl/envs/tracking_env.py`

`TrackingEnvCfg` を拡張する。

```python
course: str = "circle"  # "random_bank" を追加
random_path_cfg_path: str = "configs/random_path.yaml"
```

状態を追加する。

```python
self._is_bank: bool
self._path_bank: RandomPathBank | None
self._env_path_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
self._nearest_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
```

実装内容:

- `_build_path()` は通常 course / `random_long` / `random_bank` をすべて扱う。
- `course == "random_bank"` の場合、`RandomPathBank(P, M)` を作り、初期 placeholder として `bank[0]` を全 env row にコピーした `Path(N, M)` を返す。
- `__init__` 末尾の初回 `_reset_idx(_ALL_INDICES)` で各 env の実 bank row をサンプルし、`self.path[env]` を上書きする。
- `_compute_path_state()` は bank 分岐を持たず、常に既存の `self.path.project(...)` を使う。これにより projection hot loop を built-in course / `random_long` / `random_bank` で共通に保つ。
- bank mode の `_reset_idx()` は以下の流れにする。

```python
new_path_idx = torch.randint(0, self._path_bank.num_paths, (n_reset,), device=self.device)
self._env_path_idx[env_ids_t] = new_path_idx
self.path.s[env_ids_t] = self._path_bank.s[new_path_idx]
self.path.x[env_ids_t] = self._path_bank.x[new_path_idx]
self.path.y[env_ids_t] = self._path_bank.y[new_path_idx]
self.path.v[env_ids_t] = self._path_bank.v[new_path_idx]
self.path.psi[env_ids_t] = self._path_bank.psi[new_path_idx]
```

- reset pose / yaw / `s_reset` / warm-start speed は、bank row copy 後の `self.path` から読む。

```python
self._nearest_idx[env_ids_t] = idx
```

- `random_reset_along_path=False` の場合も `self.path[env_ids_t, 0]` を読む。`random_bank` では、これにより freshly-sampled path の start pose に spawn できる。
- `_update_progress()` は常に `self.path` の metadata を使う。

#### `configs/random_path.yaml`

Phase 2 で使う section:

- `phase2_bank.num_paths`
- `phase2_bank.length_m`
- `phase2_bank.is_loop`

#### `scripts/rl/train_ppo.py`

Phase 1 で追加した CLI だけで足りる。`--course random_bank` と `--random_path_cfg` を使う。

### Phase 2 完了条件

- `course=random_bank` で env が動く。
- env ごとに異なる `_env_path_idx` と `self.path` row を持てる。
- reset 時に `_env_path_idx` と `nearest_idx` の両方が変わる。
- env 初期化後に path 生成が発生しない。
- reset-copy 方式が実装 / docs 上で明文化されている。copy cost は `n_reset * M * 5 fields * 4 bytes` で、default の 256 env / 1000 m / ds=0.2 では全 env reset 時に約 25 MB。
- smoke command:

```powershell
c:\work\isaac\env_isaaclab\Scripts\python.exe scripts\rl\train_ppo.py --task Vehicle-Tracking-Direct-v0 --course random_bank --random_path_cfg configs\random_path.yaml --num_envs 512 --max_iterations 5 --headless
```

## Phase 3: Path Bank 再生成

未使用の path bank slot を任意で差し替える機能を追加する。この機能は default では無効にする。

### active path の定義

ある path slot は、現在どれかの env が以下の状態なら active とみなす。

```python
self._env_path_idx[env] == slot_id
```

reset-copy 方式では、走行中の env が見る道路は `self.path[env]` にコピー済みなので、bank slot を上書きしても episode 途中の道路は突然変わらない。ただし Phase 3 の初期実装では、ログの解釈と再現性を単純にするため active slot は上書きしない。unused slot が不足する場合は再生成を skip する。

### 変更ファイル

#### `src/vehicle_rl/planner/random_path.py`

追加するもの:

```python
def replace_paths(
    self,
    path_ids: Tensor,
    new_bank: "RandomPathBank",
) -> None:
    ...
```

要件:

- `new_bank.num_paths == len(path_ids)`。
- `new_bank.num_samples == self.num_samples`。
- `new_bank.ds == self.ds`。
- GPU 上で選択 slot だけコピーする。

```python
self.x[path_ids] = new_bank.x
...
```

#### `src/vehicle_rl/planner/random_path.py`

追加するもの:

```python
def random_clothoid_path_bank_slots(
    *,
    cfg: RandomPathGeneratorCfg,
    path_ids: Tensor,
    length_m: float,
    is_loop: bool,
    device: torch.device | str,
    generation_epoch: int,
) -> RandomPathBank:
    ...
```

seed rule:

```text
seed = cfg.seed + generation_epoch * 1_000_000 + path_id
```

#### `src/vehicle_rl/envs/tracking_env.py`

`TrackingEnvCfg` を拡張する。

```python
path_regeneration_enabled: bool = False
path_regeneration_interval_resets: int = 5000
path_regeneration_fraction: float = 0.05
path_regeneration_min_unused_slots: int = 16
```

状態を追加する。

```python
self._num_resets_total = torch.zeros((), dtype=torch.long, device=self.device)
self._path_generation_epoch = 0
```

method を追加する。

```python
def _maybe_regenerate_path_bank(self) -> None:
    ...

def _unused_path_ids(self) -> Tensor:
    ...
```

実装内容:

- `_reset_idx()` で `self._num_resets_total` を `n_reset` だけ増やす。
- `_maybe_regenerate_path_bank()` は以下の条件をすべて満たすときだけ動く。
  - `self._path_bank is not None`
  - regeneration が有効
  - reset counter が interval をまたいだ
- active mask を作る。

```python
active = torch.zeros(self._path_bank.num_paths, dtype=torch.bool, device=self.device)
active[self._env_path_idx] = True
unused = torch.where(~active)[0]
```

- `unused.numel() < min_unused_slots` の場合は再生成を skip する。
- 再生成数の上限は以下。

```text
ceil(path_bank.num_paths * path_regeneration_fraction)
```

- 再生成対象は `unused` からのみ選ぶ。
- 現在の env の `_env_path_idx` と `nearest_idx` は変更しない。

#### `configs/random_path.yaml`

Phase 3 で使う section:

- `phase3_regeneration.enabled`
- `phase3_regeneration.interval_resets`
- `phase3_regeneration.fraction`
- `phase3_regeneration.min_unused_slots`

### Phase 3 完了条件

- regeneration は default で無効。
- regeneration 有効時も、初期実装では active slot を上書きしない。
- regeneration をまたいでも、走行中の env に不連続な path 変更が起きない。reset-copy 方式では走行中 path が `self.path[env]` に独立しているため、これは active slot 回避に依存しない。
- TensorBoard に以下を記録する。

```text
Episode_PathBank/num_active_paths
Episode_PathBank/num_unused_paths
Episode_PathBank/num_regenerated_paths
Episode_PathBank/generation_epoch
```

## テストと検証

### 単体レベルの確認

軽量 test または smoke function で以下を確認する。

- random path 生成結果の `x/y/psi/s/v` が finite。
- `abs(kappa) <= 1 / min_radius_m`。
- local projection は、true nearest index 付近から開始した場合、slow reference の全点探索と同じ nearest sample を返す。
- open-path reset index が episode end margin を守る。
- `course=random_bank` でも plan 出力 shape が `(N, K)` になる。

候補ファイル:

- `tests/test_random_path_generation.py`
- `tests/test_projection_local.py`

まだ pytest workflow を入れない場合は、同等の smoke check を以下に置く。

- `scripts/dev/smoke_random_path.py`

### 性能確認

planner projection を以下で比較する。

- `course=circle`, local projection
- `course=random_long`, local projection
- `course=random_bank`, local projection

目標:

```text
projection cost が N * M ではなく N * W に比例すること
```

ここで `W = 2 * search_radius_samples + 1`。

### 学習 smoke check

Phase 1:

```powershell
c:\work\isaac\env_isaaclab\Scripts\python.exe scripts\rl\train_ppo.py --task Vehicle-Tracking-Direct-v0 --course random_long --random_path_cfg configs\random_path.yaml --num_envs 128 --max_iterations 5 --headless
```

Phase 2:

```powershell
c:\work\isaac\env_isaaclab\Scripts\python.exe scripts\rl\train_ppo.py --task Vehicle-Tracking-Direct-v0 --course random_bank --random_path_cfg configs\random_path.yaml --num_envs 512 --max_iterations 5 --headless
```

Phase 3:

```powershell
c:\work\isaac\env_isaaclab\Scripts\python.exe scripts\rl\train_ppo.py --task Vehicle-Tracking-Direct-v0 --course random_bank --random_path_cfg configs\random_path.yaml --num_envs 512 --max_iterations 20 --headless
```

Phase 3 では、一時 config copy で `phase3_regeneration.enabled: true` にする。

## 実装順序

1. `configs/random_path.yaml` を追加する。
2. `src/vehicle_rl/planner/random_path.py` を追加する。
3. `src/vehicle_rl/planner/path.py` の `Path.project()` を local window 版に置き換える。
4. `Path.project()` の呼び出し元である `TrackingEnv` / `run_classical.py` / 関連 docstring を新 API に合わせる。
5. `TrackingEnv` に `course=random_long` を接続する。
6. train/play scripts に `--course` と `--random_path_cfg` を追加する。
7. Phase 1 smoke test を行う。
8. `src/vehicle_rl/planner/random_path.py` に `RandomPathBank` を追加する。
9. `random_clothoid_path_bank()` を追加する。
10. `TrackingEnv` に `course=random_bank` を接続する。
11. Phase 2 smoke test を行う。
12. unused slot regeneration を追加する。
13. Phase 3 smoke test を行う。

## リスク

- `search_radius_samples` が小さすぎると、大きな外乱後に projection が遅れる可能性がある。対策として `recovery_radius_samples` を大きくし、off-track spike 後や reset 後に使う。
- 長い open path は episode 全体を走れる終端 margin が必要。対策として reset 上限を lookahead horizon だけでなく episode horizon から計算する。
- random curve が難しすぎる場合がある。対策として初期は `min_radius_m >= 50` にし、segment 速度 sample では最大曲率に基づく `v_curve_limit` を必ず使う。
- bank memory は大きくなり得る。対策として `P`, `M`, channel 数を YAML で明示し、startup 時に推定 MB を log する。
