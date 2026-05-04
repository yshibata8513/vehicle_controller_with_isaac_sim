# Phase 3 Random Path Phase 2 実装レビュー

対象:

- `src/vehicle_rl/envs/tracking_env.py`
- `src/vehicle_rl/planner/random_path.py`
- `src/vehicle_rl/planner/__init__.py`
- `configs/random_path.yaml`
- `scripts/rl/train_ppo.py`
- `scripts/rl/play.py`
- `README.md`

確認:

```powershell
python -m compileall src\vehicle_rl\planner src\vehicle_rl\envs scripts\rl\train_ppo.py scripts\rl\play.py scripts\sim\run_classical.py
```

結果: 構文チェックは通過。

軽量 bank 生成 smoke:

```powershell
$env:PYTHONPATH='src'
c:\work\isaac\env_isaaclab\Scripts\python.exe - <<'PY'
from vehicle_rl.planner.random_path import load_random_path_cfg, random_clothoid_path_bank
cfg = load_random_path_cfg('configs/random_path.yaml')
bank = random_clothoid_path_bank(cfg=cfg, num_paths=4, length_m=100.0, is_loop=False, device='cpu')
print(bank.x.shape, bank.v.min().item(), bank.v.max().item(), bank.ds, bank.is_loop)
print('finite', bank.x.isfinite().all().item(), bank.y.isfinite().all().item(), bank.psi.isfinite().all().item(), bank.v.isfinite().all().item())
PY
```

結果:

```text
torch.Size([4, 500]) 6.249... 10.140... 0.2 False
finite True True True True
```

## 指摘事項

### 1. `random_bank` + `random_reset_along_path=False` で spawn pose と path row が一致しない

該当:

- `src/vehicle_rl/envs/tracking_env.py`
- `_build_path()`
- `_reset_idx()`
- `scripts/rl/play.py`

現状:

- `random_bank` では `_build_path()` 時点で placeholder として `bank[0]` を全 env に入れる。
- その直後に `_initial_pos_local` / `_initial_yaw` が `bank[0]` から作られる。
- `_reset_idx()` では、`random_reset_along_path` の値に関係なく、bank から新しい `path_idx` を sample して `self.path[env_ids]` の行を上書きする。
- その後 `random_reset_along_path=False` の分岐では、上書き後の `self.path` ではなく、古い `_initial_pos_local` / `_initial_yaw` を使って spawn する。

問題:

`scripts/rl/play.py` は eval の再現性のために以下を設定している。

```python
env_cfg.random_reset_along_path = False
```

この状態で `--course random_bank` を使うと、env は新しく選ばれた bank path を追従対象にする一方、車両は placeholder `bank[0]` の start pose に置かれる可能性がある。`new_path_idx != 0` なら初期状態から path と車両 pose がずれる。

影響:

- `play.py --course random_bank` が初期 off-track になり得る。
- eval / 可視化結果が policy の性能ではなく reset 不整合を見てしまう。
- `random_reset_along_path=False` を使う deterministic eval が壊れる。

修正案:

`random_reset_along_path=False` の場合も、bank row 上書き後の `self.path` から sample 0 を読む。

```python
idx = torch.zeros(n_reset, dtype=torch.long, device=self.device)
pos_local_xy = torch.stack(
    [self.path.x[env_ids_t, idx], self.path.y[env_ids_t, idx]], dim=-1
)
yaw = self.path.psi[env_ids_t, idx]
s_reset = self.path.s[env_ids_t, idx]
v_warmstart = self.path.v[env_ids_t, idx]
```

この形にすると built-in course / random_long / random_bank で同じ reset source になる。`_initial_pos_local` / `_initial_yaw` は不要にできるか、built-in course のみに限定して使う。

別案:

- `random_reset_along_path=False` のときは bank path を sample しない。
- ただし eval で「path は変えたいが start は path[0] にしたい」需要があるため、上の修正案の方が自然。

### 2. Phase 2 計画の「reset 時に active path copy しない」方針から外れている

該当:

- `src/vehicle_rl/envs/tracking_env.py`
- `_reset_idx()`
- `src/vehicle_rl/planner/random_path.py`
- `RandomPathBank`

現状:

`random_bank` の reset では、選ばれた bank row を `self.path` の env row へ丸ごとコピーしている。

```python
self.path.s[env_ids_t] = self._path_bank.s[new_path_idx]
self.path.x[env_ids_t] = self._path_bank.x[new_path_idx]
self.path.y[env_ids_t] = self._path_bank.y[new_path_idx]
self.path.v[env_ids_t] = self._path_bank.v[new_path_idx]
self.path.psi[env_ids_t] = self._path_bank.psi[new_path_idx]
```

問題:

`docs/phase3_random_path_plan.md` の Phase 2 では、以下を目標としていた。

```text
No `(n_reset, M)` active path copy occurs at reset.
per-env path selection is done by path_id gather.
```

現在の実装は、reset ごとに以下のコピーが発生する。

```text
n_reset * M * 5 fields * 4 bytes
```

default の `M = 1000m / 0.2m = 5000` なら、全 128 env reset で約 12.8 MB、4096 env 全 reset なら約 409 MB の GPU copy になる。

影響:

- 通常の staggered reset では許容できる可能性がある。
- ただし early termination が多い学習初期や、全 env 一斉 reset / curriculum refresh では reset cost が目立つ。
- Phase 3 の active path regeneration 設計も、当初の path_id-gather 方式とは意味が変わる。現在の方式では running env は bank からコピー済みなので、bank slot を上書きしても走行中 path は変わらない。

判断:

これは即 functional bug ではない。Phase 2 を早く動かす実装としては成立している。ただし「計算時間を非常に重視」という最初の要件から見ると、計画との差分として明示するべき。

修正案 A: 計画どおり path_id-gather 方式へ寄せる

- `RandomPathBank.project(...)` を実装する。
- `self.path` を env ごとにコピーせず、`self._path_id` と `self._nearest_idx` で projection / reset pose gather を行う。
- reset は `path_id` と `start_idx` の更新だけにする。

修正案 B: 現在方式を採用として明文化する

- reset copy 方式を「Phase 2a」として README / plan に明記する。
- Phase 3 regeneration は active slot 回避が不要または意味が変わるため、別設計として書き直す。
- performance smoke で reset cost を測る。

推奨:

最終的に Phase 3 の bank regeneration まで進めるなら、A の path_id-gather 方式に寄せる方が設計がきれい。まず学習を回したいなら B でもよいが、その場合は「計画からの意図的逸脱」として記録する。

### 3. CLI help / README の説明が `random_bank` に完全には追従していない

該当:

- `scripts/rl/train_ppo.py`
- `scripts/rl/play.py`
- `README.md`

現状:

`train_ppo.py` の `--course` help はまだ `random_long` まで。

```text
(circle | s_curve | dlc | lemniscate | random_long)
```

`--random_path_cfg` の説明も `only used when course=random_long` のまま。

問題:

`random_bank` 自体は `TrackingEnv` で受けられるが、CLI help から分かりにくい。README は random_bank section を追加済みだが、directory tree では `random_path.yaml` が `random_long` 用のように書かれている。

修正案:

- `train_ppo.py` help:

```text
(circle | s_curve | dlc | lemniscate | random_long | random_bank)
```

- `--random_path_cfg` help:

```text
used when course is random_long or random_bank
```

- `play.py` docstring の example に `random_bank` 例を追加してもよい。
- README の tree:

```text
random_path.yaml # Phase 3 random_long/random_bank 生成パラメタ
```

### 4. `phase2_bank.is_loop=true` は実質未対応に見える

該当:

- `configs/random_path.yaml`
- `src/vehicle_rl/planner/random_path.py`
- `src/vehicle_rl/envs/tracking_env.py`

現状:

- `RandomPathBank` は `is_loop` を持つ。
- `TrackingEnv` は `self.path.is_loop` に従って reset / progress wrap を処理する。
- しかし `random_clothoid_path_bank(..., is_loop=True)` は幾何的に閉じた path を生成しない。
- `random_long` 側 docstring には `is_loop=True` は幾何的に閉じないと明記されているが、bank 側では同じ注意が薄い。

問題:

ユーザーが YAML で `phase2_bank.is_loop: true` にすると、open geometry を loop として扱い、終端から先頭へ projection / lookahead が wrap する。座標が連続していないため、終端付近で不自然な plan になる。

修正案:

Phase 2 では明示的に open path only とする。

```python
if pb.is_loop:
    raise ValueError("phase2_bank.is_loop=true is not supported yet")
```

または `random_path.py` / YAML コメントに「true は未対応」と明記する。

## 良い点

- Phase 1 の 4 指摘は保持されたまま Phase 2 に拡張されている。
- `random_clothoid_path_bank()` は `_integrate_kappa_v()` により vectorized cumsum を使っており、bank 起動時間を意識した実装になっている。
- `RandomPathBank` は `(P, M)` tensor として持っており、将来の path_id-gather 方式へ移行しやすい。
- `random_bank` 初期化時に `phase2_bank.length_m` の episode horizon check があり、短すぎる bank path を早期検出できる。

## 優先度

1. `random_bank` + `random_reset_along_path=False` の reset pose 不整合を修正する。
2. reset copy 方式を採用するのか、path_id-gather 方式へ戻すのか決める。
3. CLI help / README の `random_bank` 説明を更新する。
4. `phase2_bank.is_loop=true` を禁止または未対応として明記する。

## 追加で確認したいこと

- `course=random_bank`, `random_reset_along_path=True` で short PPO smoke が起動すること。
- `course=random_bank`, `random_reset_along_path=False` で初期 `lat_err` が 0 近傍になること。
- reset copy 方式を続ける場合、`num_envs=128/512/2048` で reset-heavy な短時間 benchmark を取り、reset cost が許容範囲か確認すること。
- `Episode_PathProj/idx_jump_violation_rate` が `random_bank` でも 0 近傍であること。
