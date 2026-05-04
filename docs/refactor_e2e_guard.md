# Refactor E2E Guard

目的:

- YAML config refactor の前後で、既存挙動が変わっていないことを 1 本の end-to-end rollout で確認する。
- PPO 学習は乱数と optimizer の揺らぎがあるため対象にしない。
- deterministic に近く、かつ path / controller / dynamics / simulator / metrics を通る `run_classical.py` の circle rollout を使う。

## テスト対象

`circle`, `mu=0.9`, `target_speed=10 m/s`, `radius=30 m`, `duration=2 s`, video なし。

この 1 本で通るもの:

- built-in course generation
- local-window `Path.project()`
- `build_observation()`
- Pure Pursuit
- speed PI
- `VehicleSimulator`
- tire / normal load / attitude damper
- metrics JSON / trajectory CSV

## Refactor 前に golden を取る

```powershell
c:\work\isaac\env_isaaclab\Scripts\python.exe scripts\sim\run_classical.py `
  --course circle `
  --mu 0.9 `
  --target_speed 10 `
  --duration 2.0 `
  --radius 30.0 `
  --pp_lookahead_min 2.0 `
  --pp_lookahead_gain 0.5 `
  --pid_kp 1.0 `
  --pid_ki 0.3 `
  --no_video `
  --headless `
  --metrics_dir .agent-work\refactor_guard\before `
  --video_dir .agent-work\refactor_guard\before_video
```

期待される出力:

```text
.agent-work/refactor_guard/before/classical_circle_mu0.90_v10.0.csv
.agent-work/refactor_guard/before/classical_circle_mu0.90_v10.0.json
```

## Refactor 後に同じ条件を YAML で走らせる

refactor 後は同条件の experiment YAML を用意する。

例:

```text
configs/experiments/classical/circle_refactor_guard.yaml
```

その YAML では以下を明示する。

- course: circle
- radius_m: 30.0
- target_speed_mps: 10.0
- mu_default: 0.9
- duration_s: 2.0
- Pure Pursuit: `lookahead_min_m=2.0`, `lookahead_gain_s=0.5`
- speed PI: `kp=1.0`, `ki=0.3`
- video disabled
- metrics_dir: `.agent-work/refactor_guard/after`

実行例:

```powershell
c:\work\isaac\env_isaaclab\Scripts\python.exe scripts\sim\run_classical.py `
  --config configs\experiments\classical\circle_refactor_guard.yaml `
  --headless
```

期待される出力:

```text
.agent-work/refactor_guard/after/classical_circle_mu0.90_v10.0.csv
.agent-work/refactor_guard/after/classical_circle_mu0.90_v10.0.json
```

## 比較

```powershell
python scripts\dev\compare_classical_rollout.py `
  --before-dir .agent-work\refactor_guard\before `
  --after-dir .agent-work\refactor_guard\after `
  --tag classical_circle_mu0.90_v10.0 `
  --atol 1e-5 `
  --rtol 1e-7
```

成功時:

```text
[PASS] rollout artifacts match
```

## 判定

CSV trajectory と JSON summary の両方を比較する。

- header / row count / JSON keys が一致すること。
- 数値差分が `atol=1e-5`, `rtol=1e-7` 以内であること。
- 差分が出た場合、最初の CSV mismatch と JSON mismatch を表示する。

YAML 化だけなら原則 bit-equivalent に近いはずなので、この tolerance を超える場合は「config の値が一致していない」か「refactor で実行順序が変わった」可能性が高い。

## 注意

- golden artifact は `.agent-work/` 配下に置き、コミットしない。
- video は無効化する。camera / encoder を挟むと余計な依存と時間が増える。
- `duration=2.0` は refactor guard 用の短時間テスト。性能評価や controller 評価には使わない。
