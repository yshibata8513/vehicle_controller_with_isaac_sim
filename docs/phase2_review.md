# Phase 2 実装レビュー

レビュー日: 2026-05-01

対象:
- `docs/PLAN.md` の Phase 2 完了意図
- `src/vehicle_rl/dynamics/`
- `src/vehicle_rl/planner/`
- `src/vehicle_rl/controller/`
- `src/vehicle_rl/envs/`
- `scripts/sim/run_classical.py`
- `scripts/sim/smoke_simulator.py`

## 総評

Phase 2 の主目的である「Phase 1.5 step 1 のタイヤ力モデル上に、Pure Pursuit + PI speed の古典制御ベースラインを構築し、Phase 3 RL と比較できる観測/行動/GT スキーマを確定する」は、概ね PLAN の意図どおり実装されています。

特に以下は意図に沿っています。

- WheeledLab 依存なしで、自前の planner / controller / simulator / sensor schema が揃っている。
- `VehicleObservation` が実車で観測可能な信号に絞られており、`delta_actual`, `vy`, `mu`, per-wheel force を policy 側へ漏らしていない。
- `VehicleSimulator.step(action)` が案 B の外力注入パイプラインをクラス化し、Phase 1.5 reference と比較する smoke test も用意されている。
- `Path.project()` は argmin / gather / body-frame 変換を batched Tensor op で実装しており、アルゴリズム構造は Phase 3 の並列化方針と整合している。
- Phase 2 の単体実行スクリプトとしては CSV / JSON / MP4 が揃い、PLAN の「エビデンス付き完了」という運用に合っている。

一方で、Phase 3 へそのまま持ち込むと問題になる箇所がいくつかあります。最優先で直すべきものは `Path.project()` 内の CPU 同期です。

## 指摘事項

### High: `Path.project()` が hot loop 内で `.item()` による GPU→CPU 同期を発生させる

該当箇所:
- `src/vehicle_rl/planner/path.py:56`
- `src/vehicle_rl/planner/path.py:137`

`Path.project()` は PLAN §0.5 で hot loop 対象とされ、「`tensor.item()` / `float(t)` / `int(t)` を使わない」ことが契約になっています。しかし `project()` 内で `ds_path = self.ds` を呼び、その property が `float((self.s[0, 1] - self.s[0, 0]).item())` を実行しています。

Phase 2 単一 env では実害は小さいですが、Phase 3 の 1024〜4096 env 学習では毎 step CPU 同期が入るため、PLAN の GPU 常駐契約に反します。

推奨修正:
- `Path` 生成時に `ds_value: float` または `lookahead_step_count` を保持する。
- 少なくとも `Path.project()` の中では `.item()` を呼ばない。
- `Path.ds` property はログ/表示用として残す場合も、hot loop からは参照しない。

例:

```python
@dataclass
class Path:
    ...
    ds_value: float

    @property
    def ds(self) -> float:
        return self.ds_value
```

### Medium: `run_classical.py` の CSV/JSON 指標が post-step state と pre-step error を混在させている

該当箇所:
- `scripts/sim/run_classical.py:239`
- `scripts/sim/run_classical.py:245`
- `scripts/sim/run_classical.py:247-260`
- `scripts/sim/run_classical.py:294-299`

ループ内では、まず現在の `state_gt` から `lat_err` / `hdg_err` を計算し、その後 `vsim.step(action)` で状態を 1 step 進めています。しかし CSV には post-step の `x, y, yaw, vx` と pre-step の `lat_err, hdg_err` が同じ行に書かれています。

200 Hz なので安定走行では差は小さいですが、低 μ で破綻するケースや DLC のように偏差が急変するケースでは、指標がわずかにズレます。Phase 2 の結論が大きく覆る可能性は低いものの、Phase 3 と apples-to-apples に比較する基準としては直しておく方が安全です。

推奨修正:
- `state_gt = vsim.step(action)` の後に、post-step state で再度 `path.project()` を呼び、ログ/指標用の `lat_err_log`, `hdg_err_log` を作る。
- もしくは CSV 行は pre-step state に統一する。

### Medium: `VehicleSimulator.reset(env_ids=...)` の返却 `VehicleStateGT` が partial reset と相性が悪い

該当箇所:
- `src/vehicle_rl/envs/simulator.py:180-227`

`reset(env_ids=...)` は Phase 3 の vectorized env で部分 reset に使う想定ですが、現在の返却値は全 env の `VehicleStateGT` です。その際、`Fz`, `Fx`, `Fy`, `slip_angle`, `ax_body`, `ay_body` が全 env 分まとめて static / zero に作り直されます。

つまり一部 env だけ reset した場合でも、返却された GT 上では reset していない env の force / accel 系フィールドがゼロまたは静荷重に見えます。次の `step()` で再計算されるため物理そのものは壊れませんが、reset 直後に観測やログを作る Phase 3 env では未 reset env の観測が汚染される可能性があります。

何が起きるかを具体化すると、例えば `num_envs=3` で env 1 だけ終了し、`reset(env_ids=tensor([1]))` を呼ぶケースです。

1. `simulator.py:198-208` では、Isaac 側の root pose / velocity / joint state は `env_ids_t` で slice されているため、実際に PhysX 上で reset されるのは env 1 だけです。ここまでは正しいです。
2. `simulator.py:211-213` でも、Python 側 actuator state と `_a_y_estimate` は env 1 だけ reset されます。ここも意図どおりです。
3. しかし返却用の `VehicleStateGT` を作る直前に、`simulator.py:216` で `zeros4 = torch.zeros(self.num_envs, 4, ...)` を作り、`simulator.py:221-226` で `Fz=self._compute_static_Fz()`, `Fx=zeros4`, `Fy=zeros4`, `slip_angle=zeros4`, `ax_body=self._zeros`, `ay_body=self._zeros` を渡しています。これらは reset 対象の env 1 だけではなく、env 0 / env 2 も含む全 env 分です。
4. そのため、返ってきた `state_gt` は pose / velocity については env 0 / env 2 の現在値を読めていますが、force / accel 系だけは env 0 / env 2 も「静荷重・タイヤ力ゼロ・加速度ゼロ」に上書きされたように見えます。

問題になるのは、Phase 3 の env wrapper が reset 直後にこの返却 `state_gt` から `build_observation()` を呼ぶ場合です。`build_observation()` は `ax_body`, `ay_body` を IMU 観測としてそのまま使うので、reset されていない env 0 / env 2 まで一瞬 `ax=0`, `ay=0` の観測になります。policy 入力、報酬ログ、episode 境界の統計にだけ出る「観測上の汚れ」で、PhysX 状態そのものが壊れるわけではありません。次の `step()` では全 env の力が再計算されるため復帰しますが、vectorized RL ではこの 1 frame の不整合も学習データに混ざります。

推奨修正:
- `VehicleSimulator` が直近の per-env force / accel を内部状態として保持し、partial reset 時は対象 env のスライスだけ更新する。
- あるいは `reset()` は reset 対象 env のみを返す API にする。ただし Phase 3 の env wrapper と明確に契約を合わせる必要があります。

### Medium: completion_rate が「経路を正しく完走したか」を過大評価しうる

該当箇所:
- `scripts/sim/run_classical.py:306-314`

loop path では `sum(abs(vx) * dt) >= path.total_length` で completion を 1.0 にしています。この計算は経路座標上の進捗ではなく車体前進速度の積分なので、コース外を大きく膨らんで走っても距離だけ稼げば完走扱いになります。

open path でも「最終 waypoint に一度でも近づいたか」で判定しており、経路順序や安定した完走までは見ていません。

PLAN 上の Phase 2 では `rms_lat`, `off_track_time` と併用しているため致命的ではありませんが、Phase 3 の評価指標としては policy が抜け道を見つける余地があります。

推奨修正:
- `Path.project()` が `closest` index または `s_proj` を返せるようにする。
- completion は `s_proj` の単調進捗または lap count で評価する。
- off-track 中の進捗を completion から除外するか、少なくとも別指標に分ける。

### Low: `FirstOrderLagActuator.step()` が毎 step scalar Tensor を生成している

該当箇所:
- `src/vehicle_rl/dynamics/actuator.py:35-36`

`tau_pos_t` / `tau_neg_t` を毎 step `torch.tensor(...)` で生成しています。CPU 同期ではありませんが、Phase 3 の hot loop では不要な Tensor allocation になります。

推奨修正:
- `__init__` で scalar Tensor として作って保持する。
- Phase 3 の DR で per-env tau にする予定があるため、最終的には `(N,)` Tensor にして reset event で更新する形が自然です。

### Low: 既存の `utils/metrics.py` と Phase 2 の 7 指標実装がまだ分離している

該当箇所:
- `scripts/sim/run_classical.py:285-330`
- `src/vehicle_rl/utils/metrics.py`

PLAN Phase 3 では「Phase 2 と同じ 7 指標を共通モジュール化」とあります。現時点では `run_classical.py` 内に Phase 2 指標が直書きされ、`utils/metrics.py` は別形式 CSV 前提の古い/汎用実装に見えます。

Phase 2 完了時点では許容範囲ですが、Phase 3 着手前に共通化しないと RL 評価と baseline 評価が微妙にズレるリスクがあります。

推奨修正:
- `TrajectoryMetrics` を Phase 2 JSON のキー名に合わせる。
- `lateral_error`, `heading_error`, `completion_rate`, `off_track_time` の算出を Phase 2 / Phase 3 で同じ関数に寄せる。

## PLAN 適合チェック

| PLAN 意図 | 実装状況 | コメント |
|---|---|---|
| 案 B: PhysX chassis + 自前タイヤ力注入 | OK | `VehicleSimulator` + `aggregate_tire_forces_to_base_link` で実装済み |
| WheeledLab 非依存 | OK | Phase 2 主要コードに WheeledLab import は見当たらない |
| 観測/行動/GT スキーマを Phase 3 へ継承 | 概ね OK | `VehicleObservation` / `VehicleAction` / `VehicleStateGT` は意図通り。partial reset 返却だけ要注意 |
| `VehicleSimulator.step(action) -> state_gt` | OK | 外力注入、操舵 target、actuator lag がまとまっている |
| `Path.project()` の GPU 並列実装 | 構造は OK / 契約違反あり | batched Tensor 実装だが、`self.ds` 経由の `.item()` が hot loop に残っている |
| `build_observation` は slice / concat + noise | OK | policy に GT を漏らさない構成 |
| Pure Pursuit + PI baseline | OK | controller 実装は batched で、Phase 2 用として妥当 |
| CSV / JSON / MP4 evidence | OK | `run_classical.py` で揃っている。ログの時刻ズレは修正推奨 |
| 7 指標 JSON | 概ね OK | 指標は出るが completion 定義は Phase 3 前に強化推奨 |
| Phase 1.5 reference との回帰 | OK | `smoke_simulator.py` あり。ただし実行結果の再確認は Isaac 実行環境依存 |

## 確認済み

- `python -m compileall src scripts` は成功。
- 今回のレビューでは Isaac Sim 実行を伴う `run_classical.py` / `smoke_simulator.py` の再実行はしていません。コード構造と PLAN への適合を中心に確認しました。

## 優先対応順

1. `Path.project()` から `.item()` を除去する。
2. `run_classical.py` のログ/指標を post-step state と同じ時刻の error に揃える。
3. Phase 3 着手前に completion を `s_proj` / path progress ベースへ変更する。
4. `VehicleSimulator.reset(env_ids=...)` の partial reset 契約を Phase 3 env wrapper と合わせて修正する。
5. 7 指標を `utils/metrics.py` に切り出し、Phase 2 / Phase 3 評価で共通化する。
