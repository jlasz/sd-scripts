# LoRA-Squeeze training / LoRA-Squeeze学習

LoRA-Squeeze trains a LoRA at a higher starting rank and compresses its effective weight update into progressively lower ranks during the same training run. The implementation uses the memory-efficient QR and core-SVD transformation described in [LoRA-Squeeze](https://arxiv.org/abs/2602.10993), without materializing the full model-sized weight update.

The final `network_dim` and `network_alpha` remain the deployment rank and alpha. `lora_squeeze_start_dim` selects the higher initial training rank.

The implementation follows sd-scripts' library boundaries: CLI registration is in `library/args.py`, scheduling and resume state are in `library/lora_squeeze_schedule.py`, network-facing factor protocols are in `library/lora_squeeze_network.py`, compression is in `library/lora_squeeze_compression.py`, optimizer mathematics are in `library/lora_squeeze_optimizer.py`, and runtime integration, metadata/logging, and Accelerate lifecycle handling are in `library/lora_squeeze_training.py`.

<details>
<summary>日本語</summary>

LoRA-Squeezeは高い初期ランクでLoRAの学習を開始し、同じ学習実行中に有効な重み更新を段階的に低いランクへ圧縮します。この実装では、モデルと同じ大きさの重み更新全体を生成せず、[LoRA-Squeeze](https://arxiv.org/abs/2602.10993)で説明されているメモリ効率の良いQR分解とコアSVD変換を使用します。

最終的な`network_dim`と`network_alpha`は、配布時に使用するランクとalphaです。`lora_squeeze_start_dim`で学習開始時の高いランクを指定します。

実装はsd-scriptsのライブラリ境界に従っています。CLI引数は`library/args.py`、スケジュールと再開状態は`library/lora_squeeze_schedule.py`、ネットワーク側の因子プロトコルは`library/lora_squeeze_network.py`、圧縮処理は`library/lora_squeeze_compression.py`、オプティマイザの計算は`library/lora_squeeze_optimizer.py`、実行時統合・メタデータ・ログ・Accelerateのライフサイクル処理は`library/lora_squeeze_training.py`にあります。

</details>

## In-Squeeze example / In-Squeezeの例

```toml
network_dim = 8
network_alpha = 4

lora_squeeze_start_dim = 64
lora_squeeze_num_squeezes = 3
```

This produces a `64 -> 32 -> 16 -> 8` rank schedule. The total configured `max_train_steps` is divided between the four rank stages, and training continues at rank 8 after the final squeeze.

The default LoRA-Squeeze behavior is:

- `lora_squeeze_rank_schedule = "geometric"`
- `lora_squeeze_step_schedule = "equal"`
- `lora_squeeze_train_after_final_squeeze = true`
- `lora_squeeze_optimizer_mode = "per_squeeze"`
- `lora_squeeze_scheduler_mode = "global"`
- `lora_squeeze_alpha_schedule = "proportional"`
- `lora_squeeze_first_segment_ratio = 1.0`
- `lora_squeeze_final_segment_ratio = 1.0`

<details>
<summary>日本語</summary>

上の設定では、ランクは`64 -> 32 -> 16 -> 8`の順に変化します。`max_train_steps`で設定した総ステップ数は4つのランク区間に配分され、最後の圧縮後もランク8で学習を続けます。

LoRA-Squeezeのデフォルトは、ランク配置が`geometric`、ステップ配分が`equal`、最終圧縮後の学習が有効、オプティマイザが`per_squeeze`、スケジューラが`global`、alphaスケジュールが`proportional`、最初と最後の区間倍率がともに`1.0`です。

</details>

## Post-Squeeze and Cont-Squeeze / Post-SqueezeとCont-Squeeze

Post-Squeeze trains at the source rank for the full training budget and performs one final compression before saving:

```toml
network_dim = 8
network_alpha = 4
lora_squeeze_start_dim = 64
lora_squeeze_num_squeezes = 1
lora_squeeze_train_after_final_squeeze = false
```

When no Accelerator state is requested, this terminal compression does not rebuild an optimizer or scheduler that can never take another step. It first switches optimizers such as schedule-free variants to their evaluation point so the compressed network matches the weights that sd-scripts would normally save. If `save_state` or `save_state_on_train_end` is enabled, the optimizer and scheduler are rebuilt so the terminal state remains loadable.

Cont-Squeeze performs one compression and then continues training at the target rank:

```toml
network_dim = 8
network_alpha = 4
lora_squeeze_start_dim = 64
lora_squeeze_num_squeezes = 1
lora_squeeze_train_after_final_squeeze = true
```

<details>
<summary>日本語</summary>

Post-Squeezeでは、学習予算の全体を元のランクで学習し、保存直前に1回だけ圧縮します。Accelerator状態を保存しない場合、以後使われないオプティマイザとスケジューラは再構築しません。schedule-free系のオプティマイザは、sd-scriptsが通常保存する重みと一致するよう、先に評価時の点へ切り替えます。`save_state`または`save_state_on_train_end`を有効にした場合は、保存した状態を読み込めるようにオプティマイザとスケジューラも再構築します。

Cont-Squeezeでは1回圧縮した後、ターゲットランクで学習を続けます。Post-Squeezeでは`lora_squeeze_train_after_final_squeeze = false`、Cont-Squeezeでは`true`を指定します。

</details>

## Rank and alpha schedules / ランクとalphaのスケジュール

`lora_squeeze_rank_schedule` controls intermediate ranks:

- `geometric` (default): approximately equal compression ratios.
- `linear`: approximately equal rank differences.

`lora_squeeze_alpha_schedule` controls pre-final-rank alpha values:

- `proportional` (default): keeps the same `alpha / rank` ratio as the final target rank:

```text
network_alpha * r / network_dim
```

For example, if the final rank/alpha is `9/3`, proportional scheduling gives rank `45` an alpha of `15`.

- `sqrt`: uses rank-stabilized square-root scaling. For a rank `r`, alpha is:

```text
network_alpha / sqrt(network_dim) * sqrt(r)
```

With the same final rank/alpha of `9/3`, square-root scheduling gives rank `45` an alpha of `sqrt(45)`, approximately `6.708`.

Each compression operates on the scaled effective LoRA update. Singular values are split evenly between the new up and down factors, and the factors are adjusted for the new alpha/rank scale.

The numerical rank is reported at every squeeze. If the target rank includes numerically zero singular directions, LoRA-Squeeze keeps the up factor at zero and initializes the corresponding down direction to a nonzero value. This leaves the effective update unchanged while allowing that channel to receive an up-factor gradient and recover during later training.

<details>
<summary>日本語</summary>

`lora_squeeze_rank_schedule`は中間ランクの配置を制御します。`geometric`（デフォルト）は各段階の圧縮率をほぼ均等にし、`linear`はランク差をほぼ均等にします。

`lora_squeeze_alpha_schedule`は最終ランクより前のalphaを制御します。`proportional`（デフォルト）は`alpha / rank`を最終ターゲットと同じ比率に保ち、`network_alpha * r / network_dim`で計算します。`sqrt`は`network_alpha / sqrt(network_dim) * sqrt(r)`を使用します。

各圧縮はスケーリング後の有効なLoRA更新に対して実行されます。特異値は新しいup因子とdown因子へ均等に分配され、新しいalphaとランクのスケールに合わせて調整されます。各圧縮時には数値ランクも報告されます。ターゲットランクに数値的にゼロの特異方向が含まれる場合、有効な更新を変えずに後の学習で回復できるよう、up因子をゼロのままにして対応するdown方向を非ゼロで初期化します。

</details>

## Step distribution / ステップ配分

`lora_squeeze_step_schedule` controls how the total training budget is divided:

- `equal` (default): equal-length rank stages.
- `rank_proportional`: more steps at larger ranks.
- `sqrt_rank_proportional`: a milder preference for larger ranks.
- `inverse_rank_proportional`: more steps at smaller ranks.
- `inverse_sqrt_rank_proportional`: a milder preference for smaller ranks.

`lora_squeeze_first_segment_ratio` multiplies the relative length of the initial, highest-rank training stage. When the final target rank has a training stage, `lora_squeeze_final_segment_ratio` multiplies the relative length of that final stage. Both ratios may be used together.

<details>
<summary>日本語</summary>

`lora_squeeze_step_schedule`は総学習ステップを各ランク区間へ配分する方法を制御します。

- `equal`（デフォルト）: 各ランク区間を同じ長さにします。
- `rank_proportional`: 大きいランクにより多くのステップを配分します。
- `sqrt_rank_proportional`: 大きいランクを優先しますが、差を緩やかにします。
- `inverse_rank_proportional`: 小さいランクにより多くのステップを配分します。
- `inverse_sqrt_rank_proportional`: 小さいランクを優先しますが、差を緩やかにします。

`lora_squeeze_first_segment_ratio`は最初の高ランク区間の相対的な長さに掛ける倍率です。最終ターゲットランクにも学習区間がある場合、`lora_squeeze_final_segment_ratio`はその最後の区間に掛ける倍率です。両方を同時に使用できます。

</details>

## Optimizer and scheduler behavior / オプティマイザとスケジューラの動作

`lora_squeeze_optimizer_mode` controls optimizer state across rank changes:

- `per_squeeze` (default): rebuilds the optimizer with fresh state after each rank change. This is supported for every optimizer.
- `global`: carries Adam-family optimizer state across rank changes. The first and second gradient moments are projected into the new rank coordinates. Unknown state is rejected before changing the network.

Global mode currently requires every optimizer parameter to be a squeezeable `lora_down.weight` or `lora_up.weight` factor. Factors intentionally omitted through a zero learning rate are allowed. Additional trainable gates, biases, embeddings, or other parameters are rejected before training because their state cannot yet be continued safely across optimizer rebuilding. Factor group membership and ordering must also remain stable after every squeeze. `per_squeeze` mode intentionally resets state and does not have this restriction.

Global mode supports only Adam-family optimizers:

- PyTorch `Adam`, `AdamW` (including `amsgrad`), `RAdam`, `NAdam`, and `Adamax`.
- bitsandbytes Adam/AdamW variants (`AdamW8bit`, `PagedAdamW8bit`, 32-bit versions, etc.). Block-wise 8-bit state is dequantized, projected, and requantized; paged variants stay paged.

Any other optimizer (SGD, Lion, Adafactor, Prodigy, D-Adaptation, schedule-free, etc.) is rejected before training in global mode; use `per_squeeze` with those.

Projecting a diagonal second moment through a rank-coordinate change is necessarily an approximation because the optimizer does not store cross-coordinate covariance. First moments are projected as covectors.

In global mode, projected state is staged on CPU one module at a time. Once projection succeeds, the old optimizer state is moved to CPU and the replacement state is moved back beside its new parameters. This avoids retaining complete old and new optimizer-state copies on the GPU at the same time; if rebuilding fails, the original state and LoRA layers are restored. If CPU staging or offload itself fails, the transition is rolled back and retried once with optimizer state kept on the parameter devices, at the cost of higher peak VRAM usage.

`lora_squeeze_scheduler_mode` controls an external LR scheduler independently:

- `global` (default): continues one LR scheduler curve over the full training run.
- `per_squeeze`: restarts the LR scheduler for each rank stage using that stage's configured step budget.

For example, `lora_squeeze_optimizer_mode = "per_squeeze"` with `lora_squeeze_scheduler_mode = "global"` resets AdamW moments at each squeeze while preserving progress through a cosine LR curve. Global optimizer-mode logs include projected, reset, and empty optimizer-state counts.

Optimizers that own their learning-rate or averaging schedule cannot split these two lifecycles. Schedule-free optimizers, ProdigyPlusScheduleFree, and Adafactor with `relative_step=True` therefore require optimizer and scheduler mode to be equal. Because they are not supported by global optimizer mode, that means both must be `per_squeeze`. This is validated before training starts.

<details>
<summary>日本語</summary>

`lora_squeeze_optimizer_mode`はランク変更前後のオプティマイザ状態を制御します。

- `per_squeeze`（デフォルト）: ランク変更後に新しい状態でオプティマイザを再構築します。すべてのオプティマイザで使用できます。
- `global`: Adam系オプティマイザの状態をランク変更後も引き継ぎます。1次・2次の勾配モーメントを新しいランク座標へ射影します。未知の状態はネットワークを変更する前に拒否します。

`global`では、すべてのオプティマイザパラメータが圧縮可能な`lora_down.weight`または`lora_up.weight`である必要があります。学習率0によって意図的に除外した因子は使用できますが、追加のgate、bias、embeddingなどの学習パラメータは未対応です。圧縮後も因子のパラメータグループ構成と順序が同じでなければなりません。`per_squeeze`は状態をリセットするため、この制約はありません。

`global`はAdam系オプティマイザのみに対応しています。

- PyTorchの`Adam`、`AdamW`（`amsgrad`を含む）、`RAdam`、`NAdam`、`Adamax`。
- bitsandbytesのAdam/AdamW系（`AdamW8bit`、`PagedAdamW8bit`、32-bit版など）。block-wise 8-bit状態は逆量子化、射影、再量子化され、paged版はpagedのまま維持されます。

それ以外のオプティマイザ（SGD、Lion、Adafactor、Prodigy、D-Adaptation、schedule-free系など）は`global`では学習開始前に拒否されます。これらには`per_squeeze`を使用してください。

対角2次モーメントの射影は、オプティマイザが座標間の共分散を保持しないため近似になります。1次モーメントは共変ベクトルとして射影します。

`global`では射影した状態をモジュール単位でCPUへ一時配置します。射影成功後、古い状態をCPUへ移し、新しい状態を新しいパラメータと同じデバイスへ戻します。これにより古い状態と新しい状態の完全なコピーが同時にGPUへ存在することを避けます。再構築に失敗した場合は元の状態とLoRA層を復元します。CPUへの一時配置またはoffloadに失敗した場合も変更を取り消し、ピークVRAMの増加を許容してパラメータのデバイス上で1回だけ再試行します。

`lora_squeeze_scheduler_mode`は外部LRスケジューラを独立に制御します。`global`（デフォルト）は学習全体で1つの曲線を継続し、`per_squeeze`は各ランク区間のステップ数で曲線を再開します。たとえばオプティマイザを`per_squeeze`、スケジューラを`global`にすると、AdamWのmomentsは圧縮ごとにリセットされますがcosine LR曲線は継続します。

学習率または平均化スケジュールをオプティマイザ自身が持つ場合、この2つのライフサイクルは分離できません。そのためschedule-free系、ProdigyPlusScheduleFree、および`relative_step=True`のAdafactorでは、オプティマイザとスケジューラのmodeを同じにする必要があります。これらは`global`に対応していないため、両方`per_squeeze`にしてください。この条件は学習開始前に検証されます。

</details>

## Network compatibility / ネットワーク互換性

LoRA-Squeeze is enabled only for network modules that explicitly declare support through `validate_lora_squeeze_support(network_args)`. This check happens before datasets, models, or caches are loaded. The instantiated network then returns the squeezeable modules it owns through `get_lora_squeeze_modules()`. Each returned module implements `lora_squeeze_get_spec()`, `lora_squeeze_replace_factors()`, `lora_squeeze_snapshot()`, and `lora_squeeze_restore()`. The built-in supported modules are:

- `networks.lora`
- `networks.lora_anima`

Other network modules must implement both the early declaration and the structural protocol. The standard protocol mixin accepts direct Linear/Linear or Conv2d/Conv2d factors and rejects grouped convolutions, bias, custom factor subclasses, hooks, parametrizations, mixed devices/dtypes, and frozen factors before training. A custom network may preserve different semantics in its own protocol implementation. Split-QKV or `ModuleList` LoRA factors, such as the current `networks.lora_flux`, `networks.lora_lumina`, `networks.lora_sd3`, and `networks.lora_hunyuan_image` modules, are not supported yet. LoRA-FA is not supported yet.

All LoRA modules in the instantiated network must match the schedule's homogeneous current rank and alpha. `network_args` that create different ranks or alphas for different blocks, such as block-specific dim settings, are not supported.

`network_weights` may be used only when the weight file's rank and alpha match the current LoRA-Squeeze rank and alpha. For a new run, the current rank is `lora_squeeze_start_dim`. For a resumed run, it is the rank recorded in the LoRA-Squeeze resume state after the completed squeezes, which may be lower than `lora_squeeze_start_dim`. `dim_from_weights` is not supported because `network_dim` is the final target rank.

LoRA alpha scalars follow the standard LoRA serialization behavior. When `save_precision` stores factor weights in FP16 or BF16, alpha is stored in the same dtype. Loading restores the runtime alpha buffer to FP32, but a fractional alpha retains any rounding introduced by the save dtype. Metadata records the configured LoRA-Squeeze alpha, so a low-precision alpha tensor can differ slightly from the metadata value after saving. Use FP32 save precision when an exact fractional alpha must survive a save/load round trip.

<details>
<summary>日本語</summary>

LoRA-Squeezeは、`validate_lora_squeeze_support(network_args)`で明示的に対応を宣言したネットワークモジュールでのみ有効にできます。この検証はデータセット、モデル、キャッシュを読み込む前に行われます。生成されたネットワークは`get_lora_squeeze_modules()`で所有する圧縮対象モジュールを返し、各モジュールは`lora_squeeze_get_spec()`、`lora_squeeze_replace_factors()`、`lora_squeeze_snapshot()`、`lora_squeeze_restore()`を実装します。組み込みで対応しているのは`networks.lora`と`networks.lora_anima`です。

標準プロトコルmixinは、直接接続されたLinear/LinearまたはConv2d/Conv2d因子に対応します。grouped convolution、bias、独自の因子subclass、hook、parametrization、異なるdevice/dtype、凍結された因子は学習前に拒否します。独自ネットワークは別の意味を維持する独自プロトコルを実装できます。現在の`networks.lora_flux`、`networks.lora_lumina`、`networks.lora_sd3`、`networks.lora_hunyuan_image`のようなsplit-QKVまたは`ModuleList`形式、およびLoRA-FAは未対応です。

生成されたネットワーク内のすべてのLoRAモジュールは、現在のスケジュールと同じランクとalphaを持つ必要があります。blockごとに異なるdimまたはalphaを作る`network_args`は使用できません。

`network_weights`を使用する場合、重みファイルのランクとalphaは現在のLoRA-Squeezeのランクとalphaに一致する必要があります。新しい学習では現在のランクは`lora_squeeze_start_dim`です。再開時は、完了した圧縮後のLoRA-Squeeze再開状態に記録されたランクであり、`lora_squeeze_start_dim`より小さい場合があります。`network_dim`は最終ターゲットランクなので、`dim_from_weights`は使用できません。

LoRAのalpha scalarは通常のLoRA保存動作に従います。`save_precision`で因子をFP16またはBF16として保存すると、alphaも同じdtypeで保存されます。読み込み時のalpha bufferはFP32へ戻りますが、小数alphaには保存dtypeによる丸めが残ります。正確な小数alphaを保存と読み込みの往復で維持する必要がある場合は、FP32の保存精度を使用してください。

</details>

## Resuming / 再開

LoRA-Squeeze can resume from an Accelerator state directory saved with LoRA-Squeeze metadata in `train_state.json`. The same LoRA-Squeeze rank, alpha, optimizer mode, scheduler mode, segment options, and total `max_train_steps` budget must be used when resuming so the generated squeeze boundaries remain identical.

When LoRA-Squeeze is active, the restored update step remains the canonical absolute training step. It controls future squeeze events, checkpoint filenames, sampling and validation cadence, tracker steps, and `ss_steps` metadata. The data-loader resume position is tracked separately as an epoch and batch offset.

The absolute step also controls the remaining progress-bar length and training termination. A resumed run therefore performs only `max_train_steps - saved_step` further optimizer updates and cannot pass the configured LoRA-Squeeze budget, including with gradient accumulation.

Without `skip_until_initial_step`, sd-scripts starts at the beginning of the saved step's current epoch, so some data can be replayed within the remaining update budget. With `skip_until_initial_step`, LoRA-Squeeze maps the saved optimizer step to the corresponding epoch and batch offset, including partial gradient-accumulation windows at epoch boundaries, and discards the preceding batches. This preserves the update budget and squeeze schedule, but a shuffled data loader is not guaranteed to reproduce the uninterrupted run's exact sample order or bit-identical final weights. A completed state may be loaded with the same `max_train_steps`; it exits without another update.

<details>
<summary>日本語</summary>

LoRA-Squeezeは、`train_state.json`にLoRA-Squeezeメタデータを含むAccelerator状態ディレクトリから再開できます。再開時は同じLoRA-Squeezeランク、alpha、オプティマイザmode、スケジューラmode、区間設定、および`max_train_steps`の総予算を指定し、生成される圧縮境界を一致させる必要があります。

復元したupdate stepは学習全体の絶対stepとして扱われ、今後の圧縮、checkpoint名、samplingとvalidationの周期、tracker step、`ss_steps`メタデータを制御します。data loaderの再開位置はepochとbatchのoffsetとして別に管理します。

再開後に実行するoptimizer updateは`max_train_steps - saved_step`だけで、設定したLoRA-Squeeze予算を超えません。gradient accumulationを使用する場合も同じです。

`skip_until_initial_step`を使用しない場合、sd-scriptsは保存stepを含むepochの先頭から開始するため、残りのupdate予算内で一部のデータが再度使われることがあります。有効にした場合は、保存したoptimizer stepを対応するepochとbatch offsetへ変換し、epoch境界の途中のgradient accumulationも考慮して先行batchを破棄します。update予算と圧縮スケジュールは維持されますが、shuffleされたdata loaderで中断なしの学習と完全に同じsample順やbit単位で同じ最終重みになる保証はありません。完了済みの状態を同じ`max_train_steps`で読み込むと、追加のupdateを行わず終了します。

</details>

## Current limitations / 現在の制限事項

- Single-process training only.
- DeepSpeed is not supported.
- `initial_step` and `initial_epoch` are not supported.
- `torch.compile` options are not supported.
- LoRA layers must match one homogeneous scheduled current rank and alpha.
- LoRA-C3Lier is supported when `conv_dim` equals the target `network_dim` and `conv_alpha` equals the target `network_alpha`. LoRA-Squeeze automatically uses the scheduled current rank and alpha for those convolutional layers while training and resuming.
- Network arguments that create other separate per-module ranks or alphas, such as block dims or Anima regex dims, are not supported.
- Supported factors are Linear/Linear and Conv2d/Conv2d pairs with a 1x1 `lora_up` convolution.

<details>
<summary>日本語</summary>

- single-process学習のみ対応しています。
- DeepSpeedは未対応です。
- `initial_step`と`initial_epoch`は未対応です。
- `torch.compile`関連のoptionは未対応です。
- すべてのLoRA層は、スケジュールされた現在のランクとalphaに統一されている必要があります。
- LoRA-C3Lierは、`conv_dim`がターゲット`network_dim`と等しく、`conv_alpha`がターゲット`network_alpha`と等しい場合に対応します。学習中と再開時は、convolution層にもスケジュールされた現在のランクとalphaを自動的に使用します。
- block dimsやAnima regex dimsなど、モジュールごとに異なるランクまたはalphaを作るnetwork引数は未対応です。
- 対応する因子はLinear/Linear、および1x1の`lora_up`を持つConv2d/Conv2dの組み合わせです。

</details>

LoRA-Squeeze schedule and progress information is written to model metadata and training logs. At a squeeze boundary, logs distinguish the rank/alpha used for the completed optimizer step (`train_dim`/`train_alpha`) from the newly installed current rank/alpha, and record transition statistics separately. Learning-rate and optimizer-derived LR metrics use one snapshot taken after the optimizer and scheduler steps and before the squeeze transition.

<details>
<summary>日本語</summary>

LoRA-Squeezeのスケジュールと進行状況はモデルメタデータと学習ログへ保存されます。圧縮境界のログでは、完了したoptimizer stepで使用したランクとalpha（`train_dim` / `train_alpha`）を、新しく設定した現在のランクとalphaから区別し、遷移統計も別に記録します。learning rateおよびオプティマイザ由来のLR metricsには、optimizer stepとscheduler stepの後、圧縮遷移の前に取得した1つのsnapshotを使用します。

</details>
