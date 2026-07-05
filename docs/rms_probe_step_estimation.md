# RMS probe step estimation

To log RMS during an ordinary training run without changing its stopping point, set:

```toml
total_rms_check_every_n_steps = 20
```

The value is logged as `strength/total_rms` to configured trackers and printed as `total_rms`. The default `0` disables periodic RMS logging.

RMS probe step estimation runs two independent trainings:

1. A local probe starts from the base model and stops after `rms_probe_steps` optimizer steps.
2. The trainer measures the effective total scaled LoRA RMS and saves the probe's final weights and full training state.
3. Probe weights and optimizer state are discarded from memory.
4. Production training starts from the base model with a fresh optimizer and scheduler.

The production step count is estimated as:

```text
adjusted_steps = round(max_train_steps * rms_probe_target / observed_probe_rms)
```

Example TOML configuration:

```toml
max_train_steps = 5000
rms_probe_steps = 500
rms_probe_target = 0.0001
```

`rms_probe_target` is the reference model's effective total scaled RMS measured at the same probe step. The probe keeps the original `max_train_steps` as its scheduler and LoRA-Squeeze horizon, so its first steps match the unadjusted production configuration.

Probe artifacts are written beneath the configured output directory in a directory named like `character-rms-probe-step500`. This includes the final LoRA, resumable training state, and `rms_probe_result.json`. Probe artifacts are not uploaded, and probe tracking, sampling, periodic checkpoints, and validation are disabled.

This is a linear approximation. RMS growth may be nonlinear, particularly with non-constant learning-rate schedulers, adaptive optimizers, regularization, or LoRA-Squeeze. Constant learning rates make the estimate easier to interpret.

Limitations:

- Use `max_train_steps`; `max_train_epochs` is not supported.
- The feature cannot be combined with `resume`, `initial_step`, `initial_epoch`, or DeepSpeed.
- Both probe settings must be provided together.
