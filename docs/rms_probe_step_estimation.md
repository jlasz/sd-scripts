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

The default `linear` policy estimates the production step count as:

```text
adjusted_steps = round(max_train_steps * rms_probe_target / observed_probe_rms)
```

Example TOML configuration:

```toml
max_train_steps = 5000
rms_probe_steps = 500
rms_probe_target = 0.0001
rms_probe_scaling_policy = "linear"
rms_probe_adjusted_steps_divisible_by = 5
```

`rms_probe_target` is the reference model's effective total scaled RMS measured at the same probe step. The probe keeps the original `max_train_steps` as its scheduler and LoRA-Squeeze horizon, so its first steps match the unadjusted production configuration.

`rms_probe_adjusted_steps_divisible_by` optionally rounds the estimated production step count to the nearest multiple of the configured value. Ties round upward. This is useful for distributing a LoRA-Squeeze run evenly across its training segments. The option affects only RMS-probe results; ordinary training step counts are unchanged.

Probe artifacts are written beneath the configured output directory in a directory named like `character-rms-probe-step500`. This includes the final LoRA, resumable training state, and `rms_probe_result.json`. Probe artifacts are not uploaded, and probe tracking, sampling, periodic checkpoints, and validation are disabled.

This is a linear approximation. RMS growth may be nonlinear, particularly with non-constant learning-rate schedulers, adaptive optimizers, regularization, or LoRA-Squeeze. Constant learning rates make the estimate easier to interpret.

## Calibrated piecewise RMS-squared policy

`piecewise_energy_v1` models squared RMS as energy. Energy grows linearly inside each LoRA-Squeeze segment and is multiplied by a calibrated retention factor at each squeeze. The probe records an RMS curve, fits its normalized energy slope between steps 100 and 500, reads the dataset's batches per epoch from the data loader, and numerically solves for a requested final RMS.

Example for the calibrated Anima configuration:

```toml
max_train_steps = 4000
gradient_accumulation_steps = 6

rms_probe_steps = 500
rms_probe_curve_every_n_steps = 20
rms_probe_target = 0.00003878279312630184
rms_probe_final_target = 0.00008425384599385171
rms_probe_scaling_policy = "piecewise_energy_v1"
rms_probe_adjusted_steps_divisible_by = 100

# Optional: reduce, but never increase, gradient accumulation so that the
# adjusted step count times accumulation remains near this compute budget.
rms_probe_gradient_accumulation_target_microbatches = 25500
rms_probe_min_gradient_accumulation_steps = 3

lora_squeeze_start_dim = 36
network_dim = 9
lora_squeeze_num_squeezes = 4
lora_squeeze_train_after_final_squeeze = true
lora_squeeze_step_schedule = "equal"
lora_squeeze_rank_schedule = "geometric"
```

`rms_probe_final_target` is the desired RMS at the end of production training. Unlike the probe-step reference RMS, it makes the final objective explicit. `rms_probe_target` remains recorded for reference and compatibility.

The current coefficients were fitted to Anima, learning rate `6e-5`, constant AdamW, rank/alpha 9/3, start rank 36, four equal geometric squeezes, and gradient accumulation 4 through 6. The trainer enforces the rank and squeeze layout. Treat other learning rates, optimizers, datasets outside the observed size range, and accumulation below 4 as extrapolation; create a separately calibrated policy instead of silently reusing these coefficients.

The result JSON records the complete probe curve, fitted energy slope, detected batches per epoch, predicted final RMS, original and adjusted gradient accumulation, and the selected policy.

Limitations:

- Use `max_train_steps`; `max_train_epochs` is not supported.
- The feature cannot be combined with `resume`, `initial_step`, `initial_epoch`, or DeepSpeed.
- Both probe settings must be provided together.
- `piecewise_energy_v1` requires a 500-step probe, a positive final RMS target, and the calibrated 36-to-9 four-squeeze schedule.
