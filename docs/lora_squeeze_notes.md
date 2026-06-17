# LoRA-Squeeze Notes

LoRA-Squeeze trains a LoRA at a higher starting rank, periodically SVD-compresses the effective delta into lower ranks, and ends at the normal `network_dim`.

Example config block:

```toml
[lora_squeeze_arguments]
lora_squeeze_start_dim = 32
lora_squeeze_num_squeezes = 4
lora_squeeze_train_after_final_squeeze = true
lora_squeeze_step_schedule = "inverse_rank_proportional"
lora_squeeze_final_segment_ratio = 2.0
```

Core settings:

- `network_dim`: final LoRA rank after all squeezes.
- `network_alpha`: final alpha after all squeezes.
- `lora_squeeze_start_dim`: initial rank. Must be greater than `network_dim`.
- `lora_squeeze_num_squeezes`: number of rank reductions to perform. Alias: `lora_squeeze_amount_of_squeezes`.
- `lora_squeeze_train_after_final_squeeze`: if `true`, the final rank LoRA gets its own training segment after the last squeeze. If `false`, the last squeeze happens at the end of training.

Rank and alpha behavior:

- Intermediate ranks are linearly spaced from `lora_squeeze_start_dim` down to `network_dim`, rounding down.
- For each rank `N`, alpha is `network_alpha / sqrt(network_dim) * sqrt(N)`.
- Each squeeze uses exact SVD on the effective LoRA delta and keeps the top singular directions for the next rank.

Step spreading:

- If `lora_squeeze_step_schedule` is omitted, all training segments are equal length.
- `rank_proportional`: larger ranks get more steps.
- `sqrt_rank_proportional`: larger ranks get more steps, but less aggressively.
- `inverse_rank_proportional`: smaller ranks get more steps.
- `inverse_sqrt_rank_proportional`: smaller ranks get more steps, but less aggressively.
- `lora_squeeze_final_segment_ratio` multiplies only the final post-squeeze segment, and only matters when `lora_squeeze_train_after_final_squeeze = true`.

Useful RMS guardrail:

```toml
target_total_rms = 0.00009
total_rms_check_every_n_steps = 20
```

This stops training once the total scaled LoRA RMS reaches the target. It does not make RMS grow by itself, so pair it with enough `max_train_steps` or a stronger learning-rate setup.
