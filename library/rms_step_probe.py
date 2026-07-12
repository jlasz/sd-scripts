import argparse
import copy
import math
import os
from typing import Dict, Sequence, Tuple


LINEAR_POLICY = "linear"
PIECEWISE_ENERGY_POLICY = "piecewise_energy_v1"
SCALING_POLICIES = (LINEAR_POLICY, PIECEWISE_ENERGY_POLICY)

# Calibrated from eight Anima 36->9, four-squeeze runs. Energy means RMS**2,
# normalized by the energy measured at the 500-step probe.
ENERGY_DATASET_INTERCEPT = 0.41309579
ENERGY_DATASET_COEFFICIENT = 120.45173548
ENERGY_STAGE_FACTORS = (0.84333973, 0.95492410, 1.08152779, 1.12020838)
ENERGY_SQUEEZE_RETENTION = (0.93776704, 0.91973453, 0.89532405, 0.86626541)


def validate_rms_probe_configuration(args: argparse.Namespace) -> bool:
    target = args.rms_probe_target
    steps = args.rms_probe_steps
    step_multiple = args.rms_probe_adjusted_steps_divisible_by
    policy = getattr(args, "rms_probe_scaling_policy", LINEAR_POLICY)
    final_target = getattr(args, "rms_probe_final_target", None)
    curve_interval = getattr(args, "rms_probe_curve_every_n_steps", 20)
    microbatch_target = getattr(args, "rms_probe_gradient_accumulation_target_microbatches", None)
    minimum_gradient_accumulation = getattr(args, "rms_probe_min_gradient_accumulation_steps", 1)

    if policy not in SCALING_POLICIES:
        raise ValueError("--rms_probe_scaling_policy must be one of: " + ", ".join(SCALING_POLICIES))

    if target is None and steps is None:
        if step_multiple is not None:
            raise ValueError(
                "--rms_probe_adjusted_steps_divisible_by requires --rms_probe_target and --rms_probe_steps"
            )
        if policy != LINEAR_POLICY or final_target is not None or microbatch_target is not None:
            raise ValueError("RMS probe policy, final-target, and compute-budget options require probe settings")
        return False
    if target is None or steps is None:
        raise ValueError("--rms_probe_target and --rms_probe_steps must be specified together")
    if not math.isfinite(target) or target <= 0:
        raise ValueError("--rms_probe_target must be a finite value greater than 0")
    if steps <= 0:
        raise ValueError("--rms_probe_steps must be greater than 0")
    if args.max_train_epochs is not None:
        raise ValueError("RMS probe step estimation requires --max_train_steps, not --max_train_epochs")
    if args.max_train_steps <= 0:
        raise ValueError("--max_train_steps must be greater than 0 when RMS probe step estimation is enabled")
    if steps > args.max_train_steps:
        raise ValueError("--rms_probe_steps cannot exceed --max_train_steps")
    if args.resume:
        raise ValueError("RMS probe step estimation starts both runs from scratch and cannot be used with --resume")
    if args.initial_step is not None or args.initial_epoch is not None:
        raise ValueError("RMS probe step estimation cannot be used with --initial_step or --initial_epoch")
    if args.deepspeed:
        raise ValueError("RMS probe step estimation does not support --deepspeed")
    if step_multiple is not None and step_multiple <= 0:
        raise ValueError("--rms_probe_adjusted_steps_divisible_by must be greater than 0")
    if microbatch_target is not None and microbatch_target <= 0:
        raise ValueError("--rms_probe_gradient_accumulation_target_microbatches must be greater than 0")
    if minimum_gradient_accumulation <= 0:
        raise ValueError("--rms_probe_min_gradient_accumulation_steps must be greater than 0")
    if minimum_gradient_accumulation > args.gradient_accumulation_steps:
        raise ValueError(
            "--rms_probe_min_gradient_accumulation_steps cannot exceed --gradient_accumulation_steps"
        )

    if policy == PIECEWISE_ENERGY_POLICY:
        if curve_interval <= 0:
            raise ValueError("--rms_probe_curve_every_n_steps must be greater than 0")
        if steps % curve_interval != 0:
            raise ValueError("--rms_probe_steps must be divisible by --rms_probe_curve_every_n_steps")
        if final_target is None or not math.isfinite(final_target) or final_target <= 0:
            raise ValueError(
                "--rms_probe_final_target must be a finite value greater than 0 for piecewise_energy_v1"
            )
        if steps != 500:
            raise ValueError("piecewise_energy_v1 is calibrated for --rms_probe_steps=500")
        if args.max_train_steps <= steps * 5:
            raise ValueError("piecewise_energy_v1 requires the first squeeze to occur after the probe")
        expected = {
            "lora_squeeze_start_dim": 36,
            "network_dim": 9,
            "lora_squeeze_num_squeezes": 4,
            "lora_squeeze_train_after_final_squeeze": True,
            "lora_squeeze_step_schedule": "equal",
            "lora_squeeze_rank_schedule": "geometric",
        }
        mismatches = [
            f"{name}={getattr(args, name, None)!r} (expected {value!r})"
            for name, value in expected.items()
            if getattr(args, name, None) != value
        ]
        if mismatches:
            raise ValueError(
                "piecewise_energy_v1 is calibrated for the Anima 36->9 four-squeeze schedule: "
                + ", ".join(mismatches)
            )
    elif final_target is not None:
        raise ValueError("--rms_probe_final_target requires --rms_probe_scaling_policy=piecewise_energy_v1")

    return True


def estimate_rms_adjusted_steps(original_steps: int, target_rms: float, observed_rms: float) -> Tuple[int, float]:
    if original_steps <= 0:
        raise ValueError("original_steps must be greater than 0")
    if not math.isfinite(target_rms) or target_rms <= 0:
        raise ValueError("target_rms must be a finite value greater than 0")
    if not math.isfinite(observed_rms) or observed_rms <= 0:
        raise ValueError("the RMS probe produced a non-finite or zero RMS, so training steps cannot be estimated")

    step_multiplier = target_rms / observed_rms
    adjusted_steps = max(1, math.floor(original_steps * step_multiplier + 0.5))
    return adjusted_steps, step_multiplier


def fit_probe_energy_slope(
    rms_curve: Sequence[Tuple[int, float]], probe_steps: int = 500, fit_start_step: int = 100
) -> float:
    """Fit normalized RMS-squared growth per 1,000 optimizer steps."""

    samples = [(int(step), float(rms)) for step, rms in rms_curve if fit_start_step <= step <= probe_steps]
    if len(samples) < 2:
        raise ValueError("RMS probe curve needs at least two samples between the fit start and probe step")
    if samples[-1][0] != probe_steps:
        raise ValueError("RMS probe curve is missing its final probe-step sample")

    probe_rms = samples[-1][1]
    if not math.isfinite(probe_rms) or probe_rms <= 0:
        raise ValueError("RMS probe curve has a non-finite or zero final RMS")
    if any(not math.isfinite(rms) or rms <= 0 for _, rms in samples):
        raise ValueError("RMS probe curve contains a non-finite or zero RMS")

    xs = [float(step) for step, _ in samples]
    ys = [(rms / probe_rms) ** 2 for _, rms in samples]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0:
        raise ValueError("RMS probe curve samples must use distinct optimizer steps")
    slope_per_step = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    slope_per_1000_steps = slope_per_step * 1000.0
    if not math.isfinite(slope_per_1000_steps) or slope_per_1000_steps <= 0:
        raise ValueError("RMS probe energy slope must be finite and greater than 0")
    return slope_per_1000_steps


def predict_piecewise_energy_final_rms(
    total_steps: int,
    probe_steps: int,
    observed_rms: float,
    dataset_batches_per_epoch: int,
    probe_energy_slope: float,
) -> float:
    """Predict final RMS for the calibrated equal five-segment squeeze schedule."""

    if total_steps <= probe_steps * 5:
        raise ValueError("piecewise energy prediction requires the first squeeze to occur after the probe")
    if dataset_batches_per_epoch <= 0:
        raise ValueError("dataset_batches_per_epoch must be greater than 0")
    if not math.isfinite(observed_rms) or observed_rms <= 0:
        raise ValueError("observed_rms must be a finite value greater than 0")
    if not math.isfinite(probe_energy_slope) or probe_energy_slope <= 0:
        raise ValueError("probe_energy_slope must be a finite value greater than 0")

    probe_energy = observed_rms**2
    segment_steps = total_steps / 5.0
    energy = probe_energy * (1.0 + probe_energy_slope * (segment_steps - probe_steps) / 1000.0)
    later_mean_slope = ENERGY_DATASET_INTERCEPT + ENERGY_DATASET_COEFFICIENT / dataset_batches_per_epoch
    for stage_factor, retention in zip(ENERGY_STAGE_FACTORS, ENERGY_SQUEEZE_RETENTION):
        energy *= retention
        energy += probe_energy * later_mean_slope * stage_factor * segment_steps / 1000.0
    return math.sqrt(max(0.0, energy))


def estimate_piecewise_energy_adjusted_steps(
    original_steps: int,
    final_target_rms: float,
    observed_rms: float,
    probe_steps: int,
    dataset_batches_per_epoch: int,
    rms_curve: Sequence[Tuple[int, float]],
) -> Tuple[int, float, Dict[str, float]]:
    """Solve the calibrated RMS-squared trajectory for the requested final RMS."""

    if original_steps <= 0:
        raise ValueError("original_steps must be greater than 0")
    if not math.isfinite(final_target_rms) or final_target_rms <= 0:
        raise ValueError("final_target_rms must be a finite value greater than 0")
    slope = fit_probe_energy_slope(rms_curve, probe_steps)
    minimum_steps = probe_steps * 5 + 1
    low = minimum_steps
    minimum_prediction = predict_piecewise_energy_final_rms(
        low, probe_steps, observed_rms, dataset_batches_per_epoch, slope
    )
    if minimum_prediction > final_target_rms:
        raise ValueError(
            "requested final RMS is below the piecewise energy model's calibrated step domain"
        )
    high = max(original_steps, minimum_steps)
    while predict_piecewise_energy_final_rms(
        high, probe_steps, observed_rms, dataset_batches_per_epoch, slope
    ) < final_target_rms:
        high *= 2
        if high > 100_000_000:
            raise ValueError("piecewise energy model could not bracket the requested final RMS")

    while low < high:
        midpoint = (low + high) // 2
        predicted = predict_piecewise_energy_final_rms(
            midpoint, probe_steps, observed_rms, dataset_batches_per_epoch, slope
        )
        if predicted < final_target_rms:
            low = midpoint + 1
        else:
            high = midpoint

    adjusted_steps = low
    details = {
        "probe_energy_slope_per_1000_steps": slope,
        "dataset_batches_per_epoch": float(dataset_batches_per_epoch),
        "predicted_final_rms": predict_piecewise_energy_final_rms(
            adjusted_steps, probe_steps, observed_rms, dataset_batches_per_epoch, slope
        ),
    }
    return adjusted_steps, adjusted_steps / original_steps, details


def choose_gradient_accumulation_steps(
    total_steps: int,
    target_microbatches: int | None,
    current_gradient_accumulation_steps: int,
    minimum_gradient_accumulation_steps: int = 1,
) -> int:
    """Keep steps * accumulation near a compute budget without increasing accumulation."""

    if total_steps <= 0 or current_gradient_accumulation_steps <= 0:
        raise ValueError("training steps and gradient accumulation must be greater than 0")
    if minimum_gradient_accumulation_steps <= 0:
        raise ValueError("minimum gradient accumulation must be greater than 0")
    if minimum_gradient_accumulation_steps > current_gradient_accumulation_steps:
        raise ValueError("minimum gradient accumulation cannot exceed current gradient accumulation")
    if target_microbatches is None:
        return current_gradient_accumulation_steps
    if target_microbatches <= 0:
        raise ValueError("target_microbatches must be greater than 0")

    nearest = math.floor(target_microbatches / total_steps + 0.5)
    return min(current_gradient_accumulation_steps, max(minimum_gradient_accumulation_steps, nearest))


def round_steps_to_nearest_multiple(steps: int, multiple: int | None) -> int:
    if steps <= 0:
        raise ValueError("steps must be greater than 0")
    if multiple is None:
        return steps
    if multiple <= 0:
        raise ValueError("multiple must be greater than 0")

    lower = (steps // multiple) * multiple
    upper = lower + multiple
    if lower == 0:
        return upper
    return lower if steps - lower < upper - steps else upper


def build_rms_probe_args(args: argparse.Namespace) -> argparse.Namespace:
    probe_args = copy.deepcopy(args)
    output_name = probe_args.output_name or "last"
    probe_dir_name = f"{output_name}-rms-probe-step{probe_args.rms_probe_steps}"

    probe_args.output_dir = os.path.join(probe_args.output_dir or ".", probe_dir_name)
    probe_args.output_name = probe_dir_name
    probe_args._training_step_limit = probe_args.rms_probe_steps
    probe_args._is_rms_probe_run = True

    # Keep the final probe weights and resumable state, but avoid periodic
    # outputs, external uploads, samples, validation, and tracker runs.
    probe_args.save_every_n_steps = None
    probe_args.save_every_n_epochs = None
    probe_args.save_n_epoch_ratio = None
    probe_args.save_state = False
    probe_args.save_state_on_train_end = True
    probe_args.save_state_to_huggingface = False
    probe_args.huggingface_repo_id = None
    probe_args.sample_every_n_steps = None
    probe_args.sample_every_n_epochs = None
    probe_args.sample_at_first = False
    probe_args.max_validation_steps = 0
    probe_args.logging_dir = None
    probe_args.log_with = None
    probe_args.total_rms_check_every_n_steps = (
        probe_args.rms_probe_curve_every_n_steps
        if probe_args.rms_probe_scaling_policy == PIECEWISE_ENERGY_POLICY
        else 0
    )

    return probe_args
