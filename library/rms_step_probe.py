import argparse
import copy
import math
import os
from typing import Dict, Sequence, Tuple


LINEAR_POLICY = "linear"
PIECEWISE_ENERGY_POLICY = "piecewise_energy_v1"
LOGLINEAR_POLICY = "loglinear_v1"
SCALING_POLICIES = (LINEAR_POLICY, PIECEWISE_ENERGY_POLICY, LOGLINEAR_POLICY)
# Policies that predict the final RMS of the Anima 36->9 four-squeeze schedule
# from the rank-36 probe and therefore need --rms_probe_final_target.
FINAL_TARGET_POLICIES = (PIECEWISE_ENERGY_POLICY, LOGLINEAR_POLICY)
CALIBRATION_PROBE_STEPS = 500
PROBE_ENERGY_FIT_START_STEP = 100
TRAJECTORY_MODEL_VERSION = 2.9
LATER_ENERGY_EXPONENT = 0.5

# Calibrated from thirty-three Anima 36->9, four-squeeze runs across M'rissi,
# Izutsumi, Neeko, Wilykit, Mutio, Rosine, Crossbreed Priscilla, Marcia, and
# Beastgirl, plus two revisions of the mixed object-and-character 001 dataset.
# Rosine has twice
# the total family weight because its ten runs intentionally span substantial
# dataset revisions; each other family has one equal share.
# Repeated runs divide their family's weight. The compressed-stage regression
# uses standardized log batches per epoch, log probe RMS, log production
# gradient accumulation, and normalized rank-36 probe energy slope. Rank 36
# continues to use normalized energy (RMS**2). After a squeeze, the trajectory
# instead advances linearly in normalized RMS, equivalent to dE/dt proportional
# to E**0.5.
COMPRESSED_RMS_VELOCITY_FEATURE_MEANS = (
    4.613142092252085,
    -10.094529015166287,
    1.7881722220563165,
    2.091907840499041,
)
COMPRESSED_RMS_VELOCITY_FEATURE_SCALES = (
    0.7768684508256919,
    0.1815147663136012,
    0.36058280372456136,
    0.07929778322517349,
)
COMPRESSED_RMS_VELOCITY_INTERCEPT = 0.3696820795204259
COMPRESSED_RMS_VELOCITY_COEFFICIENTS = (
    -0.1286926015753895,
    -0.04127748670884288,
    0.05865581303898875,
    -0.015298846803360307,
)
# Relative four-squeeze stage factors, not absolute-rank keys. This keeps the
# state equation independent of the literal 25/18/13/9 rank names.
COMPRESSED_RMS_STAGE_FACTORS = (
    0.9444806933385512,
    0.9826392772351606,
    1.0308213924899727,
    1.0420586369363154,
)
# Family-weighted means from the thirty-two trajectories with retained-energy
# telemetry. M'rissi predates that telemetry; Rosine keeps its double share.
ENERGY_SQUEEZE_RETENTION = (
    0.9346726002675882,
    0.9243092958504103,
    0.9072586769415538,
    0.8746724149492009,
)
ADJUSTED_PROBE_MIN_POST_SQUEEZE_SAMPLES = 5

# Within-dataset log-log fits from matched Anima rank-36 trajectories. Six
# exact-revision pairs span Izutsumi, Neeko, Wilykit, Mutio, Rosine_first, and
# Rosine_second. A GA 7 versus GA 2 Rosine near-pair extends the measured range;
# its datasets differ by one image (104 versus 103). Same-GA repeats are
# geometrically averaged before fitting so reruns do not gain extra weight.
PROBE_RMS_GRADIENT_ACCUMULATION_ELASTICITY = 0.13407367061621528
PROBE_ENERGY_SLOPE_GRADIENT_ACCUMULATION_ELASTICITY = 0.028683585047468874

# loglinear_v1: one log-linear fit of the final RMS, with no stage model.
#   log(final_rms) = intercept
#                    + steps_exponent * log(total optimizer steps)
#                    + steps_per_epoch_exponent * log(batches per epoch / gradient accumulation)
#                    + reference_rms_exponent * log(step-500 rank-36 reference RMS)
# Fitted by tools/refit_rms_probe_loglinear.py on thirty-seven completed Anima
# 36->9 four-squeeze trajectories (the piecewise table plus the third 001 run
# and three Ahsoka runs). Leave-one-family-out endpoint MAPE is 2.4%.
LOGLINEAR_INTERCEPT = -10.317644499406368
LOGLINEAR_STEPS_EXPONENT = 0.7835062456574756
LOGLINEAR_STEPS_PER_EPOCH_EXPONENT = -0.23862292006208405
LOGLINEAR_REFERENCE_RMS_EXPONENT = 0.48139220669062843


def validate_rms_probe_configuration(args: argparse.Namespace) -> bool:
    target = args.rms_probe_target
    steps = args.rms_probe_steps
    step_multiple = args.rms_probe_adjusted_steps_divisible_by
    policy = getattr(args, "rms_probe_scaling_policy", LINEAR_POLICY)
    final_target = getattr(args, "rms_probe_final_target", None)
    force_adjusted_probe = getattr(args, "rms_probe_force_adjusted_probe", False)
    curve_interval = getattr(args, "rms_probe_curve_every_n_steps", 20)
    microbatch_target = getattr(args, "rms_probe_gradient_accumulation_target_microbatches", None)
    minimum_gradient_accumulation = getattr(args, "rms_probe_min_gradient_accumulation_steps", 1)
    gradient_accumulation_rounding_bias = getattr(
        args, "rms_probe_gradient_accumulation_rounding_bias", 0.6
    )
    first_segment_ratio = getattr(args, "lora_squeeze_first_segment_ratio", 1.0)
    final_segment_ratio = getattr(args, "lora_squeeze_final_segment_ratio", 1.0)
    if first_segment_ratio is None:
        first_segment_ratio = 1.0
    if final_segment_ratio is None:
        final_segment_ratio = 1.0

    if policy not in SCALING_POLICIES:
        raise ValueError("--rms_probe_scaling_policy must be one of: " + ", ".join(SCALING_POLICIES))

    if target is None and steps is None:
        if step_multiple is not None:
            raise ValueError(
                "--rms_probe_adjusted_steps_divisible_by requires --rms_probe_target and --rms_probe_steps"
            )
        if (
            policy != LINEAR_POLICY
            or final_target is not None
            or microbatch_target is not None
            or force_adjusted_probe
        ):
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
    if (
        not math.isfinite(gradient_accumulation_rounding_bias)
        or gradient_accumulation_rounding_bias < 0
        or gradient_accumulation_rounding_bias > 1
    ):
        raise ValueError("--rms_probe_gradient_accumulation_rounding_bias must be between 0 and 1")

    if policy in FINAL_TARGET_POLICIES:
        if curve_interval <= 0:
            raise ValueError("--rms_probe_curve_every_n_steps must be greater than 0")
        periodic_curve_steps = set(range(curve_interval, steps + 1, curve_interval))
        periodic_curve_steps.add(steps)
        first_squeeze_step = squeeze_schedule_steps(
            args.max_train_steps,
            first_segment_ratio,
            final_segment_ratio,
        )[0]
        rank36_fit_start_step = (
            PROBE_ENERGY_FIT_START_STEP
            if first_squeeze_step > steps
            else curve_interval
        )
        rank36_fit_sample_steps = [
            step
            for step in periodic_curve_steps
            if rank36_fit_start_step <= step < first_squeeze_step
        ]
        if len(rank36_fit_sample_steps) < 2:
            raise ValueError(
                f"{policy} needs at least two rank-36 RMS curve samples "
                "before squeeze one; increase --max_train_steps, decrease "
                "--rms_probe_curve_every_n_steps, or adjust the squeeze segment ratios"
            )
        if final_target is None or not math.isfinite(final_target) or final_target <= 0:
            raise ValueError(
                f"--rms_probe_final_target must be a finite value greater than 0 for {policy}"
            )
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
                f"{policy} is calibrated for the Anima 36->9 four-squeeze schedule: "
                + ", ".join(mismatches)
            )
        if policy == LOGLINEAR_POLICY:
            if first_segment_ratio != 1.0 or final_segment_ratio != 1.0:
                raise ValueError("loglinear_v1 is calibrated for equal squeeze segments only")
            if force_adjusted_probe:
                raise ValueError("--rms_probe_force_adjusted_probe is not used by loglinear_v1")
    elif final_target is not None:
        raise ValueError(
            "--rms_probe_final_target requires --rms_probe_scaling_policy=piecewise_energy_v1 or loglinear_v1"
        )
    elif force_adjusted_probe:
        raise ValueError(
            "--rms_probe_force_adjusted_probe requires "
            "--rms_probe_scaling_policy=piecewise_energy_v1"
        )

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
    rms_curve: Sequence[Tuple[int, float]],
    probe_steps: int = CALIBRATION_PROBE_STEPS,
    fit_start_step: int = PROBE_ENERGY_FIT_START_STEP,
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


def rescale_probe_measurements_for_gradient_accumulation(
    observed_rms: float,
    probe_energy_slope: float,
    probe_gradient_accumulation_steps: int,
    production_gradient_accumulation_steps: int,
) -> Tuple[float, float, Dict[str, float]]:
    """Map rank-36 probe measurements to a different production accumulation."""

    if not math.isfinite(observed_rms) or observed_rms <= 0:
        raise ValueError("observed_rms must be a finite value greater than 0")
    if not math.isfinite(probe_energy_slope) or probe_energy_slope <= 0:
        raise ValueError("probe_energy_slope must be a finite value greater than 0")
    if probe_gradient_accumulation_steps <= 0:
        raise ValueError("probe gradient accumulation must be greater than 0")
    if production_gradient_accumulation_steps <= 0:
        raise ValueError("production gradient accumulation must be greater than 0")

    accumulation_ratio = (
        production_gradient_accumulation_steps
        / probe_gradient_accumulation_steps
    )
    rms_scale = accumulation_ratio ** PROBE_RMS_GRADIENT_ACCUMULATION_ELASTICITY
    slope_scale = (
        accumulation_ratio
        ** PROBE_ENERGY_SLOPE_GRADIENT_ACCUMULATION_ELASTICITY
    )
    adjusted_rms = observed_rms * rms_scale
    adjusted_slope = probe_energy_slope * slope_scale
    return adjusted_rms, adjusted_slope, {
        "probe_gradient_accumulation_steps": float(
            probe_gradient_accumulation_steps
        ),
        "production_gradient_accumulation_steps": float(
            production_gradient_accumulation_steps
        ),
        "probe_gradient_accumulation_ratio": accumulation_ratio,
        "observed_probe_rms_at_probe_accumulation": observed_rms,
        "production_reference_rms": adjusted_rms,
        "probe_rms_accumulation_scale": rms_scale,
        "observed_probe_energy_slope_at_probe_accumulation": probe_energy_slope,
        "production_probe_energy_slope": adjusted_slope,
        "probe_energy_slope_accumulation_scale": slope_scale,
    }


def squeeze_schedule_steps(
    total_steps: int,
    first_segment_ratio: float = 1.0,
    final_segment_ratio: float = 1.0,
) -> Tuple[int, int, int, int]:
    """Return the exact boundaries used by a ratio-adjusted equal schedule."""

    if total_steps < 5:
        raise ValueError("total_steps must be at least 5 for a five-segment schedule")
    if not math.isfinite(first_segment_ratio) or first_segment_ratio <= 0:
        raise ValueError("first_segment_ratio must be a finite value greater than 0")
    if not math.isfinite(final_segment_ratio) or final_segment_ratio <= 0:
        raise ValueError("final_segment_ratio must be a finite value greater than 0")

    segment_weights = (first_segment_ratio, 1.0, 1.0, 1.0, final_segment_ratio)
    total_weight = sum(segment_weights)
    cumulative_weight = 0.0
    steps = []
    for weight in segment_weights[:-1]:
        cumulative_weight += weight
        steps.append(math.floor(total_steps * cumulative_weight / total_weight))
    if any(current <= previous for previous, current in zip((0, *steps[:-1]), steps)):
        raise ValueError(
            "ratio-adjusted RMS probe squeeze schedule is not strictly increasing; "
            "use less extreme segment ratios or more training steps"
        )
    if steps[-1] >= total_steps:
        raise ValueError("ratio-adjusted RMS probe squeeze schedule has no final training segment")
    return tuple(steps)


def squeeze_segment_steps(
    total_steps: int,
    first_segment_ratio: float = 1.0,
    final_segment_ratio: float = 1.0,
) -> Tuple[int, int, int, int, int]:
    """Return the exact five segment lengths used by LoRA-Squeeze."""

    squeeze_steps = squeeze_schedule_steps(
        total_steps,
        first_segment_ratio,
        final_segment_ratio,
    )
    boundaries = (0, *squeeze_steps, total_steps)
    return tuple(
        boundaries[index + 1] - boundaries[index]
        for index in range(5)
    )


def epoch_accumulation_group_sizes(
    dataset_batches_per_epoch: int,
    gradient_accumulation_steps: int,
) -> Tuple[int, ...]:
    """Return the microbatch count of each optimizer update in one epoch.

    Accelerate synchronizes at the end of the dataloader by default. A
    non-divisible remainder therefore becomes a real optimizer update even
    though every constituent loss is still divided by the configured
    gradient-accumulation value.
    """

    if dataset_batches_per_epoch <= 0:
        raise ValueError("dataset_batches_per_epoch must be greater than 0")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be greater than 0")

    full_updates, remainder = divmod(
        dataset_batches_per_epoch,
        gradient_accumulation_steps,
    )
    groups = (gradient_accumulation_steps,) * full_updates
    if remainder:
        groups += (remainder,)
    return groups or (dataset_batches_per_epoch,)


def count_optimizer_step_microbatches(
    optimizer_steps: int,
    dataset_batches_per_epoch: int,
    gradient_accumulation_steps: int,
) -> int:
    """Count real microbatches consumed by epoch-synchronized updates."""

    if optimizer_steps < 0:
        raise ValueError("optimizer_steps must not be negative")
    groups = epoch_accumulation_group_sizes(
        dataset_batches_per_epoch,
        gradient_accumulation_steps,
    )
    completed_epochs, updates_in_epoch = divmod(optimizer_steps, len(groups))
    return (
        completed_epochs * dataset_batches_per_epoch
        + sum(groups[:updates_in_epoch])
    )


def optimizer_steps_for_microbatch_budget(
    target_microbatches: int,
    dataset_batches_per_epoch: int,
    gradient_accumulation_steps: int,
) -> int:
    """Return the first optimizer-step count that reaches a microbatch budget."""

    if target_microbatches <= 0:
        raise ValueError("target_microbatches must be greater than 0")
    groups = epoch_accumulation_group_sizes(
        dataset_batches_per_epoch,
        gradient_accumulation_steps,
    )
    completed_epochs, remainder = divmod(
        target_microbatches,
        dataset_batches_per_epoch,
    )
    optimizer_steps = completed_epochs * len(groups)
    if remainder == 0:
        return optimizer_steps
    consumed = 0
    for group_size in groups:
        consumed += group_size
        optimizer_steps += 1
        if consumed >= remainder:
            return optimizer_steps
    raise AssertionError("epoch accumulation groups did not cover one epoch")


def equal_squeeze_steps(total_steps: int) -> Tuple[int, int, int, int]:
    """Return the exact four boundaries used by an unweighted equal schedule."""

    return squeeze_schedule_steps(total_steps)


def probe_schedule_needs_adjusted_probe(
    total_steps: int,
    probe_steps: int,
    first_segment_ratio: float = 1.0,
    final_segment_ratio: float = 1.0,
) -> bool:
    """Whether the proposed production schedule squeezes during the probe window."""

    if probe_steps <= 0:
        raise ValueError("probe_steps must be greater than 0")
    return (
        squeeze_schedule_steps(
            total_steps,
            first_segment_ratio,
            final_segment_ratio,
        )[0]
        <= probe_steps
    )


def should_run_adjusted_probe(
    scaling_policy: str,
    total_steps: int,
    probe_steps: int,
    force_adjusted_probe: bool = False,
    first_segment_ratio: float = 1.0,
    final_segment_ratio: float = 1.0,
    has_observed_compressed_stage_evidence: bool = False,
) -> bool:
    """Whether Probe 2 adds missing schedule evidence or was explicitly forced."""

    return scaling_policy == PIECEWISE_ENERGY_POLICY and (
        force_adjusted_probe
        or (
            not has_observed_compressed_stage_evidence
            and probe_schedule_needs_adjusted_probe(
                total_steps,
                probe_steps,
                first_segment_ratio,
                final_segment_ratio,
            )
        )
    )


def choose_adjusted_probe_training_steps(
    reference_probe_steps: int,
    original_gradient_accumulation_steps: int,
    adjusted_gradient_accumulation_steps: int,
    production_steps: int,
    curve_interval: int,
    minimum_post_squeeze_samples: int = ADJUSTED_PROBE_MIN_POST_SQUEEZE_SAMPLES,
    first_segment_ratio: float = 1.0,
    final_segment_ratio: float = 1.0,
    dataset_batches_per_epoch: int | None = None,
) -> Tuple[int, Dict[str, float]]:
    """Choose Probe 2's observation horizon from compute and phase coverage."""

    if reference_probe_steps <= 0:
        raise ValueError("reference_probe_steps must be greater than 0")
    if original_gradient_accumulation_steps <= 0:
        raise ValueError("original gradient accumulation must be greater than 0")
    if adjusted_gradient_accumulation_steps <= 0:
        raise ValueError("adjusted gradient accumulation must be greater than 0")
    if production_steps < 5:
        raise ValueError("production_steps must be at least 5")
    if curve_interval <= 0:
        raise ValueError("curve_interval must be greater than 0")
    if minimum_post_squeeze_samples <= 0:
        raise ValueError("minimum_post_squeeze_samples must be greater than 0")
    if dataset_batches_per_epoch is not None and dataset_batches_per_epoch <= 0:
        raise ValueError("dataset_batches_per_epoch must be greater than 0")

    nominal_microbatch_budget = (
        reference_probe_steps * original_gradient_accumulation_steps
    )
    if dataset_batches_per_epoch is None:
        actual_microbatch_budget = nominal_microbatch_budget
        budget_based_steps = math.ceil(
            nominal_microbatch_budget / adjusted_gradient_accumulation_steps
        )
    else:
        actual_microbatch_budget = count_optimizer_step_microbatches(
            reference_probe_steps,
            dataset_batches_per_epoch,
            original_gradient_accumulation_steps,
        )
        budget_based_steps = optimizer_steps_for_microbatch_budget(
            actual_microbatch_budget,
            dataset_batches_per_epoch,
            adjusted_gradient_accumulation_steps,
        )
    rounded_budget_steps = (
        math.ceil(budget_based_steps / curve_interval) * curve_interval
    )

    first_squeeze_step = squeeze_schedule_steps(
        production_steps,
        first_segment_ratio,
        final_segment_ratio,
    )[0]
    phase_coverage_steps = (
        first_squeeze_step + minimum_post_squeeze_samples * curve_interval
    )
    rounded_phase_coverage_steps = (
        math.ceil(phase_coverage_steps / curve_interval) * curve_interval
    )

    selected_steps = min(
        reference_probe_steps,
        production_steps,
        max(rounded_budget_steps, rounded_phase_coverage_steps),
    )
    details = {
        "adjusted_probe_reference_steps": float(reference_probe_steps),
        "adjusted_probe_nominal_microbatch_budget": float(nominal_microbatch_budget),
        "adjusted_probe_actual_microbatch_budget": float(actual_microbatch_budget),
        "adjusted_probe_budget_based_steps": float(budget_based_steps),
        "adjusted_probe_rounded_budget_steps": float(rounded_budget_steps),
        "adjusted_probe_first_squeeze_step": float(first_squeeze_step),
        "adjusted_probe_first_segment_ratio": float(first_segment_ratio),
        "adjusted_probe_final_segment_ratio": float(final_segment_ratio),
        "adjusted_probe_min_post_squeeze_samples": float(minimum_post_squeeze_samples),
        "adjusted_probe_phase_coverage_steps": float(phase_coverage_steps),
        "adjusted_probe_rounded_phase_coverage_steps": float(
            rounded_phase_coverage_steps
        ),
        "adjusted_probe_training_steps": float(selected_steps),
        "adjusted_probe_nominal_training_microbatches": float(
            selected_steps * adjusted_gradient_accumulation_steps
        ),
        "adjusted_probe_actual_training_microbatches": float(
            selected_steps * adjusted_gradient_accumulation_steps
            if dataset_batches_per_epoch is None
            else count_optimizer_step_microbatches(
                selected_steps,
                dataset_batches_per_epoch,
                adjusted_gradient_accumulation_steps,
            )
        ),
    }
    return selected_steps, details


def fit_schedule_aware_probe_energy_model(
    rms_curve: Sequence[Tuple[int, float]],
    total_steps: int,
    probe_steps: int = CALIBRATION_PROBE_STEPS,
    curve_interval: int = 20,
    first_segment_ratio: float = 1.0,
    final_segment_ratio: float = 1.0,
) -> Tuple[float, float, Dict[str, float]]:
    """Fit rank-36 energy and express it at the calibration reference step."""

    if curve_interval <= 0:
        raise ValueError("curve_interval must be greater than 0")
    if probe_steps <= 0:
        raise ValueError("probe_steps must be greater than 0")
    if not rms_curve:
        raise ValueError("RMS probe curve is empty")
    observation_steps = max(int(step) for step, _ in rms_curve)
    first_squeeze_step = squeeze_schedule_steps(
        total_steps,
        first_segment_ratio,
        final_segment_ratio,
    )[0]
    calibration_sample = next(
        (
            float(rms)
            for step, rms in reversed(rms_curve)
            if int(step) == probe_steps and int(step) < first_squeeze_step
        ),
        None,
    )
    if calibration_sample is not None:
        slope = fit_probe_energy_slope(rms_curve, probe_steps)
        return calibration_sample, slope, {
            "first_squeeze_step": float(first_squeeze_step),
            "rank36_fit_samples": float(
                sum(
                    1
                    for step, _ in rms_curve
                    if PROBE_ENERGY_FIT_START_STEP <= int(step) <= probe_steps
                )
            ),
            "probe_observation_steps": float(observation_steps),
            "calibration_reference_steps": float(probe_steps),
            "reference_rms_is_extrapolated": 0.0,
            "first_segment_ratio": float(first_segment_ratio),
            "final_segment_ratio": float(final_segment_ratio),
        }
    fit_start_step = (
        PROBE_ENERGY_FIT_START_STEP
        if first_squeeze_step > observation_steps
        else curve_interval
    )
    samples = [
        (int(step), float(rms))
        for step, rms in rms_curve
        if fit_start_step <= step <= observation_steps and step < first_squeeze_step
    ]
    if len(samples) < 2:
        raise ValueError("RMS probe needs at least two rank-36 samples before squeeze one")
    if any(not math.isfinite(rms) or rms <= 0 for _, rms in samples):
        raise ValueError("RMS probe curve contains a non-finite or zero RMS")
    xs = [float(step) for step, _ in samples]
    ys = [rms**2 for _, rms in samples]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0:
        raise ValueError("RMS probe curve samples must use distinct optimizer steps")
    energy_slope_per_step = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    ) / denominator
    reference_energy = y_mean + energy_slope_per_step * (probe_steps - x_mean)
    if (
        not math.isfinite(energy_slope_per_step)
        or energy_slope_per_step <= 0
        or not math.isfinite(reference_energy)
        or reference_energy <= 0
    ):
        raise ValueError("adjusted RMS probe produced an invalid rank-36 energy fit")
    reference_rms = math.sqrt(reference_energy)
    normalized_slope = energy_slope_per_step / reference_energy * 1000.0
    return reference_rms, normalized_slope, {
        "first_squeeze_step": float(first_squeeze_step),
        "rank36_fit_samples": float(len(samples)),
        "probe_observation_steps": float(observation_steps),
        "calibration_reference_steps": float(probe_steps),
        "reference_rms_is_extrapolated": float(observation_steps != probe_steps),
        "first_segment_ratio": float(first_segment_ratio),
        "final_segment_ratio": float(final_segment_ratio),
    }


def fit_adjusted_probe_rank36_transfer(
    primary_rms_curve: Sequence[Tuple[int, float]],
    adjusted_rms_curve: Sequence[Tuple[int, float]],
    adjusted_total_steps: int,
    adjusted_probe_steps: int,
    primary_reference_rms: float,
    primary_probe_energy_slope: float,
    first_segment_ratio: float = 1.0,
    final_segment_ratio: float = 1.0,
    primary_total_steps: int | None = None,
) -> Tuple[float, float, Dict[str, float]]:
    """Transfer Probe 1's long rank-36 fit using Probe 2's shared early window.

    Probe 2 may squeeze too early to extrapolate its own short rank-36 line to
    the calibration reference reliably. Comparing both fresh probes at the
    exact same pre-squeeze steps instead measures the relative RMS and slope
    change caused by Probe 2's accumulation/update pattern, while Probe 1
    supplies the long-horizon trajectory shape.
    """

    if adjusted_total_steps <= 0 or adjusted_probe_steps <= 0:
        raise ValueError("adjusted probe and production steps must be greater than 0")
    if primary_total_steps is not None and primary_total_steps <= 0:
        raise ValueError("primary_total_steps must be greater than 0 when supplied")
    if not math.isfinite(primary_reference_rms) or primary_reference_rms <= 0:
        raise ValueError("primary_reference_rms must be finite and greater than 0")
    if (
        not math.isfinite(primary_probe_energy_slope)
        or primary_probe_energy_slope <= 0
    ):
        raise ValueError(
            "primary_probe_energy_slope must be finite and greater than 0"
        )

    adjusted_first_squeeze_step = squeeze_schedule_steps(
        adjusted_total_steps,
        first_segment_ratio,
        final_segment_ratio,
    )[0]
    primary_first_squeeze_step = (
        squeeze_schedule_steps(
            primary_total_steps,
            first_segment_ratio,
            final_segment_ratio,
        )[0]
        if primary_total_steps is not None
        else adjusted_first_squeeze_step
    )
    shared_rank36_end_step = min(
        primary_first_squeeze_step,
        adjusted_first_squeeze_step,
    )
    primary_by_step = {int(step): float(rms) for step, rms in primary_rms_curve}
    adjusted_by_step = {int(step): float(rms) for step, rms in adjusted_rms_curve}
    shared_steps = sorted(
        step
        for step in primary_by_step.keys() & adjusted_by_step.keys()
        if 0 < step <= adjusted_probe_steps and step < shared_rank36_end_step
    )
    if len(shared_steps) < 2:
        raise ValueError(
            "adjusted RMS probe needs at least two shared rank-36 samples with Probe 1"
        )

    if any(
        not math.isfinite(rms) or rms <= 0
        for step in shared_steps
        for rms in (primary_by_step[step], adjusted_by_step[step])
    ):
        raise ValueError("RMS probe curves contain a non-finite or zero RMS")
    primary_energies = [primary_by_step[step] ** 2 for step in shared_steps]
    adjusted_energies = [adjusted_by_step[step] ** 2 for step in shared_steps]

    energy_scale_denominator = sum(energy**2 for energy in primary_energies)
    energy_scale = sum(
        primary_energy * adjusted_energy
        for primary_energy, adjusted_energy in zip(
            primary_energies,
            adjusted_energies,
        )
    ) / energy_scale_denominator
    if not math.isfinite(energy_scale) or energy_scale <= 0:
        raise ValueError("adjusted RMS probe produced an invalid rank-36 energy scale")

    x_mean = sum(shared_steps) / len(shared_steps)
    denominator = sum((step - x_mean) ** 2 for step in shared_steps)
    if denominator <= 0:
        raise ValueError("shared RMS probe samples must use distinct optimizer steps")

    def fitted_energy_slope(energies: Sequence[float]) -> float:
        energy_mean = sum(energies) / len(energies)
        return sum(
            (step - x_mean) * (energy - energy_mean)
            for step, energy in zip(shared_steps, energies)
        ) / denominator

    primary_window_slope = fitted_energy_slope(primary_energies)
    adjusted_window_slope = fitted_energy_slope(adjusted_energies)
    slope_scale_is_measured = (
        math.isfinite(primary_window_slope)
        and primary_window_slope > 0
        and math.isfinite(adjusted_window_slope)
        and adjusted_window_slope > 0
    )
    normalized_slope_scale = (
        adjusted_window_slope / primary_window_slope / energy_scale
        if slope_scale_is_measured
        else 1.0
    )
    if not math.isfinite(normalized_slope_scale) or normalized_slope_scale <= 0:
        normalized_slope_scale = 1.0
        slope_scale_is_measured = False

    rms_scale = math.sqrt(energy_scale)
    transferred_reference_rms = primary_reference_rms * rms_scale
    transferred_probe_energy_slope = (
        primary_probe_energy_slope * normalized_slope_scale
    )
    return transferred_reference_rms, transferred_probe_energy_slope, {
        "rank36_transfer_first_squeeze_step": float(shared_rank36_end_step),
        "rank36_transfer_primary_first_squeeze_step": float(
            primary_first_squeeze_step
        ),
        "rank36_transfer_adjusted_first_squeeze_step": float(
            adjusted_first_squeeze_step
        ),
        "rank36_transfer_sample_count": float(len(shared_steps)),
        "rank36_transfer_first_sample_step": float(shared_steps[0]),
        "rank36_transfer_last_sample_step": float(shared_steps[-1]),
        "rank36_transfer_energy_scale": energy_scale,
        "rank36_transfer_rms_scale": rms_scale,
        "rank36_transfer_primary_window_energy_slope": primary_window_slope,
        "rank36_transfer_adjusted_window_energy_slope": adjusted_window_slope,
        "rank36_transfer_normalized_slope_scale": normalized_slope_scale,
        "rank36_transfer_slope_scale_is_measured": float(
            slope_scale_is_measured
        ),
        "rank36_transfer_reference_rms": transferred_reference_rms,
        "rank36_transfer_energy_slope_per_1000_steps": (
            transferred_probe_energy_slope
        ),
    }


def fit_observed_later_mean_rms_velocity(
    rms_curve: Sequence[Tuple[int, float]],
    total_steps: int,
    probe_steps: int,
    reference_rms: float,
    curve_interval: int = 20,
    first_segment_ratio: float = 1.0,
    final_segment_ratio: float = 1.0,
) -> Tuple[float | None, Dict[str, float]]:
    """Estimate normalized compressed-stage RMS velocity without crossing squeezes."""

    if not math.isfinite(reference_rms) or reference_rms <= 0:
        raise ValueError("reference_rms must be a finite value greater than 0")
    if curve_interval <= 0:
        raise ValueError("curve_interval must be greater than 0")
    squeeze_steps = squeeze_schedule_steps(
        total_steps,
        first_segment_ratio,
        final_segment_ratio,
    )
    boundaries = (0, *squeeze_steps, total_steps)
    weighted_velocities = []
    details: Dict[str, float] = {
        "first_segment_ratio": float(first_segment_ratio),
        "final_segment_ratio": float(final_segment_ratio),
    }
    for segment_index in range(1, 5):
        segment_start = boundaries[segment_index]
        segment_end = boundaries[segment_index + 1]
        observed_end = min(segment_end, probe_steps)
        fit_start = segment_start + curve_interval
        fit_end = observed_end
        if segment_end <= probe_steps:
            fit_end = segment_end - curve_interval
        samples = [
            (int(step), float(rms))
            for step, rms in rms_curve
            if fit_start <= step <= fit_end
        ]
        if len(samples) < 2:
            continue
        if any(not math.isfinite(rms) or rms <= 0 for _, rms in samples):
            raise ValueError("RMS probe curve contains a non-finite or zero RMS")
        xs = [float(step) for step, _ in samples]
        ys = [rms / reference_rms for _, rms in samples]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denominator = sum((x - x_mean) ** 2 for x in xs)
        if denominator <= 0:
            continue
        raw_velocity = (
            sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
            / denominator
            * 1000.0
        )
        if not math.isfinite(raw_velocity) or raw_velocity <= 0:
            continue
        base_velocity = (
            raw_velocity / COMPRESSED_RMS_STAGE_FACTORS[segment_index - 1]
        )
        observed_span = xs[-1] - xs[0]
        if observed_span <= 0:
            continue
        weighted_velocities.append((base_velocity, observed_span))
        details[
            f"observed_stage_{segment_index}_base_rms_velocity"
        ] = base_velocity
        details[f"observed_stage_{segment_index}_span"] = observed_span

    if not weighted_velocities:
        details["observed_later_stage_count"] = 0.0
        return None, details
    total_weight = sum(weight for _, weight in weighted_velocities)
    later_mean_velocity = (
        sum(velocity * weight for velocity, weight in weighted_velocities)
        / total_weight
    )
    details["observed_later_stage_count"] = float(len(weighted_velocities))
    details["observed_later_stage_span"] = total_weight
    details["observed_later_mean_rms_velocity"] = later_mean_velocity
    return later_mean_velocity, details


def rescale_observed_later_mean_rms_velocity(
    later_mean_rms_velocity: float,
    dataset_batches_per_epoch: int,
    source_reference_rms: float,
    source_gradient_accumulation_steps: int,
    source_probe_energy_slope: float,
    target_reference_rms: float,
    target_gradient_accumulation_steps: int,
    target_probe_energy_slope: float,
) -> Tuple[float, Dict[str, float]]:
    """Map an observed compressed-stage velocity to another probe condition."""

    if not math.isfinite(later_mean_rms_velocity) or later_mean_rms_velocity <= 0:
        raise ValueError(
            "later_mean_rms_velocity must be a finite value greater than 0"
        )
    source_modeled_velocity = estimate_augmented_later_mean_rms_velocity(
        dataset_batches_per_epoch,
        source_reference_rms,
        source_gradient_accumulation_steps,
        source_probe_energy_slope,
    )
    target_modeled_velocity = estimate_augmented_later_mean_rms_velocity(
        dataset_batches_per_epoch,
        target_reference_rms,
        target_gradient_accumulation_steps,
        target_probe_energy_slope,
    )
    velocity_scale = target_modeled_velocity / source_modeled_velocity
    rescaled_velocity = later_mean_rms_velocity * velocity_scale
    return rescaled_velocity, {
        "observed_later_velocity_condition_scale": velocity_scale,
        "observed_later_velocity_source_modeled_value": source_modeled_velocity,
        "observed_later_velocity_target_modeled_value": target_modeled_velocity,
        "observed_later_velocity_rescaled_value": rescaled_velocity,
    }


def combine_observed_later_mean_rms_velocities(
    observations: Sequence[Tuple[float | None, Dict[str, float]]],
) -> Tuple[float | None, Dict[str, float]]:
    """Span-weight compatible compressed-stage measurements from fresh probes."""

    weighted_observations = []
    details: Dict[str, float] = {}
    for probe_index, (velocity, observation_details) in enumerate(
        observations,
        start=1,
    ):
        if velocity is None:
            continue
        span = float(observation_details.get("observed_later_stage_span", 0.0))
        if (
            not math.isfinite(velocity)
            or velocity <= 0
            or not math.isfinite(span)
            or span <= 0
        ):
            raise ValueError(
                "observed compressed-stage velocities need finite positive values and spans"
            )
        weighted_observations.append((float(velocity), span))
        details[f"combined_observed_later_probe_{probe_index}_velocity"] = float(
            velocity
        )
        details[f"combined_observed_later_probe_{probe_index}_span"] = span

    if not weighted_observations:
        details["combined_observed_later_probe_count"] = 0.0
        return None, details
    total_span = sum(span for _, span in weighted_observations)
    combined_velocity = (
        sum(velocity * span for velocity, span in weighted_observations)
        / total_span
    )
    details.update(
        {
            "combined_observed_later_probe_count": float(
                len(weighted_observations)
            ),
            "combined_observed_later_stage_span": total_span,
            "combined_observed_later_mean_rms_velocity": combined_velocity,
        }
    )
    return combined_velocity, details


def combine_observed_squeeze_retentions(
    retention_sets: Sequence[Sequence[float | None]],
) -> Tuple[Tuple[float | None, ...], Dict[str, float]]:
    """Combine matching squeeze stages across probes without shifting stage labels."""

    if any(len(values) > len(ENERGY_SQUEEZE_RETENTION) for values in retention_sets):
        raise ValueError(
            "observed squeeze retention has more stages than the calibrated schedule"
        )
    stage_values = [[] for _ in ENERGY_SQUEEZE_RETENTION]
    for values in retention_sets:
        for stage_index, value in enumerate(values):
            if value is None:
                continue
            if not math.isfinite(value) or value <= 0 or value > 1:
                raise ValueError(
                    "observed squeeze retention values must be finite and in (0, 1]"
                )
            stage_values[stage_index].append(float(value))

    last_observed_stage = max(
        (index for index, values in enumerate(stage_values, start=1) if values),
        default=0,
    )
    combined = tuple(
        sum(values) / len(values) if values else None
        for values in stage_values[:last_observed_stage]
    )
    details: Dict[str, float] = {
        "combined_squeeze_retention_stage_count": float(
            sum(value is not None for value in combined)
        ),
        "combined_squeeze_retention_probe_count": float(
            sum(
                any(value is not None for value in values)
                for values in retention_sets
            )
        ),
    }
    for stage_index, values in enumerate(stage_values, start=1):
        if not values:
            continue
        details[f"combined_squeeze_{stage_index}_probe_count"] = float(len(values))
        details[f"combined_squeeze_{stage_index}_retained_energy_mean"] = (
            sum(values) / len(values)
        )
    return combined, details


def estimate_augmented_later_mean_rms_velocity(
    dataset_batches_per_epoch: int,
    observed_rms: float,
    gradient_accumulation_steps: int,
    probe_energy_slope: float,
) -> float:
    """Estimate normalized compressed-stage RMS velocity with the regression."""

    if dataset_batches_per_epoch <= 0:
        raise ValueError("dataset_batches_per_epoch must be greater than 0")
    if not math.isfinite(observed_rms) or observed_rms <= 0:
        raise ValueError("observed_rms must be finite and greater than 0")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be greater than 0")
    if not math.isfinite(probe_energy_slope) or probe_energy_slope <= 0:
        raise ValueError("probe_energy_slope must be finite and greater than 0")

    features = (
        math.log(dataset_batches_per_epoch),
        math.log(observed_rms),
        math.log(gradient_accumulation_steps),
        probe_energy_slope,
    )
    standardized_features = tuple(
        (feature - mean) / scale
        for feature, mean, scale in zip(
            features,
            COMPRESSED_RMS_VELOCITY_FEATURE_MEANS,
            COMPRESSED_RMS_VELOCITY_FEATURE_SCALES,
        )
    )
    later_mean_velocity = COMPRESSED_RMS_VELOCITY_INTERCEPT + sum(
        coefficient * feature
        for coefficient, feature in zip(
            COMPRESSED_RMS_VELOCITY_COEFFICIENTS,
            standardized_features,
        )
    )
    if not math.isfinite(later_mean_velocity) or later_mean_velocity <= 0:
        raise ValueError(
            "augmented compressed-stage RMS velocity is non-positive for the supplied inputs"
        )
    return later_mean_velocity


def fit_observed_squeeze_retention_scale(
    observed_retentions: Sequence[float | None],
) -> Tuple[float, Dict[str, float]]:
    """Scale calibrated retention from matching stages observed by probe runs."""

    if len(observed_retentions) > len(ENERGY_SQUEEZE_RETENTION):
        raise ValueError(
            "observed squeeze retention has more stages than the calibrated schedule"
        )
    ratios = []
    details: Dict[str, float] = {}
    for index, (observed, calibrated) in enumerate(
        zip(observed_retentions, ENERGY_SQUEEZE_RETENTION),
        start=1,
    ):
        if observed is None:
            continue
        if not math.isfinite(observed) or observed <= 0 or observed > 1:
            raise ValueError(
                "observed squeeze retention values must be finite and in (0, 1]"
            )
        ratios.append(observed / calibrated)
        details[f"observed_squeeze_{index}_retained_energy_mean"] = float(observed)
    scale = (
        math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
        if ratios
        else 1.0
    )
    details["observed_squeeze_retention_count"] = float(len(ratios))
    details["observed_squeeze_retention_scale"] = float(scale)
    return scale, details


def predict_piecewise_energy_final_rms(
    total_steps: int,
    probe_steps: int,
    observed_rms: float,
    dataset_batches_per_epoch: int,
    probe_energy_slope: float,
    gradient_accumulation_steps: int,
    later_mean_rms_velocity: float | None = None,
    first_segment_ratio: float = 1.0,
    final_segment_ratio: float = 1.0,
    squeeze_retention_scale: float = 1.0,
) -> float:
    """Predict final RMS with rank-36 energy and compressed-stage RMS velocity."""

    if total_steps < 5:
        raise ValueError("piecewise energy prediction requires at least 5 total steps")
    if probe_steps <= 0:
        raise ValueError("probe_steps must be greater than 0")
    if dataset_batches_per_epoch <= 0:
        raise ValueError("dataset_batches_per_epoch must be greater than 0")
    if not math.isfinite(observed_rms) or observed_rms <= 0:
        raise ValueError("observed_rms must be a finite value greater than 0")
    if not math.isfinite(probe_energy_slope) or probe_energy_slope <= 0:
        raise ValueError("probe_energy_slope must be a finite value greater than 0")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be greater than 0")
    if not math.isfinite(squeeze_retention_scale) or squeeze_retention_scale <= 0:
        raise ValueError("squeeze_retention_scale must be finite and greater than 0")

    probe_energy = observed_rms**2
    segment_steps = squeeze_segment_steps(
        total_steps,
        first_segment_ratio,
        final_segment_ratio,
    )
    energy = probe_energy * (
        1.0
        + probe_energy_slope
        * (segment_steps[0] - probe_steps)
        / 1000.0
    )
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError("piecewise energy prediction produced non-positive first-segment energy")
    later_mean_velocity = later_mean_rms_velocity
    if later_mean_velocity is None:
        later_mean_velocity = estimate_augmented_later_mean_rms_velocity(
            dataset_batches_per_epoch,
            observed_rms,
            gradient_accumulation_steps,
            probe_energy_slope,
        )
    if not math.isfinite(later_mean_velocity) or later_mean_velocity <= 0:
        raise ValueError("later_mean_rms_velocity must be a finite value greater than 0")
    rms = math.sqrt(energy)
    for stage_factor, retention, stage_steps in zip(
        COMPRESSED_RMS_STAGE_FACTORS,
        ENERGY_SQUEEZE_RETENTION,
        segment_steps[1:],
    ):
        rms *= math.sqrt(min(1.0, retention * squeeze_retention_scale))
        rms += (
            observed_rms
            * later_mean_velocity
            * stage_factor
            * stage_steps
            / 1000.0
        )
    if not math.isfinite(rms) or rms <= 0:
        raise ValueError("piecewise energy prediction produced a non-positive final RMS")
    return rms


def estimate_piecewise_energy_adjusted_steps(
    original_steps: int,
    final_target_rms: float,
    observed_rms: float,
    probe_steps: int,
    dataset_batches_per_epoch: int,
    rms_curve: Sequence[Tuple[int, float]],
    gradient_accumulation_steps: int,
    probe_energy_slope: float | None = None,
    later_mean_rms_velocity: float | None = None,
    first_segment_ratio: float = 1.0,
    final_segment_ratio: float = 1.0,
    squeeze_retention_scale: float = 1.0,
) -> Tuple[int, float, Dict[str, float]]:
    """Solve the calibrated hybrid trajectory for the requested final RMS."""

    if original_steps <= 0:
        raise ValueError("original_steps must be greater than 0")
    if not math.isfinite(final_target_rms) or final_target_rms <= 0:
        raise ValueError("final_target_rms must be a finite value greater than 0")
    slope = probe_energy_slope
    if slope is None:
        slope = fit_probe_energy_slope(rms_curve, probe_steps)
    if not math.isfinite(slope) or slope <= 0:
        raise ValueError("probe_energy_slope must be a finite value greater than 0")
    if probe_steps <= 0:
        raise ValueError("probe_steps must be greater than 0")
    minimum_positive_segment = probe_steps - 1000.0 / slope
    required_first_segment_steps = max(
        1,
        math.floor(minimum_positive_segment) + 1,
    )
    total_segment_weight = first_segment_ratio + 3.0 + final_segment_ratio
    minimum_steps = max(
        5,
        math.ceil(
            required_first_segment_steps
            * total_segment_weight
            / first_segment_ratio
        ),
    )
    low = minimum_steps
    minimum_prediction = predict_piecewise_energy_final_rms(
        low,
        probe_steps,
        observed_rms,
        dataset_batches_per_epoch,
        slope,
        gradient_accumulation_steps,
        later_mean_rms_velocity,
        first_segment_ratio,
        final_segment_ratio,
        squeeze_retention_scale,
    )
    if minimum_prediction > final_target_rms:
        raise ValueError(
            "requested final RMS is below the minimum schedule compatible with the "
            f"step-{probe_steps} calibration reference"
        )
    high = max(original_steps, minimum_steps)
    while predict_piecewise_energy_final_rms(
        high,
        probe_steps,
        observed_rms,
        dataset_batches_per_epoch,
        slope,
        gradient_accumulation_steps,
        later_mean_rms_velocity,
        first_segment_ratio,
        final_segment_ratio,
        squeeze_retention_scale,
    ) < final_target_rms:
        high *= 2
        if high > 100_000_000:
            raise ValueError("piecewise energy model could not bracket the requested final RMS")

    while low < high:
        midpoint = (low + high) // 2
        predicted = predict_piecewise_energy_final_rms(
            midpoint,
            probe_steps,
            observed_rms,
            dataset_batches_per_epoch,
            slope,
            gradient_accumulation_steps,
            later_mean_rms_velocity,
            first_segment_ratio,
            final_segment_ratio,
            squeeze_retention_scale,
        )
        if predicted < final_target_rms:
            low = midpoint + 1
        else:
            high = midpoint

    adjusted_steps = low
    adjusted_segment_steps = squeeze_segment_steps(
        adjusted_steps,
        first_segment_ratio,
        final_segment_ratio,
    )
    details = {
        "probe_energy_slope_per_1000_steps": slope,
        "dataset_batches_per_epoch": float(dataset_batches_per_epoch),
        "trajectory_model_version": TRAJECTORY_MODEL_VERSION,
        "later_energy_exponent": LATER_ENERGY_EXPONENT,
        "later_mean_rms_velocity_gradient_accumulation_steps": float(
            gradient_accumulation_steps
        ),
        "first_segment_ratio": float(first_segment_ratio),
        "final_segment_ratio": float(final_segment_ratio),
        "squeeze_retention_scale": float(squeeze_retention_scale),
        **{
            f"segment_{index}_steps": float(segment_steps)
            for index, segment_steps in enumerate(adjusted_segment_steps, start=1)
        },
        "later_mean_rms_velocity_is_observed": float(
            later_mean_rms_velocity is not None
        ),
        "later_mean_rms_velocity_uses_augmented_regression": float(
            later_mean_rms_velocity is None
        ),
        "later_mean_rms_velocity_per_1000_steps": (
            later_mean_rms_velocity
            if later_mean_rms_velocity is not None
            else estimate_augmented_later_mean_rms_velocity(
                dataset_batches_per_epoch,
                observed_rms,
                gradient_accumulation_steps,
                slope,
            )
        ),
        "predicted_final_rms": predict_piecewise_energy_final_rms(
            adjusted_steps,
            probe_steps,
            observed_rms,
            dataset_batches_per_epoch,
            slope,
            gradient_accumulation_steps,
            later_mean_rms_velocity,
            first_segment_ratio,
            final_segment_ratio,
            squeeze_retention_scale,
        ),
    }
    return adjusted_steps, adjusted_steps / original_steps, details


def update_piecewise_production_prediction(
    model_details: Dict[str, float],
    production_steps: int,
    probe_steps: int,
    observed_rms: float,
    dataset_batches_per_epoch: int,
    gradient_accumulation_steps: int,
    first_segment_ratio: float = 1.0,
    final_segment_ratio: float = 1.0,
) -> Dict[str, float]:
    """Recompute model output for the actual, possibly rounded production schedule."""

    updated_details = dict(model_details)
    updated_details["estimated_steps_predicted_final_rms"] = model_details[
        "predicted_final_rms"
    ]
    updated_details["predicted_final_rms"] = predict_piecewise_energy_final_rms(
        production_steps,
        probe_steps,
        observed_rms,
        dataset_batches_per_epoch,
        model_details["probe_energy_slope_per_1000_steps"],
        gradient_accumulation_steps,
        model_details["later_mean_rms_velocity_per_1000_steps"],
        first_segment_ratio,
        final_segment_ratio,
        model_details.get("squeeze_retention_scale", 1.0),
    )
    production_segment_steps = squeeze_segment_steps(
        production_steps,
        first_segment_ratio,
        final_segment_ratio,
    )
    updated_details.update(
        {
            f"segment_{index}_steps": float(segment_steps)
            for index, segment_steps in enumerate(
                production_segment_steps,
                start=1,
            )
        }
    )
    return updated_details


def choose_gradient_accumulation_steps(
    total_steps: int,
    target_microbatches: int | None,
    current_gradient_accumulation_steps: int,
    minimum_gradient_accumulation_steps: int = 1,
    rounding_bias: float = 0.6,
    dataset_batches_per_epoch: int | None = None,
) -> int:
    """Choose accumulation near a compute budget with configurable upward bias.

    When the dataset length is supplied, every configured accumulation value
    remains eligible. Candidates are compared using the real microbatch count
    produced by Accelerate's repeating epoch-boundary update pattern instead
    of assuming ``optimizer_steps * accumulation``.
    """

    if total_steps <= 0 or current_gradient_accumulation_steps <= 0:
        raise ValueError("training steps and gradient accumulation must be greater than 0")
    if minimum_gradient_accumulation_steps <= 0:
        raise ValueError("minimum gradient accumulation must be greater than 0")
    if not math.isfinite(rounding_bias) or rounding_bias < 0 or rounding_bias > 1:
        raise ValueError("rounding_bias must be between 0 and 1")
    if dataset_batches_per_epoch is not None and dataset_batches_per_epoch <= 0:
        raise ValueError("dataset_batches_per_epoch must be greater than 0")
    if target_microbatches is None:
        return current_gradient_accumulation_steps
    if target_microbatches <= 0:
        raise ValueError("target_microbatches must be greater than 0")

    ideal_accumulation = target_microbatches / total_steps
    if dataset_batches_per_epoch is None:
        rounded = (
            math.ceil(ideal_accumulation)
            if rounding_bias == 1
            else math.floor(ideal_accumulation + rounding_bias)
        )
        return max(minimum_gradient_accumulation_steps, rounded)

    maximum_automatic_accumulation = max(
        minimum_gradient_accumulation_steps,
        dataset_batches_per_epoch,
    )
    candidates = [
        (
            accumulation,
            count_optimizer_step_microbatches(
                total_steps,
                dataset_batches_per_epoch,
                accumulation,
            ),
        )
        for accumulation in range(
            minimum_gradient_accumulation_steps,
            maximum_automatic_accumulation + 1,
        )
    ]

    def representative(candidate_microbatches: int) -> int:
        same_compute = [
            accumulation
            for accumulation, microbatches in candidates
            if microbatches == candidate_microbatches
        ]
        return min(
            same_compute,
            key=lambda accumulation: (
                abs(accumulation - ideal_accumulation),
                abs(accumulation - current_gradient_accumulation_steps),
                accumulation,
            ),
        )

    candidate_counts = sorted({microbatches for _, microbatches in candidates})
    lower_counts = [
        microbatches
        for microbatches in candidate_counts
        if microbatches <= target_microbatches
    ]
    upper_counts = [
        microbatches
        for microbatches in candidate_counts
        if microbatches >= target_microbatches
    ]
    if not lower_counts:
        return representative(candidate_counts[0])
    if not upper_counts:
        return representative(candidate_counts[-1])

    lower_count = lower_counts[-1]
    upper_count = upper_counts[0]
    if lower_count == upper_count:
        return representative(lower_count)
    position = (target_microbatches - lower_count) / (upper_count - lower_count)
    selected_count = (
        upper_count if position >= 1.0 - rounding_bias else lower_count
    )
    return representative(selected_count)


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


def estimate_piecewise_training_plan(
    original_steps: int,
    final_target_rms: float,
    observed_rms: float,
    probe_steps: int,
    dataset_batches_per_epoch: int,
    rms_curve: Sequence[Tuple[int, float]],
    current_gradient_accumulation_steps: int,
    probe_gradient_accumulation_steps: int,
    target_microbatches: int | None,
    minimum_gradient_accumulation_steps: int = 1,
    gradient_accumulation_rounding_bias: float = 0.6,
    adjusted_steps_divisible_by: int | None = None,
    probe_energy_slope: float | None = None,
    later_mean_rms_velocity: float | None = None,
    later_mean_rms_velocity_reference_gradient_accumulation_steps: int | None = None,
    first_segment_ratio: float = 1.0,
    final_segment_ratio: float = 1.0,
    squeeze_retention_scale: float = 1.0,
) -> Tuple[int, float, int, int, int, Dict[str, float]]:
    """Select a stable piecewise step and gradient-accumulation plan."""

    reference_probe_energy_slope = probe_energy_slope
    if reference_probe_energy_slope is None:
        reference_probe_energy_slope = fit_probe_energy_slope(
            rms_curve,
            probe_steps,
        )
    model_gradient_accumulation_steps = choose_gradient_accumulation_steps(
        original_steps,
        target_microbatches,
        current_gradient_accumulation_steps,
        minimum_gradient_accumulation_steps,
        gradient_accumulation_rounding_bias,
        dataset_batches_per_epoch,
    )
    seen_gradient_accumulations = set()
    while model_gradient_accumulation_steps not in seen_gradient_accumulations:
        seen_gradient_accumulations.add(model_gradient_accumulation_steps)
        (
            production_reference_rms,
            production_probe_energy_slope,
            probe_accumulation_details,
        ) = rescale_probe_measurements_for_gradient_accumulation(
            observed_rms,
            reference_probe_energy_slope,
            probe_gradient_accumulation_steps,
            model_gradient_accumulation_steps,
        )
        model_later_mean_rms_velocity = later_mean_rms_velocity
        later_velocity_accumulation_scale = 1.0
        if (
            later_mean_rms_velocity is not None
            and later_mean_rms_velocity_reference_gradient_accumulation_steps
            is not None
        ):
            reference_later_velocity = estimate_augmented_later_mean_rms_velocity(
                dataset_batches_per_epoch,
                observed_rms,
                later_mean_rms_velocity_reference_gradient_accumulation_steps,
                reference_probe_energy_slope,
            )
            modeled_later_velocity = estimate_augmented_later_mean_rms_velocity(
                dataset_batches_per_epoch,
                production_reference_rms,
                model_gradient_accumulation_steps,
                production_probe_energy_slope,
            )
            later_velocity_accumulation_scale = (
                modeled_later_velocity / reference_later_velocity
            )
            model_later_mean_rms_velocity *= later_velocity_accumulation_scale
        estimated_steps, step_multiplier, model_details = (
            estimate_piecewise_energy_adjusted_steps(
                original_steps=original_steps,
                final_target_rms=final_target_rms,
                observed_rms=production_reference_rms,
                probe_steps=probe_steps,
                dataset_batches_per_epoch=dataset_batches_per_epoch,
                rms_curve=rms_curve,
                gradient_accumulation_steps=model_gradient_accumulation_steps,
                probe_energy_slope=production_probe_energy_slope,
                later_mean_rms_velocity=model_later_mean_rms_velocity,
                first_segment_ratio=first_segment_ratio,
                final_segment_ratio=final_segment_ratio,
                squeeze_retention_scale=squeeze_retention_scale,
            )
        )
        model_details.update(probe_accumulation_details)
        if later_mean_rms_velocity is not None:
            model_details.update(
                {
                    "observed_later_mean_rms_velocity_at_probe_accumulation": (
                        later_mean_rms_velocity
                    ),
                    "observed_later_mean_rms_velocity_probe_gradient_accumulation_steps": (
                        float(
                            later_mean_rms_velocity_reference_gradient_accumulation_steps
                            or model_gradient_accumulation_steps
                        )
                    ),
                    "observed_later_mean_rms_velocity_accumulation_scale": (
                        later_velocity_accumulation_scale
                    ),
                }
            )
        adjusted_steps = round_steps_to_nearest_multiple(
            estimated_steps,
            adjusted_steps_divisible_by,
        )
        uncapped_adjusted_gradient_accumulation_steps = (
            choose_gradient_accumulation_steps(
                adjusted_steps,
                target_microbatches,
                current_gradient_accumulation_steps,
                minimum_gradient_accumulation_steps,
                gradient_accumulation_rounding_bias,
            )
        )
        adjusted_gradient_accumulation_steps = choose_gradient_accumulation_steps(
            adjusted_steps,
            target_microbatches,
            current_gradient_accumulation_steps,
            minimum_gradient_accumulation_steps,
            gradient_accumulation_rounding_bias,
            dataset_batches_per_epoch,
        )
        if adjusted_gradient_accumulation_steps == model_gradient_accumulation_steps:
            accumulation_groups = epoch_accumulation_group_sizes(
                dataset_batches_per_epoch,
                adjusted_gradient_accumulation_steps,
            )
            model_details.update(
                {
                    "optimizer_updates_per_epoch": float(len(accumulation_groups)),
                    "gradient_accumulation_epoch_remainder": float(
                        dataset_batches_per_epoch
                        % adjusted_gradient_accumulation_steps
                    ),
                    "gradient_accumulation_is_epoch_compatible": float(
                        dataset_batches_per_epoch
                        % adjusted_gradient_accumulation_steps
                        == 0
                    ),
                    "actual_production_microbatches": float(
                        count_optimizer_step_microbatches(
                            adjusted_steps,
                            dataset_batches_per_epoch,
                            adjusted_gradient_accumulation_steps,
                        )
                    ),
                    "nominal_production_microbatches": float(
                        adjusted_steps * adjusted_gradient_accumulation_steps
                    ),
                }
            )
            return (
                estimated_steps,
                step_multiplier,
                adjusted_steps,
                uncapped_adjusted_gradient_accumulation_steps,
                adjusted_gradient_accumulation_steps,
                model_details,
            )
        model_gradient_accumulation_steps = adjusted_gradient_accumulation_steps

    raise ValueError(
        "RMS probe could not find a stable step and gradient-accumulation "
        "combination for the hybrid trajectory model"
    )


def predict_loglinear_final_rms(
    total_steps: int,
    dataset_batches_per_epoch: int,
    gradient_accumulation_steps: int,
    reference_rms: float,
) -> float:
    """Final RMS of the 36->9 four-squeeze schedule under loglinear_v1."""

    if total_steps <= 0 or dataset_batches_per_epoch <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("steps, batches per epoch, and gradient accumulation must be greater than 0")
    if not math.isfinite(reference_rms) or reference_rms <= 0:
        raise ValueError("reference_rms must be a finite value greater than 0")
    return math.exp(
        LOGLINEAR_INTERCEPT
        + LOGLINEAR_STEPS_EXPONENT * math.log(total_steps)
        + LOGLINEAR_STEPS_PER_EPOCH_EXPONENT
        * math.log(dataset_batches_per_epoch / gradient_accumulation_steps)
        + LOGLINEAR_REFERENCE_RMS_EXPONENT * math.log(reference_rms)
    )


def estimate_loglinear_steps(
    final_target_rms: float,
    dataset_batches_per_epoch: int,
    gradient_accumulation_steps: int,
    reference_rms: float,
) -> int:
    """Invert loglinear_v1 for the optimizer steps that reach the final target."""

    if not math.isfinite(final_target_rms) or final_target_rms <= 0:
        raise ValueError("final_target_rms must be a finite value greater than 0")
    one_step_rms = predict_loglinear_final_rms(
        1,
        dataset_batches_per_epoch,
        gradient_accumulation_steps,
        reference_rms,
    )
    steps = (final_target_rms / one_step_rms) ** (1.0 / LOGLINEAR_STEPS_EXPONENT)
    return max(1, math.floor(steps + 0.5))


def estimate_loglinear_training_plan(
    original_steps: int,
    final_target_rms: float,
    reference_rms: float,
    dataset_batches_per_epoch: int,
    current_gradient_accumulation_steps: int,
    probe_gradient_accumulation_steps: int,
    target_microbatches: int | None,
    minimum_gradient_accumulation_steps: int = 1,
    gradient_accumulation_rounding_bias: float = 0.6,
    adjusted_steps_divisible_by: int | None = None,
) -> Tuple[int, float, int, int, int, Dict[str, float]]:
    """Select a stable loglinear_v1 step and gradient-accumulation plan.

    ``reference_rms`` is the step-500 rank-36 reference measured at the probe's
    gradient accumulation. It is moved to each candidate production
    accumulation with the calibrated rank-36 elasticity before solving.
    """

    if original_steps <= 0:
        raise ValueError("original_steps must be greater than 0")
    if probe_gradient_accumulation_steps <= 0:
        raise ValueError("probe gradient accumulation must be greater than 0")
    model_gradient_accumulation_steps = choose_gradient_accumulation_steps(
        original_steps,
        target_microbatches,
        current_gradient_accumulation_steps,
        minimum_gradient_accumulation_steps,
        gradient_accumulation_rounding_bias,
        dataset_batches_per_epoch,
    )
    seen_gradient_accumulations = set()
    while model_gradient_accumulation_steps not in seen_gradient_accumulations:
        seen_gradient_accumulations.add(model_gradient_accumulation_steps)
        accumulation_ratio = (
            model_gradient_accumulation_steps / probe_gradient_accumulation_steps
        )
        production_reference_rms = reference_rms * (
            accumulation_ratio ** PROBE_RMS_GRADIENT_ACCUMULATION_ELASTICITY
        )
        estimated_steps = estimate_loglinear_steps(
            final_target_rms,
            dataset_batches_per_epoch,
            model_gradient_accumulation_steps,
            production_reference_rms,
        )
        adjusted_steps = round_steps_to_nearest_multiple(
            estimated_steps,
            adjusted_steps_divisible_by,
        )
        uncapped_adjusted_gradient_accumulation_steps = (
            choose_gradient_accumulation_steps(
                adjusted_steps,
                target_microbatches,
                current_gradient_accumulation_steps,
                minimum_gradient_accumulation_steps,
                gradient_accumulation_rounding_bias,
            )
        )
        adjusted_gradient_accumulation_steps = choose_gradient_accumulation_steps(
            adjusted_steps,
            target_microbatches,
            current_gradient_accumulation_steps,
            minimum_gradient_accumulation_steps,
            gradient_accumulation_rounding_bias,
            dataset_batches_per_epoch,
        )
        if adjusted_gradient_accumulation_steps == model_gradient_accumulation_steps:
            return (
                estimated_steps,
                estimated_steps / original_steps,
                adjusted_steps,
                uncapped_adjusted_gradient_accumulation_steps,
                adjusted_gradient_accumulation_steps,
                {
                    "model": "loglinear_v1",
                    "final_target_rms": final_target_rms,
                    "probe_gradient_accumulation_steps": float(
                        probe_gradient_accumulation_steps
                    ),
                    "production_gradient_accumulation_steps": float(
                        adjusted_gradient_accumulation_steps
                    ),
                    "observed_probe_rms_at_probe_accumulation": reference_rms,
                    "production_reference_rms": production_reference_rms,
                    "optimizer_updates_per_epoch_nominal": (
                        dataset_batches_per_epoch
                        / adjusted_gradient_accumulation_steps
                    ),
                    "predicted_final_rms_at_estimated_steps": (
                        predict_loglinear_final_rms(
                            estimated_steps,
                            dataset_batches_per_epoch,
                            adjusted_gradient_accumulation_steps,
                            production_reference_rms,
                        )
                    ),
                    "predicted_final_rms_at_adjusted_steps": (
                        predict_loglinear_final_rms(
                            adjusted_steps,
                            dataset_batches_per_epoch,
                            adjusted_gradient_accumulation_steps,
                            production_reference_rms,
                        )
                    ),
                },
            )
        model_gradient_accumulation_steps = adjusted_gradient_accumulation_steps

    raise ValueError(
        "RMS probe could not find a stable step and gradient-accumulation "
        "combination for loglinear_v1"
    )


def build_rms_probe_args(
    args: argparse.Namespace,
    probe_index: int = 1,
    production_steps: int | None = None,
    gradient_accumulation_steps: int | None = None,
    training_step_limit: int | None = None,
) -> argparse.Namespace:
    probe_args = copy.deepcopy(args)
    output_name = probe_args.output_name or "last"
    if probe_index not in (1, 2):
        raise ValueError("probe_index must be 1 or 2")
    if probe_index == 1:
        if (
            production_steps is not None
            or gradient_accumulation_steps is not None
            or training_step_limit is not None
        ):
            raise ValueError("adjusted production settings are only valid for probe 2")
        training_step_limit = probe_args.rms_probe_steps
        probe_dir_name = f"{output_name}-rms-probe-step{probe_args.rms_probe_steps}"
    else:
        if production_steps is None or production_steps < 5:
            raise ValueError("probe 2 production_steps must be at least 5")
        if gradient_accumulation_steps is None or gradient_accumulation_steps <= 0:
            raise ValueError("probe 2 gradient accumulation must be greater than 0")
        if training_step_limit is None:
            training_step_limit = probe_args.rms_probe_steps
        if training_step_limit <= 0 or training_step_limit > probe_args.rms_probe_steps:
            raise ValueError("probe 2 training_step_limit must be between 1 and rms_probe_steps")
        if training_step_limit > production_steps:
            raise ValueError("probe 2 training_step_limit cannot exceed production_steps")
        probe_args.max_train_steps = production_steps
        probe_args.gradient_accumulation_steps = gradient_accumulation_steps
        probe_dir_name = (
            f"{output_name}-rms-probe2-step{training_step_limit}"
            f"-steps{production_steps}-ga{gradient_accumulation_steps}"
        )

    probe_args.output_dir = os.path.join(probe_args.output_dir or ".", probe_dir_name)
    probe_args.output_name = probe_dir_name
    probe_args._training_step_limit = training_step_limit
    probe_args._is_rms_probe_run = True
    probe_args._rms_probe_index = probe_index

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
