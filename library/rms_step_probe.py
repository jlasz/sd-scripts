import argparse
import copy
import math
import os
from typing import Tuple


def validate_rms_probe_configuration(args: argparse.Namespace) -> bool:
    target = args.rms_probe_target
    steps = args.rms_probe_steps

    if target is None and steps is None:
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
    probe_args.total_rms_check_every_n_steps = 0

    return probe_args
