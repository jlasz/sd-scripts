"""LoRA-Squeeze scheduling, resume state, and configuration validation."""

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union


TRAIN_STATE_LORA_SQUEEZE_KEY = "lora_squeeze"
UNSUPPORTED_NETWORK_ARG_KEYS = {
    "block_alphas",
    "block_dims",
    "conv_block_alphas",
    "conv_block_dims",
    "network_reg_dims",
}


def get_remaining_training_steps(max_train_steps: int, completed_steps: int) -> int:
    return max(0, max_train_steps - completed_steps)


def is_training_budget_exhausted(max_train_steps: int, completed_steps: int) -> bool:
    return completed_steps >= max_train_steps


def get_resume_epoch_and_batch_offset(
    completed_steps: int,
    num_batches_per_epoch: int,
    gradient_accumulation_steps: int,
) -> Tuple[int, int]:
    if completed_steps < 0:
        raise ValueError("completed_steps must not be negative")
    if num_batches_per_epoch <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("batch and gradient-accumulation counts must be positive")
    updates_per_epoch = math.ceil(num_batches_per_epoch / gradient_accumulation_steps)
    completed_epochs, updates_in_epoch = divmod(completed_steps, updates_per_epoch)
    return completed_epochs, min(updates_in_epoch * gradient_accumulation_steps, num_batches_per_epoch)


class LoRASqueezeSchedule:
    RANK_SCHEDULE_CHOICES = ("linear", "geometric")
    STEP_SCHEDULE_CHOICES = (
        "equal",
        "rank_proportional",
        "sqrt_rank_proportional",
        "inverse_rank_proportional",
        "inverse_sqrt_rank_proportional",
    )
    STATE_MODE_CHOICES = ("per_squeeze", "global")
    ALPHA_SCHEDULE_CHOICES = ("proportional", "sqrt")

    def __init__(self, args: argparse.Namespace):
        self.start_dim = getattr(args, "lora_squeeze_start_dim", None)
        self.target_dim = args.network_dim
        self.target_alpha = args.network_alpha
        self.num_squeezes = getattr(args, "lora_squeeze_num_squeezes", 0)
        if self.num_squeezes < 0:
            raise ValueError("--lora_squeeze_num_squeezes must not be negative")
        self.enabled = self.start_dim is not None or self.num_squeezes > 0
        self.train_after_final_squeeze = getattr(args, "lora_squeeze_train_after_final_squeeze", True)
        self.rank_schedule = getattr(args, "lora_squeeze_rank_schedule", "geometric")
        self.step_schedule = getattr(args, "lora_squeeze_step_schedule", "equal")
        self.optimizer_mode = getattr(args, "lora_squeeze_optimizer_mode", "per_squeeze")
        self.scheduler_mode = getattr(args, "lora_squeeze_scheduler_mode", "global")
        self.alpha_schedule = getattr(args, "lora_squeeze_alpha_schedule", "proportional")
        self.first_segment_ratio = getattr(args, "lora_squeeze_first_segment_ratio", 1.0)
        self.final_segment_ratio = getattr(args, "lora_squeeze_final_segment_ratio", 1.0)
        if self.train_after_final_squeeze is None:
            self.train_after_final_squeeze = True
        if self.rank_schedule is None:
            self.rank_schedule = "geometric"
        if self.step_schedule is None:
            self.step_schedule = "equal"
        if self.optimizer_mode is None:
            self.optimizer_mode = "per_squeeze"
        if self.scheduler_mode is None:
            self.scheduler_mode = "global"
        if self.alpha_schedule is None:
            self.alpha_schedule = "proportional"
        if self.first_segment_ratio is None:
            self.first_segment_ratio = 1.0
        if self.final_segment_ratio is None:
            self.final_segment_ratio = 1.0

        self.ranks: List[int] = []
        self.squeeze_steps: List[int] = []
        self.segment_steps: List[int] = []
        self.segment_weights: List[float] = []
        self._resume_squeeze_steps: Optional[List[int]] = None
        self._resume_segment_steps: Optional[List[int]] = None
        self._resume_segment_weights: Optional[List[float]] = None
        self.next_squeeze_index = 0
        self.current_dim = self.target_dim
        self.current_alpha = self.target_alpha
        if not self.enabled:
            return

        if self.start_dim is None:
            raise ValueError("--lora_squeeze_start_dim is required when LoRA-Squeeze is enabled")
        if self.target_dim is None:
            raise ValueError("--network_dim is required when LoRA-Squeeze is enabled")
        if self.target_alpha is None:
            raise ValueError("--network_alpha is required when LoRA-Squeeze is enabled")
        if self.start_dim <= self.target_dim:
            raise ValueError("--lora_squeeze_start_dim must be greater than --network_dim")
        if self.num_squeezes <= 0:
            raise ValueError("--lora_squeeze_num_squeezes must be greater than 0")
        if self.target_dim <= 0:
            raise ValueError("--network_dim must be greater than 0 when LoRA-Squeeze is enabled")
        if not math.isfinite(self.target_alpha):
            raise ValueError("--network_alpha must be finite when LoRA-Squeeze is enabled")
        if self.target_alpha <= 0:
            raise ValueError("--network_alpha must be greater than 0 when LoRA-Squeeze is enabled")
        self._validate_choice("--lora_squeeze_rank_schedule", self.rank_schedule, self.RANK_SCHEDULE_CHOICES)
        self._validate_choice("--lora_squeeze_step_schedule", self.step_schedule, self.STEP_SCHEDULE_CHOICES)
        self._validate_choice("--lora_squeeze_optimizer_mode", self.optimizer_mode, self.STATE_MODE_CHOICES)
        self._validate_choice("--lora_squeeze_scheduler_mode", self.scheduler_mode, self.STATE_MODE_CHOICES)
        self._validate_choice("--lora_squeeze_alpha_schedule", self.alpha_schedule, self.ALPHA_SCHEDULE_CHOICES)
        if not math.isfinite(self.first_segment_ratio):
            raise ValueError("--lora_squeeze_first_segment_ratio must be finite")
        if self.first_segment_ratio <= 0:
            raise ValueError("--lora_squeeze_first_segment_ratio must be greater than 0")
        if not math.isfinite(self.final_segment_ratio):
            raise ValueError("--lora_squeeze_final_segment_ratio must be finite")
        if self.final_segment_ratio <= 0:
            raise ValueError("--lora_squeeze_final_segment_ratio must be greater than 0")

        self.ranks = self._build_rank_schedule()
        self.current_dim = self.start_dim
        self.current_alpha = self.alpha_for_rank(self.current_dim)

    @staticmethod
    def _validate_choice(name: str, value: str, choices: Tuple[str, ...]):
        if value not in choices:
            raise ValueError(f"{name} must be one of: " + ", ".join(choices))

    def _build_rank_schedule(self) -> List[int]:
        ranks = [int(self.start_dim)]
        rank_delta = self.start_dim - self.target_dim
        for index in range(1, self.num_squeezes + 1):
            if index == self.num_squeezes:
                rank = self.target_dim
            elif self.rank_schedule == "geometric":
                progress = index / self.num_squeezes
                ideal_rank = self.start_dim * math.pow(self.target_dim / self.start_dim, progress)
                rank = math.floor(ideal_rank + 0.5)
                minimum_rank = self.target_dim + self.num_squeezes - index
                rank = min(max(rank, minimum_rank), ranks[-1] - 1)
            else:
                rank = math.floor(self.start_dim - rank_delta * index / self.num_squeezes)
            if rank >= ranks[-1]:
                raise ValueError(
                    "LoRA-Squeeze rank schedule is not strictly decreasing. "
                    "Use fewer squeezes or a larger start_dim-target_dim gap."
                )
            if rank < self.target_dim:
                raise ValueError("LoRA-Squeeze rank schedule went below --network_dim")
            ranks.append(int(rank))
        return ranks

    def alpha_for_rank(self, rank: int) -> float:
        if self.alpha_schedule == "proportional":
            alpha = float(self.target_alpha * rank / self.target_dim)
        else:
            alpha = float(self.target_alpha / math.sqrt(self.target_dim) * math.sqrt(rank))
        if not math.isfinite(alpha):
            raise ValueError(f"LoRA-Squeeze alpha schedule produced a non-finite alpha for rank {rank}")
        return alpha

    def _weight_for_rank(self, rank: int) -> float:
        if self.step_schedule == "equal":
            return 1.0
        if self.step_schedule == "rank_proportional":
            return float(rank)
        if self.step_schedule == "sqrt_rank_proportional":
            return math.sqrt(rank)
        if self.step_schedule == "inverse_rank_proportional":
            return 1.0 / float(rank)
        if self.step_schedule == "inverse_sqrt_rank_proportional":
            return 1.0 / math.sqrt(rank)
        self._validate_choice("--lora_squeeze_step_schedule", self.step_schedule, self.STEP_SCHEDULE_CHOICES)
        raise AssertionError("unreachable")

    def _build_segment_weights(self) -> List[float]:
        segment_ranks = self.ranks if self.train_after_final_squeeze else self.ranks[:-1]
        weights = [self._weight_for_rank(rank) for rank in segment_ranks]
        weights[0] *= self.first_segment_ratio
        if self.train_after_final_squeeze:
            weights[-1] *= self.final_segment_ratio
        if any(not math.isfinite(weight) for weight in weights):
            raise ValueError("LoRA-Squeeze step schedule produced a non-finite segment weight")
        return weights

    def set_total_steps(self, max_train_steps: int):
        if not self.enabled:
            return
        self.segment_weights = self._build_segment_weights()
        if max_train_steps < len(self.segment_weights):
            raise ValueError(
                f"max_train_steps={max_train_steps} is too small for "
                f"{len(self.segment_weights)} LoRA-Squeeze training segments"
            )
        total_weight = sum(self.segment_weights)
        if not math.isfinite(total_weight):
            raise ValueError("LoRA-Squeeze step schedule produced a non-finite total segment weight")
        cumulative_weight = 0.0
        steps = []
        weighted_boundary_count = self.num_squeezes if self.train_after_final_squeeze else self.num_squeezes - 1
        for index in range(weighted_boundary_count):
            cumulative_weight += self.segment_weights[index]
            steps.append(math.floor(max_train_steps * cumulative_weight / total_weight))
        if not self.train_after_final_squeeze:
            # The terminal squeeze is the end of the training budget by definition.
            # Do not let two numerically different floating-point summations move it
            # one step earlier.
            steps.append(max_train_steps)
        previous_step = 0
        for step in steps:
            if step <= previous_step:
                raise ValueError(
                    "LoRA-Squeeze step schedule is not strictly increasing. "
                    "Use fewer squeezes or more max_train_steps."
                )
            previous_step = step
        if self.train_after_final_squeeze and steps[-1] >= max_train_steps:
            raise ValueError(
                "LoRA-Squeeze final segment has zero steps. "
                "Increase max_train_steps or adjust the first/final segment ratios."
            )
        self.squeeze_steps = steps
        boundaries = [0, *steps]
        self.segment_steps = [boundaries[index + 1] - boundaries[index] for index in range(len(steps))]
        if self.train_after_final_squeeze:
            self.segment_steps.append(max_train_steps - steps[-1])
        if sum(self.segment_steps) != max_train_steps:
            raise AssertionError("LoRA-Squeeze segment steps do not cover the complete training budget")
        self._validate_resume_schedule()

    def _validate_resume_schedule(self):
        comparisons = (
            ("step schedule", self._resume_squeeze_steps, self.squeeze_steps),
            ("segment steps", self._resume_segment_steps, self.segment_steps),
            ("segment weights", self._resume_segment_weights, self.segment_weights),
        )
        for label, restored, current in comparisons:
            if restored is not None and restored != current:
                raise ValueError(
                    f"LoRA-Squeeze resume state {label} does not match the current training configuration: "
                    f"{restored} != {current}"
                )

    @property
    def completed_squeezes(self) -> int:
        return self.next_squeeze_index

    @property
    def current_segment_steps(self) -> Optional[int]:
        if not self.enabled or not self.segment_steps or self.next_squeeze_index >= len(self.segment_steps):
            return None
        return self.segment_steps[self.next_squeeze_index]

    @property
    def next_segment_steps_after_squeeze(self) -> Optional[int]:
        index = self.next_squeeze_index + 1
        if not self.enabled or not self.segment_steps or index >= len(self.segment_steps):
            return None
        return self.segment_steps[index]

    @property
    def total_segments(self) -> int:
        return len(self.segment_steps)

    def next_due(self, global_step: int) -> Optional[Dict[str, Union[int, float]]]:
        if not self.enabled or self.next_squeeze_index >= self.num_squeezes:
            return None
        if global_step < self.squeeze_steps[self.next_squeeze_index]:
            return None
        next_dim = self.ranks[self.next_squeeze_index + 1]
        return {
            "step": self.squeeze_steps[self.next_squeeze_index],
            "dim": next_dim,
            "alpha": self.alpha_for_rank(next_dim),
        }

    def mark_squeezed(self, dim: int, alpha: float):
        self.current_dim = dim
        self.current_alpha = alpha
        self.next_squeeze_index += 1

    def state_dict(self) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        return {
            "version": 1,
            "start_dim": self.start_dim,
            "target_dim": self.target_dim,
            "target_alpha": self.target_alpha,
            "num_squeezes": self.num_squeezes,
            "train_after_final_squeeze": bool(self.train_after_final_squeeze),
            "optimizer_mode": self.optimizer_mode,
            "scheduler_mode": self.scheduler_mode,
            "rank_schedule": self.rank_schedule,
            "step_schedule": self.step_schedule,
            "alpha_schedule": self.alpha_schedule,
            "first_segment_ratio": self.first_segment_ratio,
            "final_segment_ratio": self.final_segment_ratio,
            "ranks": list(self.ranks),
            "squeeze_steps": list(self.squeeze_steps),
            "segment_steps": list(self.segment_steps),
            "segment_weights": list(self.segment_weights),
            "next_squeeze_index": self.next_squeeze_index,
            "current_dim": self.current_dim,
            "current_alpha": self.current_alpha,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        if not self.enabled:
            raise ValueError("cannot restore LoRA-Squeeze state when LoRA-Squeeze is not enabled")
        if not state:
            raise ValueError("LoRA-Squeeze resume state is missing")
        expected_values = {
            "version": 1,
            "start_dim": self.start_dim,
            "target_dim": self.target_dim,
            "target_alpha": self.target_alpha,
            "num_squeezes": self.num_squeezes,
            "train_after_final_squeeze": bool(self.train_after_final_squeeze),
            "optimizer_mode": self.optimizer_mode,
            "scheduler_mode": self.scheduler_mode,
            "rank_schedule": self.rank_schedule,
            "step_schedule": self.step_schedule,
            "alpha_schedule": self.alpha_schedule,
            "first_segment_ratio": self.first_segment_ratio,
            "final_segment_ratio": self.final_segment_ratio,
            "ranks": self.ranks,
        }
        for key, expected in expected_values.items():
            if key not in state:
                raise ValueError(f"LoRA-Squeeze resume state is missing {key}")
            actual = state[key]
            if key in ("target_alpha", "first_segment_ratio", "final_segment_ratio"):
                actual = float(actual)
                if not math.isfinite(actual):
                    raise ValueError(f"LoRA-Squeeze resume state {key} must be finite")
                equal = math.isclose(actual, float(expected), rel_tol=1e-6, abs_tol=1e-8)
            else:
                equal = actual == expected
            if not equal:
                raise ValueError(
                    f"LoRA-Squeeze resume state does not match current configuration for {key}: "
                    f"{actual} != {expected}"
                )
        progress_keys = (
            "next_squeeze_index",
            "current_dim",
            "current_alpha",
            "squeeze_steps",
            "segment_steps",
            "segment_weights",
        )
        for key in progress_keys:
            if key not in state:
                raise ValueError(f"LoRA-Squeeze resume state is missing {key}")
        index = int(state["next_squeeze_index"])
        if index < 0 or index > self.num_squeezes:
            raise ValueError(f"LoRA-Squeeze resume state has invalid next_squeeze_index: {index}")
        expected_dim = self.ranks[index]
        expected_alpha = self.alpha_for_rank(expected_dim)
        current_dim = int(state["current_dim"])
        current_alpha = float(state["current_alpha"])
        if current_dim != expected_dim:
            raise ValueError(
                f"LoRA-Squeeze resume state current_dim does not match completed squeezes: "
                f"{current_dim} != {expected_dim}"
            )
        if not math.isfinite(current_alpha):
            raise ValueError("LoRA-Squeeze resume state current_alpha must be finite")
        if not math.isclose(current_alpha, expected_alpha, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(
                f"LoRA-Squeeze resume state current_alpha does not match completed squeezes: "
                f"{current_alpha} != {expected_alpha}"
            )
        squeeze_steps = [int(step) for step in state["squeeze_steps"]]
        segment_steps = [int(step) for step in state["segment_steps"]]
        segment_weights = [float(weight) for weight in state["segment_weights"]]
        if any(not math.isfinite(weight) for weight in segment_weights):
            raise ValueError("LoRA-Squeeze resume state segment_weights must be finite")
        self.next_squeeze_index = index
        self.current_dim = current_dim
        self.current_alpha = current_alpha
        self.squeeze_steps = squeeze_steps
        self.segment_steps = segment_steps
        self.segment_weights = segment_weights
        self._resume_squeeze_steps = list(self.squeeze_steps) if self.squeeze_steps else None
        self._resume_segment_steps = list(self.segment_steps) if self.segment_steps else None
        self._resume_segment_weights = list(self.segment_weights) if self.segment_weights else None

    def rank_schedule_text(self) -> str:
        return " -> ".join(str(rank) for rank in self.ranks)

    def step_schedule_text(self) -> str:
        return ", ".join(
            f"step {step}: {self.ranks[index]}->{self.ranks[index + 1]}"
            for index, step in enumerate(self.squeeze_steps)
        )

    def step_distribution_text(self) -> str:
        text = self.step_schedule
        if self.first_segment_ratio != 1.0:
            text += f", first_segment_ratio={self.first_segment_ratio:g}"
        if self.train_after_final_squeeze and self.final_segment_ratio != 1.0:
            text += f", final_segment_ratio={self.final_segment_ratio:g}"
        if self.segment_steps:
            text += "; segment steps: " + " / ".join(str(step) for step in self.segment_steps)
        return text


def load_train_state_json(state_dir: str) -> Dict[str, Any]:
    train_state_file = os.path.join(state_dir, "train_state.json")
    if not os.path.exists(train_state_file):
        raise ValueError(f"LoRA-Squeeze resume requires train_state.json in state directory: {state_dir}")
    with open(train_state_file, "r", encoding="utf-8") as file:
        return json.load(file)


def restore_lora_squeeze_state_from_train_state(schedule: LoRASqueezeSchedule, train_state: Dict[str, Any]):
    state = train_state.get(TRAIN_STATE_LORA_SQUEEZE_KEY)
    if state is None:
        raise ValueError(
            "LoRA-Squeeze resume requires a state saved with LoRA-Squeeze metadata. "
            "The selected train_state.json does not contain lora_squeeze."
        )
    schedule.load_state_dict(state)


def restore_lora_squeeze_state_from_resume_dir(schedule: LoRASqueezeSchedule, state_dir: str) -> Dict[str, Any]:
    train_state = load_train_state_json(state_dir)
    restore_lora_squeeze_state_from_train_state(schedule, train_state)
    return train_state


def validate_lora_squeeze_training_args(args: argparse.Namespace):
    if getattr(args, "torch_compile", False) or getattr(args, "compile", False):
        raise ValueError(
            "LoRA-Squeeze does not support torch.compile options because it replaces LoRA layers during training"
        )


def validate_lora_squeeze_network_args(net_kwargs: Dict[str, Any]):
    unsupported = sorted(key for key in UNSUPPORTED_NETWORK_ARG_KEYS if key in net_kwargs)
    if unsupported:
        raise ValueError(
            "LoRA-Squeeze does not support network_args that create separate per-module rank or alpha semantics: "
            + ", ".join(unsupported)
        )
