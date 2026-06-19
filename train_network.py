import gc
import importlib
import argparse
import csv
import math
import os
import typing
from typing import Any, Dict, List, Optional, Union
import sys
import random
import time
import json
from multiprocessing import Value
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.types import Number
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from library import deepspeed_utils, model_util, sai_model_spec, strategy_base, strategy_sd, sai_model_spec

import library.train_util as train_util
from library.train_util import DreamBoothDataset
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.huggingface_util as huggingface_util
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
)
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


class WeightValueTracker:
    CSV_FIELDS = [
        "step",
        "epoch",
        "track_id",
        "source",
        "param_kind",
        "layer",
        "param_name",
        "shape",
        "flat_index",
        "value",
    ]

    def __init__(self, args: argparse.Namespace, accelerator: Accelerator, network: torch.nn.Module):
        self.enabled = bool(args.track_weight_values)
        self.interval = args.weight_track_interval
        self.csv_file = None
        self.writer = None
        self.csv_path: Optional[str] = None
        self.tracked_weights: List[Dict[str, Any]] = []

        if not self.enabled:
            return

        if args.weight_track_count <= 0:
            raise ValueError("weight_track_count must be greater than 0")
        if args.weight_track_interval <= 0:
            raise ValueError("weight_track_interval must be greater than 0")

        if not accelerator.is_main_process:
            self.enabled = False
            return

        log_dir = self.get_log_dir(accelerator)
        if log_dir is None:
            raise ValueError("--track_weight_values requires --logging_dir so the CSV can be written beside training logs")

        self.tracked_weights = self.sample_weights(network, args.weight_track_count, args.weight_track_seed)
        if len(self.tracked_weights) == 0:
            raise ValueError("no trainable weight parameters were found for --track_weight_values")

        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "tracked_weight_values.csv")
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.CSV_FIELDS)
        self.writer.writeheader()
        self.csv_file.flush()

        source_counts: Dict[str, int] = {}
        for item in self.tracked_weights:
            source_counts[item["source"]] = source_counts.get(item["source"], 0) + 1
        logger.info(f"tracking {len(self.tracked_weights)} weight values in {self.csv_path}: {source_counts}")

    @staticmethod
    def get_log_dir(accelerator: Accelerator) -> Optional[str]:
        logging_dir = getattr(accelerator, "logging_dir", None) or getattr(accelerator, "project_dir", None)
        if logging_dir is None:
            return None
        return os.fspath(logging_dir)

    @staticmethod
    def classify_source(name: str) -> str:
        if "lora_te1" in name:
            return "text_encoder_1"
        if "lora_te2" in name:
            return "text_encoder_2"
        if "lora_te3" in name:
            return "text_encoder_3"
        if "lora_te" in name:
            return "text_encoder"
        if "lora_unet" in name:
            return "unet"
        return "other"

    @staticmethod
    def classify_param_kind(name: str) -> str:
        if "lora_down" in name:
            return "lora_down"
        if "lora_up" in name:
            return "lora_up"
        return "weight"

    @staticmethod
    def get_layer_name(name: str) -> str:
        for token in [".lora_down.", ".lora_up."]:
            if token in name:
                return name.split(token, 1)[0]
        return name.rsplit(".", 1)[0]

    @staticmethod
    def source_sort_key(source: str) -> tuple:
        if source.startswith("text_encoder"):
            return (0, source)
        if source == "unet":
            return (1, source)
        return (2, source)

    @staticmethod
    def order_candidates_by_layer(candidates: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
        layer_groups: Dict[str, List[Dict[str, Any]]] = {}
        for candidate in candidates:
            layer_groups.setdefault(candidate["layer"], []).append(candidate)

        for layer_candidates in layer_groups.values():
            rng.shuffle(layer_candidates)

        layers = list(layer_groups.keys())
        rng.shuffle(layers)

        ordered: List[Dict[str, Any]] = []
        while any(layer_groups[layer] for layer in layers):
            for layer in layers:
                if layer_groups[layer]:
                    ordered.append(layer_groups[layer].pop(0))

        return ordered

    @classmethod
    def sample_weights(cls, network: torch.nn.Module, count: int, seed: int) -> List[Dict[str, Any]]:
        rng = random.Random(seed)
        grouped_candidates: Dict[str, List[Dict[str, Any]]] = {}
        all_candidates: List[Dict[str, Any]] = []

        for name, param in network.named_parameters():
            if not param.requires_grad or param.numel() == 0 or not name.endswith(".weight"):
                continue

            source = cls.classify_source(name)
            candidate = {
                "param": param,
                "param_name": name,
                "source": source,
                "param_kind": cls.classify_param_kind(name),
                "layer": cls.get_layer_name(name),
                "shape": "x".join(str(dim) for dim in param.shape),
            }
            grouped_candidates.setdefault(source, []).append(candidate)
            all_candidates.append(candidate)

        for source, candidates in list(grouped_candidates.items()):
            grouped_candidates[source] = cls.order_candidates_by_layer(candidates, rng)

        selected: List[Dict[str, Any]] = []
        source_names = sorted(grouped_candidates.keys(), key=cls.source_sort_key)
        while len(selected) < count and any(grouped_candidates[source] for source in source_names):
            for source in source_names:
                if grouped_candidates[source]:
                    selected.append(grouped_candidates[source].pop(0))
                    if len(selected) >= count:
                        break

        while len(selected) < count and all_candidates:
            selected.append(rng.choice(all_candidates).copy())

        for track_id, item in enumerate(selected):
            item["track_id"] = track_id
            item["flat_index"] = rng.randrange(item["param"].numel())

        return selected

    def resample(self, network: torch.nn.Module, count: int, seed: int):
        if not self.enabled:
            return
        self.tracked_weights = self.sample_weights(network, count, seed)
        if len(self.tracked_weights) == 0:
            raise ValueError("no trainable weight parameters were found for --track_weight_values")

    def record(self, step: int, epoch: int):
        if not self.enabled or self.writer is None:
            return
        if step != 0 and step % self.interval != 0:
            return

        for item in self.tracked_weights:
            value = item["param"].detach().flatten()[item["flat_index"]].float().cpu().item()
            self.writer.writerow(
                {
                    "step": step,
                    "epoch": epoch,
                    "track_id": item["track_id"],
                    "source": item["source"],
                    "param_kind": item["param_kind"],
                    "layer": item["layer"],
                    "param_name": item["param_name"],
                    "shape": item["shape"],
                    "flat_index": item["flat_index"],
                    "value": f"{value:.17g}",
                }
            )
        self.csv_file.flush()

    def close(self):
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None


class LoRASqueezeSchedule:
    RANK_SCHEDULE_CHOICES = (
        "linear",
        "geometric",
    )
    STEP_SCHEDULE_CHOICES = (
        "equal",
        "rank_proportional",
        "sqrt_rank_proportional",
        "inverse_rank_proportional",
        "inverse_sqrt_rank_proportional",
    )

    def __init__(self, args: argparse.Namespace):
        start_dim = args.lora_squeeze_start_dim
        num_squeezes = args.lora_squeeze_num_squeezes
        alias_num_squeezes = getattr(args, "lora_squeeze_amount_of_squeezes", None)
        if num_squeezes == 0 and alias_num_squeezes is not None:
            num_squeezes = alias_num_squeezes
        self.enabled = start_dim is not None or num_squeezes > 0

        self.start_dim = start_dim
        self.target_dim = args.network_dim
        self.target_alpha = args.network_alpha
        self.num_squeezes = num_squeezes
        self.train_after_final_squeeze = args.lora_squeeze_train_after_final_squeeze
        self.rank_schedule = getattr(args, "lora_squeeze_rank_schedule", "linear")
        if self.rank_schedule is None:
            self.rank_schedule = "linear"
        self.step_schedule = getattr(args, "lora_squeeze_step_schedule", None)
        self.final_segment_ratio = getattr(args, "lora_squeeze_final_segment_ratio", 1.0)
        if self.final_segment_ratio is None:
            self.final_segment_ratio = 1.0
        self.ranks: List[int] = []
        self.squeeze_steps: List[int] = []
        self.segment_steps: List[int] = []
        self.segment_weights: List[float] = []
        self.next_squeeze_index = 0
        self.current_dim = args.network_dim
        self.current_alpha = args.network_alpha

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
        if self.target_alpha <= 0:
            raise ValueError("--network_alpha must be greater than 0 when LoRA-Squeeze is enabled")
        if self.rank_schedule not in self.RANK_SCHEDULE_CHOICES:
            raise ValueError(
                "--lora_squeeze_rank_schedule must be one of: " + ", ".join(self.RANK_SCHEDULE_CHOICES)
            )
        if self.step_schedule is not None and self.step_schedule not in self.STEP_SCHEDULE_CHOICES:
            raise ValueError(
                "--lora_squeeze_step_schedule must be one of: " + ", ".join(self.STEP_SCHEDULE_CHOICES)
            )
        if self.final_segment_ratio <= 0:
            raise ValueError("--lora_squeeze_final_segment_ratio must be greater than 0")

        self.ranks = self._build_rank_schedule()
        self.current_dim = self.start_dim
        self.current_alpha = self.alpha_for_rank(self.current_dim)

    def _build_rank_schedule(self) -> List[int]:
        ranks = [int(self.start_dim)]
        rank_delta = self.start_dim - self.target_dim
        for i in range(1, self.num_squeezes + 1):
            if i == self.num_squeezes:
                rank = self.target_dim
            elif self.rank_schedule == "geometric":
                progress = i / self.num_squeezes
                ideal_rank = self.start_dim * math.pow(self.target_dim / self.start_dim, progress)
                rank = math.floor(ideal_rank + 0.5)

                # Integer rounding can repeat a rank, especially when the range is tight. Clamp the
                # candidate to leave one distinct integer rank for every remaining squeeze.
                minimum_rank = self.target_dim + self.num_squeezes - i
                maximum_rank = ranks[-1] - 1
                rank = min(max(rank, minimum_rank), maximum_rank)
            else:
                rank = math.floor(self.start_dim - rank_delta * i / self.num_squeezes)

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
        return float(self.target_alpha / math.sqrt(self.target_dim) * math.sqrt(rank))

    def _weight_for_rank(self, rank: int) -> float:
        if self.step_schedule is None or self.step_schedule == "equal":
            return 1.0
        if self.step_schedule == "rank_proportional":
            return float(rank)
        if self.step_schedule == "sqrt_rank_proportional":
            return math.sqrt(rank)
        if self.step_schedule == "inverse_rank_proportional":
            return 1.0 / float(rank)
        if self.step_schedule == "inverse_sqrt_rank_proportional":
            return 1.0 / math.sqrt(rank)
        raise ValueError("--lora_squeeze_step_schedule must be one of: " + ", ".join(self.STEP_SCHEDULE_CHOICES))

    def _build_segment_weights(self) -> List[float]:
        segment_ranks = self.ranks if self.train_after_final_squeeze else self.ranks[:-1]
        weights = [self._weight_for_rank(rank) for rank in segment_ranks]
        if self.train_after_final_squeeze:
            weights[-1] *= self.final_segment_ratio
        return weights

    def set_total_steps(self, max_train_steps: int):
        if not self.enabled:
            return

        self.segment_weights = self._build_segment_weights()
        training_segments = len(self.segment_weights)
        if max_train_steps < training_segments:
            raise ValueError(
                f"max_train_steps={max_train_steps} is too small for {training_segments} LoRA-Squeeze training segments"
            )

        total_weight = sum(self.segment_weights)
        steps = []
        cumulative_weight = 0.0
        for i in range(self.num_squeezes):
            cumulative_weight += self.segment_weights[i]
            steps.append(math.floor(max_train_steps * cumulative_weight / total_weight))

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
                "Increase max_train_steps or lora_squeeze_final_segment_ratio."
            )

        self.squeeze_steps = steps
        self.segment_steps = []
        previous_step = 0
        for step in self.squeeze_steps:
            self.segment_steps.append(step - previous_step)
            previous_step = step
        if self.train_after_final_squeeze:
            self.segment_steps.append(max_train_steps - previous_step)

    @property
    def completed_squeezes(self) -> int:
        return self.next_squeeze_index

    @property
    def current_segment_steps(self) -> Optional[int]:
        if not self.enabled or not self.segment_steps or self.next_squeeze_index >= len(self.segment_steps):
            return None
        return self.segment_steps[self.next_squeeze_index]

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

    def rank_schedule_text(self) -> str:
        return " -> ".join(str(rank) for rank in self.ranks)

    def step_schedule_text(self) -> str:
        if not self.squeeze_steps:
            return ""
        return ", ".join(
            f"step {step}: {self.ranks[i]}->{self.ranks[i + 1]}"
            for i, step in enumerate(self.squeeze_steps)
        )

    def step_distribution_text(self) -> str:
        schedule = self.step_schedule if self.step_schedule is not None else "equal"
        text = schedule
        if self.train_after_final_squeeze and self.final_segment_ratio != 1.0:
            text += f", final_segment_ratio={self.final_segment_ratio:g}"
        if self.segment_steps:
            text += "; segment steps: " + " / ".join(str(step) for step in self.segment_steps)
        return text


def resolve_total_rms_check_interval(target_total_rms: Optional[float], check_every_n_steps: Optional[int]) -> int:
    if check_every_n_steps is None:
        return 25 if target_total_rms is not None else 0
    if check_every_n_steps < 0:
        raise ValueError("--total_rms_check_every_n_steps must be 0 or greater")
    return check_every_n_steps


class NetworkTrainer:
    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False

    @staticmethod
    def get_lora_squeeze_modules(network: torch.nn.Module) -> List[torch.nn.Module]:
        modules: List[torch.nn.Module] = []
        for module in network.modules():
            if not hasattr(module, "lora_down") or not hasattr(module, "lora_up"):
                continue
            if not isinstance(module.lora_down, (nn.Linear, nn.Conv2d)):
                continue
            if not isinstance(module.lora_up, (nn.Linear, nn.Conv2d)):
                continue
            modules.append(module)
        return modules

    @staticmethod
    def get_lora_module_rank(module: torch.nn.Module) -> int:
        return int(module.lora_down.weight.shape[0])

    @staticmethod
    def get_lora_module_alpha(module: torch.nn.Module) -> float:
        alpha = getattr(module, "alpha", None)
        if alpha is None:
            return float(NetworkTrainer.get_lora_module_rank(module))
        if isinstance(alpha, torch.Tensor):
            return float(alpha.detach().float().item())
        return float(alpha)

    @staticmethod
    def get_lora_factor_matrices(module: torch.nn.Module):
        down = module.lora_down.weight.detach().float()
        up = module.lora_up.weight.detach().float()

        if isinstance(module.lora_down, nn.Linear) and isinstance(module.lora_up, nn.Linear):
            return up, down, "linear"

        if isinstance(module.lora_down, nn.Conv2d) and isinstance(module.lora_up, nn.Conv2d):
            if tuple(module.lora_up.kernel_size) != (1, 1):
                raise ValueError(f"LoRA-Squeeze only supports 1x1 lora_up convs: {module.lora_name}")
            up_2d = up[:, :, 0, 0]
            down_2d = down.reshape(down.shape[0], -1)
            return up_2d, down_2d, "conv2d"

        raise ValueError(f"LoRA-Squeeze does not support mixed LoRA layer types: {module.lora_name}")

    @staticmethod
    def compact_lora_product(up_2d: torch.Tensor, down_2d: torch.Tensor, old_scale: float, target_dim: int):
        # SVD of (up @ down * old_scale) through the rank-sized core.
        a = up_2d
        b = down_2d * old_scale
        qa, ra = torch.linalg.qr(a, mode="reduced")
        qb, rb = torch.linalg.qr(b.T, mode="reduced")
        core = ra @ rb.T
        u_core, singular_values, vh_core = torch.linalg.svd(core, full_matrices=False)

        usable_dim = min(target_dim, singular_values.numel())
        u = qa @ u_core[:, :usable_dim]
        vh = vh_core[:usable_dim, :] @ qb.T
        return u, singular_values, vh

    @staticmethod
    def replace_lora_module_layers(
        module: torch.nn.Module,
        new_up_2d: torch.Tensor,
        new_down_2d: torch.Tensor,
        target_dim: int,
        target_alpha: float,
        kind: str,
    ):
        old_down = module.lora_down
        old_up = module.lora_up
        device = old_down.weight.device
        down_dtype = old_down.weight.dtype
        up_dtype = old_up.weight.dtype

        if kind == "linear":
            new_down = nn.Linear(old_down.in_features, target_dim, bias=False, device=device, dtype=down_dtype)
            new_up = nn.Linear(target_dim, old_up.out_features, bias=False, device=device, dtype=up_dtype)
            down_weight = new_down_2d
            up_weight = new_up_2d
        elif kind == "conv2d":
            new_down = nn.Conv2d(
                old_down.in_channels,
                target_dim,
                old_down.kernel_size,
                old_down.stride,
                old_down.padding,
                old_down.dilation,
                old_down.groups,
                bias=False,
                padding_mode=old_down.padding_mode,
                device=device,
                dtype=down_dtype,
            )
            new_up = nn.Conv2d(
                target_dim,
                old_up.out_channels,
                old_up.kernel_size,
                old_up.stride,
                old_up.padding,
                old_up.dilation,
                old_up.groups,
                bias=False,
                padding_mode=old_up.padding_mode,
                device=device,
                dtype=up_dtype,
            )
            down_weight = new_down_2d.reshape(target_dim, *old_down.weight.shape[1:])
            up_weight = new_up_2d.reshape(old_up.out_channels, target_dim, *old_up.weight.shape[2:])
        else:
            raise ValueError(f"unknown LoRA layer kind: {kind}")

        new_down.train(old_down.training)
        new_up.train(old_up.training)
        with torch.no_grad():
            new_down.weight.copy_(down_weight.to(device=device, dtype=down_dtype))
            new_up.weight.copy_(up_weight.to(device=device, dtype=up_dtype))

        module.lora_down = new_down
        module.lora_up = new_up
        module.lora_dim = target_dim
        module.scale = target_alpha / target_dim
        old_alpha = getattr(module, "alpha", None)
        alpha_dtype = old_alpha.dtype if isinstance(old_alpha, torch.Tensor) else down_dtype
        alpha_tensor = torch.tensor(target_alpha, device=device, dtype=alpha_dtype)
        if isinstance(old_alpha, torch.Tensor):
            module.alpha = alpha_tensor
        else:
            module.register_buffer("alpha", alpha_tensor)

    def squeeze_lora_network(self, network: torch.nn.Module, target_dim: int, target_alpha: float) -> Dict[str, float]:
        lora_modules = self.get_lora_squeeze_modules(network)
        if len(lora_modules) == 0:
            raise ValueError("LoRA-Squeeze did not find any LoRA modules with lora_down/lora_up weights")

        source_ranks = {self.get_lora_module_rank(module) for module in lora_modules}
        if len(source_ranks) != 1:
            raise ValueError(f"LoRA-Squeeze requires a homogeneous current rank, found ranks: {sorted(source_ranks)}")
        source_rank = next(iter(source_ranks))
        if target_dim >= source_rank:
            raise ValueError(f"LoRA-Squeeze target rank {target_dim} must be less than current rank {source_rank}")

        retained_energies: List[float] = []
        with torch.no_grad():
            for module in lora_modules:
                old_rank = self.get_lora_module_rank(module)
                old_alpha = self.get_lora_module_alpha(module)
                old_scale = old_alpha / old_rank
                up_2d, down_2d, kind = self.get_lora_factor_matrices(module)

                u, singular_values, vh = self.compact_lora_product(up_2d, down_2d, old_scale, target_dim)
                usable_dim = min(target_dim, singular_values.numel())
                kept = singular_values[:usable_dim].clamp_min(0)
                total_energy = torch.sum(singular_values * singular_values).item()
                kept_energy = torch.sum(kept * kept).item()
                retained_energies.append(1.0 if total_energy == 0.0 else kept_energy / total_energy)

                factor_scale = math.sqrt(target_dim / target_alpha)
                sqrt_s = torch.sqrt(kept)
                new_up_2d = torch.zeros(
                    (up_2d.shape[0], target_dim),
                    device=up_2d.device,
                    dtype=up_2d.dtype,
                )
                new_down_2d = torch.zeros(
                    (target_dim, down_2d.shape[1]),
                    device=down_2d.device,
                    dtype=down_2d.dtype,
                )
                if usable_dim > 0:
                    new_up_2d[:, :usable_dim] = u[:, :usable_dim] * sqrt_s.unsqueeze(0) * factor_scale
                    new_down_2d[:usable_dim, :] = sqrt_s.unsqueeze(1) * vh[:usable_dim, :] * factor_scale

                self.replace_lora_module_layers(module, new_up_2d, new_down_2d, target_dim, target_alpha, kind)

        return {
            "modules": float(len(lora_modules)),
            "source_rank": float(source_rank),
            "target_rank": float(target_dim),
            "retained_energy_min": min(retained_energies),
            "retained_energy_mean": sum(retained_energies) / len(retained_energies),
        }

    # TODO 他のスクリプトと共通化する
    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer=None,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        mean_grad_norm=None,
        mean_combined_norm=None,
    ):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/max_key_norm"] = maximum_norm
        if mean_norm is not None:
            logs["norm/avg_key_norm"] = mean_norm
        if mean_grad_norm is not None:
            logs["norm/avg_grad_norm"] = mean_grad_norm
        if mean_combined_norm is not None:
            logs["norm/avg_combined_norm"] = mean_combined_norm

        lrs = lr_scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            if lr_descriptions is not None:
                lr_desc = lr_descriptions[i]
            else:
                idx = i - (0 if args.network_train_unet_only else 1)
                if idx == -1:
                    lr_desc = "textencoder"
                else:
                    if len(lrs) > 2:
                        lr_desc = f"group{i}"
                    else:
                        lr_desc = "unet"

            logs[f"lr/{lr_desc}"] = lr

            if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower().startswith("Prodigy".lower()):
                opt = lr_scheduler.optimizers[-1] if hasattr(lr_scheduler, "optimizers") else optimizer
                if opt is not None:
                    logs[f"lr/d*lr/{lr_desc}"] = opt.param_groups[i]["d"] * opt.param_groups[i]["lr"]
                    if "effective_lr" in opt.param_groups[i]:
                        logs[f"lr/d*eff_lr/{lr_desc}"] = opt.param_groups[i]["d"] * opt.param_groups[i]["effective_lr"]

        return logs
    
    @staticmethod
    def _rms_to_2d_weight(tensor: torch.Tensor, role: str) -> torch.Tensor:
        t = tensor.detach().float()

        if t.ndim == 2:
            return t

        if t.ndim >= 3:
            return t.reshape(t.shape[0], -1)

        if t.ndim == 1:
            return t.reshape(1, -1) if role == "down" else t.reshape(-1, 1)

        raise ValueError(f"Unsupported tensor rank for RMS check: {t.ndim}")


    @staticmethod
    def _rms_product_frobenius_norm(left: torch.Tensor, right: torch.Tensor) -> float:
        if left.shape[1] != right.shape[0]:
            raise ValueError(f"product shape mismatch: left={tuple(left.shape)}, right={tuple(right.shape)}")

        gram_l = left.T @ left
        gram_r = right @ right.T
        fro_sq = torch.sum(gram_l * gram_r).clamp_min(0.0)
        return float(torch.sqrt(fro_sq).item())


    @staticmethod
    @torch.no_grad()
    def compute_total_scaled_lora_rms(network) -> float:
        total_numel = 0
        total_fro_sq = 0.0

        for module in network.modules():
            if not hasattr(module, "lora_down") or not hasattr(module, "lora_up"):
                continue

            if not hasattr(module.lora_down, "weight") or not hasattr(module.lora_up, "weight"):
                continue

            down = NetworkTrainer._rms_to_2d_weight(module.lora_down.weight, role="down")
            up = NetworkTrainer._rms_to_2d_weight(module.lora_up.weight, role="up")

            if up.shape[1] != down.shape[0]:
                continue

            rank = int(down.shape[0])

            alpha = getattr(module, "alpha", None)
            if alpha is None:
                scale = 1.0
            else:
                if isinstance(alpha, torch.Tensor):
                    alpha = float(alpha.detach().float().reshape(-1)[0].item())
                else:
                    alpha = float(alpha)

                scale = alpha / rank if rank > 0 else 1.0

            raw_fro = NetworkTrainer._rms_product_frobenius_norm(up, down)
            scaled_fro = raw_fro * scale

            effective_numel = int(up.shape[0] * down.shape[1])
            total_numel += effective_numel
            total_fro_sq += scaled_fro * scaled_fro

        if total_numel <= 0:
            return 0.0

        return math.sqrt(total_fro_sq) / math.sqrt(total_numel)

    def step_logging(self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int):
        self.accelerator_logging(accelerator, logs, global_step, global_step, epoch)

    def epoch_logging(self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int):
        self.accelerator_logging(accelerator, logs, epoch, global_step, epoch)

    def val_logging(self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int, val_step: int):
        self.accelerator_logging(accelerator, logs, global_step + val_step, global_step, epoch, val_step)

    def accelerator_logging(
        self, accelerator: Accelerator, logs: dict, step_value: int, global_step: int, epoch: int, val_step: Optional[int] = None
    ):
        """
        step_value is for tensorboard, other values are for wandb
        """
        tensorboard_tracker = None
        wandb_tracker = None
        other_trackers = []
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tensorboard_tracker = accelerator.get_tracker("tensorboard")
            elif tracker.name == "wandb":
                wandb_tracker = accelerator.get_tracker("wandb")
            else:
                other_trackers.append(accelerator.get_tracker(tracker.name))

        if tensorboard_tracker is not None:
            tensorboard_tracker.log(logs, step=step_value)

        if wandb_tracker is not None:
            logs["global_step"] = global_step
            logs["epoch"] = epoch
            if val_step is not None:
                logs["val_step"] = val_step
            wandb_tracker.log(logs)

        for tracker in other_trackers:
            tracker.log(logs, step=step_value)

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        train_dataset_group.verify_bucket_reso_steps(64)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(64)

    def load_target_model(self, args, weight_dtype, accelerator) -> tuple[str, nn.Module, nn.Module, Optional[nn.Module]]:
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)

        # モデルに xformers とか memory efficient attention を組み込む
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), text_encoder, vae, unet

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, List[nn.Module]]:
        raise NotImplementedError()

    def get_tokenize_strategy(self, args):
        return strategy_sd.SdTokenizeStrategy(args.v2, args.max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sd.SdTokenizeStrategy) -> List[Any]:
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            True, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        return strategy_sd.SdTextEncodingStrategy(args.clip_skip)

    def get_text_encoder_outputs_caching_strategy(self, args):
        return None

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        """
        Returns a list of models that will be used for text encoding. SDXL uses wrapped and unwrapped models.
        FLUX.1 and SD3 may cache some outputs of the text encoder, so return the models that will be used for encoding (not cached).
        """
        return text_encoders

    # returns a list of bool values indicating whether each text encoder should be trained
    def get_text_encoders_train_flags(self, args, text_encoders):
        return [True] * len(text_encoders) if self.is_train_text_encoder(args) else [False] * len(text_encoders)

    def is_train_text_encoder(self, args):
        return not args.network_train_unet_only

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        for t_enc in text_encoders:
            t_enc.to(accelerator.device, dtype=weight_dtype)

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype, **kwargs):
        noise_pred = unet(noisy_latents, timesteps, text_conds[0]).sample
        return noise_pred

    def all_reduce_network(self, accelerator, network):
        for param in network.parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoder, unet):
        train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizers[0], text_encoder, unet)

    # region SD/SDXL

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        pass

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )
        prepare_scheduler_for_custom_training(noise_scheduler, device)
        if args.zero_terminal_snr:
            custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)
        return noise_scheduler

    def encode_images_to_latents(self, args, vae: AutoencoderKL, images: torch.FloatTensor) -> torch.FloatTensor:
        return vae.encode(images).latent_dist.sample()

    def shift_scale_latents(self, args, latents: torch.FloatTensor) -> torch.FloatTensor:
        return latents * self.vae_scale_factor

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        # Sample noise, sample a random timestep for each image, and add noise to the latents,
        # with noise offset and/or multires noise if specified
        noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            for x in noisy_latents:
                x.requires_grad_(True)
            for t in text_encoder_conds:
                t.requires_grad_(True)

        # For inpainting models: concatenate [noisy_latents, mask, masked_latents] -> 9-channel UNet input
        unet_latents = noisy_latents
        if batch.get("masked_latents") is not None:
            mask = torch.nn.functional.interpolate(
                batch["masks"].to(weight_dtype), size=noisy_latents.shape[2:]
            )
            unet_latents = torch.cat([noisy_latents, mask, batch["masked_latents"].to(weight_dtype)], dim=1)

        # Predict the noise residual
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            noise_pred = self.call_unet(
                args,
                accelerator,
                unet,
                unet_latents.requires_grad_(train_unet),
                timesteps,
                text_encoder_conds,
                batch,
                weight_dtype,
            )

        if args.v_parameterization:
            # v-parameterization training
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        # differential output preservation
        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0:
                network.set_multiplier(0.0)
                with torch.no_grad(), accelerator.autocast():
                    noise_pred_prior = self.call_unet(
                        args,
                        accelerator,
                        unet,
                        noisy_latents,
                        timesteps,
                        text_encoder_conds,
                        batch,
                        weight_dtype,
                        indices=diff_output_pr_indices,
                    )
                network.set_multiplier(1.0)  # may be overwritten by "network_multipliers" in the next step
                target[diff_output_pr_indices] = noise_pred_prior.to(target.dtype)

        return noise_pred, target, timesteps, None

    def post_process_loss(self, loss, args, timesteps: torch.IntTensor, noise_scheduler) -> torch.FloatTensor:
        if args.min_snr_gamma:
            loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
        if args.scale_v_pred_loss_like_noise_pred:
            loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
        if args.v_pred_like_loss:
            loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
        if args.debiased_estimation_loss:
            loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, args.v_parameterization)
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec(None, args, self.is_sdxl, True, False)

    def update_metadata(self, metadata, args):
        pass

    def is_text_encoder_not_needed_for_training(self, args):
        return False  # use for sample images

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # set top parameter requires_grad = True for gradient checkpointing works
        text_encoder.text_model.embeddings.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        text_encoder.text_model.embeddings.to(dtype=weight_dtype)

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        return accelerator.prepare(unet)

    def on_step_start(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train: bool = True):
        pass

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        pass

    # endregion

    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy: strategy_base.TextEncodingStrategy,
        tokenize_strategy: strategy_base.TokenizeStrategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
    ) -> torch.Tensor:
        """
        Process a batch for the network
        """
        with torch.no_grad():
            if "latents" in batch and batch["latents"] is not None:
                latents = typing.cast(torch.FloatTensor, batch["latents"].to(accelerator.device))
            else:
                # latentに変換
                if args.vae_batch_size is None or len(batch["images"]) <= args.vae_batch_size:
                    latents = self.encode_images_to_latents(args, vae, batch["images"].to(accelerator.device, dtype=vae_dtype))
                else:
                    chunks = [
                        batch["images"][i : i + args.vae_batch_size] for i in range(0, len(batch["images"]), args.vae_batch_size)
                    ]
                    list_latents = []
                    for chunk in chunks:
                        with torch.no_grad():
                            chunk = self.encode_images_to_latents(args, vae, chunk.to(accelerator.device, dtype=vae_dtype))
                            list_latents.append(chunk)
                    latents = torch.cat(list_latents, dim=0)

                # NaNが含まれていれば警告を表示し0に置き換える
                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = typing.cast(torch.FloatTensor, torch.nan_to_num(latents, 0, out=latents))

            latents = self.shift_scale_latents(args, latents)

            # Prepare inpainting masked_latents if batch contains masks
            if batch.get("masks") is not None:
                masked_latents = self.encode_images_to_latents(
                    args, vae, batch["masked_images"].to(accelerator.device, dtype=vae_dtype)
                )
                batch["masked_latents"] = self.shift_scale_latents(args, masked_latents)

        text_encoder_conds = []
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            text_encoder_conds = text_encoder_outputs_list  # List of text encoder outputs

        if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
            # TODO this does not work if 'some text_encoders are trained' and 'some are not and not cached'
            with torch.set_grad_enabled(is_train and train_text_encoder), accelerator.autocast():
                # Get the text embedding for conditioning
                if args.weighted_captions:
                    input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens_with_weights(
                        tokenize_strategy,
                        self.get_models_for_text_encoding(args, accelerator, text_encoders),
                        input_ids_list,
                        weights_list,
                    )
                else:
                    input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                        tokenize_strategy,
                        self.get_models_for_text_encoding(args, accelerator, text_encoders),
                        input_ids,
                    )
                if args.full_fp16:
                    encoded_text_encoder_conds = [c.to(weight_dtype) for c in encoded_text_encoder_conds]

            # if text_encoder_conds is not cached, use encoded_text_encoder_conds
            if len(text_encoder_conds) == 0:
                text_encoder_conds = encoded_text_encoder_conds
            else:
                # if encoded_text_encoder_conds is not None, update cached text_encoder_conds
                for i in range(len(encoded_text_encoder_conds)):
                    if encoded_text_encoder_conds[i] is not None:
                        text_encoder_conds[i] = encoded_text_encoder_conds[i]

        # sample noise, call unet, get target
        noise_pred, target, timesteps, weighting = self.get_noise_pred_and_target(
            args,
            accelerator,
            noise_scheduler,
            latents,
            batch,
            text_encoder_conds,
            unet,
            network,
            weight_dtype,
            train_unet,
            is_train=is_train,
        )

        huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
        loss = train_util.conditional_loss(noise_pred.float(), target.float(), args.loss_type, "none", huber_c)
        if weighting is not None:
            loss = loss * weighting
        if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
            loss = apply_masked_loss(loss, batch)
        loss = loss.mean(dim=list(range(1, loss.ndim)))  # mean over all dims except batch

        loss_weights = batch["loss_weights"]  # 各sampleごとのweight
        loss = loss * loss_weights

        loss = self.post_process_loss(loss, args, timesteps, noise_scheduler)

        return loss.mean()

    def cast_text_encoder(self, args):
        return True  # default for other than HunyuanImage

    def cast_vae(self, args):
        return True  # default for other than HunyuanImage

    def cast_unet(self, args):
        return True  # default for other than HunyuanImage

    def train(self, args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        deepspeed_utils.prepare_deepspeed_args(args)
        setup_logging(args, reset=True)

        args.total_rms_check_every_n_steps = resolve_total_rms_check_interval(
            args.target_total_rms, args.total_rms_check_every_n_steps
        )

        cache_latents = args.cache_latents
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        tokenize_strategy = self.get_tokenize_strategy(args)
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
        tokenizers = self.get_tokenizers(tokenize_strategy)  # will be removed after sample_image is refactored

        # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
        latents_caching_strategy = self.get_latents_caching_strategy(args)
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

        # データセットを準備する
        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
            if use_user_config:
                logger.info(f"Loading dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "reg_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    logger.warning(
                        "ignoring the following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                            ", ".join(ignored)
                        )
                    )
            else:
                if use_dreambooth_method:
                    logger.info("Using DreamBooth method.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                    args.train_data_dir, args.reg_data_dir
                                )
                            }
                        ]
                    }
                else:
                    logger.info("Training with captions.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": [
                                    {
                                        "image_dir": args.train_data_dir,
                                        "metadata_file": args.in_json,
                                    }
                                ]
                            }
                        ]
                    }

            blueprint = blueprint_generator.generate(user_config, args)
            train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
        else:
            # use arbitrary dataset class
            train_dataset_group = train_util.load_arbitrary_dataset(args)
            val_dataset_group = None  # placeholder until validation dataset supported for arbitrary

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

        if args.debug_dataset:
            train_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
            train_util.debug_dataset(train_dataset_group)

            if val_dataset_group is not None:
                val_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
                train_util.debug_dataset(val_dataset_group)
            return
        if len(train_dataset_group) == 0:
            logger.error(
                "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
            )
            return

        if cache_latents:
            assert (
                train_dataset_group.is_latent_cacheable()
            ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"
            if val_dataset_group is not None:
                assert (
                    val_dataset_group.is_latent_cacheable()
                ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

        self.assert_extra_args(args, train_dataset_group, val_dataset_group)  # may change some args

        lora_squeeze_schedule = LoRASqueezeSchedule(args)
        if lora_squeeze_schedule.enabled:
            if args.dim_from_weights:
                raise ValueError("LoRA-Squeeze does not support --dim_from_weights because --network_dim is the target rank")
            if args.deepspeed:
                raise ValueError("LoRA-Squeeze does not support DeepSpeed yet")
            if args.resume:
                raise ValueError("LoRA-Squeeze resume is not supported yet because saved states may have a different rank")
            if args.initial_step is not None or args.initial_epoch is not None:
                raise ValueError("LoRA-Squeeze does not support --initial_step or --initial_epoch yet")

        # acceleratorを準備する
        logger.info("preparing accelerator")
        accelerator = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process
        if lora_squeeze_schedule.enabled and accelerator.num_processes != 1:
            raise ValueError("LoRA-Squeeze currently supports only single-process training")

        # mixed precisionに対応した型を用意しておき適宜castする
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = (torch.float32 if args.no_half_vae else weight_dtype) if self.cast_vae(args) else None

        # load target models: unet may be None for lazy loading
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)
        if vae_dtype is None:
            vae_dtype = vae.dtype
            logger.info(f"vae_dtype is set to {vae_dtype} by the model since cast_vae() is false")

        # text_encoder is List[CLIPTextModel] or CLIPTextModel
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        # prepare dataset for latents caching if needed
        if cache_latents:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()

            train_dataset_group.new_cache_latents(vae, accelerator)
            if val_dataset_group is not None:
                val_dataset_group.new_cache_latents(vae, accelerator)

            vae.to("cpu")
            clean_memory_on_device(accelerator.device)

            accelerator.wait_for_everyone()

        # 必要ならテキストエンコーダーの出力をキャッシュする: Text Encoderはcpuまたはgpuへ移される
        # cache text encoder outputs if needed: Text Encoder is moved to cpu or gpu
        text_encoding_strategy = self.get_text_encoding_strategy(args)
        strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

        text_encoder_outputs_caching_strategy = self.get_text_encoder_outputs_caching_strategy(args)
        if text_encoder_outputs_caching_strategy is not None:
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_outputs_caching_strategy)
        self.cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, train_dataset_group, weight_dtype)
        if val_dataset_group is not None:
            self.cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, val_dataset_group, weight_dtype)

        if unet is None:
            # lazy load unet if needed. text encoders may be freed or replaced with dummy models for saving memory
            unet, text_encoders = self.load_unet_lazily(args, weight_dtype, accelerator, text_encoders)

        # 差分追加学習のためにモデルを読み込む
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)

        if args.base_weights is not None:
            # base_weights が指定されている場合は、指定された重みを読み込みマージする
            for i, weight_path in enumerate(args.base_weights):
                if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]

                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

                module, weights_sd = network_module.create_network_from_weights(
                    multiplier, weight_path, vae, text_encoder, unet, for_inference=True
                )
                module.merge_to(text_encoder, unet, weights_sd, weight_dtype, accelerator.device if args.lowram else "cpu")

            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        # prepare network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=", 1)
                net_kwargs[key] = value

        # if a new network is added in future, add if ~ then blocks for each network (;'∀')
        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet, **net_kwargs)
        else:
            if "dropout" not in net_kwargs:
                # workaround for LyCORIS (;^ω^)
                net_kwargs["dropout"] = args.network_dropout

            initial_network_dim = lora_squeeze_schedule.current_dim if lora_squeeze_schedule.enabled else args.network_dim
            initial_network_alpha = lora_squeeze_schedule.current_alpha if lora_squeeze_schedule.enabled else args.network_alpha
            if lora_squeeze_schedule.enabled:
                accelerator.print(
                    f"LoRA-Squeeze enabled. rank schedule: {lora_squeeze_schedule.rank_schedule_text()}; "
                    f"initial alpha: {initial_network_alpha:.8g}"
                )

            network = network_module.create_network(
                1.0,
                initial_network_dim,
                initial_network_alpha,
                vae,
                text_encoder,
                unet,
                neuron_dropout=args.network_dropout,
                **net_kwargs,
            )
        if network is None:
            return
        network_has_multiplier = hasattr(network, "set_multiplier")

        # TODO remove `hasattr` by setting up methods if not defined in the network like below  (hacky but will work):
        # if not hasattr(network, "prepare_network"):
        #    network.prepare_network = lambda args: None

        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
            logger.warning(
                "warning: scale_weight_norms is specified but the network does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません"
            )
            args.scale_weight_norms = False

        self.post_process_network(args, accelerator, network, text_encoders, unet)

        # apply network to unet and text_encoder
        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = self.is_train_text_encoder(args)
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        if args.network_weights is not None:
            # FIXME consider alpha of weights: this assumes that the alpha is not changed
            info = network.load_weights(args.network_weights)
            accelerator.print(f"load network weights from {args.network_weights}: {info}")

        if args.gradient_checkpointing:
            if args.cpu_offload_checkpointing:
                unet.enable_gradient_checkpointing(cpu_offload=True)
            else:
                unet.enable_gradient_checkpointing()

            for t_enc, flag in zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders)):
                if flag:
                    if t_enc.supports_gradient_checkpointing:
                        t_enc.gradient_checkpointing_enable()
            del t_enc
            network.enable_gradient_checkpointing()  # may have no effect

        # 学習に必要なクラスを準備する
        accelerator.print("prepare optimizer, data loader etc.")

        # make backward compatibility for text_encoder_lr
        support_multiple_lrs = hasattr(network, "prepare_optimizer_params_with_multiple_te_lrs")
        if support_multiple_lrs:
            text_encoder_lr = args.text_encoder_lr
        else:
            # toml backward compatibility
            if args.text_encoder_lr is None or isinstance(args.text_encoder_lr, float) or isinstance(args.text_encoder_lr, int):
                text_encoder_lr = args.text_encoder_lr
            else:
                text_encoder_lr = None if len(args.text_encoder_lr) == 0 else args.text_encoder_lr[0]
        try:
            if support_multiple_lrs:
                results = network.prepare_optimizer_params_with_multiple_te_lrs(text_encoder_lr, args.unet_lr, args.learning_rate)
            else:
                results = network.prepare_optimizer_params(text_encoder_lr, args.unet_lr, args.learning_rate)
            if type(results) is tuple:
                trainable_params = results[0]
                lr_descriptions = results[1]
            else:
                trainable_params = results
                lr_descriptions = None
        except TypeError as e:
            trainable_params = network.prepare_optimizer_params(text_encoder_lr, args.unet_lr)
            lr_descriptions = None

        # if len(trainable_params) == 0:
        #     accelerator.print("no trainable parameters found / 学習可能なパラメータが見つかりませんでした")
        # for params in trainable_params:
        #     for k, v in params.items():
        #         if type(v) == float:
        #             pass
        #         else:
        #             v = len(v)
        #         accelerator.print(f"trainable_params: {k} = {v}")

        if train_util.is_adv_optm_optimizer_type(args.optimizer_type):
            adv_optm_tag_counts = train_util.tag_adv_optm_trainable_parameters(network)
            tag_text = ", ".join(f"{name}={count}" for name, count in adv_optm_tag_counts.items() if count > 0)
            if tag_text:
                accelerator.print(f"tagged trainable parameters for adv_optm scaling: {tag_text}")

        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)
        optimizer_train_fn, optimizer_eval_fn = train_util.get_optimizer_train_eval_fn(optimizer, args)

        # prepare dataloader
        # strategies are set here because they cannot be referenced in another process. Copy them with the dataset
        # some strategies can be None
        train_dataset_group.set_current_strategies()
        if val_dataset_group is not None:
            val_dataset_group.set_current_strategies()

        # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset_group if val_dataset_group is not None else [],
            shuffle=False,
            batch_size=1,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        # 学習ステップ数を計算する
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
            )
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
            )

        # データセット側にも学習ステップを送信
        train_dataset_group.set_max_train_steps(args.max_train_steps)
        lora_squeeze_schedule.set_total_steps(args.max_train_steps)
        if lora_squeeze_schedule.enabled:
            accelerator.print(f"LoRA-Squeeze step distribution: {lora_squeeze_schedule.step_distribution_text()}")
            accelerator.print(f"LoRA-Squeeze step schedule: {lora_squeeze_schedule.step_schedule_text()}")

        # lr schedulerを用意する
        scheduler_training_steps = lora_squeeze_schedule.current_segment_steps if lora_squeeze_schedule.enabled else None
        lr_scheduler = train_util.get_scheduler_fix(
            args, optimizer, accelerator.num_processes, training_steps=scheduler_training_steps
        )
        if lora_squeeze_schedule.enabled:
            accelerator.print(
                "LoRA-Squeeze scheduler: independent cycle per rank segment; "
                f"segment 1/{lora_squeeze_schedule.total_segments}, steps={scheduler_training_steps}"
            )

        # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
        if args.full_fp16:
            assert (
                args.mixed_precision == "fp16"
            ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert (
                args.mixed_precision == "bf16"
            ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            accelerator.print("enable full bf16 training.")
            network.to(weight_dtype)

        unet_weight_dtype = te_weight_dtype = weight_dtype
        # Experimental Feature: Put base model into fp8 to save vram
        if args.fp8_base or args.fp8_base_unet:
            assert torch.__version__ >= "2.1.0", "fp8_base requires torch>=2.1.0 / fp8を使う場合はtorch>=2.1.0が必要です。"
            assert (
                args.mixed_precision != "no"
            ), "fp8_base requires mixed precision='fp16' or 'bf16' / fp8を使う場合はmixed_precision='fp16'または'bf16'が必要です。"
            accelerator.print("enable fp8 training for U-Net.")
            unet_weight_dtype = torch.float8_e4m3fn

            if not args.fp8_base_unet:
                accelerator.print("enable fp8 training for Text Encoder.")
            te_weight_dtype = weight_dtype if args.fp8_base_unet else torch.float8_e4m3fn

            # unet.to(accelerator.device)  # this makes faster `to(dtype)` below, but consumes 23 GB VRAM
            # unet.to(dtype=unet_weight_dtype)  # without moving to gpu, this takes a lot of time and main memory

            # logger.info(f"set U-Net weight dtype to {unet_weight_dtype}, device to {accelerator.device}")
            # unet.to(accelerator.device, dtype=unet_weight_dtype)  # this seems to be safer than above
            logger.info(f"set U-Net weight dtype to {unet_weight_dtype}")
            unet.to(dtype=unet_weight_dtype)  # do not move to device because unet is not prepared by accelerator

        unet.requires_grad_(False)
        if self.cast_unet(args):
            unet.to(dtype=unet_weight_dtype)
        for i, t_enc in enumerate(text_encoders):
            t_enc.requires_grad_(False)

            # in case of cpu, dtype is already set to fp32 because cpu does not support fp8/fp16/bf16
            if t_enc.device.type != "cpu" and self.cast_text_encoder(args):
                t_enc.to(dtype=te_weight_dtype)

                # nn.Embedding not support FP8
                if te_weight_dtype != weight_dtype:
                    self.prepare_text_encoder_fp8(i, t_enc, te_weight_dtype, weight_dtype)

        # acceleratorがなんかよろしくやってくれるらしい / accelerator will do something good
        if args.deepspeed:
            flags = self.get_text_encoders_train_flags(args, text_encoders)
            ds_model = deepspeed_utils.prepare_deepspeed_model(
                args,
                unet=unet if train_unet else None,
                text_encoder1=text_encoders[0] if flags[0] else None,
                text_encoder2=(text_encoders[1] if flags[1] else None) if len(text_encoders) > 1 else None,
                network=network,
            )
            ds_model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
                ds_model, optimizer, train_dataloader, val_dataloader, lr_scheduler
            )
            training_model = ds_model
        else:
            if train_unet:
                # default implementation is:  unet = accelerator.prepare(unet)
                unet = self.prepare_unet_with_accelerator(args, accelerator, unet)  # accelerator does some magic here
            else:
                # move to device because unet is not prepared by accelerator
                unet.to(accelerator.device, dtype=unet_weight_dtype if self.cast_unet(args) else None)
            if train_text_encoder:
                text_encoders = [
                    (accelerator.prepare(t_enc) if flag else t_enc)
                    for t_enc, flag in zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders))
                ]
                if len(text_encoders) > 1:
                    text_encoder = text_encoders
                else:
                    text_encoder = text_encoders[0]
            else:
                pass  # if text_encoder is not trained, no need to prepare. and device and dtype are already set

            network, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
                network, optimizer, train_dataloader, val_dataloader, lr_scheduler
            )
            training_model = network

        if args.gradient_checkpointing:
            # according to TI example in Diffusers, train is required
            unet.train()
            for i, (t_enc, frag) in enumerate(zip(text_encoders, self.get_text_encoders_train_flags(args, text_encoders))):
                t_enc.train()

                # set top parameter requires_grad = True for gradient checkpointing works
                if frag:
                    self.prepare_text_encoder_grad_ckpt_workaround(i, t_enc)

        else:
            unet.eval()
            for t_enc in text_encoders:
                t_enc.eval()

        del t_enc

        accelerator.unwrap_model(network).prepare_grad_etc(text_encoder, unet)

        if not cache_latents:  # キャッシュしない場合はVAEを使うのでVAEを準備する
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)

        # before resuming make hook for saving/loading to save/load the network weights only
        def save_model_hook(models, weights, output_dir):
            # pop weights of other models than network to save only network weights
            # only main process or deepspeed https://github.com/huggingface/diffusers/issues/2606
            if accelerator.is_main_process or args.deepspeed:
                remove_indices = []
                for i, model in enumerate(models):
                    if not isinstance(model, type(accelerator.unwrap_model(network))):
                        remove_indices.append(i)
                for i in reversed(remove_indices):
                    if len(weights) > i:
                        weights.pop(i)
                # print(f"save model hook: {len(weights)} weights will be saved")

            # save current ecpoch and step
            train_state_file = os.path.join(output_dir, "train_state.json")
            # +1 is needed because the state is saved before current_step is set from global_step
            logger.info(f"save train state to {train_state_file} at epoch {current_epoch.value} step {current_step.value+1}")
            with open(train_state_file, "w", encoding="utf-8") as f:
                json.dump({"current_epoch": current_epoch.value, "current_step": current_step.value + 1}, f)

        steps_from_state = None

        def load_model_hook(models, input_dir):
            # remove models except network
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(network))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                models.pop(i)
            # print(f"load model hook: {len(models)} models will be loaded")

            # load current epoch and step to
            nonlocal steps_from_state
            train_state_file = os.path.join(input_dir, "train_state.json")
            if os.path.exists(train_state_file):
                with open(train_state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                steps_from_state = data["current_step"]
                logger.info(f"load train state from {train_state_file}: {data}")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        # resumeする
        train_util.resume_from_local_or_hf_if_specified(accelerator, args)

        # epoch数を計算する
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

        # 学習する
        # TODO: find a way to handle total batch size when there are multiple datasets
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        accelerator.print("running training / 学習開始")
        accelerator.print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
        accelerator.print(
            f"  num validation images * repeats / 学習画像の数×繰り返し回数: {val_dataset_group.num_train_images if val_dataset_group is not None else 0}"
        )
        accelerator.print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
        accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
        accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
        accelerator.print(
            f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
        )
        # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
        accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

        # TODO refactor metadata creation and move to util
        metadata = {
            "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
            "ss_training_started_at": training_started_at,  # unix timestamp
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_text_encoder_lr": text_encoder_lr,
            "ss_unet_lr": args.unet_lr,
            "ss_num_train_images": train_dataset_group.num_train_images,
            "ss_num_validation_images": val_dataset_group.num_train_images if val_dataset_group is not None else 0,
            "ss_num_reg_images": train_dataset_group.num_reg_images,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            "ss_network_module": args.network_module,
            "ss_network_dim": lora_squeeze_schedule.current_dim if lora_squeeze_schedule.enabled else args.network_dim,  # None means default because another network than LoRA may have another default dim
            "ss_network_alpha": lora_squeeze_schedule.current_alpha if lora_squeeze_schedule.enabled else args.network_alpha,  # some networks may not have alpha
            "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
            "ss_mixed_precision": args.mixed_precision,
            "ss_full_fp16": bool(args.full_fp16),
            "ss_v2": bool(args.v2),
            "ss_base_model_version": model_version,
            "ss_clip_skip": args.clip_skip,
            "ss_max_token_length": args.max_token_length,
            "ss_cache_latents": bool(args.cache_latents),
            "ss_seed": args.seed,
            "ss_lowram": args.lowram,
            "ss_noise_offset": args.noise_offset,
            "ss_multires_noise_iterations": args.multires_noise_iterations,
            "ss_multires_noise_discount": args.multires_noise_discount,
            "ss_adaptive_noise_scale": args.adaptive_noise_scale,
            "ss_zero_terminal_snr": args.zero_terminal_snr,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_caption_dropout_rate": args.caption_dropout_rate,
            "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
            "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
            "ss_face_crop_aug_range": args.face_crop_aug_range,
            "ss_prior_loss_weight": args.prior_loss_weight,
            "ss_min_snr_gamma": args.min_snr_gamma,
            "ss_scale_weight_norms": args.scale_weight_norms,
            "ss_ip_noise_gamma": args.ip_noise_gamma,
            "ss_debiased_estimation": bool(args.debiased_estimation_loss),
            "ss_noise_offset_random_strength": args.noise_offset_random_strength,
            "ss_ip_noise_gamma_random_strength": args.ip_noise_gamma_random_strength,
            "ss_loss_type": args.loss_type,
            "ss_huber_schedule": args.huber_schedule,
            "ss_huber_scale": args.huber_scale,
            "ss_huber_c": args.huber_c,
            "ss_fp8_base": bool(args.fp8_base),
            "ss_fp8_base_unet": bool(args.fp8_base_unet),
            "ss_target_total_rms": args.target_total_rms,
            "ss_total_rms_check_every_n_steps": args.total_rms_check_every_n_steps,
            "ss_lora_squeeze_enabled": bool(lora_squeeze_schedule.enabled),
            "ss_lora_squeeze_start_dim": lora_squeeze_schedule.start_dim,
            "ss_lora_squeeze_target_dim": lora_squeeze_schedule.target_dim,
            "ss_lora_squeeze_target_alpha": lora_squeeze_schedule.target_alpha,
            "ss_lora_squeeze_num_squeezes": lora_squeeze_schedule.num_squeezes,
            "ss_lora_squeeze_train_after_final_squeeze": bool(lora_squeeze_schedule.train_after_final_squeeze),
            "ss_lora_squeeze_rank_distribution": lora_squeeze_schedule.rank_schedule,
            "ss_lora_squeeze_step_distribution": lora_squeeze_schedule.step_schedule,
            "ss_lora_squeeze_final_segment_ratio": lora_squeeze_schedule.final_segment_ratio,
            "ss_lora_squeeze_rank_schedule": json.dumps(lora_squeeze_schedule.ranks),
            "ss_lora_squeeze_step_schedule": json.dumps(lora_squeeze_schedule.squeeze_steps),
            "ss_lora_squeeze_segment_steps": json.dumps(lora_squeeze_schedule.segment_steps),
            "ss_lora_squeeze_segment_weights": json.dumps(lora_squeeze_schedule.segment_weights),
            "ss_lora_squeeze_scheduler_mode": "independent_per_segment" if lora_squeeze_schedule.enabled else None,
            "ss_lora_squeeze_current_segment_steps": lora_squeeze_schedule.current_segment_steps,
            "ss_lora_squeeze_current_dim": lora_squeeze_schedule.current_dim,
            "ss_lora_squeeze_current_alpha": lora_squeeze_schedule.current_alpha,
            "ss_lora_squeeze_completed_squeezes": lora_squeeze_schedule.completed_squeezes,
            "ss_validation_seed": args.validation_seed,
            "ss_validation_split": args.validation_split,
            "ss_max_validation_steps": args.max_validation_steps,
            "ss_validate_every_n_epochs": args.validate_every_n_epochs,
            "ss_validate_every_n_steps": args.validate_every_n_steps,
            "ss_resize_interpolation": args.resize_interpolation,
        }

        self.update_metadata(metadata, args)  # architecture specific metadata

        if use_user_config:
            # save metadata of multiple datasets
            # NOTE: pack "ss_datasets" value as json one time
            #   or should also pack nested collections as json?
            datasets_metadata = []
            tag_frequency = {}  # merge tag frequency for metadata editor
            dataset_dirs_info = {}  # merge subset dirs for metadata editor

            for dataset in train_dataset_group.datasets:
                is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
                dataset_metadata = {
                    "is_dreambooth": is_dreambooth_dataset,
                    "batch_size_per_device": dataset.batch_size,
                    "num_train_images": dataset.num_train_images,  # includes repeating
                    "num_reg_images": dataset.num_reg_images,
                    "resolution": (dataset.width, dataset.height),
                    "enable_bucket": bool(dataset.enable_bucket),
                    "min_bucket_reso": dataset.min_bucket_reso,
                    "max_bucket_reso": dataset.max_bucket_reso,
                    "skip_image_resolution": dataset.skip_image_resolution,
                    "tag_frequency": dataset.tag_frequency,
                    "bucket_info": dataset.bucket_info,
                    "resize_interpolation": dataset.resize_interpolation,
                }

                subsets_metadata = []
                for subset in dataset.subsets:
                    subset_metadata = {
                        "img_count": subset.img_count,
                        "num_repeats": subset.num_repeats,
                        "color_aug": bool(subset.color_aug),
                        "flip_aug": bool(subset.flip_aug),
                        "random_crop": bool(subset.random_crop),
                        "shuffle_caption": bool(subset.shuffle_caption),
                        "keep_tokens": subset.keep_tokens,
                        "keep_tokens_separator": subset.keep_tokens_separator,
                        "secondary_separator": subset.secondary_separator,
                        "enable_wildcard": bool(subset.enable_wildcard),
                        "caption_prefix": subset.caption_prefix,
                        "caption_suffix": subset.caption_suffix,
                        "resize_interpolation": subset.resize_interpolation,
                    }

                    image_dir_or_metadata_file = None
                    if subset.image_dir:
                        image_dir = os.path.basename(subset.image_dir)
                        subset_metadata["image_dir"] = image_dir
                        image_dir_or_metadata_file = image_dir

                    if is_dreambooth_dataset:
                        subset_metadata["class_tokens"] = subset.class_tokens
                        subset_metadata["is_reg"] = subset.is_reg
                        if subset.is_reg:
                            image_dir_or_metadata_file = None  # not merging reg dataset
                    else:
                        metadata_file = os.path.basename(subset.metadata_file)
                        subset_metadata["metadata_file"] = metadata_file
                        image_dir_or_metadata_file = metadata_file  # may overwrite

                    subsets_metadata.append(subset_metadata)

                    # merge dataset dir: not reg subset only
                    # TODO update additional-network extension to show detailed dataset config from metadata
                    if image_dir_or_metadata_file is not None:
                        # datasets may have a certain dir multiple times
                        v = image_dir_or_metadata_file
                        i = 2
                        while v in dataset_dirs_info:
                            v = image_dir_or_metadata_file + f" ({i})"
                            i += 1
                        image_dir_or_metadata_file = v

                        dataset_dirs_info[image_dir_or_metadata_file] = {
                            "n_repeats": subset.num_repeats,
                            "img_count": subset.img_count,
                        }

                dataset_metadata["subsets"] = subsets_metadata
                datasets_metadata.append(dataset_metadata)

                # merge tag frequency:
                for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
                    # あるディレクトリが複数のdatasetで使用されている場合、一度だけ数える
                    # もともと繰り返し回数を指定しているので、キャプション内でのタグの出現回数と、それが学習で何度使われるかは一致しない
                    # なので、ここで複数datasetの回数を合算してもあまり意味はない
                    if ds_dir_name in tag_frequency:
                        continue
                    tag_frequency[ds_dir_name] = ds_freq_for_dir

            metadata["ss_datasets"] = json.dumps(datasets_metadata)
            metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
            metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
        else:
            # conserving backward compatibility when using train_dataset_dir and reg_dataset_dir
            assert (
                len(train_dataset_group.datasets) == 1
            ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found. This seems to be a bug. / データセットは1個だけ存在するはずですが、実際には{len(train_dataset_group.datasets)}個でした。プログラムのバグかもしれません。"

            dataset = train_dataset_group.datasets[0]

            dataset_dirs_info = {}
            reg_dataset_dirs_info = {}
            if use_dreambooth_method:
                for subset in dataset.subsets:
                    info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
                    info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats, "img_count": subset.img_count}
            else:
                for subset in dataset.subsets:
                    dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                        "n_repeats": subset.num_repeats,
                        "img_count": subset.img_count,
                    }

            metadata.update(
                {
                    "ss_batch_size_per_device": args.train_batch_size,
                    "ss_total_batch_size": total_batch_size,
                    "ss_resolution": args.resolution,
                    "ss_color_aug": bool(args.color_aug),
                    "ss_flip_aug": bool(args.flip_aug),
                    "ss_random_crop": bool(args.random_crop),
                    "ss_shuffle_caption": bool(args.shuffle_caption),
                    "ss_enable_bucket": bool(dataset.enable_bucket),
                    "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
                    "ss_min_bucket_reso": dataset.min_bucket_reso,
                    "ss_max_bucket_reso": dataset.max_bucket_reso,
                    "ss_skip_image_resolution": dataset.skip_image_resolution,
                    "ss_keep_tokens": args.keep_tokens,
                    "ss_dataset_dirs": json.dumps(dataset_dirs_info),
                    "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
                    "ss_tag_frequency": json.dumps(dataset.tag_frequency),
                    "ss_bucket_info": json.dumps(dataset.bucket_info),
                }
            )

        # add extra args
        if args.network_args:
            metadata["ss_network_args"] = json.dumps(net_kwargs)

        # model name and hash
        if args.pretrained_model_name_or_path is not None:
            sd_model_name = args.pretrained_model_name_or_path
            if os.path.exists(sd_model_name):
                metadata["ss_sd_model_hash"] = train_util.model_hash(sd_model_name)
                metadata["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name)
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name

        if args.vae is not None:
            vae_name = args.vae
            if os.path.exists(vae_name):
                metadata["ss_vae_hash"] = train_util.model_hash(vae_name)
                metadata["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name

        metadata = {k: str(v) for k, v in metadata.items()}

        # make minimum metadata for filtering
        minimum_metadata = {}
        for key in train_util.SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]

        def update_lora_squeeze_metadata():
            if not lora_squeeze_schedule.enabled:
                return

            dynamic_values = {
                "ss_network_dim": lora_squeeze_schedule.current_dim,
                "ss_network_alpha": lora_squeeze_schedule.current_alpha,
                "ss_lora_squeeze_current_dim": lora_squeeze_schedule.current_dim,
                "ss_lora_squeeze_current_alpha": lora_squeeze_schedule.current_alpha,
                "ss_lora_squeeze_completed_squeezes": lora_squeeze_schedule.completed_squeezes,
                "ss_lora_squeeze_step_schedule": json.dumps(lora_squeeze_schedule.squeeze_steps),
                "ss_lora_squeeze_segment_steps": json.dumps(lora_squeeze_schedule.segment_steps),
                "ss_lora_squeeze_segment_weights": json.dumps(lora_squeeze_schedule.segment_weights),
                "ss_lora_squeeze_current_segment_steps": lora_squeeze_schedule.current_segment_steps,
            }
            for key, value in dynamic_values.items():
                metadata[key] = str(value)
                if key in minimum_metadata:
                    minimum_metadata[key] = str(value)

        # calculate steps to skip when resuming or starting from a specific step
        initial_step = 0
        if args.initial_epoch is not None or args.initial_step is not None:
            # if initial_epoch or initial_step is specified, steps_from_state is ignored even when resuming
            if steps_from_state is not None:
                logger.warning(
                    "steps from the state is ignored because initial_step is specified / initial_stepが指定されているため、stateからのステップ数は無視されます"
                )
            if args.initial_step is not None:
                initial_step = args.initial_step
            else:
                # num steps per epoch is calculated by num_processes and gradient_accumulation_steps
                initial_step = (args.initial_epoch - 1) * math.ceil(
                    len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
                )
        else:
            # if initial_epoch and initial_step are not specified, steps_from_state is used when resuming
            if steps_from_state is not None:
                initial_step = steps_from_state
                steps_from_state = None

        if initial_step > 0:
            assert (
                args.max_train_steps > initial_step
            ), f"max_train_steps should be greater than initial step / max_train_stepsは初期ステップより大きい必要があります: {args.max_train_steps} vs {initial_step}"

        epoch_to_start = 0
        if initial_step > 0:
            if args.skip_until_initial_step:
                # if skip_until_initial_step is specified, load data and discard it to ensure the same data is used
                if not args.resume:
                    logger.info(
                        f"initial_step is specified but not resuming. lr scheduler will be started from the beginning / initial_stepが指定されていますがresumeしていないため、lr schedulerは最初から始まります"
                    )
                logger.info(f"skipping {initial_step} steps / {initial_step}ステップをスキップします")
                initial_step *= args.gradient_accumulation_steps

                # set epoch to start to make initial_step less than len(train_dataloader)
                epoch_to_start = initial_step // math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            else:
                # if not, only epoch no is skipped for informative purpose
                epoch_to_start = initial_step // math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
                initial_step = 0  # do not skip

        global_step = 0
        weight_value_tracker = WeightValueTracker(args, accelerator, accelerator.unwrap_model(network))
        weight_value_tracker.record(global_step, current_epoch.value)

        noise_scheduler = self.get_noise_scheduler(args, accelerator.device)

        train_util.init_trackers(accelerator, args, "network_train")

        loss_recorder = train_util.LossRecorder()
        val_step_loss_recorder = train_util.LossRecorder()
        val_epoch_loss_recorder = train_util.LossRecorder()

        del train_dataset_group
        if val_dataset_group is not None:
            del val_dataset_group

        # callback for step start
        if hasattr(accelerator.unwrap_model(network), "on_step_start"):
            on_step_start_for_network = accelerator.unwrap_model(network).on_step_start
        else:
            on_step_start_for_network = lambda *args, **kwargs: None

        def rebuild_optimizer_for_current_network():
            nonlocal optimizer_name, optimizer_args, optimizer, optimizer_train_fn, optimizer_eval_fn, lr_scheduler, lr_descriptions

            segment_steps = lora_squeeze_schedule.current_segment_steps
            if segment_steps is None:
                return

            old_optimizer = optimizer
            old_lr_scheduler = lr_scheduler
            unwrapped_network = accelerator.unwrap_model(network)
            try:
                if support_multiple_lrs:
                    results = unwrapped_network.prepare_optimizer_params_with_multiple_te_lrs(
                        text_encoder_lr, args.unet_lr, args.learning_rate
                    )
                else:
                    results = unwrapped_network.prepare_optimizer_params(text_encoder_lr, args.unet_lr, args.learning_rate)
                if type(results) is tuple:
                    trainable_params = results[0]
                    lr_descriptions = results[1]
                else:
                    trainable_params = results
                    lr_descriptions = None
            except TypeError:
                trainable_params = unwrapped_network.prepare_optimizer_params(text_encoder_lr, args.unet_lr)
                lr_descriptions = None

            if train_util.is_adv_optm_optimizer_type(args.optimizer_type):
                train_util.tag_adv_optm_trainable_parameters(unwrapped_network)

            optimizer_name, optimizer_args, new_optimizer = train_util.get_optimizer(args, trainable_params)
            new_lr_scheduler = train_util.get_scheduler_fix(
                args, new_optimizer, accelerator.num_processes, training_steps=segment_steps
            )
            optimizer, lr_scheduler = accelerator.prepare(new_optimizer, new_lr_scheduler)
            optimizer_train_fn, optimizer_eval_fn = train_util.get_optimizer_train_eval_fn(optimizer, args)
            optimizer_train_fn()

            accelerator.print(
                "LoRA-Squeeze scheduler restarted: "
                f"segment {lora_squeeze_schedule.completed_squeezes + 1}/{lora_squeeze_schedule.total_segments}, "
                f"rank={lora_squeeze_schedule.current_dim}, steps={segment_steps}"
            )

            raw_old_optimizer = getattr(old_optimizer, "optimizer", old_optimizer)
            if hasattr(raw_old_optimizer, "state"):
                raw_old_optimizer.state.clear()
            if hasattr(accelerator, "_optimizers"):
                accelerator._optimizers = [opt for opt in accelerator._optimizers if opt is not old_optimizer]
            if hasattr(accelerator, "_schedulers"):
                accelerator._schedulers = [scheduler for scheduler in accelerator._schedulers if scheduler is not old_lr_scheduler]

        def run_lora_squeeze_if_due(global_step: int):
            event = lora_squeeze_schedule.next_due(global_step)
            if event is None:
                return

            target_dim = int(event["dim"])
            target_alpha = float(event["alpha"])
            unwrapped_network = accelerator.unwrap_model(network)
            accelerator.wait_for_everyone()
            stats = self.squeeze_lora_network(unwrapped_network, target_dim, target_alpha)
            lora_squeeze_schedule.mark_squeezed(target_dim, target_alpha)
            update_lora_squeeze_metadata()
            unwrapped_network.prepare_grad_etc(text_encoder, unet)
            weight_value_tracker.resample(unwrapped_network, args.weight_track_count, args.weight_track_seed)

            accelerator.print(
                f"LoRA-Squeeze at step {global_step}: rank {int(stats['source_rank'])} -> {target_dim}, "
                f"alpha={target_alpha:.8g}, retained_energy_min={stats['retained_energy_min']:.6f}, "
                f"retained_energy_mean={stats['retained_energy_mean']:.6f}"
            )

            rebuild_optimizer_for_current_network()
            clean_memory_on_device(accelerator.device)

        # function for saving/removing
        def save_model(ckpt_name, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)

            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)
            update_lora_squeeze_metadata()

            metadata_to_save = minimum_metadata if args.no_metadata else metadata
            sai_metadata = self.get_sai_model_spec(args)
            metadata_to_save.update(sai_metadata)

            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)
            if args.huggingface_repo_id is not None:
                huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        # if text_encoder is not needed for training, delete it to save memory.
        # TODO this can be automated after SDXL sample prompt cache is implemented
        if self.is_text_encoder_not_needed_for_training(args):
            logger.info("text_encoder is not needed for training. deleting to save memory.")
            for t_enc in text_encoders:
                del t_enc
            text_encoders = []
            text_encoder = None
            gc.collect()
            clean_memory_on_device(accelerator.device)

        # For --sample_at_first
        optimizer_eval_fn()
        self.sample_images(accelerator, args, 0, global_step, accelerator.device, vae, tokenizers, text_encoder, unet)
        optimizer_train_fn()
        is_tracking = len(accelerator.trackers) > 0
        if is_tracking:
            # log empty object to commit the sample images to wandb
            accelerator.log({}, step=0)

        # training loop
        if initial_step > 0:  # only if skip_until_initial_step is specified
            for skip_epoch in range(epoch_to_start):  # skip epochs
                logger.info(f"skipping epoch {skip_epoch+1} because initial_step (multiplied) is {initial_step}")
                initial_step -= len(train_dataloader)
            global_step = initial_step

        # log device and dtype for each model
        logger.info(f"unet dtype: {unet_weight_dtype}, device: {unet.device}")
        for i, t_enc in enumerate(text_encoders):
            params_itr = t_enc.parameters()
            params_itr.__next__()  # skip the first parameter
            params_itr.__next__()  # skip the second parameter. because CLIP first two parameters are embeddings
            param_3rd = params_itr.__next__()
            logger.info(f"text_encoder [{i}] dtype: {param_3rd.dtype}, device: {t_enc.device}")

        clean_memory_on_device(accelerator.device)

        progress_bar = tqdm(
            range(args.max_train_steps - initial_step), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps"
        )

        validation_steps = (
            min(args.max_validation_steps, len(val_dataloader)) if args.max_validation_steps is not None else len(val_dataloader)
        )
        NUM_VALIDATION_TIMESTEPS = 4  # 200, 400, 600, 800 TODO make this configurable
        min_timestep = 0 if args.min_timestep is None else args.min_timestep
        max_timestep = noise_scheduler.config.num_train_timesteps if args.max_timestep is None else args.max_timestep
        validation_timesteps = np.linspace(min_timestep, max_timestep, (NUM_VALIDATION_TIMESTEPS + 2), dtype=int)[1:-1]
        validation_total_steps = validation_steps * len(validation_timesteps)
        original_args_min_timestep = args.min_timestep
        original_args_max_timestep = args.max_timestep

        def switch_rng_state(seed: int) -> tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple]:
            cpu_rng_state = torch.get_rng_state()
            if accelerator.device.type == "cuda":
                gpu_rng_state = torch.cuda.get_rng_state()
            elif accelerator.device.type == "xpu":
                gpu_rng_state = torch.xpu.get_rng_state()
            elif accelerator.device.type == "mps":
                gpu_rng_state = torch.cuda.get_rng_state()
            else:
                gpu_rng_state = None
            python_rng_state = random.getstate()

            torch.manual_seed(seed)
            random.seed(seed)

            return (cpu_rng_state, gpu_rng_state, python_rng_state)

        def restore_rng_state(rng_states: tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple]):
            cpu_rng_state, gpu_rng_state, python_rng_state = rng_states
            torch.set_rng_state(cpu_rng_state)
            if gpu_rng_state is not None:
                if accelerator.device.type == "cuda":
                    torch.cuda.set_rng_state(gpu_rng_state)
                elif accelerator.device.type == "xpu":
                    torch.xpu.set_rng_state(gpu_rng_state)
                elif accelerator.device.type == "mps":
                    torch.cuda.set_rng_state(gpu_rng_state)
            random.setstate(python_rng_state)

        stop_due_to_total_rms = False
        last_total_rms = None

        for epoch in range(epoch_to_start, num_train_epochs):
            accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}\n")
            current_epoch.value = epoch + 1

            metadata["ss_epoch"] = str(epoch + 1)

            accelerator.unwrap_model(network).on_epoch_start(text_encoder, unet)  # network.train() is called here

            # TRAINING
            skipped_dataloader = None
            if initial_step > 0:
                skipped_dataloader = accelerator.skip_first_batches(train_dataloader, initial_step - 1)
                initial_step = 1

            for step, batch in enumerate(skipped_dataloader or train_dataloader):
                current_step.value = global_step
                if initial_step > 0:
                    initial_step -= 1
                    continue

                with accelerator.accumulate(training_model):
                    on_step_start_for_network(text_encoder, unet)

                    # preprocess batch for each model
                    self.on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=True)

                    loss = self.process_batch(
                        batch,
                        text_encoders,
                        unet,
                        network,
                        vae,
                        noise_scheduler,
                        vae_dtype,
                        weight_dtype,
                        accelerator,
                        args,
                        text_encoding_strategy,
                        tokenize_strategy,
                        is_train=True,
                        train_text_encoder=train_text_encoder,
                        train_unet=train_unet,
                    )

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        self.all_reduce_network(accelerator, network)  # sync DDP grad manually
                        if args.max_grad_norm != 0.0:
                            params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                        if hasattr(network, "update_grad_norms"):
                            network.update_grad_norms()
                        if hasattr(network, "update_norms"):
                            network.update_norms()

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if args.scale_weight_norms:
                    keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(network).apply_max_norm_regularization(
                        args.scale_weight_norms, accelerator.device
                    )
                    mean_grad_norm = None
                    mean_combined_norm = None
                    max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                else:
                    if hasattr(network, "weight_norms"):
                        weight_norms = network.weight_norms()
                        mean_norm = weight_norms.mean().item() if weight_norms is not None else None
                        grad_norms = network.grad_norms()
                        mean_grad_norm = grad_norms.mean().item() if grad_norms is not None else None
                        combined_weight_norms = network.combined_weight_norms()
                        mean_combined_norm = combined_weight_norms.mean().item() if combined_weight_norms is not None else None
                        maximum_norm = weight_norms.max().item() if weight_norms is not None else None
                        keys_scaled = None
                        max_mean_logs = {}
                    else:
                        keys_scaled, mean_norm, maximum_norm = None, None, None
                        mean_grad_norm = None
                        mean_combined_norm = None
                        max_mean_logs = {}

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    weight_value_tracker.record(global_step, current_epoch.value)

                    if (
                        args.total_rms_check_every_n_steps > 0
                        and global_step % args.total_rms_check_every_n_steps == 0
                    ):
                        total_rms = self.compute_total_scaled_lora_rms(accelerator.unwrap_model(network))
                        last_total_rms = total_rms

                        if args.target_total_rms is None:
                            accelerator.print(f"total_rms={total_rms:.8g}")
                        else:
                            accelerator.print(
                                f"total_rms={total_rms:.8g}, target_total_rms={args.target_total_rms:.8g}"
                            )

                        if args.target_total_rms is not None and total_rms >= args.target_total_rms:
                            accelerator.print(
                                f"target_total_rms reached: {total_rms:.8g} >= {args.target_total_rms:.8g}"
                            )
                            stop_due_to_total_rms = True

                    run_lora_squeeze_if_due(global_step)

                    optimizer_eval_fn()
                    self.sample_images(
                        accelerator, args, None, global_step, accelerator.device, vae, tokenizers, text_encoder, unet
                    )
                    progress_bar.unpause()

                    # 指定ステップごとにモデルを保存
                    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                            save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch)

                            if args.save_state:
                                train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

                            remove_step_no = train_util.get_remove_step_no(args, global_step)
                            if remove_step_no is not None:
                                remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                                remove_model(remove_ckpt_name)
                    optimizer_train_fn()

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                logs = {"avr_loss": avr_loss}

                if last_total_rms is not None:
                    logs["total_rms"] = last_total_rms
                if lora_squeeze_schedule.enabled:
                    logs["lora_squeeze_dim"] = lora_squeeze_schedule.current_dim

                progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if is_tracking:
                    logs = self.generate_step_logs(
                        args,
                        current_loss,
                        avr_loss,
                        lr_scheduler,
                        lr_descriptions,
                        optimizer,
                        keys_scaled,
                        mean_norm,
                        maximum_norm,
                        mean_grad_norm,
                        mean_combined_norm,
                    )
                    if last_total_rms is not None:
                        logs["strength/total_rms"] = last_total_rms
                    if lora_squeeze_schedule.enabled:
                        logs["lora_squeeze/current_dim"] = lora_squeeze_schedule.current_dim
                        logs["lora_squeeze/current_alpha"] = lora_squeeze_schedule.current_alpha
                        logs["lora_squeeze/completed_squeezes"] = lora_squeeze_schedule.completed_squeezes
                    self.step_logging(accelerator, logs, global_step, epoch + 1)

                # VALIDATION PER STEP: global_step is already incremented
                # for example, if validate_every_n_steps=100, validate at step 100, 200, 300, ...
                should_validate_step = args.validate_every_n_steps is not None and global_step % args.validate_every_n_steps == 0
                if accelerator.sync_gradients and validation_steps > 0 and should_validate_step:
                    optimizer_eval_fn()
                    accelerator.unwrap_model(network).eval()
                    rng_states = switch_rng_state(args.validation_seed if args.validation_seed is not None else args.seed)

                    val_progress_bar = tqdm(
                        range(validation_total_steps),
                        smoothing=0,
                        disable=not accelerator.is_local_main_process,
                        desc="validation steps",
                    )
                    val_timesteps_step = 0
                    for val_step, batch in enumerate(val_dataloader):
                        if val_step >= validation_steps:
                            break

                        for timestep in validation_timesteps:
                            self.on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=False)

                            args.min_timestep = args.max_timestep = timestep  # dirty hack to change timestep

                            loss = self.process_batch(
                                batch,
                                text_encoders,
                                unet,
                                network,
                                vae,
                                noise_scheduler,
                                vae_dtype,
                                weight_dtype,
                                accelerator,
                                args,
                                text_encoding_strategy,
                                tokenize_strategy,
                                is_train=False,
                                train_text_encoder=train_text_encoder,  # this is needed for validation because Text Encoders must be called if train_text_encoder is True
                                train_unet=train_unet,
                            )

                            current_loss = loss.detach().item()
                            val_step_loss_recorder.add(epoch=epoch, step=val_timesteps_step, loss=current_loss)
                            val_progress_bar.update(1)
                            val_progress_bar.set_postfix(
                                {"val_avg_loss": val_step_loss_recorder.moving_average, "timestep": timestep}
                            )

                            # if is_tracking:
                            #     logs = {f"loss/validation/step_current_{timestep}": current_loss}
                            #     self.val_logging(accelerator, logs, global_step, epoch + 1, val_step)

                            self.on_validation_step_end(args, accelerator, network, text_encoders, unet, batch, weight_dtype)
                            val_timesteps_step += 1

                    if is_tracking:
                        loss_validation_divergence = val_step_loss_recorder.moving_average - loss_recorder.moving_average
                        logs = {
                            "loss/validation/step_average": val_step_loss_recorder.moving_average,
                            "loss/validation/step_divergence": loss_validation_divergence,
                        }
                        self.step_logging(accelerator, logs, global_step, epoch=epoch + 1)

                    restore_rng_state(rng_states)
                    args.min_timestep = original_args_min_timestep
                    args.max_timestep = original_args_max_timestep
                    optimizer_train_fn()
                    accelerator.unwrap_model(network).train()
                    progress_bar.unpause()

                if global_step >= args.max_train_steps or stop_due_to_total_rms:
                    break

            # EPOCH VALIDATION
            should_validate_epoch = (
                (epoch + 1) % args.validate_every_n_epochs == 0 if args.validate_every_n_epochs is not None else True
            )

            if should_validate_epoch and len(val_dataloader) > 0:
                optimizer_eval_fn()
                accelerator.unwrap_model(network).eval()
                rng_states = switch_rng_state(args.validation_seed if args.validation_seed is not None else args.seed)

                val_progress_bar = tqdm(
                    range(validation_total_steps),
                    smoothing=0,
                    disable=not accelerator.is_local_main_process,
                    desc="epoch validation steps",
                )

                val_timesteps_step = 0
                for val_step, batch in enumerate(val_dataloader):
                    if val_step >= validation_steps:
                        break

                    for timestep in validation_timesteps:
                        args.min_timestep = args.max_timestep = timestep

                        # temporary, for batch processing
                        self.on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=False)

                        loss = self.process_batch(
                            batch,
                            text_encoders,
                            unet,
                            network,
                            vae,
                            noise_scheduler,
                            vae_dtype,
                            weight_dtype,
                            accelerator,
                            args,
                            text_encoding_strategy,
                            tokenize_strategy,
                            is_train=False,
                            train_text_encoder=train_text_encoder,
                            train_unet=train_unet,
                        )

                        current_loss = loss.detach().item()
                        val_epoch_loss_recorder.add(epoch=epoch, step=val_timesteps_step, loss=current_loss)
                        val_progress_bar.update(1)
                        val_progress_bar.set_postfix(
                            {"val_epoch_avg_loss": val_epoch_loss_recorder.moving_average, "timestep": timestep}
                        )

                        # if is_tracking:
                        #     logs = {f"loss/validation/epoch_current_{timestep}": current_loss}
                        #     self.val_logging(accelerator, logs, global_step, epoch + 1, val_step)

                        self.on_validation_step_end(args, accelerator, network, text_encoders, unet, batch, weight_dtype)
                        val_timesteps_step += 1

                if is_tracking:
                    avr_loss: float = val_epoch_loss_recorder.moving_average
                    loss_validation_divergence = val_epoch_loss_recorder.moving_average - loss_recorder.moving_average
                    logs = {
                        "loss/validation/epoch_average": avr_loss,
                        "loss/validation/epoch_divergence": loss_validation_divergence,
                    }
                    self.epoch_logging(accelerator, logs, global_step, epoch + 1)

                restore_rng_state(rng_states)
                args.min_timestep = original_args_min_timestep
                args.max_timestep = original_args_max_timestep
                optimizer_train_fn()
                accelerator.unwrap_model(network).train()
                progress_bar.unpause()

            # END OF EPOCH
            if is_tracking:
                logs = {"loss/epoch_average": loss_recorder.moving_average}
                self.epoch_logging(accelerator, logs, global_step, epoch + 1)

            accelerator.wait_for_everyone()

            # 指定エポックごとにモデルを保存
            optimizer_eval_fn()
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)

                    remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                        remove_model(remove_ckpt_name)

                    if args.save_state:
                        train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

            self.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizers, text_encoder, unet)
            progress_bar.unpause()
            optimizer_train_fn()

            # end of epoch
            if stop_due_to_total_rms:
                break

        # metadata["ss_epoch"] = str(num_train_epochs)
        metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process:
            network = accelerator.unwrap_model(network)

        accelerator.end_training()
        optimizer_eval_fn()

        if is_main_process and (args.save_state or args.save_state_on_train_end):
            train_util.save_state_on_train_end(args, accelerator)

        if is_main_process:
            ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)

            logger.info("model saved.")

        weight_value_tracker.close()


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument(
        "--target_total_rms",
        type=float,
        default=None,
        help="stop training when total scaled adapter RMS reaches this value",
    )

    parser.add_argument(
        "--total_rms_check_every_n_steps",
        type=int,
        default=None,
        help=(
            "log total scaled adapter RMS every N optimizer steps; omit to disable unless "
            "--target_total_rms is set, in which case the default interval is 25. Set 0 to disable"
        ),
    )

    parser.add_argument(
        "--lora_squeeze_start_dim",
        type=int,
        default=None,
        help="initial LoRA rank for LoRA-Squeeze; --network_dim remains the final target rank",
    )

    parser.add_argument(
        "--lora_squeeze_num_squeezes",
        "--lora_squeeze_amount_of_squeezes",
        type=int,
        default=0,
        help="number of scheduled LoRA-Squeeze rank reductions to perform",
    )

    parser.add_argument(
        "--lora_squeeze_train_after_final_squeeze",
        action="store_true",
        help="continue training for one final segment after squeezing to --network_dim",
    )

    parser.add_argument(
        "--lora_squeeze_rank_schedule",
        type=str,
        default="linear",
        choices=LoRASqueezeSchedule.RANK_SCHEDULE_CHOICES,
        help=(
            "choose how intermediate LoRA-Squeeze ranks are spaced; linear uses equal rank differences, "
            "geometric uses approximately equal compression ratios"
        ),
    )

    parser.add_argument(
        "--lora_squeeze_step_schedule",
        "--squeeze_step_schedule",
        type=str,
        default=None,
        choices=LoRASqueezeSchedule.STEP_SCHEDULE_CHOICES,
        help=(
            "spread LoRA-Squeeze training steps by current rank; use equal or omit for equal segments. "
            "Use inverse modes to give smaller ranks more steps"
        ),
    )

    parser.add_argument(
        "--lora_squeeze_final_segment_ratio",
        "--squeeze_final_segment_ratio",
        type=float,
        default=1.0,
        help="multiply the final post-squeeze training segment length when --lora_squeeze_train_after_final_squeeze is used",
    )

    parser.add_argument(
        "--track_weight_values",
        action="store_true",
        help="track a random sample of trainable weight values to a CSV file in the logging directory",
    )

    parser.add_argument(
        "--weight_track_count",
        type=int,
        default=32,
        help="number of trainable weight values to track when --track_weight_values is enabled",
    )

    parser.add_argument(
        "--weight_track_interval",
        type=int,
        default=1,
        help="record tracked weight values every N optimizer steps",
    )

    parser.add_argument(
        "--weight_track_seed",
        type=int,
        default=0,
        help="random seed used to choose which trainable weight values are tracked",
    )

    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="[EXPERIMENTAL] enable offloading of tensors to CPU during checkpointing for U-Net or DiT, if supported"
        " / 勾配チェックポイント時にテンソルをCPUにオフロードする（U-NetまたはDiTのみ、サポートされている場合）",
    )
    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない"
    )
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=None,
        nargs="*",
        help="learning rate for Text Encoder, can be multiple / Text Encoderの学習率、複数指定可能",
    )
    parser.add_argument(
        "--fp8_base_unet",
        action="store_true",
        help="use fp8 for U-Net (or DiT), Text Encoder is fp16 or bf16"
        " / U-Net（またはDiT）にfp8を使用する。Text Encoderはfp16またはbf16",
    )

    parser.add_argument(
        "--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み"
    )
    parser.add_argument(
        "--network_module", type=str, default=None, help="network module to train / 学習対象のネットワークのモジュール"
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=None,
        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    )
    parser.add_argument(
        "--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net関連部分のみ学習する"
    )
    parser.add_argument(
        "--network_train_text_encoder_only",
        action="store_true",
        help="only training Text Encoder part / Text Encoder関連部分のみ学習する",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--skip_until_initial_step",
        action="store_true",
        help="skip training until initial_step is reached / initial_stepに到達するまで学習をスキップする",
    )
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=None,
        help="initial epoch number, 1 means first epoch (same as not specifying). NOTE: initial_epoch/step doesn't affect to lr scheduler. Which means lr scheduler will start from 0 without `--resume`."
        + " / 初期エポック数、1で最初のエポック（未指定時と同じ）。注意：initial_epoch/stepはlr schedulerに影響しないため、`--resume`しない場合はlr schedulerは0から始まる",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=None,
        help="initial step number including all epochs, 0 means first step (same as not specifying). overwrites initial_epoch."
        + " / 初期ステップ数、全エポックを含むステップ数、0で最初のステップ（未指定時と同じ）。initial_epochを上書きする",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=None,
        help="Validation seed for shuffling validation dataset, training `--seed` used otherwise / 検証データセットをシャッフルするための検証シード、それ以外の場合はトレーニング `--seed` を使用する",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.0,
        help="Split for validation images out of the training dataset / 学習画像から検証画像に分割する割合",
    )
    parser.add_argument(
        "--validate_every_n_steps",
        type=int,
        default=None,
        help="Run validation on validation dataset every N steps. By default, validation will only occur every epoch if a validation dataset is available / 検証データセットの検証をNステップごとに実行します。デフォルトでは、検証データセットが利用可能な場合にのみ、検証はエポックごとに実行されます",
    )
    parser.add_argument(
        "--validate_every_n_epochs",
        type=int,
        default=None,
        help="Run validation dataset every N epochs. By default, validation will run every epoch if a validation dataset is available / 検証データセットをNエポックごとに実行します。デフォルトでは、検証データセットが利用可能な場合、検証はエポックごとに実行されます",
    )
    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=None,
        help="Max number of validation dataset items processed. By default, validation will run the entire validation dataset / 処理される検証データセット項目の最大数。デフォルトでは、検証は検証データセット全体を実行します",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = NetworkTrainer()
    trainer.train(args)
