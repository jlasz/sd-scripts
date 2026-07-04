"""LoRA-Squeeze optimizer, scheduler, and Accelerate lifecycle management."""

import argparse
import copy
import json
import math
from dataclasses import dataclass
from multiprocessing import Value
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from library.accelerator_utils import (
    find_replaceable_optimizer,
    find_replaceable_scheduler,
    make_replaceable_optimizer_scheduler,
)
from library.lora_squeeze_schedule import (
    LoRASqueezeSchedule,
    TRAIN_STATE_LORA_SQUEEZE_KEY,
    get_remaining_training_steps,
    get_resume_epoch_and_batch_offset,
    is_training_budget_exhausted,
    restore_lora_squeeze_state_from_resume_dir,
    restore_lora_squeeze_state_from_train_state,
    validate_lora_squeeze_network_args,
    validate_lora_squeeze_training_args,
)
from library.lora_squeeze_compression import (
    get_lora_squeeze_optimizer_parameter_layout,
    restore_lora_module_layers,
    preserve_lora_squeeze_alpha_precision,
    snapshot_lora_module_layers,
    squeeze_lora_network,
    validate_lora_squeeze_network,
    validate_lora_squeeze_optimizer_parameters,
)
from library.lora_squeeze_network import validate_lora_squeeze_network_module
from library.lora_squeeze_optimizer import (
    OptimizerStateCPUStagingError,
    copy_optimizer_param_group_state,
    move_optimizer_state_to_parameter_devices,
    offload_optimizer_state_to_cpu,
    prepare_optimizer_state_transfer,
    restore_offloaded_optimizer_state,
    validate_optimizer_scheduler_modes,
)


@dataclass
class LoRASqueezeStepContext:
    train_dim: int
    train_alpha: float
    learning_rates: List[float]
    optimizer_param_groups: List[Dict[str, float]]
    transition_stats: Optional[Dict[str, float]] = None


class LoRASqueezeRuntime:
    """Small integration surface between NetworkTrainer and LoRA-Squeeze."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.schedule = LoRASqueezeSchedule(args)
        self.absolute_update_step = None
        self.uses_conv_lora = False
        if not self.enabled:
            return
        self.absolute_update_step = Value("i", 0)
        validate_lora_squeeze_training_args(args)
        if args.dim_from_weights:
            raise ValueError(
                "LoRA-Squeeze does not support --dim_from_weights because --network_dim is the target rank"
            )
        if args.deepspeed:
            raise ValueError("LoRA-Squeeze does not support DeepSpeed yet")
        if args.initial_step is not None or args.initial_epoch is not None:
            raise ValueError("LoRA-Squeeze does not support --initial_step or --initial_epoch yet")

    @property
    def enabled(self) -> bool:
        return self.schedule.enabled

    @property
    def current_dim(self) -> int:
        return self.schedule.current_dim

    @property
    def current_alpha(self) -> float:
        return self.schedule.current_alpha

    def restore_from_resume_dir(self, state_dir: str):
        restore_lora_squeeze_state_from_resume_dir(self.schedule, state_dir)

    def restore_from_train_state(self, train_state: Dict[str, Any]):
        if self.enabled:
            restore_lora_squeeze_state_from_train_state(self.schedule, train_state)

    def validate_process_count(self, num_processes: int):
        if self.enabled and num_processes != 1:
            raise ValueError("LoRA-Squeeze currently supports only single-process training")

    def validate_network_module(self, network_module: Any, network_args: Dict[str, Any]):
        if not self.enabled:
            return
        validate_lora_squeeze_network_args(network_args)
        validate_lora_squeeze_network_module(network_module, network_args)

    def prepare_network_args(self, network_args: Dict[str, Any]) -> Dict[str, Any]:
        """Translate homogeneous C3Lier target settings to the current squeeze stage."""

        if not self.enabled:
            return network_args
        if "conv_alpha" in network_args and "conv_dim" not in network_args:
            raise ValueError("LoRA-Squeeze requires conv_dim when conv_alpha is specified")
        if "conv_dim" not in network_args:
            return network_args

        try:
            target_conv_dim = int(network_args["conv_dim"])
            target_conv_alpha = float(network_args.get("conv_alpha", 1.0))
        except (TypeError, ValueError) as error:
            raise ValueError("LoRA-Squeeze requires numeric conv_dim and conv_alpha values") from error
        if target_conv_dim != self.schedule.target_dim:
            raise ValueError(
                "LoRA-Squeeze requires homogeneous target ranks: "
                f"conv_dim={target_conv_dim} must equal --network_dim={self.schedule.target_dim}"
            )
        if not math.isfinite(target_conv_alpha) or not math.isclose(
            target_conv_alpha, self.schedule.target_alpha, rel_tol=1e-5, abs_tol=1e-8
        ):
            raise ValueError(
                "LoRA-Squeeze requires homogeneous target alphas: "
                f"conv_alpha={target_conv_alpha:.8g} must equal --network_alpha={self.schedule.target_alpha:.8g}"
            )

        self.uses_conv_lora = True
        network_args["conv_dim"] = str(self.current_dim)
        network_args["conv_alpha"] = str(self.current_alpha)
        return network_args

    def prepare_optimizer_scheduler_for_accelerator(self, optimizer: Any, lr_scheduler: Any) -> Tuple[Any, Any]:
        if not self.enabled:
            return optimizer, lr_scheduler
        return make_replaceable_optimizer_scheduler(optimizer, lr_scheduler)

    def validate_network(self, network: torch.nn.Module) -> Dict[str, float]:
        return validate_lora_squeeze_network(network, self.schedule)

    def preserve_alpha_precision(self, network: torch.nn.Module) -> int:
        if not self.enabled:
            return 0
        return preserve_lora_squeeze_alpha_precision(network)

    def initial_network_rank_alpha(self, network_dim: int, network_alpha: float) -> Tuple[int, float]:
        if not self.enabled:
            return network_dim, network_alpha
        return self.current_dim, self.current_alpha

    def set_total_steps(self, max_train_steps: int):
        self.schedule.set_total_steps(max_train_steps)

    def scheduler_training_steps(self) -> Optional[int]:
        if self.enabled and self.schedule.scheduler_mode == "per_squeeze":
            return self.schedule.current_segment_steps
        return None

    def preflight_optimizer(self, optimizer: torch.optim.Optimizer, network: torch.nn.Module):
        if not self.enabled:
            return None
        validate_optimizer_scheduler_modes(
            optimizer,
            self.schedule.optimizer_mode,
            self.schedule.scheduler_mode,
        )
        if self.schedule.optimizer_mode != "global":
            return None
        validate_lora_squeeze_optimizer_parameters(network, optimizer)
        return prepare_optimizer_state_transfer(optimizer)

    def state_dict(self) -> Dict[str, Any]:
        return self.schedule.state_dict()

    def add_to_train_state(self, train_state: Dict[str, Any]):
        if self.enabled:
            train_state[TRAIN_STATE_LORA_SQUEEZE_KEY] = self.state_dict()

    def set_initial_step(self, initial_step: int):
        if self.enabled:
            self.absolute_update_step.value = initial_step

    def progress_step(self) -> Optional[int]:
        return self.absolute_update_step.value if self.enabled else None

    def _validate_absolute_step(self, global_step: int):
        if self.enabled and global_step != self.absolute_update_step.value:
            raise RuntimeError(
                "LoRA-Squeeze absolute update step diverged from NetworkTrainer global_step: "
                f"{self.absolute_update_step.value} != {global_step}"
            )

    def remaining_steps(self, max_train_steps: int, global_step: int) -> int:
        self._validate_absolute_step(global_step)
        return get_remaining_training_steps(max_train_steps, global_step)

    def budget_exhausted(self, max_train_steps: int, global_step: int) -> bool:
        self._validate_absolute_step(global_step)
        return is_training_budget_exhausted(max_train_steps, global_step)

    def resume_epoch_and_batch_offset(
        self,
        completed_steps: int,
        num_batches_per_epoch: int,
        gradient_accumulation_steps: int,
    ) -> Tuple[int, int]:
        return get_resume_epoch_and_batch_offset(
            completed_steps,
            num_batches_per_epoch,
            gradient_accumulation_steps,
        )

    def capture_step_context(self, optimizer: Any) -> Optional[LoRASqueezeStepContext]:
        if not self.enabled:
            return None
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        optimizer_param_groups = []
        for group in raw_optimizer.param_groups:
            snapshot = {}
            for key in ("lr", "d", "effective_lr"):
                if key not in group:
                    continue
                try:
                    snapshot[key] = float(group[key])
                except (TypeError, ValueError):
                    continue
            optimizer_param_groups.append(snapshot)
        return LoRASqueezeStepContext(
            train_dim=self.current_dim,
            train_alpha=self.current_alpha,
            learning_rates=[group["lr"] for group in optimizer_param_groups],
            optimizer_param_groups=optimizer_param_groups,
        )

    def run_after_optimizer_step(
        self,
        controller: "LoRASqueezeTrainingController",
        context: Optional[LoRASqueezeStepContext],
        global_step: int,
    ) -> bool:
        if not self.enabled:
            return False
        expected_step = self.absolute_update_step.value + 1
        if global_step != expected_step:
            raise RuntimeError(
                f"LoRA-Squeeze expected absolute update step {expected_step}, received {global_step}"
            )
        self.absolute_update_step.value = global_step
        squeezed = controller.run_if_due(global_step)
        if context is not None and squeezed:
            context.transition_stats = dict(controller.last_squeeze_stats or {})
        return squeezed

    def append_step_logs(self, logs: Dict[str, Any], context: Optional[LoRASqueezeStepContext]):
        if not self.enabled or context is None:
            return
        logs["lora_squeeze/train_dim"] = context.train_dim
        logs["lora_squeeze/train_alpha"] = context.train_alpha
        logs["lora_squeeze/current_dim"] = self.current_dim
        logs["lora_squeeze/current_alpha"] = self.current_alpha
        logs["lora_squeeze/completed_squeezes"] = self.schedule.completed_squeezes
        logs["lora_squeeze/transition"] = int(context.transition_stats is not None)
        if context.transition_stats:
            for key in (
                "modules",
                "source_rank",
                "target_rank",
                "retained_energy_min",
                "retained_energy_mean",
                "numerical_rank_min",
                "numerical_rank_mean",
                "rank_deficient_modules",
                "revived_rank_channels",
                "optimizer_state_projected",
                "optimizer_state_reset",
                "optimizer_state_empty",
                "optimizer_state_warm_restarted",
            ):
                if key in context.transition_stats:
                    logs[f"lora_squeeze/{key}"] = context.transition_stats[key]

    def metadata_values(self) -> Dict[str, Any]:
        schedule = self.schedule
        if not self.enabled:
            return {}
        return {
            "ss_network_dim": schedule.current_dim,
            "ss_network_alpha": schedule.current_alpha,
            "ss_lora_squeeze_enabled": True,
            "ss_lora_squeeze_start_dim": schedule.start_dim,
            "ss_lora_squeeze_target_dim": schedule.target_dim,
            "ss_lora_squeeze_target_alpha": schedule.target_alpha,
            "ss_lora_squeeze_num_squeezes": schedule.num_squeezes,
            "ss_lora_squeeze_train_after_final_squeeze": bool(schedule.train_after_final_squeeze),
            "ss_lora_squeeze_optimizer_mode": schedule.optimizer_mode,
            "ss_lora_squeeze_scheduler_mode": schedule.scheduler_mode,
            "ss_lora_squeeze_rank_distribution": schedule.rank_schedule,
            "ss_lora_squeeze_step_distribution": schedule.step_schedule,
            "ss_lora_squeeze_alpha_schedule": schedule.alpha_schedule,
            "ss_lora_squeeze_first_segment_ratio": schedule.first_segment_ratio,
            "ss_lora_squeeze_final_segment_ratio": schedule.final_segment_ratio,
            "ss_lora_squeeze_rank_schedule": json.dumps(schedule.ranks),
            "ss_lora_squeeze_step_schedule": json.dumps(schedule.squeeze_steps),
            "ss_lora_squeeze_segment_steps": json.dumps(schedule.segment_steps),
            "ss_lora_squeeze_segment_weights": json.dumps(schedule.segment_weights),
            "ss_lora_squeeze_current_segment_steps": schedule.current_segment_steps,
            "ss_lora_squeeze_current_dim": schedule.current_dim,
            "ss_lora_squeeze_current_alpha": schedule.current_alpha,
            "ss_lora_squeeze_completed_squeezes": schedule.completed_squeezes,
        }

    def update_metadata(self, metadata: Dict[str, Any], minimum_metadata: Dict[str, Any]):
        for key, value in self.metadata_values().items():
            value = str(value)
            metadata[key] = value
            if key in minimum_metadata:
                minimum_metadata[key] = value
        if self.uses_conv_lora:
            for target in (metadata, minimum_metadata):
                serialized_args = target.get("ss_network_args")
                if serialized_args is None:
                    continue
                network_args = json.loads(serialized_args)
                network_args["conv_dim"] = str(self.current_dim)
                network_args["conv_alpha"] = str(self.current_alpha)
                target["ss_network_args"] = json.dumps(network_args)

    def resume_status_message(self, is_resuming: bool) -> Optional[str]:
        if not self.enabled or not is_resuming:
            return None
        schedule = self.schedule
        return (
            "LoRA-Squeeze resume state restored: "
            f"completed_squeezes={schedule.completed_squeezes}, "
            f"current_rank={schedule.current_dim}, current_alpha={schedule.current_alpha:.8g}"
        )

    def network_creation_status_message(self, network_dim: int, network_alpha: float) -> Optional[str]:
        if not self.enabled:
            return None
        schedule = self.schedule
        return (
            f"LoRA-Squeeze enabled. rank schedule: {schedule.rank_schedule_text()}; "
            f"alpha schedule: {schedule.alpha_schedule}; "
            f"current rank: {network_dim}; current alpha: {network_alpha:.8g}"
        )

    def preflight_status_message(self, stats: Dict[str, float]) -> Optional[str]:
        if not self.enabled:
            return None
        return (
            "LoRA-Squeeze preflight: "
            f"modules={int(stats['modules'])}, current_rank={int(stats['source_rank'])}"
        )

    def step_schedule_status_messages(self) -> List[str]:
        if not self.enabled:
            return []
        return [
            f"LoRA-Squeeze step distribution: {self.schedule.step_distribution_text()}",
            f"LoRA-Squeeze step schedule: {self.schedule.step_schedule_text()}",
        ]

    def optimizer_preflight_status_messages(self, optimizer_state_adapter: Optional[Any]) -> List[str]:
        if (
            not self.enabled
            or optimizer_state_adapter is None
            or optimizer_state_adapter.warm_restart_reason is None
        ):
            return []
        return [
            "LoRA-Squeeze global optimizer policy will preserve learned global scale and warm-restart "
            f"coupled state: {optimizer_state_adapter.warm_restart_reason}"
        ]

    def optimizer_scheduler_status_messages(self, scheduler_training_steps: Optional[int]) -> List[str]:
        if not self.enabled:
            return []
        schedule = self.schedule
        messages = []
        if schedule.optimizer_mode == "global":
            messages.append("LoRA-Squeeze optimizer mode: global; optimizer-specific state transfer is enabled")
        else:
            messages.append("LoRA-Squeeze optimizer mode: per_squeeze; state will reset after each squeeze")
        if schedule.scheduler_mode == "per_squeeze":
            messages.append(
                "LoRA-Squeeze scheduler mode: per_squeeze; "
                f"segment {schedule.completed_squeezes + 1}/{schedule.total_segments}, "
                f"steps={scheduler_training_steps}"
            )
        else:
            messages.append("LoRA-Squeeze scheduler mode: global; continuing one global training curve")
        return messages

    def network_weights_load_error(self, error: RuntimeError) -> RuntimeError:
        schedule = self.schedule
        return RuntimeError(
            "Failed to load --network_weights while LoRA-Squeeze is enabled. "
            "A rank mismatch is a possible cause: the weight file must match "
            f"the current LoRA-Squeeze rank {schedule.current_dim}; alpha values must also match "
            f"the current LoRA-Squeeze alpha {schedule.current_alpha:.8g}. "
            f"Original error: {error}"
        )


class LoRASqueezeTrainingController:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        accelerator: Any,
        schedule: LoRASqueezeSchedule,
        network: torch.nn.Module,
        optimizer_util: Any,
        optimizer_name: str,
        optimizer_args: str,
        optimizer: torch.optim.Optimizer,
        optimizer_train_fn: Callable[[], None],
        optimizer_eval_fn: Callable[[], None],
        lr_scheduler: Any,
        lr_descriptions: Any,
        prepare_optimizer_params_fn: Callable[[], Tuple[Any, Any]],
        prepare_grad_fn: Callable[[], None],
        update_metadata_fn: Callable[[], None],
        clean_memory_fn: Callable[[], None],
        logger: Optional[Any] = None,
        optimizer_factory_args: Optional[argparse.Namespace] = None,
    ):
        self.args = args
        self.accelerator = accelerator
        self.schedule = schedule
        self.network = network
        self.optimizer_util = optimizer_util
        self.optimizer_name = optimizer_name
        self.optimizer_args = optimizer_args
        self.optimizer = optimizer
        self.optimizer_train_fn = optimizer_train_fn
        self.optimizer_eval_fn = optimizer_eval_fn
        self.lr_scheduler = lr_scheduler
        self.lr_descriptions = lr_descriptions
        self.prepare_optimizer_params_fn = prepare_optimizer_params_fn
        self.prepare_grad_fn = prepare_grad_fn
        self.update_metadata_fn = update_metadata_fn
        self.clean_memory_fn = clean_memory_fn
        self.logger = logger
        self.optimizer_factory_args = optimizer_factory_args if optimizer_factory_args is not None else args
        self.last_squeeze_stats: Optional[Dict[str, float]] = None
        self.replaceable_optimizer = find_replaceable_optimizer(optimizer)
        self.replaceable_scheduler = find_replaceable_scheduler(lr_scheduler)

    def export_training_state(self) -> Tuple[str, str, Any, Callable[[], None], Callable[[], None], Any, Any]:
        return (
            self.optimizer_name,
            self.optimizer_args,
            self.optimizer,
            self.optimizer_train_fn,
            self.optimizer_eval_fn,
            self.lr_scheduler,
            self.lr_descriptions,
        )

    def _terminal_state_save_requested(self) -> bool:
        return bool(getattr(self.args, "save_state", False) or getattr(self.args, "save_state_on_train_end", False))

    def _release_terminal_optimizer_parameters(self):
        raw_optimizer = (
            self.replaceable_optimizer.optimizer
            if self.replaceable_optimizer is not None
            else getattr(self.optimizer, "optimizer", self.optimizer)
        )
        if hasattr(raw_optimizer, "state"):
            raw_optimizer.state.clear()
        for group in getattr(raw_optimizer, "param_groups", []):
            group["params"] = []

    def rebuild_optimizer(
        self,
        optimizer_state_transfers: List[Tuple[torch.nn.Parameter, Dict[str, Any]]],
        squeeze_stats: Dict[str, float],
        segment_steps: Optional[int],
        expected_parameter_layout: Optional[Tuple[Tuple[str, ...], ...]],
        offload_optimizer_state: bool = True,
    ):
        terminal_squeeze = segment_steps is None
        old_optimizer = self.optimizer
        old_lr_scheduler = self.lr_scheduler
        raw_old_optimizer = (
            self.replaceable_optimizer.optimizer
            if self.replaceable_optimizer is not None
            else getattr(old_optimizer, "optimizer", old_optimizer)
        )
        global_optimizer_mode = self.schedule.optimizer_mode == "global"
        global_scheduler_mode = self.schedule.scheduler_mode == "global"
        old_scheduler_state = (
            old_lr_scheduler.state_dict()
            if global_scheduler_mode and hasattr(old_lr_scheduler, "state_dict")
            else None
        )
        old_group_lrs = [group["lr"] for group in raw_old_optimizer.param_groups] if global_scheduler_mode else None

        trainable_params, new_lr_descriptions = self.prepare_optimizer_params_fn()
        optimizer_factory_args = copy.copy(self.optimizer_factory_args)
        new_optimizer_name, new_optimizer_args, new_optimizer = self.optimizer_util.get_optimizer(
            optimizer_factory_args, trainable_params
        )
        if global_optimizer_mode:
            unwrapped_network = self.accelerator.unwrap_model(self.network)
            new_parameter_layout = get_lora_squeeze_optimizer_parameter_layout(unwrapped_network, new_optimizer)
            if new_parameter_layout != expected_parameter_layout:
                raise ValueError(
                    "LoRA-Squeeze global optimizer mode changed factor parameter-group membership or ordering: "
                    f"{expected_parameter_layout} -> {new_parameter_layout}"
                )

        projected_state_count = 0
        old_optimizer_offload_records = []
        try:
            if global_optimizer_mode:
                if offload_optimizer_state:
                    old_optimizer_offload_records = offload_optimizer_state_to_cpu(raw_old_optimizer)
                copy_optimizer_param_group_state(raw_old_optimizer, new_optimizer, optimizer_state_transfers)
                for parameter, state in optimizer_state_transfers:
                    if state:
                        new_optimizer.state[parameter] = state
                        projected_state_count += 1
                move_optimizer_state_to_parameter_devices(new_optimizer)

            if global_scheduler_mode:
                new_lr_scheduler = self.optimizer_util.get_scheduler_fix(
                    self.args,
                    new_optimizer,
                    self.accelerator.num_processes,
                )
            else:
                new_lr_scheduler = self.optimizer_util.get_scheduler_fix(
                    self.args,
                    new_optimizer,
                    self.accelerator.num_processes,
                    training_steps=1 if terminal_squeeze else segment_steps,
                )
            if global_scheduler_mode:
                if old_scheduler_state is not None and hasattr(new_lr_scheduler, "load_state_dict"):
                    new_lr_scheduler.load_state_dict(old_scheduler_state)
                if len(new_optimizer.param_groups) != len(old_group_lrs):
                    raise ValueError("LoRA-Squeeze global scheduler mode changed the optimizer parameter-group count")
                for group, lr in zip(new_optimizer.param_groups, old_group_lrs):
                    group["lr"] = lr

            new_optimizer_train_fn, new_optimizer_eval_fn = self.optimizer_util.get_optimizer_train_eval_fn(
                new_optimizer, self.args
            )
            new_optimizer_train_fn()
        except Exception:
            if hasattr(new_optimizer, "state"):
                for state in new_optimizer.state.values():
                    state.clear()
                new_optimizer.state.clear()
            restore_offloaded_optimizer_state(old_optimizer_offload_records)
            raise

        if self.replaceable_optimizer is not None:
            self.replaceable_optimizer.replace(new_optimizer)
            prepared_optimizer = old_optimizer
        else:
            prepared_optimizer = new_optimizer
        if self.replaceable_scheduler is not None:
            self.replaceable_scheduler.replace(new_lr_scheduler)
            prepared_lr_scheduler = old_lr_scheduler
        else:
            prepared_lr_scheduler = new_lr_scheduler

        if hasattr(raw_old_optimizer, "state"):
            raw_old_optimizer.state.clear()

        self.optimizer_name = new_optimizer_name
        self.optimizer_args = new_optimizer_args
        self.optimizer = prepared_optimizer
        self.lr_scheduler = prepared_lr_scheduler
        self.optimizer_train_fn, self.optimizer_eval_fn = self.optimizer_util.get_optimizer_train_eval_fn(
            prepared_optimizer, self.args
        )
        self.lr_descriptions = new_lr_descriptions

        target_rank = int(squeeze_stats.get("target_rank", self.schedule.current_dim))
        if global_optimizer_mode:
            reset_state_count = int(squeeze_stats.get("optimizer_state_reset", 0))
            empty_state_count = int(squeeze_stats.get("optimizer_state_empty", 0))
            warm_restart_count = int(squeeze_stats.get("optimizer_state_warm_restarted", 0))
            self.accelerator.print(
                "LoRA-Squeeze optimizer continued globally: "
                f"rank={target_rank}, projected_states={projected_state_count}, "
                f"warm_restarted_states={warm_restart_count}, reset_states={reset_state_count}, "
                f"empty_states={empty_state_count}"
            )
            if warm_restart_count > 0:
                self.accelerator.print(
                    "LoRA-Squeeze optimizer used a coherent warm restart for unprojectable state; "
                    "the optimizer's learned global scale was preserved"
                )
            if reset_state_count > 0:
                self.accelerator.print(
                    "LoRA-Squeeze warning: global optimizer mode continued optimizer param-group state, "
                    "but reset unsupported per-parameter optimizer state for some squeezed parameters"
                )
        else:
            self.accelerator.print(f"LoRA-Squeeze optimizer reset for rank={target_rank}")

        if global_scheduler_mode:
            self.accelerator.print(f"LoRA-Squeeze scheduler continued globally for rank={target_rank}")
        elif terminal_squeeze:
            self.accelerator.print(f"LoRA-Squeeze scheduler refreshed after terminal squeeze: rank={target_rank}")
        else:
            self.accelerator.print(
                "LoRA-Squeeze scheduler restarted: "
                f"segment {self.schedule.completed_squeezes + 2}/{self.schedule.total_segments}, "
                f"rank={target_rank}, steps={segment_steps}"
            )

    def run_if_due(self, lora_squeeze_step: int) -> bool:
        self.last_squeeze_stats = None
        event = self.schedule.next_due(lora_squeeze_step)
        if event is None:
            return False

        target_dim = int(event["dim"])
        target_alpha = float(event["alpha"])
        unwrapped_network = self.accelerator.unwrap_model(self.network)
        self.accelerator.wait_for_everyone()
        raw_optimizer = (
            self.replaceable_optimizer.optimizer
            if self.replaceable_optimizer is not None
            else getattr(self.optimizer, "optimizer", self.optimizer)
        )
        next_segment_steps = self.schedule.next_segment_steps_after_squeeze
        terminal_squeeze = next_segment_steps is None
        skip_terminal_rebuild = terminal_squeeze and not self._terminal_state_save_requested()
        optimizer_for_state_transfer = (
            raw_optimizer if self.schedule.optimizer_mode == "global" and not skip_terminal_rebuild else None
        )
        optimizer_parameter_layout = (
            get_lora_squeeze_optimizer_parameter_layout(unwrapped_network, raw_optimizer)
            if optimizer_for_state_transfer is not None
            else None
        )
        terminal_optimizer_in_eval_mode = False
        if terminal_squeeze:
            self.optimizer_eval_fn()
            terminal_optimizer_in_eval_mode = True
        layer_snapshots = snapshot_lora_module_layers(unwrapped_network)

        def squeeze_and_rebuild(optimizer_state_staging_device, offload_optimizer_state):
            stats, optimizer_state_transfers = squeeze_lora_network(
                unwrapped_network,
                target_dim,
                target_alpha,
                optimizer_for_state_transfer=optimizer_for_state_transfer,
                optimizer_state_staging_device=optimizer_state_staging_device,
            )
            if not skip_terminal_rebuild:
                self.prepare_grad_fn()
                self.rebuild_optimizer(
                    optimizer_state_transfers,
                    stats,
                    next_segment_steps,
                    optimizer_parameter_layout,
                    offload_optimizer_state=offload_optimizer_state,
                )
            return stats

        try:
            cpu_staging_error = None
            try:
                stats = squeeze_and_rebuild(
                    "cpu" if optimizer_for_state_transfer is not None else None,
                    True,
                )
            except OptimizerStateCPUStagingError as error:
                cpu_staging_error = str(error)

            if cpu_staging_error is not None:
                restore_lora_module_layers(layer_snapshots)
                self.prepare_grad_fn()
                self.clean_memory_fn()
                self.accelerator.print(
                    "LoRA-Squeeze CPU optimizer-state staging failed; retrying on the parameter device "
                    f"with higher peak VRAM usage. Error: {cpu_staging_error}"
                )
                stats = squeeze_and_rebuild(None, False)
        except Exception:
            restore_lora_module_layers(layer_snapshots)
            try:
                if terminal_optimizer_in_eval_mode:
                    self.optimizer_train_fn()
                self.prepare_grad_fn()
            except Exception as restore_error:
                if self.logger is not None:
                    self.logger.warning(f"failed to refresh LoRA gradients after rollback: {restore_error}")
            raise

        layer_snapshots.clear()
        self.schedule.mark_squeezed(target_dim, target_alpha)
        self.last_squeeze_stats = dict(stats)
        self.update_metadata_fn()
        self.accelerator.print(
            f"LoRA-Squeeze at step {lora_squeeze_step}: rank {int(stats['source_rank'])} -> {target_dim}, "
            f"alpha={target_alpha:.8g}, retained_energy_min={stats['retained_energy_min']:.6f}, "
            f"retained_energy_mean={stats['retained_energy_mean']:.6f}, "
            f"numerical_rank_min={int(stats['numerical_rank_min'])}"
        )
        revived_channels = int(stats.get("revived_rank_channels", 0))
        if revived_channels > 0:
            self.accelerator.print(
                "LoRA-Squeeze warning: revived "
                f"{revived_channels} numerically zero rank channels across "
                f"{int(stats['rank_deficient_modules'])} modules with zero-product, learnable initialization"
            )
        if skip_terminal_rebuild:
            self._release_terminal_optimizer_parameters()
            self.accelerator.print(
                "LoRA-Squeeze terminal compression complete; skipped optimizer and scheduler rebuild "
                "and compressed the optimizer evaluation point because no resumable state was requested"
            )
        elif terminal_squeeze:
            self.accelerator.print(
                "LoRA-Squeeze terminal compression complete; rebuilt resumable optimizer and scheduler state "
                "from the optimizer evaluation point"
            )
        self.clean_memory_fn()
        return True
