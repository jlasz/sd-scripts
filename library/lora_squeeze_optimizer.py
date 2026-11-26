"""Optimizer state transfer for in-training LoRA rank changes.

Global optimizer mode is limited to Adam-family optimizers (PyTorch Adam, AdamW,
RAdam, NAdam and Adamax, plus the bitsandbytes Adam/AdamW variants). Their
per-parameter state is gradient moments, which are projected as covectors.
Every other optimizer must use --lora_squeeze_optimizer_mode=per_squeeze.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


class OptimizerStateCPUStagingError(RuntimeError):
    """Raised when optimizer state cannot be staged on CPU."""


GRADIENT = "gradient"
GRADIENT_SQUARED = "gradient_squared"
GRADIENT_ABS_MAX = "gradient_abs_max"


def _move_optimizer_state_value(value: Any, device: torch.device, move_scalar_tensors: bool = True) -> Any:
    if isinstance(value, torch.Tensor):
        # bitsandbytes paged optimizer buffers intentionally report a CPU
        # device while being backed by CUDA managed memory. Moving them with
        # Tensor.to() would replace them with ordinary CUDA allocations and
        # silently disable paging after an optimizer rebuild.
        if getattr(value, "is_paged", False):
            return value
        if not move_scalar_tensors and value.numel() == 1:
            return value
        return value.to(device=device)
    if isinstance(value, dict):
        return {
            key: _move_optimizer_state_value(item, device, move_scalar_tensors)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_move_optimizer_state_value(item, device, move_scalar_tensors) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_optimizer_state_value(item, device, move_scalar_tensors) for item in value)
    return value


def stage_optimizer_state(state: Dict[str, Any], device: Optional[Union[str, torch.device]]) -> Dict[str, Any]:
    if device is None or not state:
        return state
    target_device = torch.device(device)
    try:
        return {
            key: _move_optimizer_state_value(value, target_device, move_scalar_tensors=False)
            for key, value in state.items()
        }
    except Exception as error:
        if target_device.type == "cpu":
            raise OptimizerStateCPUStagingError("Failed to stage projected optimizer state on CPU") from error
        raise


def _move_optimizer_state_value_with_device_record(value: Any, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device), value.device
    if isinstance(value, dict):
        moved, devices = {}, {}
        for key, item in value.items():
            moved[key], devices[key] = _move_optimizer_state_value_with_device_record(item, device)
        return moved, devices
    if isinstance(value, list):
        moved_items, device_items = [], []
        for item in value:
            moved, original_device = _move_optimizer_state_value_with_device_record(item, device)
            moved_items.append(moved)
            device_items.append(original_device)
        return moved_items, device_items
    if isinstance(value, tuple):
        moved_items, device_items = [], []
        for item in value:
            moved, original_device = _move_optimizer_state_value_with_device_record(item, device)
            moved_items.append(moved)
            device_items.append(original_device)
        return tuple(moved_items), tuple(device_items)
    return value, None


def _restore_optimizer_state_value_devices(value: Any, devices: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=devices) if devices is not None else value
    if isinstance(value, dict):
        return {key: _restore_optimizer_state_value_devices(item, devices[key]) for key, item in value.items()}
    if isinstance(value, list):
        return [
            _restore_optimizer_state_value_devices(item, item_devices)
            for item, item_devices in zip(value, devices)
        ]
    if isinstance(value, tuple):
        return tuple(
            _restore_optimizer_state_value_devices(item, item_devices)
            for item, item_devices in zip(value, devices)
        )
    return value


def offload_optimizer_state_to_cpu(optimizer: torch.optim.Optimizer):
    records = []
    try:
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                moved, original_devices = _move_optimizer_state_value_with_device_record(value, torch.device("cpu"))
                state[key] = moved
                records.append((state, key, original_devices))
    except Exception as error:
        try:
            restore_offloaded_optimizer_state(records)
        except Exception as restore_error:
            raise RuntimeError("CPU optimizer-state offload failed and could not be rolled back") from restore_error
        raise OptimizerStateCPUStagingError("Failed to offload existing optimizer state to CPU") from error
    return records


def restore_offloaded_optimizer_state(records):
    for state, key, original_devices in records:
        if key in state:
            state[key] = _restore_optimizer_state_value_devices(state[key], original_devices)


def move_optimizer_state_to_parameter_devices(optimizer: torch.optim.Optimizer):
    for parameter, state in optimizer.state.items():
        for key, value in list(state.items()):
            state[key] = _move_optimizer_state_value(value, parameter.device, move_scalar_tensors=False)


def _qualified_name(optimizer: torch.optim.Optimizer) -> str:
    optimizer_type = type(optimizer)
    return f"{optimizer_type.__module__}.{optimizer_type.__name__}"


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


def _state_tensor_to_parameter_shape(value: torch.Tensor, parameter: torch.nn.Parameter) -> Tuple[torch.Tensor, bool]:
    if tuple(value.shape) == tuple(parameter.shape):
        return value, False
    if value.ndim == 1 and value.numel() == parameter.numel():
        return value.reshape(parameter.shape), True
    raise ValueError(
        f"optimizer state shape {tuple(value.shape)} does not match parameter shape {tuple(parameter.shape)}"
    )


def _parameter_tensor_to_2d(tensor: torch.Tensor, role: str) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim >= 3:
        return tensor.reshape(tensor.shape[0], -1)
    if tensor.ndim == 1:
        return tensor.reshape(1, -1) if role == "down" else tensor.reshape(-1, 1)
    raise ValueError(f"unsupported optimizer state tensor rank: {tensor.ndim}")


def _restore_projected_layout(projected: torch.Tensor, target_shape: torch.Size, was_flat: bool) -> torch.Tensor:
    projected = projected.reshape(target_shape)
    return projected.flatten() if was_flat else projected


def _project_2d(
    tensor_2d: torch.Tensor,
    projection: torch.Tensor,
    role: str,
) -> torch.Tensor:
    if role == "up":
        if tensor_2d.shape[1] != projection.shape[0]:
            raise ValueError("up-state rank does not match the LoRA-Squeeze projection")
        return tensor_2d @ projection
    if role == "down":
        if tensor_2d.shape[0] != projection.shape[1]:
            raise ValueError("down-state rank does not match the LoRA-Squeeze projection")
        return projection @ tensor_2d
    raise ValueError(f"unknown LoRA optimizer-state role: {role}")


def _project_full_state_tensor(
    value: torch.Tensor,
    old_parameter: torch.nn.Parameter,
    new_parameter: torch.nn.Parameter,
    projection: torch.Tensor,
    role: str,
    kind: str,
) -> torch.Tensor:
    shaped, was_flat = _state_tensor_to_parameter_shape(value, old_parameter)
    state_2d = _parameter_tensor_to_2d(shaped.detach().float(), role)
    projection = projection.detach().to(device=state_2d.device, dtype=state_2d.dtype)

    if kind == GRADIENT_SQUARED:
        projection = projection.square()

    if kind == GRADIENT_ABS_MAX:
        # Adamax stores a component-wise maximum absolute gradient. Coordinate
        # mixing has no exact diagonal representation, so retain a conservative
        # envelope using the absolute projection coefficients.
        projected = _project_2d(state_2d.abs(), projection.abs(), role).clamp_min_(0)
    else:
        projected = _project_2d(state_2d, projection, role)
        if kind == GRADIENT_SQUARED:
            projected.clamp_min_(0)

    if projected.numel() != new_parameter.numel() or not torch.isfinite(projected).all():
        raise ValueError("optimizer state projection produced an invalid target tensor")
    projected = _restore_projected_layout(projected, new_parameter.shape, was_flat)
    return projected.to(device=new_parameter.device, dtype=value.dtype)


class OptimizerStateTransferAdapter:
    display_name = "optimizer"
    scalar_keys = {"step"}
    state_rules: Dict[str, str] = {}

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.qualified_name = _qualified_name(optimizer)

    def _unexpected_state_keys(self) -> List[str]:
        accepted = set(self.scalar_keys) | set(self.state_rules)
        return sorted({key for state in self.optimizer.state.values() for key in state if key not in accepted})

    def preflight(self):
        unexpected = self._unexpected_state_keys()
        if unexpected:
            raise ValueError(
                f"LoRA-Squeeze global optimizer mode does not know how {self.qualified_name} uses state "
                f"{unexpected}. Use --lora_squeeze_optimizer_mode=per_squeeze."
            )

    def project_parameter_state(
        self,
        old_parameter: torch.nn.Parameter,
        new_parameter: torch.nn.Parameter,
        projection: torch.Tensor,
        role: str,
    ) -> Tuple[Dict[str, Any], str]:
        state = self.optimizer.state.get(old_parameter, {})
        if not state:
            return {}, "empty"

        projected: Dict[str, Any] = {}
        for key, value in state.items():
            if key in self.scalar_keys:
                if isinstance(value, torch.Tensor) and value.numel() != 1:
                    raise ValueError(f"{self.qualified_name} state {key!r} was expected to be scalar")
                projected[key] = _clone_value(value)
                continue
            if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
                raise ValueError(f"{self.qualified_name} state {key!r} is not a floating-point tensor")
            projected[key] = _project_full_state_tensor(
                value, old_parameter, new_parameter, projection, role, self.state_rules[key]
            )
        return projected, "projected"

    def copy_group_state(
        self,
        new_optimizer: torch.optim.Optimizer,
        transfers: List[Tuple[torch.nn.Parameter, Dict[str, Any]]],
    ):
        _validate_group_layout(self.optimizer, new_optimizer)


class RuleBasedAdapter(OptimizerStateTransferAdapter):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        display_name: str,
        state_rules: Dict[str, str],
        scalar_keys=None,
    ):
        super().__init__(optimizer)
        self.display_name = display_name
        self.state_rules = state_rules
        if scalar_keys is not None:
            self.scalar_keys = set(scalar_keys)


def _validate_group_layout(old_optimizer: torch.optim.Optimizer, new_optimizer: torch.optim.Optimizer):
    if _qualified_name(old_optimizer) != _qualified_name(new_optimizer):
        raise ValueError(
            "LoRA-Squeeze global optimizer mode rebuilt a different optimizer class: "
            f"{_qualified_name(old_optimizer)} -> {_qualified_name(new_optimizer)}"
        )
    if len(old_optimizer.param_groups) != len(new_optimizer.param_groups):
        raise ValueError("LoRA-Squeeze global optimizer mode changed the optimizer parameter-group count")


class BitsAndBytesAdapter(OptimizerStateTransferAdapter):
    scalar_keys = {"step", "unorm_vec", "gnorm_vec"}

    def __init__(self, optimizer, state1_kind, has_state2):
        super().__init__(optimizer)
        self.state1_kind = state1_kind
        self.has_state2 = has_state2

    def preflight(self):
        args = getattr(self.optimizer, "args", None)
        if (
            args is not None
            and getattr(args, "optim_bits", None) == 8
            and not getattr(args, "block_wise", True)
        ):
            raise ValueError(
                "LoRA-Squeeze global mode supports block-wise but not tensor-wise bitsandbytes 8-bit state"
            )
        accepted = self.scalar_keys | {"state1", "qmap1", "absmax1"}
        if self.has_state2:
            accepted |= {"state2", "qmap2", "absmax2"}
        unexpected = sorted({key for state in self.optimizer.state.values() for key in state if key not in accepted})
        if unexpected:
            raise ValueError(
                f"LoRA-Squeeze global mode supports only block-wise bitsandbytes state; found {unexpected}"
            )

    @staticmethod
    def _project_bnb_buffer(state, prefix, kind, old_parameter, new_parameter, projection, role):
        value = state[prefix]
        if value.dtype != torch.uint8:
            return _project_full_state_tensor(value, old_parameter, new_parameter, projection, role, kind), {}
        try:
            import bitsandbytes.functional as F
        except ImportError as error:
            raise ValueError("bitsandbytes is required to transfer quantized optimizer state") from error
        suffix = prefix[-1]
        qmap_key = f"qmap{suffix}"
        absmax_key = f"absmax{suffix}"
        # Paged bitsandbytes buffers use CUDA managed memory but present as
        # CPU tensors. bitsandbytes' blockwise CUDA kernel consequently tries
        # to obtain a CUDA stream for device index None when given the managed
        # tensor directly. Materialize only the moment currently being
        # projected as an ordinary tensor on the parameter device.
        dequantize_value = value
        if value.device != old_parameter.device or getattr(value, "is_paged", False):
            dequantize_value = torch.empty(
                value.shape,
                dtype=value.dtype,
                device=old_parameter.device,
            )
            dequantize_value.copy_(value)
        absmax = state[absmax_key].to(device=old_parameter.device)
        qmap = state[qmap_key].to(device=old_parameter.device)
        dequantized = F.dequantize_blockwise(
            dequantize_value,
            absmax=absmax,
            code=qmap,
            blocksize=256,
        ).reshape(old_parameter.shape)
        projected = _project_full_state_tensor(
            dequantized, old_parameter, new_parameter, projection, role, kind
        ).reshape(new_parameter.shape).float()
        quantized, quant_state = F.quantize_blockwise(
            projected, code=qmap, blocksize=256
        )
        return quantized, {qmap_key: quant_state.code, absmax_key: quant_state.absmax}

    def project_parameter_state(self, old_parameter, new_parameter, projection, role):
        state = self.optimizer.state.get(old_parameter, {})
        if not state:
            return {}, "empty"
        projected = {key: _clone_value(value) for key, value in state.items() if key in self.scalar_keys}
        state1, metadata = self._project_bnb_buffer(
            state, "state1", self.state1_kind, old_parameter, new_parameter, projection, role
        )
        projected["state1"] = state1
        projected.update(metadata)
        if self.has_state2:
            state2, metadata = self._project_bnb_buffer(
                state, "state2", GRADIENT_SQUARED, old_parameter, new_parameter, projection, role
            )
            projected["state2"] = state2
            projected.update(metadata)
        return projected, "projected"

    def copy_group_state(self, new_optimizer, transfers):
        _validate_group_layout(self.optimizer, new_optimizer)
        get_state_buffer = getattr(new_optimizer, "get_state_buffer", None)
        if get_state_buffer is None:
            return
        # Paged variants rely on their own managed-memory allocator. Re-home
        # projected moment buffers through it instead of silently turning a
        # paged optimizer into an ordinary CUDA-resident one after a squeeze.
        for parameter, state in transfers:
            for key in ("state1", "state2"):
                if key not in state:
                    continue
                source = state[key]
                target = get_state_buffer(parameter, dtype=source.dtype)
                target.copy_(source)
                state[key] = target


def _torch_adapter(optimizer, class_name) -> Optional[OptimizerStateTransferAdapter]:
    policies = {
        "Adam": {"exp_avg": GRADIENT, "exp_avg_sq": GRADIENT_SQUARED, "max_exp_avg_sq": GRADIENT_SQUARED},
        "AdamW": {"exp_avg": GRADIENT, "exp_avg_sq": GRADIENT_SQUARED, "max_exp_avg_sq": GRADIENT_SQUARED},
        "RAdam": {"exp_avg": GRADIENT, "exp_avg_sq": GRADIENT_SQUARED},
        "NAdam": {"exp_avg": GRADIENT, "exp_avg_sq": GRADIENT_SQUARED},
        "Adamax": {"exp_avg": GRADIENT, "exp_inf": GRADIENT_ABS_MAX},
    }
    scalar_keys = {
        "Adam": {"step"},
        "AdamW": {"step"},
        "RAdam": {"step"},
        "NAdam": {"step", "mu_product"},
        "Adamax": {"step"},
    }
    if class_name not in policies:
        return None
    return RuleBasedAdapter(optimizer, class_name, policies[class_name], scalar_keys[class_name])


def prepare_optimizer_state_transfer(optimizer: torch.optim.Optimizer) -> OptimizerStateTransferAdapter:
    module = type(optimizer).__module__
    class_name = type(optimizer).__name__
    adapter: Optional[OptimizerStateTransferAdapter] = None

    if module.startswith("torch.optim"):
        adapter = _torch_adapter(optimizer, class_name)
    elif module.startswith("bitsandbytes") and "Adam" in class_name and "AdEMAMix" not in class_name:
        adapter = BitsAndBytesAdapter(optimizer, GRADIENT, has_state2=True)

    if adapter is None:
        raise ValueError(
            "LoRA-Squeeze global optimizer mode supports only Adam-family optimizers "
            "(Adam, AdamW, RAdam, NAdam, Adamax and the bitsandbytes Adam/AdamW variants), not "
            f"{_qualified_name(optimizer)}. Use --lora_squeeze_optimizer_mode=per_squeeze."
        )
    adapter.preflight()
    return adapter


def optimizer_owns_lr_schedule(optimizer: torch.optim.Optimizer) -> bool:
    """Return whether LR/averaging progress is inseparable from optimizer state."""
    optimizer = getattr(optimizer, "optimizer", optimizer)
    module = type(optimizer).__module__
    class_name = type(optimizer).__name__
    if module.startswith("schedulefree") or (
        module.startswith("prodigyplus") and class_name == "ProdigyPlusScheduleFree"
    ):
        return True
    if module.startswith("transformers") and class_name == "Adafactor":
        return any(bool(group.get("relative_step", False)) for group in optimizer.param_groups)
    return False


def validate_optimizer_scheduler_modes(
    optimizer: torch.optim.Optimizer,
    optimizer_mode: str,
    scheduler_mode: str,
):
    if optimizer_owns_lr_schedule(optimizer) and optimizer_mode != scheduler_mode:
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        raise ValueError(
            "LoRA-Squeeze cannot independently restart optimizer and scheduler state for "
            f"{_qualified_name(raw_optimizer)}. Its learning-rate/averaging schedule is owned by the optimizer; "
            "set --lora_squeeze_optimizer_mode and --lora_squeeze_scheduler_mode to the same value."
        )


def copy_optimizer_param_group_state(
    old_optimizer: torch.optim.Optimizer,
    new_optimizer: torch.optim.Optimizer,
    transfers: Optional[List[Tuple[torch.nn.Parameter, Dict[str, Any]]]] = None,
):
    adapter = prepare_optimizer_state_transfer(old_optimizer)
    adapter.copy_group_state(new_optimizer, transfers or [])
