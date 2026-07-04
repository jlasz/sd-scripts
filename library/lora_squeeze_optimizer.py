"""Optimizer-aware state transfer for in-training LoRA rank changes.

Optimizer tensors do not all live in the same coordinate space. Gradient moments
are covectors, update/momentum buffers are vectors, and anchors/averages are
points in parameter space.  LoRA-Squeeze therefore uses explicit optimizer
policies instead of guessing from state-key names.
"""

import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch


class OptimizerStateCPUStagingError(RuntimeError):
    """Raised when optimizer state cannot be staged on CPU."""


GRADIENT = "gradient"
GRADIENT_SQUARED = "gradient_squared"
GRADIENT_ABS_MAX = "gradient_abs_max"
VECTOR = "vector"
VECTOR_SQUARED = "vector_squared"
POSITION = "position"
MAX_VECTOR_PROJECTION_ABS = 1e6


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


def _vector_projection(projection: torch.Tensor) -> torch.Tensor:
    # Gradients are covectors. Parameter displacements use the pseudoinverse of
    # the transpose so that <gradient, displacement> is retained in the kept
    # subspace as closely as possible.
    result = torch.linalg.pinv(projection.T)
    if not torch.isfinite(result).all():
        raise ValueError("optimizer vector-state projection produced non-finite values")
    if result.numel() > 0 and float(result.detach().abs().max()) > MAX_VECTOR_PROJECTION_ABS:
        raise ValueError(
            "optimizer vector-state projection is ill-conditioned; use "
            "--lora_squeeze_optimizer_mode=per_squeeze"
        )
    return result


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

    if kind in (VECTOR, VECTOR_SQUARED):
        projection = _vector_projection(projection)
    if kind in (GRADIENT_SQUARED, VECTOR_SQUARED):
        projection = projection.square()

    if kind == GRADIENT_ABS_MAX:
        # Adamax stores a component-wise maximum absolute gradient. Coordinate
        # mixing has no exact diagonal representation, so retain a conservative
        # envelope using the absolute projection coefficients.
        projected = _project_2d(state_2d.abs(), projection.abs(), role).clamp_min_(0)
    else:
        projected = _project_2d(state_2d, projection, role)
        if kind in (GRADIENT_SQUARED, VECTOR_SQUARED):
            projected.clamp_min_(0)

    if projected.numel() != new_parameter.numel() or not torch.isfinite(projected).all():
        raise ValueError("optimizer state projection produced an invalid target tensor")
    projected = _restore_projected_layout(projected, new_parameter.shape, was_flat)
    return projected.to(device=new_parameter.device, dtype=value.dtype)


def _project_position_state(
    value: torch.Tensor,
    old_parameter: torch.nn.Parameter,
    new_parameter: torch.nn.Parameter,
    projection: torch.Tensor,
    role: str,
) -> torch.Tensor:
    was_flat = value.ndim == 1 and value.numel() == old_parameter.numel()
    if value.numel() == 1 and value.item() == 0:
        # Prodigy stores an all-zero anchor as a scalar to save memory.
        shaped = torch.zeros_like(old_parameter)
        was_flat = True
    else:
        shaped, was_flat = _state_tensor_to_parameter_shape(value, old_parameter)

    displacement = shaped.detach().float() - old_parameter.detach().float()
    mapped = _project_full_state_tensor(
        displacement,
        old_parameter,
        new_parameter,
        projection,
        role,
        VECTOR,
    )
    mapped_shaped = mapped.reshape(new_parameter.shape).float()
    position = new_parameter.detach().float() + mapped_shaped
    position = position.flatten() if was_flat else position
    return position.to(device=new_parameter.device, dtype=value.dtype)


class OptimizerStateTransferAdapter:
    display_name = "optimizer"
    scalar_keys = {"step"}
    state_rules: Dict[str, str] = {}
    warm_restart_reason: Optional[str] = None

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
        if self.warm_restart_reason is not None:
            return {}, "warm_restart:" + self.warm_restart_reason

        projected: Dict[str, Any] = {}
        for key, value in state.items():
            if key in self.scalar_keys:
                if isinstance(value, torch.Tensor) and value.numel() != 1:
                    raise ValueError(f"{self.qualified_name} state {key!r} was expected to be scalar")
                projected[key] = _clone_value(value)
                continue
            if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
                raise ValueError(f"{self.qualified_name} state {key!r} is not a floating-point tensor")
            kind = self.state_rules[key]
            if kind == POSITION:
                projected[key] = _project_position_state(
                    value, old_parameter, new_parameter, projection, role
                )
            else:
                projected[key] = _project_full_state_tensor(
                    value, old_parameter, new_parameter, projection, role, kind
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


def _copy_group_keys(old_optimizer, new_optimizer, keys: Iterable[str]):
    _validate_group_layout(old_optimizer, new_optimizer)
    for old_group, new_group in zip(old_optimizer.param_groups, new_optimizer.param_groups):
        for key in keys:
            if key in old_group:
                new_group[key] = _clone_value(old_group[key])


def _all_states(optimizer: torch.optim.Optimizer) -> Iterable[Dict[str, Any]]:
    return optimizer.state.values()


def _state_abs_sum(states: Iterable[Dict[str, Any]], key: str) -> float:
    return sum(float(state[key].detach().float().abs().sum()) for state in states if key in state)


def _state_l2_norm(states: Iterable[Dict[str, Any]], key: str) -> float:
    squared = sum(float(state[key].detach().float().square().sum()) for state in states if key in state)
    return math.sqrt(max(0.0, squared))


def _transferred_states(transfers) -> Iterable[Dict[str, Any]]:
    return (state for _, state in transfers)


def _scaled_history(value: Any, old_norm: float, new_norm: float) -> Any:
    if old_norm <= 0 or new_norm <= 0:
        return type(value)(0) if not isinstance(value, torch.Tensor) else torch.zeros_like(value)
    ratio = new_norm / old_norm
    return value * ratio


class ProdigyAdapter(RuleBasedAdapter):
    def __init__(self, optimizer):
        super().__init__(
            optimizer,
            "Prodigy",
            {"s": GRADIENT, "p0": POSITION, "exp_avg": GRADIENT, "exp_avg_sq": GRADIENT_SQUARED},
        )

    def preflight(self):
        unexpected = self._unexpected_state_keys()
        slice_values = {group.get("slice_p", 1) for group in self.optimizer.param_groups}
        if slice_values != {1}:
            self.warm_restart_reason = "Prodigy slice_p state cannot be unsliced after a rank change"
        elif unexpected:
            self.warm_restart_reason = "unsupported Prodigy state: " + ",".join(unexpected)

    def copy_group_state(self, new_optimizer, transfers):
        _validate_group_layout(self.optimizer, new_optimizer)
        if self.warm_restart_reason is not None:
            _copy_group_keys(self.optimizer, new_optimizer, ("d", "d_max"))
            return

        old_norm = _state_abs_sum(_all_states(self.optimizer), "s")
        new_norm = _state_abs_sum(_transferred_states(transfers), "s")
        for old_group, new_group in zip(self.optimizer.param_groups, new_optimizer.param_groups):
            for key in ("d", "d_max", "k"):
                if key in old_group:
                    new_group[key] = _clone_value(old_group[key])
            numerator = _scaled_history(old_group.get("d_numerator", 0.0), old_norm, new_norm)
            new_group["d_numerator"] = numerator
            new_group["d_denom"] = new_norm
            d_coef = new_group.get("d_coef", 1.0)
            new_group["d_hat"] = new_group.get("d", 0.0) if new_norm == 0 else d_coef * numerator / new_norm


class ProdigyPlusWarmRestartAdapter(OptimizerStateTransferAdapter):
    """Preserve the learned LR while restarting inseparable sliced/SF state.

    ProdigyPlusScheduleFree 1.9.x stores its Prodigy anchor/accumulator at a
    fixed 1-in-11 slice and combines them with schedule-free points and optional
    factored/experimental moments. Rank mixing makes that slice non-invertible.
    Restarting every per-parameter buffer together is safer than constructing a
    hybrid, while retaining ``d`` avoids relearning the main LR estimate.
    """

    def __init__(self, optimizer):
        super().__init__(optimizer)
        self.warm_restart_reason = "ProdigyPlus sliced and schedule-free state is not jointly invertible"

    def preflight(self):
        pass

    def copy_group_state(self, new_optimizer, transfers):
        _copy_group_keys(self.optimizer, new_optimizer, ("d", "d_prev", "shared_d"))


class DAdaptRatioAdapter(RuleBasedAdapter):
    def __init__(self, optimizer, display_name, rules, history_key, norm="l1"):
        super().__init__(optimizer, display_name, rules)
        self.history_key = history_key
        self.norm = norm

    def preflight(self):
        unexpected = self._unexpected_state_keys()
        if unexpected:
            self.warm_restart_reason = "unsupported D-Adaptation state: " + ",".join(unexpected)

    def copy_group_state(self, new_optimizer, transfers):
        _validate_group_layout(self.optimizer, new_optimizer)
        if self.warm_restart_reason is not None:
            _copy_group_keys(self.optimizer, new_optimizer, ("d",))
            return
        norm_fn = _state_l2_norm if self.norm == "l2" else _state_abs_sum
        old_norm = norm_fn(_all_states(self.optimizer), "s")
        new_norm = norm_fn(_transferred_states(transfers), "s")
        for old_group, new_group in zip(self.optimizer.param_groups, new_optimizer.param_groups):
            for key in ("d", "k"):
                if key in old_group:
                    new_group[key] = _clone_value(old_group[key])
            value = _scaled_history(old_group.get(self.history_key, 0.0), old_norm, new_norm)
            new_group[self.history_key] = value


def _adan_metrics(states: Iterable[Dict[str, Any]], eps: float) -> Tuple[float, float]:
    weighted_sq = 0.0
    l1 = 0.0
    for state in states:
        if "s" not in state or "exp_avg_sq" not in state:
            continue
        s = state["s"].detach().float()
        denom = state["exp_avg_sq"].detach().float().clamp_min(0).sqrt().add(eps)
        weighted_sq += float((s.square() / denom).sum())
        l1 += float(s.abs().sum())
    return weighted_sq, l1


class DAdaptAdanAdapter(RuleBasedAdapter):
    def __init__(self, optimizer, display_name):
        super().__init__(
            optimizer,
            display_name,
            {
                "s": GRADIENT,
                "exp_avg": GRADIENT,
                "exp_avg_diff": GRADIENT,
                "exp_avg_sq": GRADIENT_SQUARED,
                "pre_grad": GRADIENT,
            },
        )

    def preflight(self):
        unexpected = self._unexpected_state_keys()
        if unexpected:
            self.warm_restart_reason = "unsupported D-Adaptation Adan state: " + ",".join(unexpected)

    def copy_group_state(self, new_optimizer, transfers):
        _validate_group_layout(self.optimizer, new_optimizer)
        if self.warm_restart_reason is not None:
            _copy_group_keys(self.optimizer, new_optimizer, ("d",))
            return
        old_group = self.optimizer.param_groups[0]
        eps = float(old_group.get("eps", 1e-8))
        beta = float(old_group.get("betas", (0.98, 0.92, 0.99))[-1])
        old_sq, old_l1 = _adan_metrics(_all_states(self.optimizer), eps)
        new_sq, new_l1 = _adan_metrics(_transferred_states(transfers), eps)
        old_history = float(old_group.get("gsq_weighted", 0.0))
        old_d_hat = 0.0 if old_l1 == 0 else (old_sq / (1 - beta) - old_history) / old_l1
        new_history = max(0.0, new_sq / (1 - beta) - old_d_hat * new_l1)
        for source, target in zip(self.optimizer.param_groups, new_optimizer.param_groups):
            target["d"] = _clone_value(source.get("d", target.get("d")))
            target["k"] = _clone_value(source.get("k", target.get("k", 0)))
            target["gsq_weighted"] = new_history


class DAdaptSGDAdapter(RuleBasedAdapter):
    def __init__(self, optimizer):
        super().__init__(optimizer, "DAdaptSGD", {"s": VECTOR, "x0": POSITION, "z": POSITION})

    def project_parameter_state(self, old_parameter, new_parameter, projection, role):
        projected, status = super().project_parameter_state(old_parameter, new_parameter, projection, role)
        if projected and "x0" in projected and "s" in projected:
            projected["z"] = projected["x0"] - projected["s"]
        return projected, status

    def copy_group_state(self, new_optimizer, transfers):
        _validate_group_layout(self.optimizer, new_optimizer)
        old_norm = _state_l2_norm(_all_states(self.optimizer), "s")
        new_norm = _state_l2_norm(_transferred_states(transfers), "s")
        for old_group, new_group in zip(self.optimizer.param_groups, new_optimizer.param_groups):
            new_group["d"] = _clone_value(old_group.get("d", new_group.get("d")))
            new_group["numerator_weighted"] = _scaled_history(
                old_group.get("numerator_weighted", 0.0), old_norm, new_norm
            )
            # Re-measure the initial gradient norm in the new coordinates.
            new_group["k"] = 0
            new_group.pop("g0_norm", None)


class DAdaptAdaGradAdapter(RuleBasedAdapter):
    def __init__(self, optimizer):
        super().__init__(optimizer, "DAdaptAdaGrad", {"alphak": GRADIENT_SQUARED, "sk": GRADIENT, "x0": POSITION})

    @staticmethod
    def _metrics(states, eps):
        weighted_sq = 0.0
        l1 = 0.0
        for state in states:
            if "sk" not in state or "alphak" not in state:
                continue
            sk = state["sk"].detach().float()
            denom = state["alphak"].detach().float().clamp_min(0).sqrt().add(eps)
            weighted_sq += float((sk.square() / denom).sum())
            l1 += float(sk.abs().sum())
        return weighted_sq, l1

    def copy_group_state(self, new_optimizer, transfers):
        _validate_group_layout(self.optimizer, new_optimizer)
        group = self.optimizer.param_groups[0]
        eps = float(group.get("eps", 1e-6))
        old_sq, old_l1 = self._metrics(_all_states(self.optimizer), eps)
        new_sq, new_l1 = self._metrics(_transferred_states(transfers), eps)
        old_gsq = float(group.get("gsq_weighted", 0.0))
        old_d_hat = 0.0 if old_l1 == 0 else (old_sq - old_gsq) / old_l1
        new_gsq = max(0.0, new_sq - old_d_hat * new_l1)
        for source, target in zip(self.optimizer.param_groups, new_optimizer.param_groups):
            for key in ("d", "k"):
                target[key] = _clone_value(source.get(key, target.get(key)))
            target["sksq_weighted"] = new_sq
            target["skl1"] = new_l1
            target["gsq_weighted"] = new_gsq


class AdafactorAdapter(OptimizerStateTransferAdapter):
    scalar_keys = {"step", "RMS"}
    state_rules = {"exp_avg": VECTOR, "exp_avg_sq": GRADIENT_SQUARED}

    def preflight(self):
        accepted = self.scalar_keys | set(self.state_rules) | {"exp_avg_sq_row", "exp_avg_sq_col"}
        unexpected = sorted({key for state in self.optimizer.state.values() for key in state if key not in accepted})
        if unexpected:
            raise ValueError(f"unsupported Adafactor state for LoRA-Squeeze global mode: {unexpected}")

    def project_parameter_state(self, old_parameter, new_parameter, projection, role):
        state = self.optimizer.state.get(old_parameter, {})
        if not state:
            return {}, "empty"
        projected = {key: _clone_value(value) for key, value in state.items() if key in self.scalar_keys}
        if "exp_avg" in state:
            projected["exp_avg"] = _project_full_state_tensor(
                # Adafactor accumulates its already-preconditioned, LR-scaled
                # update here (not the raw gradient), so this is a parameter
                # vector rather than a gradient covector.
                state["exp_avg"], old_parameter, new_parameter, projection, role, VECTOR
            )
        if "exp_avg_sq" in state:
            projected["exp_avg_sq"] = _project_full_state_tensor(
                state["exp_avg_sq"], old_parameter, new_parameter, projection, role, GRADIENT_SQUARED
            )
        elif "exp_avg_sq_row" in state and "exp_avg_sq_col" in state:
            row = state["exp_avg_sq_row"].detach().float()
            col = state["exp_avg_sq_col"].detach().float()
            full = (row / row.mean(dim=-1, keepdim=True).clamp_min(1e-30)).unsqueeze(-1) * col.unsqueeze(-2)
            full = full.to(device=old_parameter.device)
            mapped = _project_full_state_tensor(
                full, old_parameter, new_parameter, projection, role, GRADIENT_SQUARED
            ).reshape(new_parameter.shape).float()
            projected["exp_avg_sq_row"] = mapped.mean(dim=-1).to(
                device=new_parameter.device, dtype=state["exp_avg_sq_row"].dtype
            )
            projected["exp_avg_sq_col"] = mapped.mean(dim=-2).to(
                device=new_parameter.device, dtype=state["exp_avg_sq_col"].dtype
            )
        return projected, "projected"


class ScheduleFreeAdapter(RuleBasedAdapter):
    def __init__(self, optimizer, rules):
        super().__init__(optimizer, type(optimizer).__name__, rules)

    def copy_group_state(self, new_optimizer, transfers):
        _copy_group_keys(
            self.optimizer,
            new_optimizer,
            ("k", "train_mode", "weight_sum", "lr_max", "scheduled_lr"),
        )


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
        "SparseAdam": {"exp_avg": GRADIENT, "exp_avg_sq": GRADIENT_SQUARED},
        "SGD": {"momentum_buffer": VECTOR},
        "Adagrad": {"sum": GRADIENT_SQUARED},
        "RMSprop": {"square_avg": GRADIENT_SQUARED, "grad_avg": GRADIENT, "momentum_buffer": VECTOR},
        "Adamax": {"exp_avg": GRADIENT, "exp_inf": GRADIENT_ABS_MAX},
        "Adadelta": {"square_avg": GRADIENT_SQUARED, "acc_delta": VECTOR_SQUARED},
        "ASGD": {"ax": POSITION},
    }
    scalar_keys = {
        "Adam": {"step"}, "AdamW": {"step"}, "RAdam": {"step"}, "NAdam": {"step", "mu_product"},
        "SparseAdam": {"step"}, "SGD": {"step"}, "Adagrad": {"step"}, "RMSprop": {"step"},
        "Adamax": {"step"}, "Adadelta": {"step"}, "ASGD": {"step", "eta", "mu"},
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
    elif module.startswith("lion_pytorch") and class_name == "Lion":
        adapter = RuleBasedAdapter(optimizer, "Lion", {"exp_avg": GRADIENT})
    elif module.startswith("prodigyopt") and class_name == "Prodigy":
        adapter = ProdigyAdapter(optimizer)
    elif module.startswith("prodigyplus") and class_name == "ProdigyPlusScheduleFree":
        adapter = ProdigyPlusWarmRestartAdapter(optimizer)
    elif module.startswith("dadaptation"):
        if class_name == "DAdaptAdam":
            adapter = DAdaptRatioAdapter(
                optimizer,
                class_name,
                {"s": GRADIENT, "exp_avg": GRADIENT, "exp_avg_sq": GRADIENT_SQUARED},
                "numerator_weighted",
            )
        elif class_name == "DAdaptLion":
            adapter = DAdaptRatioAdapter(
                optimizer, class_name, {"s": VECTOR, "exp_avg": GRADIENT}, "numerator_weighted"
            )
        elif class_name == "DAdaptAdanIP":
            adapter = DAdaptRatioAdapter(
                optimizer,
                class_name,
                # dadaptation 3.2's experimental IP implementation assigns the
                # squared-update accumulator to exp_avg_diff and the signed
                # gradient-difference accumulator to exp_avg_sq (despite their
                # names), so follow the implementation actually stepped.
                {
                    "s": GRADIENT,
                    "exp_avg": GRADIENT,
                    "exp_avg_diff": GRADIENT_SQUARED,
                    "exp_avg_sq": GRADIENT,
                    "pre_grad": GRADIENT,
                },
                "numerator_weighted",
            )
        elif class_name in ("DAdaptAdan", "DAdaptAdamPreprint"):
            rules_name = class_name
            adapter = DAdaptAdanAdapter(optimizer, rules_name)
            if class_name == "DAdaptAdamPreprint":
                adapter.state_rules = {"s": GRADIENT, "exp_avg": GRADIENT, "exp_avg_sq": GRADIENT_SQUARED}
        elif class_name == "DAdaptSGD":
            adapter = DAdaptSGDAdapter(optimizer)
        elif class_name == "DAdaptAdaGrad":
            adapter = DAdaptAdaGradAdapter(optimizer)
    elif module.startswith("transformers") and class_name == "Adafactor":
        adapter = AdafactorAdapter(optimizer)
    elif module.startswith("schedulefree"):
        if class_name in ("AdamWScheduleFree", "RAdamScheduleFree"):
            adapter = ScheduleFreeAdapter(
                optimizer, {"z": POSITION, "exp_avg_sq": GRADIENT_SQUARED, "exp_avg": GRADIENT}
            )
        elif class_name == "SGDScheduleFree":
            adapter = ScheduleFreeAdapter(optimizer, {"z": POSITION})
    elif module.startswith("bitsandbytes"):
        if "Adam" in class_name and "AdEMAMix" not in class_name:
            adapter = BitsAndBytesAdapter(optimizer, GRADIENT, has_state2=True)
        elif "Lion" in class_name:
            adapter = BitsAndBytesAdapter(optimizer, GRADIENT, has_state2=False)
        elif "SGD" in class_name:
            adapter = BitsAndBytesAdapter(optimizer, VECTOR, has_state2=False)

    if adapter is None:
        raise ValueError(
            "LoRA-Squeeze global optimizer mode is not implemented for "
            f"{_qualified_name(optimizer)}. Its state will not be guessed; use "
            "--lora_squeeze_optimizer_mode=per_squeeze."
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
