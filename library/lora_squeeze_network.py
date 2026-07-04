"""Network-facing LoRA-Squeeze protocols and standard factor implementation."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Protocol

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LoRASqueezeModuleSpec:
    """The factor data exposed by a network-owned squeeze module."""

    name: str
    rank: int
    alpha: float
    scale: float
    kind: str
    up_2d: torch.Tensor
    down_2d: torch.Tensor
    up_parameter: torch.nn.Parameter
    down_parameter: torch.nn.Parameter


class LoRASqueezeModuleProtocol(Protocol):
    def lora_squeeze_get_spec(self) -> LoRASqueezeModuleSpec: ...

    def lora_squeeze_replace_factors(
        self,
        up_2d: torch.Tensor,
        down_2d: torch.Tensor,
        target_dim: int,
        target_alpha: float,
    ) -> None: ...

    def lora_squeeze_snapshot(self) -> Any: ...

    def lora_squeeze_restore(self, snapshot: Any) -> None: ...


class LoRASqueezeNetworkProtocol(Protocol):
    def get_lora_squeeze_modules(self) -> Iterable[LoRASqueezeModuleProtocol]: ...


def validate_lora_squeeze_network_module(network_module: Any, network_args: Dict[str, Any]):
    """Require an explicit, early compatibility contract from a network module."""
    validator = getattr(network_module, "validate_lora_squeeze_support", None)
    if not callable(validator):
        module_name = getattr(network_module, "__name__", network_module.__class__.__name__)
        raise ValueError(
            f"LoRA-Squeeze is not supported by {module_name}: the network module does not implement "
            "validate_lora_squeeze_support(network_args)"
        )
    validator(dict(network_args))


def _factor_layer_has_hooks(layer: nn.Module) -> bool:
    hook_attributes = (
        "_forward_pre_hooks",
        "_forward_hooks",
        "_backward_pre_hooks",
        "_backward_hooks",
    )
    return any(bool(getattr(layer, attribute, None)) for attribute in hook_attributes)


def _validate_standard_factor_layer(layer: nn.Module, name: str, role: str):
    if type(layer) not in (nn.Linear, nn.Conv2d):
        raise ValueError(
            f"{name} has unsupported {role} type {type(layer).__name__}; "
            "custom factor layers must implement their own LoRA-Squeeze protocol methods"
        )
    if layer.bias is not None:
        raise ValueError(f"{name} has a biased {role}; standard LoRA-Squeeze factors must be bias-free")
    if torch.nn.utils.parametrize.is_parametrized(layer):
        raise ValueError(f"{name} has a parametrized {role}; replacement would discard its parametrization")
    if _factor_layer_has_hooks(layer):
        raise ValueError(f"{name} has hooks attached to {role}; replacement would discard those hooks")
    if not layer.weight.requires_grad:
        raise ValueError(f"{name} has a frozen {role} factor; LoRA-Squeeze requires trainable factors")


def _get_device_rng_state(device: torch.device) -> Optional[torch.Tensor]:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_rng_state(device)
    if device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.get_rng_state(device)
    return None


def _set_device_rng_state(device: torch.device, state: Optional[torch.Tensor]):
    if state is None:
        return
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.set_rng_state(state, device)


class StandardLoRASqueezeModuleMixin:
    """Protocol implementation for sd-scripts' ordinary Linear/Conv2d LoRA."""

    @staticmethod
    def _stored_alpha_matches_runtime_scale(alpha: Any, effective_alpha: float) -> bool:
        if isinstance(alpha, torch.Tensor):
            if alpha.numel() != 1:
                return False
            stored_alpha = float(alpha.detach().float().item())
            if not math.isfinite(stored_alpha):
                return False
            if torch.is_floating_point(alpha):
                serialized_dtypes = (alpha.dtype, torch.float16, torch.bfloat16)
                for dtype in serialized_dtypes:
                    expected = torch.tensor(effective_alpha, dtype=dtype)
                    expected_alpha = float(expected.detach().float().item())
                    if stored_alpha == expected_alpha:
                        return True
                return False
            return stored_alpha == effective_alpha
        return math.isclose(float(alpha), effective_alpha, rel_tol=1e-7, abs_tol=1e-8)

    def lora_squeeze_get_spec(self) -> LoRASqueezeModuleSpec:
        name = str(getattr(self, "lora_name", self.__class__.__name__))
        if not hasattr(self, "lora_down") or not hasattr(self, "lora_up"):
            raise ValueError(f"{name} must have both lora_down and lora_up")

        down = self.lora_down
        up = self.lora_up
        _validate_standard_factor_layer(down, name, "lora_down")
        _validate_standard_factor_layer(up, name, "lora_up")
        if isinstance(down, nn.Linear) != isinstance(up, nn.Linear):
            raise ValueError(f"{name} mixes Linear and Conv2d LoRA factors")
        if down.weight.device != up.weight.device or down.weight.dtype != up.weight.dtype:
            raise ValueError(f"{name} has LoRA factors on different devices or with different dtypes")

        if isinstance(down, nn.Linear):
            rank = int(down.out_features)
            if up.in_features != rank:
                raise ValueError(f"{name} lora_up input features do not match rank: {up.in_features} != {rank}")
            up_2d = up.weight.detach().float()
            down_2d = down.weight.detach().float()
            kind = "linear"
        else:
            if down.groups != 1 or up.groups != 1:
                raise ValueError(
                    f"{name} uses grouped LoRA convolutions, which require a custom squeeze implementation"
                )
            if tuple(up.kernel_size) != (1, 1):
                raise ValueError(f"{name} has unsupported lora_up kernel size: {tuple(up.kernel_size)}")
            if tuple(up.stride) != (1, 1) or tuple(up.padding) != (0, 0) or tuple(up.dilation) != (1, 1):
                raise ValueError(f"{name} has non-standard lora_up convolution semantics")
            rank = int(down.out_channels)
            if up.in_channels != rank:
                raise ValueError(f"{name} lora_up input channels do not match rank: {up.in_channels} != {rank}")
            up_2d = up.weight.detach().float()[:, :, 0, 0]
            down_2d = down.weight.detach().float().reshape(rank, -1)
            kind = "conv2d"

        if rank <= 0:
            raise ValueError(f"{name} has invalid LoRA rank: {rank}")
        stored_alpha = getattr(self, "alpha", rank)
        if isinstance(stored_alpha, torch.Tensor):
            if stored_alpha.numel() != 1:
                raise ValueError(f"{name} has a non-scalar LoRA alpha buffer")
            stored_alpha_value = float(stored_alpha.detach().float().item())
        else:
            stored_alpha_value = float(stored_alpha)
        scale = getattr(self, "scale", stored_alpha_value / rank)
        if isinstance(scale, torch.Tensor):
            scale = float(scale.detach().float().item())
        else:
            scale = float(scale)
        effective_alpha = scale * rank
        if not self._stored_alpha_matches_runtime_scale(stored_alpha, effective_alpha):
            raise ValueError(
                f"{name} has inconsistent LoRA alpha/scale values: scale={scale:.8g}, "
                f"serialized_alpha/rank={stored_alpha_value / rank:.8g}. "
                "Check --network_weights alpha compatibility."
            )

        return LoRASqueezeModuleSpec(
            name=name,
            rank=rank,
            alpha=effective_alpha,
            scale=scale,
            kind=kind,
            up_2d=up_2d,
            down_2d=down_2d,
            up_parameter=up.weight,
            down_parameter=down.weight,
        )

    def lora_squeeze_preserve_alpha_precision(self):
        if "alpha" not in getattr(self, "_buffers", {}):
            return
        alpha = self._buffers["alpha"]
        if alpha is None:
            return
        spec = self.lora_squeeze_get_spec()
        self._buffers["alpha"] = torch.tensor(spec.alpha, device=alpha.device, dtype=torch.float32)

    def lora_squeeze_snapshot(self) -> Dict[str, Any]:
        return {
            "lora_down": self.lora_down,
            "lora_up": self.lora_up,
            "lora_dim": getattr(self, "lora_dim", None),
            "scale": getattr(self, "scale", None),
            "has_alpha": hasattr(self, "alpha"),
            "alpha": getattr(self, "alpha", None),
            "alpha_is_buffer": "alpha" in getattr(self, "_buffers", {}),
        }

    def lora_squeeze_restore(self, snapshot: Dict[str, Any]) -> None:
        self.lora_down = snapshot["lora_down"]
        self.lora_up = snapshot["lora_up"]
        if snapshot["lora_dim"] is not None:
            self.lora_dim = snapshot["lora_dim"]
        if snapshot["scale"] is not None:
            self.scale = snapshot["scale"]
        if snapshot["alpha_is_buffer"]:
            self._buffers["alpha"] = snapshot["alpha"]
        elif snapshot["has_alpha"]:
            self.alpha = snapshot["alpha"]
        elif hasattr(self, "alpha"):
            delattr(self, "alpha")

    def lora_squeeze_replace_factors(
        self,
        up_2d: torch.Tensor,
        down_2d: torch.Tensor,
        target_dim: int,
        target_alpha: float,
    ) -> None:
        spec = self.lora_squeeze_get_spec()
        old_down = self.lora_down
        old_up = self.lora_up
        device = old_down.weight.device
        cpu_rng_state = torch.get_rng_state()
        device_rng_state = _get_device_rng_state(device)
        try:
            if spec.kind == "linear":
                new_down = nn.Linear(
                    old_down.in_features,
                    target_dim,
                    bias=False,
                    device=device,
                    dtype=old_down.weight.dtype,
                )
                new_up = nn.Linear(
                    target_dim,
                    old_up.out_features,
                    bias=False,
                    device=device,
                    dtype=old_up.weight.dtype,
                )
                down_weight = down_2d
                up_weight = up_2d
            else:
                new_down = nn.Conv2d(
                    old_down.in_channels,
                    target_dim,
                    old_down.kernel_size,
                    old_down.stride,
                    old_down.padding,
                    old_down.dilation,
                    groups=1,
                    bias=False,
                    padding_mode=old_down.padding_mode,
                    device=device,
                    dtype=old_down.weight.dtype,
                )
                new_up = nn.Conv2d(
                    target_dim,
                    old_up.out_channels,
                    old_up.kernel_size,
                    old_up.stride,
                    old_up.padding,
                    old_up.dilation,
                    groups=1,
                    bias=False,
                    padding_mode=old_up.padding_mode,
                    device=device,
                    dtype=old_up.weight.dtype,
                )
                down_weight = down_2d.reshape(target_dim, old_down.in_channels, *old_down.kernel_size)
                up_weight = up_2d[:, :, None, None]
        finally:
            torch.set_rng_state(cpu_rng_state)
            _set_device_rng_state(device, device_rng_state)

        new_down.train(old_down.training)
        new_up.train(old_up.training)
        new_down.weight.requires_grad_(old_down.weight.requires_grad)
        new_up.weight.requires_grad_(old_up.weight.requires_grad)
        with torch.no_grad():
            new_down.weight.copy_(down_weight.to(device=device, dtype=old_down.weight.dtype))
            new_up.weight.copy_(up_weight.to(device=device, dtype=old_up.weight.dtype))

        self.lora_down = new_down
        self.lora_up = new_up
        self.lora_dim = target_dim
        self.scale = target_alpha / target_dim
        old_alpha = getattr(self, "alpha", None)
        alpha_tensor = torch.tensor(target_alpha, device=device, dtype=torch.float32)
        if isinstance(old_alpha, torch.Tensor):
            self.alpha = alpha_tensor
        elif hasattr(self, "alpha"):
            self.alpha = float(target_alpha)
        else:
            self.register_buffer("alpha", alpha_tensor)
