import math

import torch


def validate_rms_log_interval(interval: int) -> None:
    if interval < 0:
        raise ValueError("--total_rms_check_every_n_steps must be 0 or greater")


def _as_matrix(weight: torch.Tensor, role: str) -> torch.Tensor:
    weight = weight.detach().float()
    if weight.ndim == 2:
        return weight
    if weight.ndim >= 3:
        return weight.reshape(weight.shape[0], -1)
    if weight.ndim == 1:
        return weight.reshape(1, -1) if role == "down" else weight.reshape(-1, 1)
    raise ValueError(f"unsupported LoRA weight rank for RMS measurement: {weight.ndim}")


def _product_frobenius_norm(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape[1] != right.shape[0]:
        raise ValueError(f"LoRA factor shape mismatch: left={tuple(left.shape)}, right={tuple(right.shape)}")

    # ||AB||_F^2 = tr((A^T A)(B B^T)). This avoids materializing the
    # potentially very large effective adapter weight AB.
    left_gram = left.T @ left
    right_gram = right @ right.T
    squared_norm = torch.sum(left_gram * right_gram).clamp_min(0.0)
    return float(torch.sqrt(squared_norm).item())


def _module_scale(module, rank: int) -> float:
    scale = getattr(module, "scale", None)
    if scale is None:
        alpha = getattr(module, "alpha", rank)
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().reshape(-1)[0].item()
        scale = float(alpha) / rank if rank > 0 else 1.0
    elif isinstance(scale, torch.Tensor):
        scale = scale.detach().float().reshape(-1)[0].item()
    return float(scale)


@torch.no_grad()
def compute_total_scaled_lora_rms(network: torch.nn.Module) -> float:
    """Return the RMS of all effective, alpha-scaled LoRA weight deltas."""
    total_numel = 0
    total_squared_frobenius_norm = 0.0

    for module in network.modules():
        lora_down = getattr(module, "lora_down", None)
        lora_up = getattr(module, "lora_up", None)
        if not hasattr(lora_down, "weight") or not hasattr(lora_up, "weight"):
            continue

        down = _as_matrix(lora_down.weight, role="down")
        up = _as_matrix(lora_up.weight, role="up")
        if up.shape[1] != down.shape[0]:
            continue

        raw_frobenius_norm = _product_frobenius_norm(up, down)
        scaled_frobenius_norm = raw_frobenius_norm * _module_scale(module, int(down.shape[0]))
        total_numel += int(up.shape[0] * down.shape[1])
        total_squared_frobenius_norm += scaled_frobenius_norm**2

    if total_numel == 0:
        return 0.0
    return math.sqrt(total_squared_frobenius_norm / total_numel)
