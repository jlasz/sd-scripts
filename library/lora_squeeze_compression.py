"""LoRA-Squeeze validation, rank compression, and optimizer-state projection."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from library.lora_squeeze_schedule import LoRASqueezeSchedule
from library.lora_squeeze_optimizer import prepare_optimizer_state_transfer, stage_optimizer_state
from library.lora_squeeze_network import LoRASqueezeModuleProtocol, LoRASqueezeModuleSpec


def _require_protocol_method(owner: Any, method_name: str, owner_name: str):
    method = getattr(owner, method_name, None)
    if not callable(method):
        raise ValueError(f"{owner_name} does not implement required LoRA-Squeeze method {method_name}()")
    return method


def _validate_protocol_spec(spec: LoRASqueezeModuleSpec, registered_parameter_ids: set[int]):
    if not isinstance(spec, LoRASqueezeModuleSpec):
        raise ValueError("lora_squeeze_get_spec() must return LoRASqueezeModuleSpec")
    if (
        spec.rank <= 0
        or not math.isfinite(spec.alpha)
        or not math.isfinite(spec.scale)
        or spec.alpha <= 0
        or spec.scale <= 0
    ):
        raise ValueError(f"{spec.name} returned invalid rank, alpha, or scale")
    if not math.isclose(spec.scale, spec.alpha / spec.rank, rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError(f"{spec.name} returned inconsistent alpha/rank scaling")
    if spec.up_2d.ndim != 2 or spec.down_2d.ndim != 2:
        raise ValueError(f"{spec.name} must expose two-dimensional factor matrices")
    if spec.up_2d.shape[1] != spec.rank or spec.down_2d.shape[0] != spec.rank:
        raise ValueError(f"{spec.name} factor matrices do not match declared rank {spec.rank}")
    if spec.up_2d.device != spec.down_2d.device or spec.up_2d.dtype != spec.down_2d.dtype:
        raise ValueError(f"{spec.name} exposed factor matrices on different devices or with different dtypes")
    if not torch.isfinite(spec.up_2d).all() or not torch.isfinite(spec.down_2d).all():
        raise ValueError(f"{spec.name} exposed non-finite LoRA factors")
    if spec.up_parameter.numel() != spec.up_2d.numel() or spec.down_parameter.numel() != spec.down_2d.numel():
        raise ValueError(f"{spec.name} factor parameters do not match exposed matrices")
    if spec.up_parameter.ndim < 2 or spec.down_parameter.ndim < 2:
        raise ValueError(f"{spec.name} factor parameters must have matrix-compatible layouts")
    up_layout = (spec.up_parameter.shape[0], spec.up_parameter.numel() // spec.up_parameter.shape[0])
    down_layout = (spec.down_parameter.shape[0], spec.down_parameter.numel() // spec.down_parameter.shape[0])
    if up_layout != tuple(spec.up_2d.shape) or down_layout != tuple(spec.down_2d.shape):
        raise ValueError(f"{spec.name} factor parameter layouts do not match exposed matrices")
    if not spec.up_parameter.requires_grad or not spec.down_parameter.requires_grad:
        raise ValueError(f"{spec.name} exposed frozen factor parameters")
    if id(spec.up_parameter) not in registered_parameter_ids or id(spec.down_parameter) not in registered_parameter_ids:
        raise ValueError(f"{spec.name} exposed factor parameters that are not registered on the network")


def get_lora_squeeze_modules(network: torch.nn.Module) -> List[LoRASqueezeModuleProtocol]:
    provider = _require_protocol_method(network, "get_lora_squeeze_modules", network.__class__.__name__)
    modules = list(provider())
    if not modules:
        raise ValueError("LoRA-Squeeze network protocol returned no modules")
    if len({id(module) for module in modules}) != len(modules):
        raise ValueError("LoRA-Squeeze network protocol returned duplicate modules")
    registered_parameter_ids = {id(parameter) for parameter in network.parameters()}
    for module in modules:
        if not isinstance(module, nn.Module):
            raise ValueError("LoRA-Squeeze network protocol must return torch.nn.Module instances")
        name = str(getattr(module, "lora_name", module.__class__.__name__))
        _require_protocol_method(module, "lora_squeeze_get_spec", name)
        _require_protocol_method(module, "lora_squeeze_replace_factors", name)
        _require_protocol_method(module, "lora_squeeze_snapshot", name)
        _require_protocol_method(module, "lora_squeeze_restore", name)
        _validate_protocol_spec(module.lora_squeeze_get_spec(), registered_parameter_ids)
    return modules


def preserve_lora_squeeze_alpha_precision(network: torch.nn.Module) -> int:
    preserved = 0
    for module in get_lora_squeeze_modules(network):
        preserve = getattr(module, "lora_squeeze_preserve_alpha_precision", None)
        if callable(preserve):
            preserve()
            preserved += 1
    return preserved


def validate_lora_squeeze_optimizer_parameters(
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    modules = get_lora_squeeze_modules(network)
    factor_parameters: Dict[int, str] = {}
    for module_index, module in enumerate(modules):
        spec = module.lora_squeeze_get_spec()
        named_factors = (
            (spec.up_parameter, f"{module_index}:{spec.name}.lora_up.weight"),
            (spec.down_parameter, f"{module_index}:{spec.name}.lora_down.weight"),
        )
        for parameter, factor_name in named_factors:
            if id(parameter) in factor_parameters:
                raise ValueError(
                    "LoRA-Squeeze global optimizer mode found a factor parameter shared by multiple protocol roles: "
                    f"{factor_parameters[id(parameter)]} and {factor_name}"
                )
            factor_parameters[id(parameter)] = factor_name

    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    optimizer_parameters = [parameter for group in raw_optimizer.param_groups for parameter in group["params"]]
    optimizer_parameter_ids = [id(parameter) for parameter in optimizer_parameters]
    if len(set(optimizer_parameter_ids)) != len(optimizer_parameter_ids):
        raise ValueError("LoRA-Squeeze global optimizer mode found duplicate parameters in optimizer groups")

    named_parameters = {id(parameter): name for name, parameter in network.named_parameters()}
    unexpected = [
        named_parameters.get(id(parameter), f"unnamed parameter {tuple(parameter.shape)}")
        for parameter in optimizer_parameters
        if id(parameter) not in factor_parameters
    ]
    if unexpected:
        raise ValueError(
            "LoRA-Squeeze global optimizer mode requires every optimizer parameter to be a LoRA squeeze factor; "
            "non-factor optimizer parameters: " + ", ".join(unexpected[:8]) + ". Use per_squeeze optimizer mode."
        )


def get_lora_squeeze_optimizer_parameter_layout(
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[Tuple[str, ...], ...]:
    validate_lora_squeeze_optimizer_parameters(network, optimizer)
    factor_parameters: Dict[int, str] = {}
    for module_index, module in enumerate(get_lora_squeeze_modules(network)):
        spec = module.lora_squeeze_get_spec()
        factor_parameters[id(spec.up_parameter)] = f"{module_index}:{spec.name}.lora_up.weight"
        factor_parameters[id(spec.down_parameter)] = f"{module_index}:{spec.name}.lora_down.weight"
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    return tuple(
        tuple(factor_parameters[id(parameter)] for parameter in group["params"])
        for group in raw_optimizer.param_groups
    )


def validate_lora_squeeze_network(network: torch.nn.Module, schedule: LoRASqueezeSchedule) -> Dict[str, float]:
    modules = get_lora_squeeze_modules(network)
    specs = [module.lora_squeeze_get_spec() for module in modules]
    source_ranks = {spec.rank for spec in specs}
    if len(source_ranks) != 1:
        raise ValueError(f"LoRA-Squeeze requires a homogeneous current rank, found ranks: {sorted(source_ranks)}")
    source_rank = next(iter(source_ranks))
    if schedule.enabled and source_rank != schedule.current_dim:
        raise ValueError(
            f"LoRA-Squeeze expected all LoRA modules to have current rank {schedule.current_dim}, "
            f"but found rank {source_rank}. Check --network_args, --network_weights, and resume state."
        )
    if schedule.enabled:
        mismatched_alphas = [
            f"{spec.name}={spec.alpha:.8g}"
            for spec in specs
            if not math.isclose(spec.alpha, schedule.current_alpha, rel_tol=1e-5, abs_tol=1e-8)
        ]
        if mismatched_alphas:
            raise ValueError(
                f"LoRA-Squeeze expected every LoRA module to have current alpha {schedule.current_alpha:.8g}, "
                "but found: " + ", ".join(mismatched_alphas[:8])
            )
    if schedule.enabled:
        future_ranks = schedule.ranks[schedule.completed_squeezes + 1 :]
        invalid_targets = [rank for rank in future_ranks if rank >= source_rank]
        if invalid_targets:
            raise ValueError(
                f"LoRA-Squeeze target ranks must be less than current rank {source_rank}, found: {invalid_targets}"
            )
    return {"modules": float(len(modules)), "source_rank": float(source_rank)}


def get_lora_factor_matrices(module: LoRASqueezeModuleProtocol):
    spec = module.lora_squeeze_get_spec()
    return spec.up_2d, spec.down_2d, spec.kind


def compact_lora_product(
    up_2d: torch.Tensor,
    down_2d: torch.Tensor,
    old_scale: float,
    target_dim: int,
    target_alpha: float,
    build_optimizer_projections: bool = False,
):
    """Compress a scaled LoRA product through its rank-sized interaction core."""
    a = up_2d
    b = down_2d * old_scale
    qa, ra = torch.linalg.qr(a, mode="reduced")
    qb, rb = torch.linalg.qr(b.T, mode="reduced")
    core = ra @ rb.T
    u_core, singular_values, vh_core = torch.linalg.svd(core, full_matrices=False)

    usable_dim = min(target_dim, singular_values.numel())
    kept = singular_values[:usable_dim].clamp_min(0)
    factor_scale = math.sqrt(target_dim / target_alpha)
    sqrt_s = torch.sqrt(kept)
    largest = singular_values.max() if singular_values.numel() > 0 else singular_values.new_zeros(())
    tolerance = max(up_2d.shape[0], down_2d.shape[1]) * torch.finfo(singular_values.dtype).eps * largest
    retained_active_mask = kept > tolerance
    numerical_rank = int((singular_values > tolerance).sum().item())
    revived_rank_mask = torch.ones(target_dim, device=ra.device, dtype=torch.bool)
    revived_rank_mask[:usable_dim] = ~retained_active_mask

    new_up_core = torch.zeros((ra.shape[0], target_dim), device=ra.device, dtype=ra.dtype)
    new_down_core = torch.zeros((target_dim, rb.shape[0]), device=rb.device, dtype=rb.dtype)
    if retained_active_mask.any():
        active_indices = torch.nonzero(retained_active_mask, as_tuple=False).flatten()
        active_scales = sqrt_s[active_indices] * factor_scale
        new_up_core[:, active_indices] = u_core[:, active_indices] * active_scales.unsqueeze(0)
        new_down_core[active_indices, :] = active_scales.unsqueeze(1) * vh_core[active_indices, :]

    if revived_rank_mask.any():
        active_factor_norms = sqrt_s[retained_active_mask] * factor_scale
        old_down_norms = down_2d.norm(dim=1)
        if active_factor_norms.numel() > 0:
            revival_norm = active_factor_norms.median()
        elif old_down_norms.numel() > 0:
            revival_norm = old_down_norms.median()
        else:
            revival_norm = down_2d.new_tensor(1.0)
        if not torch.isfinite(revival_norm) or revival_norm <= 0:
            revival_norm = down_2d.new_tensor(1.0)
        for revived_index in torch.nonzero(revived_rank_mask, as_tuple=False).flatten().tolist():
            if revived_index < usable_dim:
                direction = vh_core[revived_index]
            else:
                direction = torch.zeros(rb.shape[0], device=rb.device, dtype=rb.dtype)
                direction[revived_index % rb.shape[0]] = 1.0
            new_down_core[revived_index] = direction * revival_norm

    new_up_2d = qa @ new_up_core
    new_down_2d = new_down_core @ qb.T
    up_grad_projection = None
    down_grad_projection = None
    if build_optimizer_projections:
        new_scale = target_alpha / target_dim
        up_grad_projection = new_scale * torch.linalg.pinv(rb) @ new_down_core.T
        down_grad_projection = (new_scale / old_scale) * (torch.linalg.pinv(ra) @ new_up_core).T
        if not torch.isfinite(up_grad_projection).all() or not torch.isfinite(down_grad_projection).all():
            raise ValueError("LoRA-Squeeze optimizer-state projection produced non-finite values")
    return (
        new_up_2d,
        singular_values,
        new_down_2d,
        up_grad_projection,
        down_grad_projection,
        numerical_rank,
        revived_rank_mask,
    )


def snapshot_lora_module_layers(network: torch.nn.Module) -> List[Tuple[LoRASqueezeModuleProtocol, Any]]:
    return [(module, module.lora_squeeze_snapshot()) for module in get_lora_squeeze_modules(network)]


def restore_lora_module_layers(snapshots: List[Tuple[LoRASqueezeModuleProtocol, Any]]):
    for module, snapshot in snapshots:
        module.lora_squeeze_restore(snapshot)


def squeeze_lora_network(
    network: torch.nn.Module,
    target_dim: int,
    target_alpha: float,
    optimizer_for_state_transfer: Optional[torch.optim.Optimizer] = None,
    optimizer_state_staging_device: Optional[Union[str, torch.device]] = None,
) -> Tuple[Dict[str, float], List[Tuple[torch.nn.Parameter, Dict[str, Any]]]]:
    modules = get_lora_squeeze_modules(network)
    source_ranks = {module.lora_squeeze_get_spec().rank for module in modules}
    if len(source_ranks) != 1:
        raise ValueError(f"LoRA-Squeeze requires a homogeneous current rank, found ranks: {sorted(source_ranks)}")
    source_rank = next(iter(source_ranks))
    if target_dim >= source_rank:
        raise ValueError(f"LoRA-Squeeze target rank {target_dim} must be less than current rank {source_rank}")
    optimizer_state_adapter = (
        prepare_optimizer_state_transfer(optimizer_for_state_transfer)
        if optimizer_for_state_transfer is not None
        else None
    )

    retained_energies: List[float] = []
    numerical_ranks: List[int] = []
    revived_rank_channels = 0
    rank_deficient_modules = 0
    transfers: List[Tuple[torch.nn.Parameter, Dict[str, Any]]] = []
    state_counts = {"projected": 0, "reset": 0, "empty": 0, "warm_restarted": 0}
    with torch.no_grad():
        for module in modules:
            old_spec = module.lora_squeeze_get_spec()
            (
                new_up_2d,
                singular_values,
                new_down_2d,
                up_projection,
                down_projection,
                numerical_rank,
                revived_rank_mask,
            ) = compact_lora_product(
                old_spec.up_2d,
                old_spec.down_2d,
                old_spec.scale,
                target_dim,
                target_alpha,
                build_optimizer_projections=optimizer_for_state_transfer is not None,
            )
            usable_dim = min(target_dim, singular_values.numel())
            kept = singular_values[:usable_dim].clamp_min(0)
            total_energy = torch.sum(singular_values * singular_values).item()
            kept_energy = torch.sum(kept * kept).item()
            retained_energies.append(1.0 if total_energy == 0.0 else kept_energy / total_energy)
            numerical_ranks.append(numerical_rank)
            revived_count = int(revived_rank_mask.sum().item())
            revived_rank_channels += revived_count
            rank_deficient_modules += int(revived_count > 0)

            module.lora_squeeze_replace_factors(new_up_2d, new_down_2d, target_dim, target_alpha)
            new_spec = module.lora_squeeze_get_spec()
            _validate_protocol_spec(new_spec, {id(parameter) for parameter in network.parameters()})
            if new_spec.rank != target_dim or not math.isclose(
                new_spec.alpha, target_alpha, rel_tol=1e-5, abs_tol=1e-8
            ):
                raise ValueError(
                    f"{new_spec.name} did not install the requested LoRA-Squeeze rank/alpha: "
                    f"{new_spec.rank}/{new_spec.alpha:.8g} != {target_dim}/{target_alpha:.8g}"
                )
            if optimizer_state_adapter is not None:
                new_up_state, up_status = optimizer_state_adapter.project_parameter_state(
                    old_spec.up_parameter, new_spec.up_parameter, up_projection, "up"
                )
                new_up_state = stage_optimizer_state(new_up_state, optimizer_state_staging_device)
                new_down_state, down_status = optimizer_state_adapter.project_parameter_state(
                    old_spec.down_parameter, new_spec.down_parameter, down_projection, "down"
                )
                new_down_state = stage_optimizer_state(new_down_state, optimizer_state_staging_device)
                for status in (up_status, down_status):
                    if status == "projected":
                        state_counts["projected"] += 1
                    elif status == "empty":
                        state_counts["empty"] += 1
                    elif status.startswith("warm_restart:"):
                        state_counts["warm_restarted"] += 1
                    else:
                        state_counts["reset"] += 1
                transfers.append((new_spec.up_parameter, new_up_state))
                transfers.append((new_spec.down_parameter, new_down_state))

    return (
        {
            "modules": float(len(modules)),
            "source_rank": float(source_rank),
            "target_rank": float(target_dim),
            "retained_energy_min": min(retained_energies),
            "retained_energy_mean": sum(retained_energies) / len(retained_energies),
            "numerical_rank_min": float(min(numerical_ranks)),
            "numerical_rank_mean": float(sum(numerical_ranks) / len(numerical_ranks)),
            "rank_deficient_modules": float(rank_deficient_modules),
            "revived_rank_channels": float(revived_rank_channels),
            "optimizer_state_projected": float(state_counts["projected"]),
            "optimizer_state_reset": float(state_counts["reset"]),
            "optimizer_state_empty": float(state_counts["empty"]),
            "optimizer_state_warm_restarted": float(state_counts["warm_restarted"]),
        },
        transfers,
    )
