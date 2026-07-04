import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from library.lora_squeeze_compression import (
    get_lora_squeeze_optimizer_parameter_layout,
    preserve_lora_squeeze_alpha_precision,
    restore_lora_module_layers,
    snapshot_lora_module_layers,
    squeeze_lora_network,
    validate_lora_squeeze_network,
    validate_lora_squeeze_optimizer_parameters,
)
from library.lora_squeeze_network import validate_lora_squeeze_network_module
from tests.lora_squeeze_test_utils import (
    FakeFloatAlphaLoRAModule,
    FakeLoRAModule,
    FakeMultiNetwork,
    FakeNetwork,
    effective_delta,
    make_schedule,
)


@pytest.mark.parametrize("kernel_size", [1, 3], ids=["linear", "conv2d"])
def test_compression_matches_direct_truncated_svd(kernel_size):
    torch.manual_seed(123)
    module = FakeLoRAModule(11, 13, 8, 5.5, kernel_size=kernel_size)
    nn.init.normal_(module.lora_down.weight)
    nn.init.normal_(module.lora_up.weight)
    before = effective_delta(module)
    u, s, vh = torch.linalg.svd(before, full_matrices=False)
    expected = (u[:, :4] * s[:4]) @ vh[:4]

    stats, transfers = squeeze_lora_network(FakeNetwork(module), 4, target_alpha=2.75)

    assert torch.allclose(effective_delta(module), expected, rtol=2e-5, atol=2e-5)
    assert stats["retained_energy_mean"] == pytest.approx(float((s[:4] ** 2).sum() / (s**2).sum()))
    assert transfers == []
    assert module.lora_dim == 4
    assert module.alpha.dtype == torch.float32


def test_repeated_squeeze_matches_direct_lower_rank_approximation():
    torch.manual_seed(321)
    module = FakeLoRAModule(11, 13, 8, 5.5)
    nn.init.normal_(module.lora_down.weight)
    nn.init.normal_(module.lora_up.weight)
    before = effective_delta(module)
    u, s, vh = torch.linalg.svd(before, full_matrices=False)
    expected = (u[:, :2] * s[:2]) @ vh[:2]

    squeeze_lora_network(FakeNetwork(module), 5, target_alpha=3.0)
    squeeze_lora_network(FakeNetwork(module), 2, target_alpha=1.5)

    assert torch.allclose(effective_delta(module), expected, rtol=2e-5, atol=2e-5)


def test_float_alpha_and_low_precision_buffer_are_handled_without_false_mismatch():
    float_alpha = FakeFloatAlphaLoRAModule(7, 5, 4, 2.0)
    squeeze_lora_network(FakeNetwork(float_alpha), 2, target_alpha=1.25)
    assert isinstance(float_alpha.alpha, float)
    assert float_alpha.alpha == pytest.approx(1.25)

    buffered = FakeLoRAModule(7, 5, 4, 1.234567)
    buffered.alpha = buffered.alpha.to(torch.float16)
    buffered.scale = float(buffered.alpha.float()) / buffered.lora_dim
    assert preserve_lora_squeeze_alpha_precision(FakeNetwork(buffered)) == 1
    assert buffered.alpha.dtype == torch.float32


def test_squeeze_preserves_cpu_rng_state():
    torch.manual_seed(2025)
    module = FakeLoRAModule(7, 5, 4, 2.0)
    nn.init.normal_(module.lora_down.weight)
    nn.init.normal_(module.lora_up.weight)
    state = torch.get_rng_state()

    squeeze_lora_network(FakeNetwork(module), 2, target_alpha=1.0)

    assert torch.equal(torch.get_rng_state(), state)


def test_rank_deficient_squeeze_revives_a_trainable_zero_channel():
    torch.manual_seed(2026)
    module = FakeLoRAModule(7, 5, 4, 4.0)
    with torch.no_grad():
        module.lora_down.weight.normal_()
        module.lora_up.weight.zero_()
        module.lora_up.weight[:, 0].normal_()
    before = effective_delta(module)

    stats, _ = squeeze_lora_network(FakeNetwork(module), 2, target_alpha=2.0)

    assert torch.allclose(effective_delta(module), before, rtol=2e-5, atol=2e-5)
    assert stats["numerical_rank_min"] == 1
    assert stats["rank_deficient_modules"] == 1
    assert stats["revived_rank_channels"] == 1

    revived_index = int(torch.argmin(module.lora_up.weight.detach().norm(dim=0)))
    assert module.lora_up.weight[:, revived_index].count_nonzero() == 0
    assert module.lora_down.weight[revived_index].norm() > 0

    target = torch.randn_like(before)
    delta = module.lora_up.weight @ module.lora_down.weight * module.scale
    ((delta - target) ** 2).mean().backward()
    assert module.lora_up.weight.grad[:, revived_index].abs().sum() > 0


def test_snapshot_restores_replaced_factor_objects():
    module = FakeLoRAModule(7, 5, 4, 2.0)
    network = FakeNetwork(module)
    original_down = module.lora_down
    original_up = module.lora_up
    snapshots = snapshot_lora_module_layers(network)

    squeeze_lora_network(network, 2, target_alpha=1.0)
    restore_lora_module_layers(snapshots)

    assert module.lora_down is original_down
    assert module.lora_up is original_up
    assert module.lora_dim == 4
    assert module.scale == pytest.approx(0.5)


def test_optimizer_parameter_validation_rejects_non_factor_parameters():
    network = FakeNetwork(FakeLoRAModule(7, 5, 4, 4.0))
    optimizer = torch.optim.AdamW(network.parameters())
    validate_lora_squeeze_optimizer_parameters(network, optimizer)
    assert get_lora_squeeze_optimizer_parameter_layout(network, optimizer) == (
        ("0:test_lora.lora_down.weight", "0:test_lora.lora_up.weight"),
    )

    network.extra_gate = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW(network.parameters())
    with pytest.raises(ValueError, match="non-factor optimizer parameters: extra_gate"):
        validate_lora_squeeze_optimizer_parameters(network, optimizer)


def test_network_module_requires_an_explicit_support_contract():
    unsupported = SimpleNamespace(__name__="networks.unsupported")
    with pytest.raises(ValueError, match="does not implement validate_lora_squeeze_support"):
        validate_lora_squeeze_network_module(unsupported, {})

    received = []
    supported = SimpleNamespace(
        __name__="networks.supported",
        validate_lora_squeeze_support=lambda network_args: received.append(network_args),
    )
    validate_lora_squeeze_network_module(supported, {"foo": "bar"})
    assert received == [{"foo": "bar"}]


def test_resumed_validation_checks_the_current_and_future_ranks():
    schedule = make_schedule(lora_squeeze_start_dim=16, network_dim=4, lora_squeeze_num_squeezes=2)
    schedule.set_total_steps(120)
    resumed_rank = schedule.ranks[1]
    schedule.mark_squeezed(resumed_rank, schedule.alpha_for_rank(resumed_rank))

    stats = validate_lora_squeeze_network(
        FakeNetwork(FakeLoRAModule(7, 5, resumed_rank, schedule.current_alpha)),
        schedule,
    )

    assert stats["source_rank"] == resumed_rank
    assert stats["modules"] == 1


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda module: setattr(module, "lora_down", nn.Linear(7, 3, bias=False)), "input features do not match rank"),
        (lambda module: setattr(module, "scale", module.scale * 2), "alpha/scale"),
    ],
)
def test_network_validation_rejects_inconsistent_factor_metadata(mutate, message):
    schedule = make_schedule(lora_squeeze_start_dim=4, network_dim=2, network_alpha=2, lora_squeeze_num_squeezes=1)
    module = FakeLoRAModule(7, 5, 4, schedule.current_alpha)
    mutate(module)

    with pytest.raises(ValueError, match=message):
        validate_lora_squeeze_network(FakeNetwork(module), schedule)


def test_mixed_current_ranks_are_rejected_before_training():
    schedule = make_schedule(lora_squeeze_start_dim=4, network_dim=2, network_alpha=2, lora_squeeze_num_squeezes=1)
    network = FakeMultiNetwork(
        FakeLoRAModule(7, 5, 4, schedule.current_alpha),
        FakeLoRAModule(7, 5, 3, schedule.current_alpha),
    )

    with pytest.raises(ValueError, match="rank"):
        validate_lora_squeeze_network(network, schedule)


@pytest.mark.parametrize("module_name", ["networks.lora", "networks.lora_anima"])
def test_builtin_lora_modules_validate_and_squeeze(module_name):
    network_module = importlib.import_module(module_name)
    schedule = make_schedule(lora_squeeze_start_dim=8, network_dim=4, lora_squeeze_num_squeezes=1)
    validate_lora_squeeze_network_module(network_module, {})
    module = network_module.LoRAModule(
        "test",
        nn.Linear(7, 5, bias=False),
        lora_dim=8,
        alpha=schedule.current_alpha,
    )

    validate_lora_squeeze_network(FakeNetwork(module), schedule)
    squeeze_lora_network(FakeNetwork(module), 4, target_alpha=2.0)

    assert module.lora_dim == 4


def test_standard_protocol_rejects_unsafe_factor_semantics():
    schedule = make_schedule(lora_squeeze_start_dim=4, network_dim=2, lora_squeeze_num_squeezes=1)

    grouped = FakeLoRAModule(4, 5, 4, 4.0, kernel_size=3)
    grouped.lora_down = nn.Conv2d(4, 4, 1, groups=2, bias=False)
    with pytest.raises(ValueError, match="grouped LoRA convolutions"):
        validate_lora_squeeze_network(FakeNetwork(grouped), schedule)

    hooked = FakeLoRAModule(7, 5, 4, 4.0)
    hooked.lora_down.register_forward_hook(lambda _module, _inputs, output: output)
    with pytest.raises(ValueError, match="hooks attached"):
        validate_lora_squeeze_network(FakeNetwork(hooked), schedule)

    frozen = FakeLoRAModule(7, 5, 4, 4.0)
    frozen.lora_up.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="frozen lora_up"):
        validate_lora_squeeze_network(FakeNetwork(frozen), schedule)
