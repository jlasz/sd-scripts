import pytest
import torch
from torch import nn

from library.lora_squeeze_compression import squeeze_lora_network
from library.lora_squeeze_optimizer import (
    OptimizerStateCPUStagingError,
    copy_optimizer_param_group_state,
    move_optimizer_state_to_parameter_devices,
    offload_optimizer_state_to_cpu,
    prepare_optimizer_state_transfer,
    stage_optimizer_state,
    validate_optimizer_scheduler_modes,
)
from tests.lora_squeeze_test_utils import FakeLoRAModule, FakeNetwork


def install_transfers(old_optimizer, new_optimizer, transfers):
    copy_optimizer_param_group_state(old_optimizer, new_optimizer, transfers)
    for parameter, state in transfers:
        if state:
            new_optimizer.state[parameter] = state


def test_cpu_staging_uses_a_distinct_error(monkeypatch):
    def fail_move(value, device, move_scalar_tensors=True):
        raise RuntimeError("move failed")

    monkeypatch.setattr("library.lora_squeeze_optimizer._move_optimizer_state_value", fail_move)
    with pytest.raises(OptimizerStateCPUStagingError, match="projected optimizer state"):
        stage_optimizer_state({"buffer": torch.ones(2)}, "cpu")
    with pytest.raises(RuntimeError, match="move failed"):
        stage_optimizer_state({"buffer": torch.ones(2)}, "cuda")


def test_partial_optimizer_offload_rolls_back(monkeypatch):
    first = nn.Parameter(torch.zeros(2))
    second = nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([first, second], lr=0.1)
    optimizer.state[first]["buffer"] = torch.ones_like(first)
    optimizer.state[second]["buffer"] = torch.ones_like(second)

    from library import lora_squeeze_optimizer as optimizer_module

    original_move = optimizer_module._move_optimizer_state_value_with_device_record
    calls = 0

    def fail_second_move(value, device):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("CPU transfer failed")
        return original_move(value, device)

    monkeypatch.setattr(optimizer_module, "_move_optimizer_state_value_with_device_record", fail_second_move)

    with pytest.raises(OptimizerStateCPUStagingError, match="existing optimizer state"):
        offload_optimizer_state_to_cpu(optimizer)

    assert calls == 2
    assert torch.equal(optimizer.state[first]["buffer"], torch.ones_like(first))
    assert torch.equal(optimizer.state[second]["buffer"], torch.ones_like(second))


def test_sgd_momentum_uses_parameter_displacement_projection():
    old_parameter = nn.Parameter(torch.zeros(3, 4))
    new_parameter = nn.Parameter(torch.zeros(3, 2))
    optimizer = torch.optim.SGD([old_parameter], lr=0.1, momentum=0.9)
    momentum = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    optimizer.state[old_parameter]["momentum_buffer"] = momentum.clone()
    projection = torch.tensor([[2.0, 0.0], [0.0, 0.5], [0.0, 0.0], [0.0, 0.0]])

    projected, status = prepare_optimizer_state_transfer(optimizer).project_parameter_state(
        old_parameter, new_parameter, projection, "up"
    )
    expected = momentum @ torch.linalg.pinv(projection.T)

    assert status == "projected"
    assert torch.allclose(projected["momentum_buffer"], expected)
    assert not torch.allclose(projected["momentum_buffer"], momentum @ projection)

    next_optimizer = torch.optim.SGD([new_parameter], lr=0.1, momentum=0.9)
    next_optimizer.state[new_parameter] = projected
    new_parameter.grad = torch.zeros_like(new_parameter)
    next_optimizer.step()
    assert torch.allclose(new_parameter, -0.09 * expected)


def test_ill_conditioned_projection_fails_closed():
    old_parameter = nn.Parameter(torch.zeros(1, 2))
    new_parameter = nn.Parameter(torch.zeros(1, 1))
    optimizer = torch.optim.SGD([old_parameter], momentum=0.9)
    optimizer.state[old_parameter]["momentum_buffer"] = torch.ones_like(old_parameter)

    with pytest.raises(ValueError, match="ill-conditioned.*per_squeeze"):
        prepare_optimizer_state_transfer(optimizer).project_parameter_state(
            old_parameter,
            new_parameter,
            torch.tensor([[1e-8], [0.0]]),
            "up",
        )


def test_convolutional_adam_state_can_take_another_step_after_squeeze():
    torch.manual_seed(14)
    module = FakeLoRAModule(7, 5, 4, 2.0, kernel_size=3)
    nn.init.normal_(module.lora_up.weight)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3, amsgrad=True)
    for parameter in module.parameters():
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    optimizer.zero_grad()

    stats, transfers = squeeze_lora_network(
        FakeNetwork(module), 2, target_alpha=1.0, optimizer_for_state_transfer=optimizer
    )
    new_optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3, amsgrad=True)
    install_transfers(optimizer, new_optimizer, transfers)
    for parameter in module.parameters():
        parameter.grad = torch.randn_like(parameter)
    new_optimizer.step()

    assert stats["optimizer_state_projected"] == 2
    assert all(torch.isfinite(parameter).all() for parameter in module.parameters())


def test_unknown_optimizer_state_is_rejected_without_changing_layers():
    module = FakeLoRAModule(7, 5, 4, 2.0)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    for parameter in module.parameters():
        optimizer.state[parameter]["p0"] = torch.randn(parameter.numel())

    with pytest.raises(ValueError, match="state.*p0"):
        squeeze_lora_network(FakeNetwork(module), 2, target_alpha=1.0, optimizer_for_state_transfer=optimizer)
    assert module.lora_dim == 4


def test_adafactor_preserves_relative_step_and_refactors_second_moment():
    from transformers.optimization import Adafactor

    torch.manual_seed(32)
    module = FakeLoRAModule(7, 5, 4, 2.0)
    nn.init.normal_(module.lora_up.weight)
    kwargs = dict(lr=None, beta1=0.9, relative_step=True, scale_parameter=True, warmup_init=False)
    optimizer = Adafactor(module.parameters(), **kwargs)
    for _ in range(3):
        for parameter in module.parameters():
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        optimizer.zero_grad()

    stats, transfers = squeeze_lora_network(
        FakeNetwork(module), 2, target_alpha=1.0, optimizer_for_state_transfer=optimizer
    )
    new_optimizer = Adafactor(module.parameters(), **kwargs)
    install_transfers(optimizer, new_optimizer, transfers)

    assert stats["optimizer_state_projected"] == 2
    for parameter in module.parameters():
        state = new_optimizer.state[parameter]
        assert state["step"] == 3
        assert state["exp_avg"].shape == parameter.shape
        assert state["exp_avg_sq_row"].shape == parameter.shape[:-1]
        assert state["exp_avg_sq_col"].shape == parameter.shape[:-2] + parameter.shape[-1:]


def test_prodigy_preserves_learned_scale_and_complete_state():
    prodigyopt = pytest.importorskip("prodigyopt")
    torch.manual_seed(31)
    module = FakeLoRAModule(7, 5, 4, 2.0)
    nn.init.normal_(module.lora_up.weight)
    optimizer = prodigyopt.Prodigy(module.parameters(), lr=1.0, use_bias_correction=True)
    for _ in range(3):
        for parameter in module.parameters():
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        optimizer.zero_grad()

    old_d = optimizer.param_groups[0]["d"]
    old_k = optimizer.param_groups[0]["k"]
    stats, transfers = squeeze_lora_network(
        FakeNetwork(module), 2, target_alpha=1.0, optimizer_for_state_transfer=optimizer
    )
    new_optimizer = prodigyopt.Prodigy(module.parameters(), lr=1.0, use_bias_correction=True)
    install_transfers(optimizer, new_optimizer, transfers)

    assert stats["optimizer_state_projected"] == 2
    assert stats["optimizer_state_warm_restarted"] == 0
    assert new_optimizer.param_groups[0]["d"] == old_d
    assert new_optimizer.param_groups[0]["k"] == old_k
    assert all(
        set(new_optimizer.state[parameter]) == {"step", "s", "p0", "exp_avg", "exp_avg_sq"}
        for parameter in module.parameters()
    )


def test_prodigy_plus_uses_a_coherent_warm_restart():
    prodigyplus = pytest.importorskip("prodigyplus")
    torch.manual_seed(35)
    module = FakeLoRAModule(64, 64, 16, 8.0)
    nn.init.normal_(module.lora_up.weight)
    kwargs = {"lr": 1.0, "factored": True, "stochastic_rounding": False}
    optimizer = prodigyplus.ProdigyPlusScheduleFree(module.parameters(), **kwargs)
    for _ in range(2):
        for parameter in module.parameters():
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        optimizer.zero_grad()
    optimizer.param_groups[0]["d"] = 0.023

    stats, transfers = squeeze_lora_network(
        FakeNetwork(module), 8, target_alpha=4.0, optimizer_for_state_transfer=optimizer
    )
    new_optimizer = prodigyplus.ProdigyPlusScheduleFree(module.parameters(), **kwargs)
    install_transfers(optimizer, new_optimizer, transfers)

    assert stats["optimizer_state_warm_restarted"] == 2
    assert [state for _, state in transfers] == [{}, {}]
    assert new_optimizer.param_groups[0]["d"] == pytest.approx(0.023)


def test_schedule_free_preserves_iteration_history():
    schedulefree = pytest.importorskip("schedulefree")
    torch.manual_seed(33)
    module = FakeLoRAModule(7, 5, 4, 2.0)
    nn.init.normal_(module.lora_up.weight)
    optimizer = schedulefree.AdamWScheduleFree(module.parameters(), lr=0.01)
    optimizer.train()
    for _ in range(3):
        for parameter in module.parameters():
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        optimizer.zero_grad()

    old_k = optimizer.param_groups[0]["k"]
    stats, transfers = squeeze_lora_network(
        FakeNetwork(module), 2, target_alpha=1.0, optimizer_for_state_transfer=optimizer
    )
    new_optimizer = schedulefree.AdamWScheduleFree(module.parameters(), lr=0.01)
    new_optimizer.train()
    install_transfers(optimizer, new_optimizer, transfers)

    assert stats["optimizer_state_projected"] == 2
    assert new_optimizer.param_groups[0]["k"] == old_k
    assert all(new_optimizer.state[p]["z"].shape == p.shape for p in module.parameters())


@pytest.mark.parametrize(
    ("optimizer_mode", "scheduler_mode"),
    [("per_squeeze", "global"), ("global", "per_squeeze")],
)
def test_ordinary_optimizer_allows_independent_state_modes(optimizer_mode, scheduler_mode):
    optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(1))])
    validate_optimizer_scheduler_modes(optimizer, optimizer_mode, scheduler_mode)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for paged bitsandbytes coverage")
def test_paged_bitsandbytes_state_projects_and_remains_paged():
    bnb = pytest.importorskip("bitsandbytes")
    if not hasattr(bnb.optim, "PagedAdamW8bit"):
        pytest.skip("PagedAdamW8bit is not available")

    torch.manual_seed(36)
    module = FakeLoRAModule(4096, 4096, 41, 41.0).cuda()
    optimizer = bnb.optim.PagedAdamW8bit(module.parameters(), lr=1e-3, min_8bit_size=4096)
    for parameter in module.parameters():
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    old_buffers = [
        value
        for state in optimizer.state.values()
        for key, value in state.items()
        if key in ("state1", "state2")
    ]
    if not old_buffers or not all(getattr(value, "is_paged", False) for value in old_buffers):
        pytest.skip("bitsandbytes did not allocate managed paged buffers")

    stats, transfers = squeeze_lora_network(
        FakeNetwork(module),
        28,
        target_alpha=28.0,
        optimizer_for_state_transfer=optimizer,
        optimizer_state_staging_device="cpu",
    )
    new_optimizer = bnb.optim.PagedAdamW8bit(module.parameters(), lr=1e-3, min_8bit_size=4096)
    install_transfers(optimizer, new_optimizer, transfers)
    move_optimizer_state_to_parameter_devices(new_optimizer)

    new_buffers = [
        value
        for state in new_optimizer.state.values()
        for key, value in state.items()
        if key in ("state1", "state2")
    ]
    assert new_buffers
    assert all(getattr(value, "is_paged", False) for value in new_buffers)
    assert stats["optimizer_state_projected"] == 2
