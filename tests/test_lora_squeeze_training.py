from types import SimpleNamespace

import pytest
import torch
from torch import nn

from library.accelerator_utils import (
    ReplaceableLRScheduler,
    ReplaceableOptimizer,
    find_replaceable_optimizer,
    find_replaceable_scheduler,
    make_replaceable_optimizer_scheduler,
)
from library.args import get_sanitized_config_or_none, resume_from_local_or_hf_if_specified
from library.lora_squeeze_optimizer import OptimizerStateCPUStagingError, stage_optimizer_state
from library.lora_squeeze_training import LoRASqueezeRuntime, LoRASqueezeTrainingController
from tests.lora_squeeze_test_utils import (
    FakeAccelerator,
    FakeLoRAModule,
    FakeNetwork,
    FakeOptimizerUtil,
    FakeScheduler,
    build_controller,
    effective_delta,
    make_schedule,
    make_schedule_args,
)


def test_ordinary_resume_does_not_mutate_args_or_tracker_config():
    class ResumeAccelerator:
        def __init__(self):
            self.loaded_paths = []

        def load_state(self, path):
            self.loaded_paths.append(path)

    args = SimpleNamespace(
        resume="state-a",
        resume_from_huggingface=False,
        log_config=True,
        lora_squeeze_start_dim=None,
        lora_squeeze_num_squeezes=0,
        lora_squeeze_optimizer_mode="per_squeeze",
    )
    accelerator = ResumeAccelerator()

    resume_from_local_or_hf_if_specified(accelerator, args)
    args.resume = "state-b"
    resume_from_local_or_hf_if_specified(accelerator, args)

    assert accelerator.loaded_paths == ["state-a", "state-b"]
    assert not hasattr(args, "_resume_state_dir")
    tracker_config = get_sanitized_config_or_none(args)
    assert "_resume_state_dir" not in tracker_config
    assert not any(key.startswith("lora_squeeze_") for key in tracker_config)


def test_enabled_squeeze_arguments_remain_in_tracker_config():
    args = SimpleNamespace(
        log_config=True,
        lora_squeeze_start_dim=16,
        lora_squeeze_num_squeezes=2,
        lora_squeeze_optimizer_mode="global",
    )
    tracker_config = get_sanitized_config_or_none(args)
    assert tracker_config["lora_squeeze_start_dim"] == 16
    assert tracker_config["lora_squeeze_num_squeezes"] == 2


def test_replaceable_objects_keep_stable_public_identities():
    parameter = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    replaceable_optimizer, replaceable_scheduler = make_replaceable_optimizer_scheduler(optimizer, scheduler)

    new_parameter = nn.Parameter(torch.ones(1))
    new_optimizer = torch.optim.AdamW([new_parameter], lr=0.2)
    new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=2)
    replaceable_optimizer.replace(new_optimizer)
    replaceable_scheduler.replace(new_scheduler)

    assert isinstance(replaceable_optimizer, ReplaceableOptimizer)
    assert isinstance(replaceable_scheduler, ReplaceableLRScheduler)
    assert find_replaceable_optimizer(replaceable_optimizer) is replaceable_optimizer
    assert find_replaceable_scheduler(replaceable_scheduler) is replaceable_scheduler
    assert replaceable_optimizer.optimizer is new_optimizer
    assert replaceable_scheduler.scheduler is new_scheduler


def test_replaceable_objects_survive_real_accelerate_save_load(tmp_path):
    from accelerate import Accelerator

    accelerator = Accelerator()
    parameter = nn.Parameter(torch.ones(1, device=accelerator.device))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    replaceable_optimizer, replaceable_scheduler = make_replaceable_optimizer_scheduler(optimizer, scheduler)
    prepared_optimizer, prepared_scheduler = accelerator.prepare(replaceable_optimizer, replaceable_scheduler)
    optimizer_identity = id(prepared_optimizer)
    scheduler_identity = id(prepared_scheduler)

    new_parameter = nn.Parameter(torch.ones(1, device=accelerator.device))
    new_optimizer = torch.optim.SGD([new_parameter], lr=0.2)
    new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1, gamma=0.5)
    find_replaceable_optimizer(prepared_optimizer).replace(new_optimizer)
    find_replaceable_scheduler(prepared_scheduler).replace(new_scheduler)

    new_parameter.grad = torch.ones_like(new_parameter)
    prepared_optimizer.step()
    prepared_scheduler.step()

    assert id(prepared_optimizer) == optimizer_identity
    assert id(prepared_scheduler) == scheduler_identity
    assert torch.allclose(new_parameter, torch.tensor([0.8], device=accelerator.device))

    accelerator.save_state(str(tmp_path))
    new_optimizer.param_groups[0]["lr"] = 0.9
    accelerator.load_state(str(tmp_path))
    assert new_optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_runtime_reports_current_rank_in_weight_errors():
    runtime = LoRASqueezeRuntime(
        make_schedule_args(
            lora_squeeze_start_dim=4,
            network_dim=2,
            network_alpha=2,
            lora_squeeze_num_squeezes=1,
        )
    )

    error = runtime.network_weights_load_error(RuntimeError("unexpected tensor key"))
    assert "current LoRA-Squeeze rank 4" in str(error)
    assert "Original error: unexpected tensor key" in str(error)

    runtime.schedule.mark_squeezed(2, 2.0)
    assert "current LoRA-Squeeze rank 2" in str(runtime.network_weights_load_error(RuntimeError("mismatch")))


def test_boundary_metrics_distinguish_trained_and_new_ranks():
    runtime = LoRASqueezeRuntime(
        make_schedule_args(
            lora_squeeze_start_dim=4,
            network_dim=2,
            network_alpha=2,
            lora_squeeze_num_squeezes=1,
        )
    )
    runtime.set_total_steps(10)
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.25)
    context = runtime.capture_step_context(optimizer)

    class BoundaryController:
        last_squeeze_stats = {"source_rank": 4.0, "target_rank": 2.0}

        def run_if_due(self, global_step):
            runtime.schedule.mark_squeezed(2, runtime.schedule.alpha_for_rank(2))
            return True

    boundary = runtime.schedule.squeeze_steps[0]
    runtime.set_initial_step(boundary - 1)
    assert runtime.run_after_optimizer_step(BoundaryController(), context, boundary)
    logs = {}
    runtime.append_step_logs(logs, context)

    assert logs["lora_squeeze/train_dim"] == 4
    assert logs["lora_squeeze/current_dim"] == 2
    assert logs["lora_squeeze/transition"] == 1


def test_controller_rebuild_does_not_reprepare_or_replace_registrations(monkeypatch):
    schedule = make_schedule(
        lora_squeeze_start_dim=4,
        network_dim=2,
        network_alpha=2,
        lora_squeeze_num_squeezes=1,
        lora_squeeze_optimizer_mode="global",
        lora_squeeze_scheduler_mode="global",
    )
    schedule.set_total_steps(10)
    network = FakeNetwork(FakeLoRAModule(7, 5, 4, schedule.current_alpha))
    controller, accelerator, _, optimizer, scheduler, grad_refreshes = build_controller(schedule, network)
    original_optimizer_registry = tuple(accelerator._optimizers)
    original_scheduler_registry = tuple(accelerator._schedulers)

    def fail_prepare(*args):
        raise AssertionError("must not reprepare")

    monkeypatch.setattr(accelerator, "prepare", fail_prepare)
    assert controller.run_if_due(schedule.squeeze_steps[0])

    assert tuple(map(id, accelerator._optimizers)) == tuple(map(id, original_optimizer_registry))
    assert tuple(map(id, accelerator._schedulers)) == tuple(map(id, original_scheduler_registry))
    assert controller.optimizer is optimizer
    assert controller.lr_scheduler is scheduler
    assert network.adapter.lora_dim == 2
    assert schedule.completed_squeezes == 1
    assert len(grad_refreshes) == 1


def test_controller_rolls_back_on_factor_group_reordering():
    schedule = make_schedule(
        lora_squeeze_start_dim=4,
        network_dim=2,
        network_alpha=2,
        lora_squeeze_num_squeezes=1,
        lora_squeeze_optimizer_mode="global",
        lora_squeeze_scheduler_mode="global",
    )
    schedule.set_total_steps(10)
    network = FakeNetwork(FakeLoRAModule(7, 5, 4, schedule.current_alpha))
    controller, _, _, _, _, grad_refreshes = build_controller(schedule, network)
    controller.prepare_optimizer_params_fn = lambda: (list(reversed(list(network.parameters()))), None)

    with pytest.raises(ValueError, match="membership or ordering"):
        controller.run_if_due(schedule.squeeze_steps[0])

    assert network.adapter.lora_dim == 4
    assert schedule.completed_squeezes == 0
    assert len(grad_refreshes) == 2


def test_cpu_staging_failure_retries_on_parameter_device(monkeypatch):
    schedule = make_schedule(
        lora_squeeze_start_dim=4,
        network_dim=2,
        network_alpha=2,
        lora_squeeze_num_squeezes=1,
        lora_squeeze_optimizer_mode="global",
        lora_squeeze_scheduler_mode="global",
    )
    schedule.set_total_steps(10)
    network = FakeNetwork(FakeLoRAModule(7, 5, 4, schedule.current_alpha))
    controller, accelerator, _, optimizer, _, grad_refreshes = build_controller(schedule, network)
    for parameter in optimizer.param_groups[0]["params"]:
        optimizer.state[parameter]["momentum_buffer"] = torch.ones_like(parameter)

    staging_devices = []

    def fail_cpu_staging(state, device):
        staging_devices.append(device)
        if device is not None and torch.device(device).type == "cpu":
            raise OptimizerStateCPUStagingError("simulated CPU staging failure")
        return stage_optimizer_state(state, device)

    monkeypatch.setattr("library.lora_squeeze_compression.stage_optimizer_state", fail_cpu_staging)
    assert controller.run_if_due(schedule.squeeze_steps[0])

    assert staging_devices == ["cpu", None, None]
    assert network.adapter.lora_dim == 2
    assert len(controller.optimizer.state) == 2
    assert len(grad_refreshes) == 2
    assert any("retrying on the parameter device" in message for message in accelerator.messages)


def test_terminal_squeeze_skips_rebuild_without_resumable_state():
    schedule = make_schedule(
        lora_squeeze_start_dim=4,
        network_dim=2,
        network_alpha=2,
        lora_squeeze_num_squeezes=1,
        lora_squeeze_train_after_final_squeeze=False,
        lora_squeeze_optimizer_mode="global",
        lora_squeeze_scheduler_mode="global",
    )
    schedule.set_total_steps(10)
    network = FakeNetwork(FakeLoRAModule(7, 5, 4, schedule.current_alpha))
    controller, accelerator, optimizer_util, optimizer, scheduler, grad_refreshes = build_controller(schedule, network)
    for parameter in optimizer.param_groups[0]["params"]:
        optimizer.state[parameter]["momentum_buffer"] = torch.ones_like(parameter)

    assert controller.run_if_due(schedule.squeeze_steps[0])

    assert optimizer_util.training_steps == []
    assert accelerator._optimizers == [optimizer]
    assert accelerator._schedulers == [scheduler]
    assert controller.optimizer.param_groups[0]["params"] == []
    assert len(controller.optimizer.state) == 0
    assert grad_refreshes == []


def test_terminal_squeeze_rebuilds_when_end_state_will_be_saved():
    schedule = make_schedule(
        lora_squeeze_start_dim=4,
        network_dim=2,
        network_alpha=2,
        lora_squeeze_num_squeezes=1,
        lora_squeeze_train_after_final_squeeze=False,
        lora_squeeze_optimizer_mode="per_squeeze",
        lora_squeeze_scheduler_mode="per_squeeze",
    )
    schedule.set_total_steps(10)
    network = FakeNetwork(FakeLoRAModule(7, 5, 4, schedule.current_alpha))
    controller, _, optimizer_util, optimizer, scheduler, grad_refreshes = build_controller(
        schedule, network, args=SimpleNamespace(save_state_on_train_end=True)
    )

    assert controller.run_if_due(schedule.squeeze_steps[0])

    assert optimizer_util.training_steps[-1] == 1
    assert controller.optimizer is optimizer
    assert controller.lr_scheduler is scheduler
    assert [tuple(parameter.shape) for parameter in controller.optimizer.param_groups[0]["params"]] == [
        (2, 7),
        (5, 2),
    ]
    assert len(grad_refreshes) == 1


def test_real_accelerate_controller_rebuild_keeps_prepared_objects():
    from accelerate import Accelerator

    schedule = make_schedule(
        lora_squeeze_start_dim=4,
        network_dim=2,
        network_alpha=2,
        lora_squeeze_num_squeezes=1,
        lora_squeeze_optimizer_mode="global",
        lora_squeeze_scheduler_mode="global",
    )
    schedule.set_total_steps(10)
    accelerator = Accelerator()
    network = FakeNetwork(FakeLoRAModule(7, 5, 4, schedule.current_alpha)).to(accelerator.device)
    raw_optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
    loss = (network.adapter.lora_up.weight @ network.adapter.lora_down.weight).square().mean()
    loss.backward()
    raw_optimizer.step()
    raw_optimizer.zero_grad(set_to_none=True)
    raw_scheduler = torch.optim.lr_scheduler.StepLR(raw_optimizer, step_size=2)
    optimizer, scheduler = make_replaceable_optimizer_scheduler(raw_optimizer, raw_scheduler)
    prepared_optimizer, prepared_scheduler = accelerator.prepare(optimizer, scheduler)
    optimizer_identity = id(prepared_optimizer)
    scheduler_identity = id(prepared_scheduler)

    controller = LoRASqueezeTrainingController(
        args=SimpleNamespace(),
        accelerator=accelerator,
        schedule=schedule,
        network=network,
        optimizer_util=FakeOptimizerUtil(torch.optim.AdamW, lr=0.01),
        optimizer_name="torch.optim.AdamW",
        optimizer_args="",
        optimizer=prepared_optimizer,
        optimizer_train_fn=lambda: None,
        optimizer_eval_fn=lambda: None,
        lr_scheduler=prepared_scheduler,
        lr_descriptions=None,
        prepare_optimizer_params_fn=lambda: (list(network.parameters()), None),
        prepare_grad_fn=lambda: None,
        update_metadata_fn=lambda: None,
        clean_memory_fn=lambda: None,
    )

    assert controller.run_if_due(schedule.squeeze_steps[0])
    assert id(controller.optimizer) == optimizer_identity
    assert id(controller.lr_scheduler) == scheduler_identity
    assert network.adapter.lora_dim == 2
    assert len(controller.optimizer.state) == 2
    assert find_replaceable_optimizer(controller.optimizer).optimizer is not raw_optimizer
