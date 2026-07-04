import argparse
import json
import math
from types import SimpleNamespace

import pytest

from library.args import add_lora_squeeze_arguments
from library.lora_squeeze_schedule import (
    get_remaining_training_steps,
    get_resume_epoch_and_batch_offset,
    is_training_budget_exhausted,
    restore_lora_squeeze_state_from_resume_dir,
    validate_lora_squeeze_network_args,
    validate_lora_squeeze_training_args,
)
from library.lora_squeeze_training import LoRASqueezeRuntime
from tests.lora_squeeze_test_utils import make_schedule, make_schedule_args


def test_arguments_are_registered_with_safe_defaults():
    parser = argparse.ArgumentParser()
    add_lora_squeeze_arguments(parser)
    args = parser.parse_args([])

    assert args.lora_squeeze_start_dim is None
    assert args.lora_squeeze_num_squeezes == 0
    assert args.lora_squeeze_optimizer_mode == "per_squeeze"
    assert args.lora_squeeze_scheduler_mode == "global"
    assert args.lora_squeeze_alpha_schedule == "proportional"
    assert args.lora_squeeze_first_segment_ratio == 1.0
    assert args.lora_squeeze_final_segment_ratio == 1.0


def test_legacy_namespace_keeps_lora_squeeze_disabled(monkeypatch):
    args = SimpleNamespace(network_dim=4, network_alpha=1)

    def fail_allocation(*args, **kwargs):
        raise AssertionError("must not allocate")

    monkeypatch.setattr("library.lora_squeeze_training.Value", fail_allocation)

    runtime = LoRASqueezeRuntime(args)

    assert not runtime.enabled
    assert runtime.absolute_update_step is None


@pytest.mark.parametrize(
    ("rank_schedule", "expected"),
    [
        ("geometric", [41, 28, 19, 13, 9]),
        ("linear", [41, 33, 25, 17, 9]),
    ],
)
def test_rank_schedules(rank_schedule, expected):
    schedule = make_schedule(lora_squeeze_rank_schedule=rank_schedule)
    assert schedule.ranks == expected


def test_geometric_schedule_remains_strict_in_a_tight_integer_range():
    schedule = make_schedule(lora_squeeze_start_dim=13, network_dim=9)
    assert schedule.ranks == [13, 12, 11, 10, 9]


@pytest.mark.parametrize(
    ("step_schedule", "expected_steps"),
    [
        ("equal", [750, 750, 750, 750]),
        ("rank_proportional", [1600, 800, 400, 200]),
    ],
)
def test_step_schedules_distribute_the_complete_budget(step_schedule, expected_steps):
    schedule = make_schedule(
        lora_squeeze_start_dim=16,
        network_dim=2,
        lora_squeeze_num_squeezes=3,
        lora_squeeze_step_schedule=step_schedule,
    )
    schedule.set_total_steps(3000)

    assert schedule.segment_steps == expected_steps
    assert sum(schedule.segment_steps) == 3000


def test_first_and_final_segment_ratios_can_be_combined():
    schedule = make_schedule(
        lora_squeeze_start_dim=16,
        network_dim=2,
        lora_squeeze_num_squeezes=3,
        lora_squeeze_first_segment_ratio=2.0,
        lora_squeeze_final_segment_ratio=3.0,
    )
    schedule.set_total_steps(700)

    assert schedule.segment_weights == [2.0, 1.0, 1.0, 3.0]
    assert schedule.segment_steps == [200, 100, 100, 300]


def test_terminal_boundary_uses_the_exact_budget_with_fractional_weights():
    schedule = make_schedule(
        lora_squeeze_start_dim=5,
        network_dim=1,
        network_alpha=1,
        lora_squeeze_num_squeezes=4,
        lora_squeeze_rank_schedule="linear",
        lora_squeeze_step_schedule="sqrt_rank_proportional",
        lora_squeeze_train_after_final_squeeze=False,
    )
    schedule.set_total_steps(257)

    assert schedule.squeeze_steps[-1] == 257
    assert sum(schedule.segment_steps) == 257


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"lora_squeeze_rank_schedule": "quadratic"}, "rank_schedule must be one of"),
        ({"lora_squeeze_alpha_schedule": "quadratic"}, "alpha_schedule must be one of"),
        ({"network_alpha": math.inf}, "network_alpha must be finite"),
        ({"lora_squeeze_first_segment_ratio": 0}, "first_segment_ratio must be greater than 0"),
        ({"lora_squeeze_final_segment_ratio": math.nan}, "final_segment_ratio must be finite"),
    ],
)
def test_invalid_schedule_configuration_is_rejected(override, message):
    with pytest.raises(ValueError, match=message):
        make_schedule(**override)


@pytest.mark.parametrize("option", ["torch_compile", "compile"])
def test_compile_modes_are_rejected(option):
    args = SimpleNamespace(torch_compile=False, compile=False)
    setattr(args, option, True)
    with pytest.raises(ValueError, match="torch.compile"):
        validate_lora_squeeze_training_args(args)


def test_heterogeneous_network_args_are_rejected_but_c3lier_is_allowed():
    validate_lora_squeeze_network_args({"conv_dim": "4", "conv_alpha": "1"})
    with pytest.raises(ValueError, match="network_reg_dims"):
        validate_lora_squeeze_network_args({"network_reg_dims": ".*=4"})


def test_c3lier_args_and_metadata_follow_the_current_squeeze_stage():
    runtime = LoRASqueezeRuntime(
        make_schedule_args(
            lora_squeeze_start_dim=8,
            network_dim=4,
            network_alpha=2,
            lora_squeeze_num_squeezes=1,
        )
    )
    network_args = runtime.prepare_network_args({"conv_dim": "4", "conv_alpha": "2"})
    metadata = {"ss_network_args": json.dumps(network_args)}

    assert network_args["conv_dim"] == "8"
    assert float(network_args["conv_alpha"]) == pytest.approx(runtime.current_alpha)

    runtime.schedule.mark_squeezed(4, 2.0)
    runtime.update_metadata(metadata, {})
    saved_args = json.loads(metadata["ss_network_args"])
    assert saved_args["conv_dim"] == "4"
    assert saved_args["conv_alpha"] == "2.0"


def test_schedule_state_round_trip_restores_progress():
    schedule = make_schedule(lora_squeeze_start_dim=16, network_dim=4, lora_squeeze_num_squeezes=2)
    schedule.set_total_steps(120)
    schedule.mark_squeezed(schedule.ranks[1], schedule.alpha_for_rank(schedule.ranks[1]))

    restored = make_schedule(lora_squeeze_start_dim=16, network_dim=4, lora_squeeze_num_squeezes=2)
    restored.load_state_dict(schedule.state_dict())

    assert restored.completed_squeezes == 1
    assert restored.current_dim == schedule.ranks[1]
    assert restored.current_alpha == pytest.approx(schedule.alpha_for_rank(schedule.ranks[1]))
    assert restored.squeeze_steps == schedule.squeeze_steps


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda state: state.pop("next_squeeze_index"), "missing next_squeeze_index"),
        (lambda state: state.__setitem__("version", 2), "configuration for version"),
        (lambda state: state.__setitem__("current_alpha", math.inf), "current_alpha must be finite"),
    ],
)
def test_invalid_resume_state_is_rejected(mutation, message):
    schedule = make_schedule(lora_squeeze_start_dim=16, network_dim=4, lora_squeeze_num_squeezes=2)
    schedule.set_total_steps(120)
    state = schedule.state_dict()
    mutation(state)

    restored = make_schedule(lora_squeeze_start_dim=16, network_dim=4, lora_squeeze_num_squeezes=2)
    with pytest.raises(ValueError, match=message):
        restored.load_state_dict(state)


def test_resume_directory_restores_current_rank(tmp_path):
    schedule = make_schedule(lora_squeeze_start_dim=16, network_dim=4, lora_squeeze_num_squeezes=2)
    schedule.set_total_steps(120)
    schedule.mark_squeezed(schedule.ranks[1], schedule.alpha_for_rank(schedule.ranks[1]))
    (tmp_path / "train_state.json").write_text(
        json.dumps({"current_epoch": 0, "current_step": 40, "lora_squeeze": schedule.state_dict()}),
        encoding="utf-8",
    )

    restored = make_schedule(lora_squeeze_start_dim=16, network_dim=4, lora_squeeze_num_squeezes=2)
    restore_lora_squeeze_state_from_resume_dir(restored, str(tmp_path))

    assert restored.completed_squeezes == 1
    assert restored.current_dim == schedule.ranks[1]


def test_resume_budget_helpers_use_absolute_optimizer_steps():
    assert get_remaining_training_steps(11, completed_steps=4) == 7
    assert not is_training_budget_exhausted(11, completed_steps=10)
    assert is_training_budget_exhausted(11, completed_steps=11)
    assert get_resume_epoch_and_batch_offset(2, 5, 2) == (0, 4)
    assert get_resume_epoch_and_batch_offset(3, 5, 2) == (1, 0)
