import os
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.rms_step_probe import (
    build_rms_probe_args,
    estimate_rms_adjusted_steps,
    round_steps_to_nearest_multiple,
    validate_rms_probe_configuration,
)


def make_args(**overrides):
    values = {
        "rms_probe_target": 0.0001,
        "rms_probe_steps": 500,
        "rms_probe_adjusted_steps_divisible_by": None,
        "max_train_steps": 5000,
        "max_train_epochs": None,
        "resume": None,
        "initial_step": None,
        "initial_epoch": None,
        "deepspeed": False,
        "output_dir": "output",
        "output_name": "character",
        "save_every_n_steps": 100,
        "save_every_n_epochs": 1,
        "save_n_epoch_ratio": 4,
        "save_state": True,
        "save_state_on_train_end": False,
        "save_state_to_huggingface": True,
        "huggingface_repo_id": "owner/repo",
        "sample_every_n_steps": 100,
        "sample_every_n_epochs": 1,
        "sample_at_first": True,
        "max_validation_steps": None,
        "logging_dir": "logs",
        "log_with": "wandb",
        "total_rms_check_every_n_steps": 25,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class RMSStepProbeTest(unittest.TestCase):
    def test_estimate_scales_original_production_steps(self):
        adjusted_steps, multiplier = estimate_rms_adjusted_steps(5000, 0.0001, 0.00008)

        self.assertEqual(adjusted_steps, 6250)
        self.assertAlmostEqual(multiplier, 1.25)

    def test_estimate_rounds_to_nearest_step_and_never_returns_zero(self):
        self.assertEqual(estimate_rms_adjusted_steps(5, 1.0, 2.0)[0], 3)
        self.assertEqual(estimate_rms_adjusted_steps(5, 1.0, 100.0)[0], 1)

    def test_zero_observed_rms_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero RMS"):
            estimate_rms_adjusted_steps(5000, 0.0001, 0.0)

    def test_adjusted_steps_can_be_rounded_to_nearest_multiple(self):
        self.assertEqual(round_steps_to_nearest_multiple(4398, 5), 4400)
        self.assertEqual(round_steps_to_nearest_multiple(4397, 5), 4395)
        self.assertEqual(round_steps_to_nearest_multiple(10, 4), 12)
        self.assertEqual(round_steps_to_nearest_multiple(1, 5), 5)
        self.assertEqual(round_steps_to_nearest_multiple(4398, None), 4398)

    def test_adjusted_step_multiple_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            validate_rms_probe_configuration(make_args(rms_probe_adjusted_steps_divisible_by=0))

    def test_adjusted_step_multiple_requires_probe(self):
        with self.assertRaisesRegex(ValueError, "requires --rms_probe_target"):
            validate_rms_probe_configuration(
                make_args(
                    rms_probe_target=None,
                    rms_probe_steps=None,
                    rms_probe_adjusted_steps_divisible_by=5,
                )
            )

    def test_probe_parameters_must_be_given_together(self):
        with self.assertRaisesRegex(ValueError, "must be specified together"):
            validate_rms_probe_configuration(make_args(rms_probe_steps=None))

    def test_probe_rejects_resume_and_initial_position(self):
        with self.assertRaisesRegex(ValueError, "cannot be used with --resume"):
            validate_rms_probe_configuration(make_args(resume="state"))
        with self.assertRaisesRegex(ValueError, "cannot be used with --initial_step"):
            validate_rms_probe_configuration(make_args(initial_step=100))

    def test_probe_cannot_exceed_original_training_horizon(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed --max_train_steps"):
            validate_rms_probe_configuration(make_args(rms_probe_steps=5001))

    def test_probe_rejects_deepspeed(self):
        with self.assertRaisesRegex(ValueError, "does not support --deepspeed"):
            validate_rms_probe_configuration(make_args(deepspeed=True))

    def test_probe_keeps_scheduler_horizon_but_stops_at_probe_step(self):
        probe_args = build_rms_probe_args(make_args())

        self.assertEqual(probe_args.max_train_steps, 5000)
        self.assertEqual(probe_args._training_step_limit, 500)
        self.assertEqual(probe_args.output_dir, os.path.join("output", "character-rms-probe-step500"))
        self.assertTrue(probe_args.save_state_on_train_end)
        self.assertIsNone(probe_args.save_n_epoch_ratio)
        self.assertIsNone(probe_args.huggingface_repo_id)
        self.assertIsNone(probe_args.logging_dir)
        self.assertEqual(probe_args.total_rms_check_every_n_steps, 0)


if __name__ == "__main__":
    unittest.main()
