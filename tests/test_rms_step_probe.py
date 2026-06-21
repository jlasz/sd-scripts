import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.rms_step_probe import build_rms_probe_args, estimate_rms_adjusted_steps, validate_rms_probe_configuration


def make_args(**overrides):
    args = {
        "rms_probe_target": 0.0001,
        "rms_probe_steps": 500,
        "max_train_steps": 5000,
        "max_train_epochs": None,
        "resume": None,
        "initial_step": None,
        "initial_epoch": None,
        "target_total_rms": None,
        "deepspeed": False,
        "output_dir": "output",
        "output_name": "character",
        "save_every_n_steps": 100,
        "save_every_n_epochs": 1,
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
        "track_weight_values": True,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


class RMSStepProbeTest(unittest.TestCase):
    def test_estimate_scales_original_production_steps(self):
        adjusted_steps, multiplier = estimate_rms_adjusted_steps(5000, 0.0001, 0.00008)

        self.assertEqual(adjusted_steps, 6250)
        self.assertAlmostEqual(multiplier, 1.25)

    def test_estimate_can_make_fresh_production_run_shorter_than_probe(self):
        adjusted_steps, multiplier = estimate_rms_adjusted_steps(500, 0.0001, 0.001)

        self.assertEqual(adjusted_steps, 50)
        self.assertAlmostEqual(multiplier, 0.1)

    def test_zero_observed_rms_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero RMS"):
            estimate_rms_adjusted_steps(5000, 0.0001, 0.0)

    def test_probe_parameters_must_be_given_together(self):
        with self.assertRaisesRegex(ValueError, "must be specified together"):
            validate_rms_probe_configuration(make_args(rms_probe_steps=None))

    def test_probe_rejects_resume_and_rms_cutoff(self):
        with self.assertRaisesRegex(ValueError, "cannot be used with --resume"):
            validate_rms_probe_configuration(make_args(resume="state"))
        with self.assertRaisesRegex(ValueError, "cannot be combined with --target_total_rms"):
            validate_rms_probe_configuration(make_args(target_total_rms=0.0002))

    def test_probe_cannot_exceed_original_training_horizon(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed --max_train_steps"):
            validate_rms_probe_configuration(make_args(rms_probe_steps=5001))

    def test_probe_rejects_deepspeed(self):
        with self.assertRaisesRegex(ValueError, "does not support --deepspeed"):
            validate_rms_probe_configuration(make_args(deepspeed=True))

    def test_probe_keeps_production_horizon_but_stops_at_probe_step(self):
        probe_args = build_rms_probe_args(make_args())

        self.assertEqual(probe_args.max_train_steps, 5000)
        self.assertEqual(probe_args._training_step_limit, 500)
        self.assertEqual(probe_args.output_dir, os.path.join("output", "character-rms-probe-step500"))
        self.assertTrue(probe_args.save_state_on_train_end)
        self.assertIsNone(probe_args.huggingface_repo_id)
        self.assertIsNone(probe_args.logging_dir)
        self.assertFalse(probe_args.track_weight_values)


if __name__ == "__main__":
    unittest.main()
