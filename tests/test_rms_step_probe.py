import os
import sys
import math
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.rms_step_probe import (
    build_rms_probe_args,
    choose_gradient_accumulation_steps,
    estimate_piecewise_energy_adjusted_steps,
    estimate_rms_adjusted_steps,
    fit_probe_energy_slope,
    predict_piecewise_energy_final_rms,
    round_steps_to_nearest_multiple,
    validate_rms_probe_configuration,
)


def make_args(**overrides):
    values = {
        "rms_probe_target": 0.0001,
        "rms_probe_steps": 500,
        "rms_probe_scaling_policy": "linear",
        "rms_probe_final_target": None,
        "rms_probe_curve_every_n_steps": 20,
        "rms_probe_adjusted_steps_divisible_by": None,
        "rms_probe_gradient_accumulation_target_microbatches": None,
        "rms_probe_min_gradient_accumulation_steps": 1,
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
        "gradient_accumulation_steps": 6,
        "lora_squeeze_start_dim": 36,
        "network_dim": 9,
        "lora_squeeze_num_squeezes": 4,
        "lora_squeeze_train_after_final_squeeze": True,
        "lora_squeeze_step_schedule": "equal",
        "lora_squeeze_rank_schedule": "geometric",
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

    def test_probe_energy_slope_uses_squared_rms(self):
        observed = 4e-5
        curve = [(step, observed * math.sqrt(1.0 + 0.002 * (step - 500))) for step in range(100, 501, 20)]

        self.assertAlmostEqual(fit_probe_energy_slope(curve), 2.0)

    def test_piecewise_energy_solver_hits_final_target(self):
        observed = 3.527385845165899e-5
        curve = [(step, observed * math.sqrt(1.0 + 0.002024 * (step - 500))) for step in range(100, 501, 20)]

        steps, multiplier, details = estimate_piecewise_energy_adjusted_steps(
            original_steps=4000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=observed,
            probe_steps=500,
            dataset_batches_per_epoch=250,
            rms_curve=curve,
        )

        self.assertEqual(steps, 6499)
        self.assertAlmostEqual(multiplier, steps / 4000)
        self.assertGreaterEqual(details["predicted_final_rms"], 8.425384599385171e-5)
        self.assertLess(
            predict_piecewise_energy_final_rms(steps - 1, 500, observed, 250, 2.024),
            8.425384599385171e-5,
        )

    def test_piecewise_energy_calibration_reproduces_held_out_mrissi_endpoint(self):
        predicted = predict_piecewise_energy_final_rms(
            total_steps=4000,
            probe_steps=500,
            observed_rms=3.8782791e-5,
            dataset_batches_per_epoch=132,
            probe_energy_slope=2.103,
        )

        self.assertAlmostEqual(predicted, 8.4042434e-5, delta=1e-11)

    def test_gradient_accumulation_compute_budget_thresholds(self):
        self.assertEqual(choose_gradient_accumulation_steps(4400, 25500, 6), 6)
        self.assertEqual(choose_gradient_accumulation_steps(5500, 25500, 6), 5)
        self.assertEqual(choose_gradient_accumulation_steps(6500, 25500, 6), 4)
        self.assertEqual(choose_gradient_accumulation_steps(9000, 25500, 6), 3)
        self.assertEqual(choose_gradient_accumulation_steps(6500, None, 6), 6)
        self.assertEqual(choose_gradient_accumulation_steps(2000, 25500, 6), 6)

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

    def test_piecewise_policy_requires_final_target_and_calibrated_schedule(self):
        with self.assertRaisesRegex(ValueError, "rms_probe_final_target"):
            validate_rms_probe_configuration(make_args(rms_probe_scaling_policy="piecewise_energy_v1"))
        with self.assertRaisesRegex(ValueError, "36->9"):
            validate_rms_probe_configuration(
                make_args(
                    rms_probe_scaling_policy="piecewise_energy_v1",
                    rms_probe_final_target=8e-5,
                    network_dim=16,
                )
            )

    def test_piecewise_policy_accepts_calibrated_configuration(self):
        self.assertTrue(
            validate_rms_probe_configuration(
                make_args(rms_probe_scaling_policy="piecewise_energy_v1", rms_probe_final_target=8e-5)
            )
        )

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

    def test_piecewise_probe_collects_the_configured_curve(self):
        probe_args = build_rms_probe_args(
            make_args(rms_probe_scaling_policy="piecewise_energy_v1", rms_probe_final_target=8e-5)
        )

        self.assertEqual(probe_args.total_rms_check_every_n_steps, 20)


if __name__ == "__main__":
    unittest.main()
