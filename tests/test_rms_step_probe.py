import os
import sys
import math
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.rms_step_probe import (
    COMPRESSED_RMS_STAGE_FACTORS,
    ENERGY_SQUEEZE_RETENTION,
    build_rms_probe_args,
    choose_adjusted_probe_training_steps,
    choose_gradient_accumulation_steps,
    combine_observed_later_mean_rms_velocities,
    combine_observed_squeeze_retentions,
    count_optimizer_step_microbatches,
    equal_squeeze_steps,
    epoch_accumulation_group_sizes,
    estimate_augmented_later_mean_rms_velocity,
    estimate_loglinear_steps,
    estimate_loglinear_training_plan,
    estimate_piecewise_energy_adjusted_steps,
    estimate_piecewise_training_plan,
    estimate_rms_adjusted_steps,
    fit_adjusted_probe_rank36_transfer,
    fit_observed_later_mean_rms_velocity,
    fit_observed_squeeze_retention_scale,
    fit_probe_energy_slope,
    fit_schedule_aware_probe_energy_model,
    optimizer_steps_for_microbatch_budget,
    probe_schedule_needs_adjusted_probe,
    predict_loglinear_final_rms,
    predict_piecewise_energy_final_rms,
    rescale_observed_later_mean_rms_velocity,
    rescale_probe_measurements_for_gradient_accumulation,
    round_steps_to_nearest_multiple,
    should_run_adjusted_probe,
    squeeze_schedule_steps,
    squeeze_segment_steps,
    update_piecewise_production_prediction,
    validate_rms_probe_configuration,
)
from library.lora_squeeze_schedule import LoRASqueezeSchedule


def make_args(**overrides):
    values = {
        "rms_probe_target": 0.0001,
        "rms_probe_steps": 500,
        "rms_probe_scaling_policy": "linear",
        "rms_probe_final_target": None,
        "rms_probe_curve_every_n_steps": 20,
        "rms_probe_adjusted_steps_divisible_by": None,
        "rms_probe_force_adjusted_probe": False,
        "rms_probe_gradient_accumulation_target_microbatches": None,
        "rms_probe_gradient_accumulation_rounding_bias": 0.6,
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
        "network_alpha": 3,
        "lora_squeeze_num_squeezes": 4,
        "lora_squeeze_train_after_final_squeeze": True,
        "lora_squeeze_step_schedule": "equal",
        "lora_squeeze_rank_schedule": "geometric",
        "lora_squeeze_first_segment_ratio": 1.0,
        "lora_squeeze_final_segment_ratio": 1.0,
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

    def test_short_probe_projects_rank36_energy_to_step_500_reference(self):
        reference_rms = 4e-5
        reference_energy = reference_rms**2
        reference_slope = 2.0
        curve = []
        for step in (*range(20, 141, 20), 150):
            energy = reference_energy * (
                1.0 + reference_slope * (step - 500) / 1000.0
            )
            curve.append((step, math.sqrt(energy)))

        projected_rms, projected_slope, details = (
            fit_schedule_aware_probe_energy_model(
                curve,
                total_steps=4000,
                probe_steps=500,
                curve_interval=20,
            )
        )

        self.assertAlmostEqual(projected_rms, reference_rms)
        self.assertAlmostEqual(projected_slope, reference_slope)
        self.assertEqual(details["probe_observation_steps"], 150.0)
        self.assertEqual(details["calibration_reference_steps"], 500.0)
        self.assertEqual(details["reference_rms_is_extrapolated"], 1.0)

    def test_probe_measurements_rescale_with_calibrated_accumulation_elasticity(self):
        adjusted_rms, adjusted_slope, details = (
            rescale_probe_measurements_for_gradient_accumulation(
                observed_rms=4.19757912778963e-5,
                probe_energy_slope=2.1832932595622987,
                probe_gradient_accumulation_steps=6,
                production_gradient_accumulation_steps=9,
            )
        )

        self.assertAlmostEqual(adjusted_rms, 4.43208512565915e-5)
        self.assertAlmostEqual(adjusted_slope, 2.208833613089547)
        self.assertEqual(details["probe_gradient_accumulation_ratio"], 1.5)

    def test_probe_measurement_rescaling_is_reversible(self):
        observed_rms = 3.804286006720449e-5
        observed_slope = 2.0634011093724816
        adjusted_rms, adjusted_slope, _ = (
            rescale_probe_measurements_for_gradient_accumulation(
                observed_rms,
                observed_slope,
                6,
                5,
            )
        )
        restored_rms, restored_slope, _ = (
            rescale_probe_measurements_for_gradient_accumulation(
                adjusted_rms,
                adjusted_slope,
                5,
                6,
            )
        )

        self.assertAlmostEqual(restored_rms, observed_rms)
        self.assertAlmostEqual(restored_slope, observed_slope)

    def test_probe_measurement_rescaling_improves_calibration_pairs(self):
        # Same-GA reruns are represented by their geometric means, matching the
        # calibration fit. The final wide-range Rosine pair differs by one image.
        pairs = (
            (6, 3.54396179318428e-5, 2.038203269953988, 5, 3.469298826530576e-5, 2.006614663143924),
            (6, 3.634618288860564e-5, 2.059958669045801, 5, 3.633632877608761e-5, 2.071582634208073),
            (6, 3.804286006720449e-5, 2.0634011093724816, 5, 3.7372468152551915e-5, 2.0134947818470312),
            (6, 3.6507251192282416e-5, 1.996520566790334, 5, 3.585959711926989e-5, 1.966833287803121),
            (6, 4.31827932468486e-5, 2.1156507756813077, 8, 4.517141292410292e-5, 2.1815015111656755),
            (6, 4.19757912778963e-5, 2.1832932595622987, 9, 4.448729100949966e-5, 2.232738325480896),
            (7, 4.329497199592824e-5, 2.134595960542286, 2, 3.648383790277876e-5, 2.085742066127855),
        )
        raw_rms_errors = []
        adjusted_rms_errors = []
        raw_slope_errors = []
        adjusted_slope_errors = []
        for probe_ga, probe_rms, probe_slope, production_ga, actual_rms, actual_slope in pairs:
            adjusted_rms, adjusted_slope, _ = (
                rescale_probe_measurements_for_gradient_accumulation(
                    probe_rms,
                    probe_slope,
                    probe_ga,
                    production_ga,
                )
            )
            raw_rms_errors.append(abs(probe_rms / actual_rms - 1.0))
            adjusted_rms_errors.append(abs(adjusted_rms / actual_rms - 1.0))
            raw_slope_errors.append(abs(probe_slope / actual_slope - 1.0))
            adjusted_slope_errors.append(abs(adjusted_slope / actual_slope - 1.0))

        mean = lambda values: sum(values) / len(values)
        self.assertAlmostEqual(mean(raw_rms_errors), 0.049279778527764516)
        self.assertAlmostEqual(mean(adjusted_rms_errors), 0.007654149229480108)
        self.assertAlmostEqual(mean(raw_slope_errors), 0.01956964287055209)
        self.assertAlmostEqual(mean(adjusted_slope_errors), 0.013719752333265707)

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
            gradient_accumulation_steps=6,
        )

        self.assertEqual(steps, 5711)
        self.assertAlmostEqual(multiplier, steps / 4000)
        self.assertEqual(
            details["later_mean_rms_velocity_gradient_accumulation_steps"], 6.0
        )
        self.assertEqual(
            details["later_mean_rms_velocity_uses_augmented_regression"], 1.0
        )
        self.assertEqual(details["trajectory_model_version"], 2.9)
        self.assertEqual(details["later_energy_exponent"], 0.5)
        self.assertGreaterEqual(details["predicted_final_rms"], 8.425384599385171e-5)
        self.assertLess(
            predict_piecewise_energy_final_rms(steps - 1, 500, observed, 250, 2.024, 6),
            8.425384599385171e-5,
        )

    def test_piecewise_energy_solver_supports_a_squeeze_before_probe_end(self):
        observed = 5.3288192e-5
        curve = [
            (step, observed * math.sqrt(1.0 + 0.002 * (step - 500)))
            for step in range(100, 501, 20)
        ]

        steps, _, _ = estimate_piecewise_energy_adjusted_steps(
            original_steps=4000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=observed,
            probe_steps=500,
            dataset_batches_per_epoch=22,
            rms_curve=curve,
            gradient_accumulation_steps=5,
        )

        self.assertEqual(steps, 2043)
        self.assertTrue(probe_schedule_needs_adjusted_probe(steps, 500))

    def test_piecewise_energy_solver_can_extrapolate_below_calibration_reference(self):
        observed = 4e-5
        target_steps = 350
        target = predict_piecewise_energy_final_rms(
            total_steps=target_steps,
            probe_steps=500,
            observed_rms=observed,
            dataset_batches_per_epoch=50,
            probe_energy_slope=2.0,
            gradient_accumulation_steps=20,
            later_mean_rms_velocity=1.0,
        )

        steps, _, _ = estimate_piecewise_energy_adjusted_steps(
            original_steps=4000,
            final_target_rms=target,
            observed_rms=observed,
            probe_steps=500,
            dataset_batches_per_epoch=50,
            rms_curve=[],
            gradient_accumulation_steps=20,
            probe_energy_slope=2.0,
            later_mean_rms_velocity=1.0,
        )

        self.assertEqual(steps, target_steps)

    def test_out_of_range_augmented_estimate_selects_adjusted_probe_settings(self):
        curve = [
            (20, 7.7112147e-6),
            (40, 1.0898868e-5),
            (60, 1.3565062e-5),
            (80, 1.6082649e-5),
            (100, 1.87522e-5),
            (120, 2.0835095e-5),
            (140, 2.2580525e-5),
            (160, 2.4658147e-5),
            (180, 2.6919618e-5),
            (200, 2.8844781e-5),
            (220, 3.0423066e-5),
            (240, 3.245155e-5),
            (260, 3.4024333e-5),
            (280, 3.5367862e-5),
            (300, 3.7224761e-5),
            (320, 3.8830702e-5),
            (340, 4.0604795e-5),
            (360, 4.2479001e-5),
            (380, 4.4004631e-5),
            (400, 4.5833497e-5),
            (420, 4.7086027e-5),
            (440, 4.8636081e-5),
            (460, 5.0261792e-5),
            (480, 5.1694753e-5),
            (500, 5.3288192e-5),
        ]

        steps, _, details = estimate_piecewise_energy_adjusted_steps(
            original_steps=4000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=curve[-1][1],
            probe_steps=500,
            dataset_batches_per_epoch=50,
            rms_curve=curve,
            gradient_accumulation_steps=20,
        )
        rounded_steps = round_steps_to_nearest_multiple(steps, 100)
        uncapped_accumulation = choose_gradient_accumulation_steps(
            rounded_steps, 24000, 6, rounding_bias=0.7
        )

        self.assertEqual(steps, 1939)
        self.assertAlmostEqual(details["probe_energy_slope_per_1000_steps"], 2.2019441770164145)
        self.assertEqual(
            details["later_mean_rms_velocity_uses_augmented_regression"], 1.0
        )
        self.assertEqual(rounded_steps, 1900)
        self.assertEqual(uncapped_accumulation, 13)
        self.assertEqual(min(uncapped_accumulation, 50), 13)
        self.assertEqual(equal_squeeze_steps(rounded_steps), (380, 760, 1140, 1520))

    def test_beastgirl_adjusted_probe_uses_rounded_budget_and_phase_coverage(self):
        probe_steps, details = choose_adjusted_probe_training_steps(
            reference_probe_steps=500,
            original_gradient_accumulation_steps=6,
            adjusted_gradient_accumulation_steps=20,
            production_steps=700,
            curve_interval=20,
        )

        self.assertEqual(probe_steps, 240)
        self.assertEqual(details["adjusted_probe_nominal_microbatch_budget"], 3000.0)
        self.assertEqual(details["adjusted_probe_budget_based_steps"], 150.0)
        self.assertEqual(details["adjusted_probe_rounded_budget_steps"], 160.0)
        self.assertEqual(details["adjusted_probe_first_squeeze_step"], 140.0)
        self.assertEqual(details["adjusted_probe_phase_coverage_steps"], 240.0)
        self.assertEqual(details["adjusted_probe_rounded_phase_coverage_steps"], 240.0)
        self.assertEqual(details["adjusted_probe_nominal_training_microbatches"], 4800.0)

    def test_adjusted_probe_is_capped_by_sub_reference_production_horizon(self):
        probe_steps, details = choose_adjusted_probe_training_steps(
            reference_probe_steps=500,
            original_gradient_accumulation_steps=6,
            adjusted_gradient_accumulation_steps=20,
            production_steps=120,
            curve_interval=20,
        )

        self.assertEqual(probe_steps, 120)
        self.assertEqual(details["adjusted_probe_training_steps"], 120.0)

    def test_adjusted_probe_rounds_unaligned_phase_coverage_upward(self):
        probe_steps, details = choose_adjusted_probe_training_steps(
            reference_probe_steps=500,
            original_gradient_accumulation_steps=6,
            adjusted_gradient_accumulation_steps=100,
            production_steps=706,
            curve_interval=20,
        )

        self.assertEqual(details["adjusted_probe_first_squeeze_step"], 141.0)
        self.assertEqual(details["adjusted_probe_phase_coverage_steps"], 241.0)
        self.assertEqual(details["adjusted_probe_rounded_phase_coverage_steps"], 260.0)
        self.assertEqual(probe_steps, 260)

    def test_adjusted_probe_uses_ratio_adjusted_first_squeeze(self):
        probe_steps, details = choose_adjusted_probe_training_steps(
            reference_probe_steps=500,
            original_gradient_accumulation_steps=6,
            adjusted_gradient_accumulation_steps=20,
            production_steps=700,
            curve_interval=20,
            first_segment_ratio=2.0,
            final_segment_ratio=1.0,
        )

        self.assertEqual(details["adjusted_probe_first_squeeze_step"], 233.0)
        self.assertEqual(details["adjusted_probe_phase_coverage_steps"], 333.0)
        self.assertEqual(probe_steps, 340)

    def test_adjusted_probe_budget_counts_epoch_remainder_updates(self):
        probe_steps, details = choose_adjusted_probe_training_steps(
            reference_probe_steps=2,
            original_gradient_accumulation_steps=18,
            adjusted_gradient_accumulation_steps=11,
            production_steps=100,
            curve_interval=1,
            minimum_post_squeeze_samples=1,
            dataset_batches_per_epoch=22,
        )

        self.assertEqual(details["adjusted_probe_nominal_microbatch_budget"], 36.0)
        self.assertEqual(details["adjusted_probe_actual_microbatch_budget"], 22.0)
        self.assertEqual(details["adjusted_probe_budget_based_steps"], 2.0)
        self.assertEqual(probe_steps, 2)
        self.assertEqual(details["adjusted_probe_actual_training_microbatches"], 22.0)

    def test_shortened_adjusted_probe_can_fit_pre_and_post_squeeze_growth(self):
        production_steps = 700
        observation_steps = 240
        reference_rms = 4e-5
        reference_energy = reference_rms**2
        probe_slope = 2.0
        later_base_velocity = 1.0
        first_squeeze = equal_squeeze_steps(production_steps)[0]
        energy_at_squeeze = reference_energy * (
            1.0 + probe_slope * (first_squeeze - 500) / 1000.0
        )

        curve = []
        for step in range(20, observation_steps + 1, 20):
            if step < first_squeeze:
                energy = reference_energy * (
                    1.0 + probe_slope * (step - 500) / 1000.0
                )
                rms = math.sqrt(energy)
            else:
                rms = math.sqrt(energy_at_squeeze * ENERGY_SQUEEZE_RETENTION[0])
                rms += (
                    reference_rms
                    * later_base_velocity
                    * COMPRESSED_RMS_STAGE_FACTORS[0]
                    * (step - first_squeeze)
                    / 1000.0
                )
            curve.append((step, rms))

        fitted_rms, fitted_probe_slope, _ = fit_schedule_aware_probe_energy_model(
            curve, production_steps, 500, 20
        )
        fitted_later_velocity, details = fit_observed_later_mean_rms_velocity(
            curve, production_steps, observation_steps, fitted_rms, 20
        )

        self.assertAlmostEqual(fitted_rms, reference_rms)
        self.assertAlmostEqual(fitted_probe_slope, probe_slope)
        self.assertAlmostEqual(fitted_later_velocity, later_base_velocity)
        self.assertEqual(details["observed_later_stage_count"], 1.0)

    def test_adjusted_probe_transfers_relative_rank36_scale_without_short_extrapolation(self):
        primary_curve = (
            (20, 1.0e-5),
            (40, 1.4e-5),
            (60, 1.8e-5),
            (80, 2.2e-5),
            (100, 2.6e-5),
        )
        adjusted_curve = tuple(
            (step, rms * 1.1)
            for step, rms in primary_curve
        )

        reference_rms, energy_slope, details = (
            fit_adjusted_probe_rank36_transfer(
                primary_curve,
                adjusted_curve,
                adjusted_total_steps=400,
                adjusted_probe_steps=100,
                primary_reference_rms=5.0e-5,
                primary_probe_energy_slope=2.0,
            )
        )

        self.assertAlmostEqual(reference_rms, 5.5e-5)
        self.assertAlmostEqual(energy_slope, 2.0)
        self.assertAlmostEqual(details["rank36_transfer_energy_scale"], 1.21)
        self.assertEqual(details["rank36_transfer_sample_count"], 3.0)

    def test_rank36_transfer_excludes_samples_after_either_probe_squeezes(self):
        primary_curve = []
        adjusted_curve = []
        for step in range(20, 501, 20):
            rank36_rms = 1.0e-5 + step * 1.0e-7
            primary_curve.append(
                (step, rank36_rms if step < 320 else rank36_rms * 10.0)
            )
            adjusted_curve.append((step, rank36_rms * 1.1))

        reference_rms, energy_slope, details = (
            fit_adjusted_probe_rank36_transfer(
                primary_curve,
                adjusted_curve,
                adjusted_total_steps=4000,
                adjusted_probe_steps=500,
                primary_reference_rms=5.0e-5,
                primary_probe_energy_slope=2.0,
                primary_total_steps=1600,
            )
        )

        self.assertAlmostEqual(reference_rms, 5.5e-5)
        self.assertAlmostEqual(energy_slope, 2.0)
        self.assertEqual(details["rank36_transfer_last_sample_step"], 300.0)
        self.assertEqual(
            details["rank36_transfer_primary_first_squeeze_step"], 320.0
        )
        self.assertEqual(
            details["rank36_transfer_adjusted_first_squeeze_step"], 800.0
        )

    def test_schedule_aware_probe_fits_each_observed_rank_segment_separately(self):
        reference_rms = 4e-5
        reference_energy = reference_rms**2
        later_base_velocity = 1.0
        squeeze_steps = equal_squeeze_steps(1000)

        def rms_at(step):
            rms = math.sqrt(
                reference_energy
                * (1.0 + 2.0 * (squeeze_steps[0] - 500) / 1000.0)
            )
            segment_start = squeeze_steps[0]
            for stage_index, segment_end in enumerate((*squeeze_steps[1:], 1000)):
                rms *= math.sqrt(
                    ENERGY_SQUEEZE_RETENTION[stage_index]
                )
                observed_end = min(step, segment_end)
                if observed_end > segment_start:
                    rms += (
                        reference_rms
                        * later_base_velocity
                        * COMPRESSED_RMS_STAGE_FACTORS[stage_index]
                        * (observed_end - segment_start)
                        / 1000.0
                    )
                if step <= segment_end:
                    return rms
                segment_start = segment_end
            return rms

        curve = []
        for step in range(20, 501, 20):
            if step < squeeze_steps[0]:
                energy = reference_energy * (1.0 + 2.0 * (step - 500) / 1000.0)
                rms = math.sqrt(energy)
            else:
                rms = rms_at(step)
            curve.append((step, rms))

        fitted_rms, fitted_probe_slope, _ = fit_schedule_aware_probe_energy_model(
            curve, 1000, 500, 20
        )
        fitted_later_velocity, details = fit_observed_later_mean_rms_velocity(
            curve, 1000, 500, fitted_rms, 20
        )

        self.assertAlmostEqual(fitted_rms, reference_rms)
        self.assertAlmostEqual(fitted_probe_slope, 2.0)
        self.assertAlmostEqual(fitted_later_velocity, later_base_velocity)
        self.assertEqual(details["observed_later_stage_count"], 2.0)

    def test_compressed_rms_velocity_is_independent_of_entry_rms(self):
        reference_rms = 4e-5
        base_velocity = 0.5
        first_squeeze = equal_squeeze_steps(1000)[0]
        stage_factor = COMPRESSED_RMS_STAGE_FACTORS[0]

        def make_curve(entry_rms):
            return [
                (
                    step,
                    entry_rms
                    + reference_rms
                    * base_velocity
                    * stage_factor
                    * (step - first_squeeze)
                    / 1000.0,
                )
                for step in range(220, 381, 20)
            ]

        low_curve = make_curve(3e-5)
        high_curve = make_curve(7e-5)
        low_velocity, _ = fit_observed_later_mean_rms_velocity(
            low_curve, 1000, 380, reference_rms, 20
        )
        high_velocity, _ = fit_observed_later_mean_rms_velocity(
            high_curve, 1000, 380, reference_rms, 20
        )

        self.assertAlmostEqual(low_velocity, base_velocity)
        self.assertAlmostEqual(high_velocity, base_velocity)
        low_energy_slope = (
            (low_curve[-1][1] / reference_rms) ** 2
            - (low_curve[0][1] / reference_rms) ** 2
        ) / (low_curve[-1][0] - low_curve[0][0]) * 1000.0
        high_energy_slope = (
            (high_curve[-1][1] / reference_rms) ** 2
            - (high_curve[0][1] / reference_rms) ** 2
        ) / (high_curve[-1][0] - high_curve[0][0]) * 1000.0
        self.assertNotAlmostEqual(low_energy_slope, high_energy_slope)

    def test_observed_compressed_velocities_are_combined_by_usable_span(self):
        combined, details = combine_observed_later_mean_rms_velocities(
            (
                (0.4, {"observed_later_stage_span": 100.0}),
                (0.8, {"observed_later_stage_span": 300.0}),
            )
        )

        self.assertAlmostEqual(combined, 0.7)
        self.assertEqual(details["combined_observed_later_probe_count"], 2.0)
        self.assertEqual(details["combined_observed_later_stage_span"], 400.0)

    def test_observed_compressed_velocity_can_be_mapped_between_probe_conditions(self):
        source_modeled = estimate_augmented_later_mean_rms_velocity(
            28, 4.0e-5, 7, 2.0
        )
        target_modeled = estimate_augmented_later_mean_rms_velocity(
            28, 4.5e-5, 14, 2.1
        )
        rescaled, details = rescale_observed_later_mean_rms_velocity(
            0.5,
            28,
            4.0e-5,
            7,
            2.0,
            4.5e-5,
            14,
            2.1,
        )

        self.assertAlmostEqual(rescaled, 0.5 * target_modeled / source_modeled)
        self.assertAlmostEqual(
            details["observed_later_velocity_condition_scale"],
            target_modeled / source_modeled,
        )

    def test_squeeze_retentions_are_combined_by_matching_stage(self):
        combined, details = combine_observed_squeeze_retentions(
            ((0.90, 0.80), (0.95,))
        )

        self.assertEqual(combined, (0.925, 0.80))
        self.assertEqual(details["combined_squeeze_1_probe_count"], 2.0)
        self.assertEqual(details["combined_squeeze_2_probe_count"], 1.0)

    def test_missing_squeeze_retention_does_not_shift_later_stage(self):
        combined, _ = combine_observed_squeeze_retentions(
            ((None, 0.80),)
        )
        _, details = fit_observed_squeeze_retention_scale(combined)

        self.assertEqual(combined, (None, 0.80))
        self.assertNotIn("observed_squeeze_1_retained_energy_mean", details)
        self.assertEqual(details["observed_squeeze_2_retained_energy_mean"], 0.80)

    def test_adjusted_probe_trigger_uses_exact_floor_boundaries(self):
        self.assertTrue(probe_schedule_needs_adjusted_probe(2504, 500))
        self.assertFalse(probe_schedule_needs_adjusted_probe(2505, 500))
        self.assertFalse(
            probe_schedule_needs_adjusted_probe(
                1600,
                500,
                first_segment_ratio=2.0,
                final_segment_ratio=1.0,
            )
        )

    def test_adjusted_probe_can_be_forced_outside_probe_window(self):
        self.assertFalse(
            should_run_adjusted_probe("piecewise_energy_v1", 4000, 500)
        )
        self.assertTrue(
            should_run_adjusted_probe(
                "piecewise_energy_v1",
                4000,
                500,
                force_adjusted_probe=True,
            )
        )
        forced_probe_steps, details = choose_adjusted_probe_training_steps(
            reference_probe_steps=500,
            original_gradient_accumulation_steps=6,
            adjusted_gradient_accumulation_steps=6,
            production_steps=4000,
            curve_interval=20,
        )
        self.assertEqual(forced_probe_steps, 500)
        self.assertEqual(details["adjusted_probe_first_squeeze_step"], 800.0)
        self.assertTrue(
            should_run_adjusted_probe("piecewise_energy_v1", 2500, 500)
        )
        self.assertFalse(
            should_run_adjusted_probe(
                "linear",
                4000,
                500,
                force_adjusted_probe=True,
            )
        )

    def test_primary_compressed_evidence_avoids_redundant_adjusted_probe(self):
        self.assertFalse(
            should_run_adjusted_probe(
                "piecewise_energy_v1",
                1600,
                500,
                has_observed_compressed_stage_evidence=True,
            )
        )
        self.assertTrue(
            should_run_adjusted_probe(
                "piecewise_energy_v1",
                1600,
                500,
                force_adjusted_probe=True,
                has_observed_compressed_stage_evidence=True,
            )
        )

    def test_ratio_adjusted_schedule_matches_lora_squeeze_weighting(self):
        self.assertEqual(
            squeeze_schedule_steps(
                4000,
                first_segment_ratio=0.5,
                final_segment_ratio=2.0,
            ),
            (363, 1090, 1818, 2545),
        )
        self.assertEqual(
            squeeze_segment_steps(
                4000,
                first_segment_ratio=0.5,
                final_segment_ratio=2.0,
            ),
            (363, 727, 728, 727, 1455),
        )

    def test_ratio_adjusted_boundaries_match_lora_squeeze_schedule(self):
        for total_steps, first_ratio, final_ratio in (
            (503, 1.0, 1.0),
            (4000, 0.75, 1.5),
            (5379, 2.0, 0.5),
        ):
            args = make_args(
                max_train_steps=total_steps,
                lora_squeeze_first_segment_ratio=first_ratio,
                lora_squeeze_final_segment_ratio=final_ratio,
            )
            schedule = LoRASqueezeSchedule(args)
            schedule.set_total_steps(total_steps)

            self.assertEqual(
                squeeze_schedule_steps(
                    total_steps,
                    first_ratio,
                    final_ratio,
                ),
                tuple(schedule.squeeze_steps),
            )
            self.assertEqual(
                squeeze_segment_steps(
                    total_steps,
                    first_ratio,
                    final_ratio,
                ),
                tuple(schedule.segment_steps),
            )

    def test_ratio_adjusted_prediction_uses_exact_integer_segments(self):
        observed = 4e-5
        probe_slope = 2.0
        later_velocity = 1.0
        segment_steps = squeeze_segment_steps(
            503,
            first_segment_ratio=2.0,
            final_segment_ratio=1.0,
        )
        rms = math.sqrt(observed**2 * (
            1.0 + probe_slope * (segment_steps[0] - 500) / 1000.0
        ))
        for stage_factor, retention, stage_steps in zip(
            COMPRESSED_RMS_STAGE_FACTORS,
            ENERGY_SQUEEZE_RETENTION,
            segment_steps[1:],
        ):
            rms *= math.sqrt(retention)
            rms += (
                observed
                * later_velocity
                * stage_factor
                * stage_steps
                / 1000.0
            )

        predicted = predict_piecewise_energy_final_rms(
            total_steps=503,
            probe_steps=500,
            observed_rms=observed,
            dataset_batches_per_epoch=200,
            probe_energy_slope=probe_slope,
            gradient_accumulation_steps=5,
            later_mean_rms_velocity=later_velocity,
            first_segment_ratio=2.0,
            final_segment_ratio=1.0,
        )

        self.assertAlmostEqual(predicted, rms)

    def test_ratio_adjusted_solver_hits_the_exact_schedule_target(self):
        observed = 4e-5
        target_steps = 4379
        target = predict_piecewise_energy_final_rms(
            total_steps=target_steps,
            probe_steps=500,
            observed_rms=observed,
            dataset_batches_per_epoch=200,
            probe_energy_slope=2.0,
            gradient_accumulation_steps=5,
            later_mean_rms_velocity=1.0,
            first_segment_ratio=2.0,
            final_segment_ratio=0.75,
        )

        steps, _, details = estimate_piecewise_energy_adjusted_steps(
            original_steps=4000,
            final_target_rms=target,
            observed_rms=observed,
            probe_steps=500,
            dataset_batches_per_epoch=200,
            rms_curve=[],
            gradient_accumulation_steps=5,
            probe_energy_slope=2.0,
            later_mean_rms_velocity=1.0,
            first_segment_ratio=2.0,
            final_segment_ratio=0.75,
        )

        self.assertEqual(steps, target_steps)
        self.assertAlmostEqual(details["predicted_final_rms"], target)
        self.assertEqual(
            tuple(details[f"segment_{index}_steps"] for index in range(1, 6)),
            tuple(
                float(value)
                for value in squeeze_segment_steps(
                    target_steps,
                    first_segment_ratio=2.0,
                    final_segment_ratio=0.75,
                )
            ),
        )

    def test_production_prediction_is_recomputed_after_step_rounding(self):
        observed = 3.527385845165899e-5
        curve = [
            (
                step,
                observed * math.sqrt(1.0 + 0.002024 * (step - 500)),
            )
            for step in range(100, 501, 20)
        ]
        estimated_steps, _, details = estimate_piecewise_energy_adjusted_steps(
            original_steps=4000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=observed,
            probe_steps=500,
            dataset_batches_per_epoch=250,
            rms_curve=curve,
            gradient_accumulation_steps=6,
        )
        rounded_steps = round_steps_to_nearest_multiple(estimated_steps, 100)

        updated = update_piecewise_production_prediction(
            details,
            production_steps=rounded_steps,
            probe_steps=500,
            observed_rms=observed,
            dataset_batches_per_epoch=250,
            gradient_accumulation_steps=6,
        )

        self.assertEqual(estimated_steps, 5711)
        self.assertEqual(rounded_steps, 5700)
        self.assertEqual(
            updated["estimated_steps_predicted_final_rms"],
            details["predicted_final_rms"],
        )
        self.assertAlmostEqual(
            updated["predicted_final_rms"],
            predict_piecewise_energy_final_rms(
                rounded_steps,
                500,
                observed,
                250,
                details["probe_energy_slope_per_1000_steps"],
                6,
            ),
        )
        self.assertEqual(
            tuple(updated[f"segment_{index}_steps"] for index in range(1, 6)),
            tuple(float(value) for value in squeeze_segment_steps(rounded_steps)),
        )

    def test_augmented_later_velocity_uses_probe_rms_bpe_and_production_accumulation(self):
        self.assertAlmostEqual(
            estimate_augmented_later_mean_rms_velocity(
                dataset_batches_per_epoch=132,
                observed_rms=3.8782791e-5,
                gradient_accumulation_steps=6,
                probe_energy_slope=2.1028907100460046,
            ),
            0.33780371929157293,
        )
        self.assertAlmostEqual(
            estimate_augmented_later_mean_rms_velocity(
                dataset_batches_per_epoch=157,
                observed_rms=3.804286006720449e-5,
                gradient_accumulation_steps=5,
                probe_energy_slope=2.063401097961805,
            ),
            0.29141287368000257,
        )
        self.assertGreater(
            estimate_augmented_later_mean_rms_velocity(
                dataset_batches_per_epoch=157,
                observed_rms=3.804286006720449e-5,
                gradient_accumulation_steps=6,
                probe_energy_slope=2.063401097961805,
            ),
            estimate_augmented_later_mean_rms_velocity(
                dataset_batches_per_epoch=157,
                observed_rms=3.804286006720449e-5,
                gradient_accumulation_steps=4,
                probe_energy_slope=2.063401097961805,
            ),
        )

    def test_augmented_later_velocity_extrapolates_out_of_range_features(self):
        self.assertAlmostEqual(
            estimate_augmented_later_mean_rms_velocity(
                dataset_batches_per_epoch=99,
                observed_rms=3.8782791e-5,
                gradient_accumulation_steps=6,
                probe_energy_slope=2.1028907100460046,
            ),
            0.3854598627365411,
        )
        self.assertAlmostEqual(
            estimate_augmented_later_mean_rms_velocity(
                dataset_batches_per_epoch=20,
                observed_rms=5.3288192e-5,
                gradient_accumulation_steps=20,
                probe_energy_slope=2.2019441770164145,
            ),
            0.754891222778256,
        )

    def test_crossbreed_priscilla_repeats_share_one_calibration_family(self):
        production_reference_rms = 3.858837416025104e-5
        production_energy_slope = 2.126722772584375
        velocity = estimate_augmented_later_mean_rms_velocity(
            dataset_batches_per_epoch=93,
            observed_rms=production_reference_rms,
            gradient_accumulation_steps=7,
            probe_energy_slope=production_energy_slope,
        )

        self.assertAlmostEqual(velocity, 0.4174372650476631)
        self.assertAlmostEqual(
            predict_piecewise_energy_final_rms(
                total_steps=3300,
                probe_steps=500,
                observed_rms=production_reference_rms,
                dataset_batches_per_epoch=93,
                probe_energy_slope=production_energy_slope,
                gradient_accumulation_steps=7,
            ),
            7.622351066935624e-5,
            delta=1e-11,
        )
        second_velocity = estimate_augmented_later_mean_rms_velocity(
            dataset_batches_per_epoch=93,
            observed_rms=3.842611053225904e-5,
            gradient_accumulation_steps=7,
            probe_energy_slope=2.0053641727895593,
        )
        self.assertAlmostEqual(second_velocity, 0.4418091197806908)
        self.assertAlmostEqual(
            predict_piecewise_energy_final_rms(
                total_steps=3700,
                probe_steps=500,
                observed_rms=3.842611053225904e-5,
                dataset_batches_per_epoch=93,
                probe_energy_slope=2.0053641727895593,
                gradient_accumulation_steps=7,
            ),
            8.50853442072068e-5,
            delta=1e-11,
        )

    def test_marcia_small_dataset_probe_horizons_are_consistent(self):
        curve = [
            (20, 7.500369974877685e-6),
            (40, 1.0642542292771395e-5),
            (60, 1.3171358659747057e-5),
            (80, 1.5526458810199983e-5),
            (100, 1.7697382645565085e-5),
            (120, 1.988203075597994e-5),
            (140, 2.183806645916775e-5),
            (160, 2.371787377342116e-5),
            (180, 2.5830004233284853e-5),
            (200, 2.7880607376573607e-5),
            (220, 2.9897479180363007e-5),
            (240, 3.14799472107552e-5),
            (260, 3.329338869662024e-5),
            (280, 3.5424644011072814e-5),
            (300, 3.713290425366722e-5),
            (320, 3.8707796193193644e-5),
            (340, 4.04687161790207e-5),
            (360, 4.204341894364916e-5),
            (380, 4.373223055154085e-5),
            (400, 4.530819569481537e-5),
            (420, 4.6773591748205945e-5),
            (440, 4.8581343435216695e-5),
            (460, 5.0131999159930274e-5),
            (480, 5.068505561212078e-5),
            (500, 5.124977542436682e-5),
        ]
        cases = (
            (250, 3.238666795368772e-5, 2392, 0.5421428184963475, False),
            (400, None, 2313, 0.5098433373718041, False),
            (500, None, 2264, 0.5255771690775687, True),
        )

        for (
            horizon,
            final_sample,
            expected_steps,
            expected_velocity,
            expected_probe_two,
        ) in cases:
            truncated = [sample for sample in curve if sample[0] <= horizon]
            if final_sample is not None:
                truncated.append((horizon, final_sample))
            reference_rms, slope, _ = fit_schedule_aware_probe_energy_model(
                truncated,
                total_steps=2400,
                probe_steps=500,
                curve_interval=20,
            )
            plan = estimate_piecewise_training_plan(
                original_steps=2400,
                final_target_rms=8.425384599385171e-5,
                observed_rms=reference_rms,
                probe_steps=500,
                dataset_batches_per_epoch=22,
                rms_curve=truncated,
                current_gradient_accumulation_steps=5,
                probe_gradient_accumulation_steps=5,
                target_microbatches=None,
                probe_energy_slope=slope,
            )

            self.assertEqual(plan[0], expected_steps)
            self.assertAlmostEqual(
                plan[5]["later_mean_rms_velocity_per_1000_steps"],
                expected_velocity,
            )
            self.assertEqual(
                should_run_adjusted_probe(
                    "piecewise_energy_v1",
                    plan[2],
                    horizon,
                    False,
                ),
                expected_probe_two,
            )

    def test_marcia_ga11_run_updates_the_shared_family_calibration(self):
        reference_rms = 6.54549402987649e-5
        energy_slope = 2.306234594209267
        velocity = estimate_augmented_later_mean_rms_velocity(
            dataset_batches_per_epoch=22,
            observed_rms=reference_rms,
            gradient_accumulation_steps=11,
            probe_energy_slope=energy_slope,
        )
        predicted_rms = predict_piecewise_energy_final_rms(
            total_steps=1700,
            probe_steps=500,
            observed_rms=reference_rms,
            dataset_batches_per_epoch=22,
            probe_energy_slope=energy_slope,
            gradient_accumulation_steps=11,
        )
        estimated_steps, _, _ = estimate_piecewise_energy_adjusted_steps(
            original_steps=1850,
            final_target_rms=8.425384599385171e-5,
            observed_rms=reference_rms,
            probe_steps=500,
            dataset_batches_per_epoch=22,
            rms_curve=(),
            gradient_accumulation_steps=11,
            probe_energy_slope=energy_slope,
        )

        self.assertAlmostEqual(velocity, 0.5749667396649231)
        self.assertAlmostEqual(predicted_rms, 9.027195362013741e-5, delta=1e-11)
        self.assertEqual(estimated_steps, 1564)

    def test_adjusted_probe_retention_scale_uses_observed_stage_means(self):
        scale, details = fit_observed_squeeze_retention_scale(
            (0.9446578025817871, 0.9350574016571045)
        )

        self.assertAlmostEqual(scale, 1.0111555687212488)
        self.assertEqual(details["observed_squeeze_retention_count"], 2.0)
        self.assertGreater(
            predict_piecewise_energy_final_rms(
                total_steps=2400,
                probe_steps=500,
                observed_rms=5.0705232640351776e-5,
                dataset_batches_per_epoch=22,
                probe_energy_slope=2.1984010211552754,
                gradient_accumulation_steps=5,
                later_mean_rms_velocity=0.4718075142017018,
                squeeze_retention_scale=scale,
            ),
            predict_piecewise_energy_final_rms(
                total_steps=2400,
                probe_steps=500,
                observed_rms=5.0705232640351776e-5,
                dataset_batches_per_epoch=22,
                probe_energy_slope=2.1984010211552754,
                gradient_accumulation_steps=5,
                later_mean_rms_velocity=0.4718075142017018,
            ),
        )

    def test_beastgirl_extends_calibration_to_fifteen_batches_per_epoch(self):
        reference_rms = 5.019000006813414e-5
        energy_slope = 2.1833027981860034
        velocity = estimate_augmented_later_mean_rms_velocity(
            dataset_batches_per_epoch=15,
            observed_rms=reference_rms,
            gradient_accumulation_steps=4,
            probe_energy_slope=energy_slope,
        )
        predicted_rms = predict_piecewise_energy_final_rms(
            total_steps=2200,
            probe_steps=500,
            observed_rms=reference_rms,
            dataset_batches_per_epoch=15,
            probe_energy_slope=energy_slope,
            gradient_accumulation_steps=4,
        )
        estimated_steps, _, _ = estimate_piecewise_energy_adjusted_steps(
            original_steps=2200,
            final_target_rms=8.425384599385171e-5,
            observed_rms=reference_rms,
            probe_steps=500,
            dataset_batches_per_epoch=15,
            rms_curve=(),
            gradient_accumulation_steps=4,
            probe_energy_slope=energy_slope,
        )

        self.assertAlmostEqual(velocity, 0.5579587838277081)
        self.assertAlmostEqual(predicted_rms, 8.421124087177094e-5, delta=1e-11)
        self.assertEqual(estimated_steps, 2202)

    def test_latest_marcia_and_beastgirl_runs_update_their_family_calibrations(self):
        cases = (
            (
                27,
                5.4083582557090604e-5,
                9,
                2.17195124877972,
                2000,
                0.5777016831591206,
                8.574137905547667e-5,
                1957,
            ),
            (
                22,
                6.370470275181598e-5,
                11,
                2.2177239901442296,
                1600,
                0.5982065162195691,
                8.588830549370317e-5,
                1563,
            ),
            (
                28,
                6.351642683420896e-5,
                14,
                2.1520303378229664,
                1600,
                0.6108336723106391,
                8.698079864106743e-5,
                1538,
            ),
            (
                24,
                6.72222426902599e-5,
                12,
                2.2565366390239823,
                1450,
                0.5782365025230364,
                8.197173866498265e-5,
                1500,
            ),
        )

        for bpe, reference_rms, ga, slope, steps, velocity, predicted_rms, target_steps in cases:
            self.assertAlmostEqual(
                estimate_augmented_later_mean_rms_velocity(
                    dataset_batches_per_epoch=bpe,
                    observed_rms=reference_rms,
                    gradient_accumulation_steps=ga,
                    probe_energy_slope=slope,
                ),
                velocity,
            )
            self.assertAlmostEqual(
                predict_piecewise_energy_final_rms(
                    total_steps=steps,
                    probe_steps=500,
                    observed_rms=reference_rms,
                    dataset_batches_per_epoch=bpe,
                    probe_energy_slope=slope,
                    gradient_accumulation_steps=ga,
                ),
                predicted_rms,
                delta=1e-11,
            )
            estimated_steps, _, _ = estimate_piecewise_energy_adjusted_steps(
                original_steps=steps,
                final_target_rms=8.425384599385171e-5,
                observed_rms=reference_rms,
                probe_steps=500,
                dataset_batches_per_epoch=bpe,
                rms_curve=(),
                gradient_accumulation_steps=ga,
                probe_energy_slope=slope,
            )
            self.assertEqual(estimated_steps, target_steps)

    def test_object_character_001_revisions_share_one_calibration_family(self):
        cases = (
            (
                122,
                3.8101557082479796e-5,
                2.0491012957960426,
                4200,
                0.3652617154442301,
                8.423485255994249e-5,
                4202,
            ),
            (
                119,
                3.8070918e-5,
                2.0700419993815973,
                4600,
                0.36552900391642135,
                9.034639913841768e-5,
                4199,
            ),
        )

        for bpe, reference_rms, slope, steps, velocity, predicted_rms, target_steps in cases:
            self.assertAlmostEqual(
                estimate_augmented_later_mean_rms_velocity(
                    dataset_batches_per_epoch=bpe,
                    observed_rms=reference_rms,
                    gradient_accumulation_steps=6,
                    probe_energy_slope=slope,
                ),
                velocity,
            )
            self.assertAlmostEqual(
                predict_piecewise_energy_final_rms(
                    total_steps=steps,
                    probe_steps=500,
                    observed_rms=reference_rms,
                    dataset_batches_per_epoch=bpe,
                    probe_energy_slope=slope,
                    gradient_accumulation_steps=6,
                ),
                predicted_rms,
                delta=1e-11,
            )
            estimated_steps, _, _ = estimate_piecewise_energy_adjusted_steps(
                original_steps=steps,
                final_target_rms=8.425384599385171e-5,
                observed_rms=reference_rms,
                probe_steps=500,
                dataset_batches_per_epoch=bpe,
                rms_curve=(),
                gradient_accumulation_steps=6,
                probe_energy_slope=slope,
            )
            self.assertEqual(estimated_steps, target_steps)

    def test_double_weighted_ten_rosine_revisions_are_in_augmented_calibration(self):
        first_later_velocity = estimate_augmented_later_mean_rms_velocity(
            dataset_batches_per_epoch=99,
            observed_rms=4.469153373812116e-5,
            gradient_accumulation_steps=8,
            probe_energy_slope=2.1870861288967656,
        )
        second_later_velocity = estimate_augmented_later_mean_rms_velocity(
            dataset_batches_per_epoch=95,
            observed_rms=4.4754658783445186e-5,
            gradient_accumulation_steps=9,
            probe_energy_slope=2.216775240715489,
        )
        third_later_velocity = estimate_augmented_later_mean_rms_velocity(
            dataset_batches_per_epoch=104,
            observed_rms=4.29566212128835e-5,
            gradient_accumulation_steps=7,
            probe_energy_slope=2.139091157515416,
        )
        low_accumulation_later_velocity = estimate_augmented_later_mean_rms_velocity(
            dataset_batches_per_epoch=103,
            observed_rms=3.648383790277876e-5,
            gradient_accumulation_steps=2,
            probe_energy_slope=2.085742066127855,
        )
        fourth_later_velocity = estimate_augmented_later_mean_rms_velocity(
            dataset_batches_per_epoch=110,
            observed_rms=4.176764188992321e-5,
            gradient_accumulation_steps=6,
            probe_energy_slope=2.1384678815539027,
        )
        second_low_accumulation_later_velocity = (
            estimate_augmented_later_mean_rms_velocity(
                dataset_batches_per_epoch=117,
                observed_rms=3.7124011e-5,
                gradient_accumulation_steps=2,
                probe_energy_slope=2.01871264275116,
            )
        )
        fifth_later_velocity = estimate_augmented_later_mean_rms_velocity(
            dataset_batches_per_epoch=112,
            observed_rms=4.4364149e-5,
            gradient_accumulation_steps=9,
            probe_energy_slope=2.1487726165980137,
        )
        sixth_later_velocity = estimate_augmented_later_mean_rms_velocity(
            dataset_batches_per_epoch=119,
            observed_rms=4.293570495580571e-5,
            gradient_accumulation_steps=6,
            probe_energy_slope=2.0936642515103694,
        )
        seventh_later_velocity = estimate_augmented_later_mean_rms_velocity(
            dataset_batches_per_epoch=111,
            observed_rms=4.5584457060263114e-5,
            gradient_accumulation_steps=7,
            probe_energy_slope=2.1592672694926716,
        )
        ga10_later_velocity = estimate_augmented_later_mean_rms_velocity(
            dataset_batches_per_epoch=103,
            observed_rms=4.6760884669408374e-5,
            gradient_accumulation_steps=10,
            probe_energy_slope=2.166961419148628,
        )
        self.assertAlmostEqual(first_later_velocity, 0.3837653883036998)
        self.assertAlmostEqual(second_later_velocity, 0.40370835130625743)
        self.assertAlmostEqual(third_later_velocity, 0.3721452449511561)
        self.assertAlmostEqual(low_accumulation_later_velocity, 0.21739231770890893)
        self.assertAlmostEqual(fourth_later_velocity, 0.34428137146932386)
        self.assertAlmostEqual(second_low_accumulation_later_velocity, 0.20525664481186984)
        self.assertAlmostEqual(fifth_later_velocity, 0.3915504334595167)
        self.assertAlmostEqual(sixth_later_velocity, 0.3336253531105056)
        self.assertAlmostEqual(seventh_later_velocity, 0.34395955655614197)
        self.assertAlmostEqual(ga10_later_velocity, 0.4070921405697654)
        self.assertAlmostEqual(
            predict_piecewise_energy_final_rms(
                total_steps=3400,
                probe_steps=500,
                observed_rms=4.4364149e-5,
                dataset_batches_per_epoch=112,
                probe_energy_slope=2.1487726165980137,
                gradient_accumulation_steps=9,
            ),
            8.684874159585946e-5,
            delta=1e-11,
        )
        self.assertAlmostEqual(
            predict_piecewise_energy_final_rms(
                total_steps=4200,
                probe_steps=500,
                observed_rms=4.293570495580571e-5,
                dataset_batches_per_epoch=119,
                probe_energy_slope=2.0936642515103694,
                gradient_accumulation_steps=6,
            ),
            9.091778286432574e-5,
            delta=1e-11,
        )
        fifth_estimated_steps, _, _ = estimate_piecewise_energy_adjusted_steps(
            original_steps=3400,
            final_target_rms=8.425384599385171e-5,
            observed_rms=4.4364149e-5,
            probe_steps=500,
            dataset_batches_per_epoch=112,
            rms_curve=(),
            gradient_accumulation_steps=9,
            probe_energy_slope=2.1487726165980137,
        )
        newest_estimated_steps, _, _ = estimate_piecewise_energy_adjusted_steps(
            original_steps=4200,
            final_target_rms=8.425384599385171e-5,
            observed_rms=4.293570495580571e-5,
            probe_steps=500,
            dataset_batches_per_epoch=119,
            rms_curve=(),
            gradient_accumulation_steps=6,
            probe_energy_slope=2.0936642515103694,
        )
        seventh_estimated_steps, _, _ = estimate_piecewise_energy_adjusted_steps(
            original_steps=3600,
            final_target_rms=8.425384599385171e-5,
            observed_rms=4.5584457060263114e-5,
            probe_steps=500,
            dataset_batches_per_epoch=111,
            rms_curve=(),
            gradient_accumulation_steps=7,
            probe_energy_slope=2.1592672694926716,
        )
        ga10_estimated_steps, _, _ = estimate_piecewise_energy_adjusted_steps(
            original_steps=2900,
            final_target_rms=8.425384599385171e-5,
            observed_rms=4.6760884669408374e-5,
            probe_steps=500,
            dataset_batches_per_epoch=103,
            rms_curve=(),
            gradient_accumulation_steps=10,
            probe_energy_slope=2.166961419148628,
        )
        self.assertEqual(fifth_estimated_steps, 3269)
        self.assertEqual(newest_estimated_steps, 3795)
        self.assertEqual(seventh_estimated_steps, 3425)
        self.assertEqual(ga10_estimated_steps, 2973)
        self.assertAlmostEqual(
            predict_piecewise_energy_final_rms(
                total_steps=3300,
                probe_steps=500,
                observed_rms=4.469153373812116e-5,
                dataset_batches_per_epoch=99,
                probe_energy_slope=2.1870861288967656,
                gradient_accumulation_steps=8,
            ),
            8.476787276067509e-5,
            delta=1e-11,
        )
        self.assertAlmostEqual(
            predict_piecewise_energy_final_rms(
                total_steps=3000,
                probe_steps=500,
                observed_rms=4.4754658783445186e-5,
                dataset_batches_per_epoch=95,
                probe_energy_slope=2.216775240715489,
                gradient_accumulation_steps=9,
            ),
            8.09645622362097e-5,
            delta=1e-11,
        )
        first_estimated_steps, _, _ = estimate_piecewise_energy_adjusted_steps(
            original_steps=3000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=4.469153373812116e-5,
            probe_steps=500,
            dataset_batches_per_epoch=99,
            rms_curve=(),
            gradient_accumulation_steps=8,
            probe_energy_slope=2.1870861288967656,
        )
        second_estimated_steps, _, _ = estimate_piecewise_energy_adjusted_steps(
            original_steps=3000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=4.4754658783445186e-5,
            probe_steps=500,
            dataset_batches_per_epoch=95,
            rms_curve=(),
            gradient_accumulation_steps=9,
            probe_energy_slope=2.216775240715489,
        )
        self.assertEqual(first_estimated_steps, 3275)
        self.assertEqual(second_estimated_steps, 3160)

    def test_piecewise_energy_calibration_reproduces_augmented_predictions(self):
        mrissi_predicted = predict_piecewise_energy_final_rms(
            total_steps=4000,
            probe_steps=500,
            observed_rms=3.8782791e-5,
            dataset_batches_per_epoch=132,
            probe_energy_slope=2.103,
            gradient_accumulation_steps=6,
        )
        wilykit_predicted = predict_piecewise_energy_final_rms(
            total_steps=4600,
            probe_steps=500,
            observed_rms=3.804286006720449e-5,
            dataset_batches_per_epoch=157,
            probe_energy_slope=2.063401097961805,
            gradient_accumulation_steps=5,
        )
        wilykit_second_predicted = predict_piecewise_energy_final_rms(
            total_steps=5200,
            probe_steps=500,
            observed_rms=3.804286006720449e-5,
            dataset_batches_per_epoch=157,
            probe_energy_slope=2.063401097961805,
            gradient_accumulation_steps=5,
        )

        self.assertAlmostEqual(mrissi_predicted, 7.96845072766962e-5, delta=1e-11)
        self.assertAlmostEqual(wilykit_predicted, 8.067478284928558e-5, delta=1e-11)
        self.assertAlmostEqual(wilykit_second_predicted, 8.83497147449849e-5, delta=1e-11)

    def test_gradient_accumulation_uses_biased_nearest_microbatch_budget(self):
        self.assertEqual(choose_gradient_accumulation_steps(3000, 24000, 6), 8)
        self.assertEqual(choose_gradient_accumulation_steps(3500, 24000, 6), 7)
        self.assertEqual(choose_gradient_accumulation_steps(3999, 24000, 6), 6)
        self.assertEqual(choose_gradient_accumulation_steps(4500, 24000, 6), 5)
        self.assertEqual(choose_gradient_accumulation_steps(5000, 24000, 6), 5)
        self.assertEqual(choose_gradient_accumulation_steps(5500, 24000, 6), 4)
        self.assertEqual(choose_gradient_accumulation_steps(6500, 24000, 6), 4)
        self.assertEqual(choose_gradient_accumulation_steps(6500, None, 6), 6)

    def test_epoch_accumulation_groups_include_underfilled_final_update(self):
        self.assertEqual(epoch_accumulation_group_sizes(22, 18), (18, 4))
        self.assertEqual(epoch_accumulation_group_sizes(22, 5), (5, 5, 5, 5, 2))
        self.assertEqual(epoch_accumulation_group_sizes(22, 22), (22,))

    def test_epoch_aware_microbatch_count_tracks_optimizer_step_phase(self):
        self.assertEqual(count_optimizer_step_microbatches(1, 22, 18), 18)
        self.assertEqual(count_optimizer_step_microbatches(2, 22, 18), 22)
        self.assertEqual(count_optimizer_step_microbatches(3, 22, 18), 40)
        self.assertEqual(count_optimizer_step_microbatches(1350, 22, 18), 14850)
        self.assertEqual(optimizer_steps_for_microbatch_budget(14850, 22, 18), 1350)

    def test_automatic_accumulation_matches_actual_epoch_microbatches(self):
        self.assertEqual(
            choose_gradient_accumulation_steps(
                1350,
                24000,
                22,
                rounding_bias=0.7,
                dataset_batches_per_epoch=22,
            ),
            22,
        )
        self.assertEqual(
            choose_gradient_accumulation_steps(
                4000,
                24000,
                6,
                rounding_bias=0.7,
                dataset_batches_per_epoch=95,
            ),
            6,
        )
        self.assertNotEqual(95 % 6, 0)
        self.assertEqual(count_optimizer_step_microbatches(4000, 95, 6), 23750)
        marcia_ga18_microbatches = count_optimizer_step_microbatches(
            1351,
            22,
            18,
        )
        self.assertEqual(marcia_ga18_microbatches, 14868)
        self.assertEqual(
            choose_gradient_accumulation_steps(
                1351,
                marcia_ga18_microbatches,
                22,
                rounding_bias=0.7,
                dataset_batches_per_epoch=22,
            ),
            18,
        )
        self.assertEqual(
            choose_gradient_accumulation_steps(
                6500,
                None,
                6,
                dataset_batches_per_epoch=95,
            ),
            6,
        )
        self.assertEqual(
            choose_gradient_accumulation_steps(
                6500,
                None,
                30,
                dataset_batches_per_epoch=22,
            ),
            30,
        )

    def test_piecewise_plan_reapplies_microbatch_target_after_second_estimate(self):
        observed_rms = 4.4754658783445186e-5
        probe_energy_slope = 2.216775240715489
        later_mean_rms_velocity = 0.40280578972662107
        final_target_rms = predict_piecewise_energy_final_rms(
            total_steps=3018,
            probe_steps=500,
            observed_rms=observed_rms,
            dataset_batches_per_epoch=95,
            probe_energy_slope=probe_energy_slope,
            gradient_accumulation_steps=9,
            later_mean_rms_velocity=later_mean_rms_velocity,
        )

        (
            estimated_steps,
            _,
            adjusted_steps,
            uncapped_accumulation,
            adjusted_accumulation,
            details,
        ) = estimate_piecewise_training_plan(
            original_steps=4000,
            final_target_rms=final_target_rms,
            observed_rms=observed_rms,
            probe_steps=500,
            dataset_batches_per_epoch=95,
            rms_curve=(),
            current_gradient_accumulation_steps=6,
            probe_gradient_accumulation_steps=9,
            target_microbatches=24000,
            gradient_accumulation_rounding_bias=0.7,
            adjusted_steps_divisible_by=100,
            probe_energy_slope=probe_energy_slope,
            later_mean_rms_velocity=later_mean_rms_velocity,
            later_mean_rms_velocity_reference_gradient_accumulation_steps=9,
        )

        self.assertEqual(estimated_steps, 3153)
        self.assertEqual(adjusted_steps, 3200)
        self.assertEqual(uncapped_accumulation, 8)
        self.assertEqual(adjusted_accumulation, 8)
        self.assertEqual(details["gradient_accumulation_is_epoch_compatible"], 0.0)
        self.assertEqual(details["actual_production_microbatches"], 25334.0)
        self.assertEqual(
            adjusted_accumulation,
            choose_gradient_accumulation_steps(
                adjusted_steps,
                24000,
                6,
                rounding_bias=0.7,
                dataset_batches_per_epoch=95,
            ),
        )
        self.assertEqual(
            details["later_mean_rms_velocity_gradient_accumulation_steps"],
            float(adjusted_accumulation),
        )
        self.assertAlmostEqual(
            details["observed_later_mean_rms_velocity_accumulation_scale"],
            0.9650090239948089,
        )

    def test_marcia_probe_two_transfers_probe_one_rank36_anchor(self):
        probe_one_curve = (
            (20, 8.384812746583882e-6),
            (40, 1.2532975534542193e-5),
            (60, 1.6333030095169823e-5),
            (80, 2.0194843333880055e-5),
            (100, 2.406638620843219e-5),
            (120, 2.8065770467594773e-5),
            (140, 3.2178582850411865e-5),
            (160, 3.591705169945463e-5),
            (180, 3.9572608661434296e-5),
            (200, 4.328498669908549e-5),
        )
        probe_two_curve = (
            (20, 8.431589772589938e-6),
            (40, 1.2581572540427063e-5),
            (60, 1.635862070439193e-5),
            (80, 1.8600551448695187e-5),
            (100, 1.9645133859058497e-5),
            (120, 2.063964607003356e-5),
            (140, 2.1107788438254896e-5),
            (160, 2.2122571828044655e-5),
            (180, 2.308801673276316e-5),
            (200, 2.406842729930015e-5),
        )
        primary_reference_rms, primary_slope, _ = (
            fit_schedule_aware_probe_energy_model(
                probe_one_curve,
                total_steps=4000,
                probe_steps=500,
                curve_interval=20,
            )
        )
        adjusted_reference_rms, adjusted_slope, _ = (
            fit_schedule_aware_probe_energy_model(
                probe_two_curve,
                total_steps=350,
                probe_steps=500,
                curve_interval=20,
            )
        )
        old_observed_later_velocity, _ = fit_observed_later_mean_rms_velocity(
            probe_two_curve,
            total_steps=350,
            probe_steps=200,
            reference_rms=adjusted_reference_rms,
            curve_interval=20,
        )
        transferred_reference_rms, transferred_slope, transfer_details = (
            fit_adjusted_probe_rank36_transfer(
                probe_one_curve,
                probe_two_curve,
                adjusted_total_steps=350,
                adjusted_probe_steps=200,
                primary_reference_rms=primary_reference_rms,
                primary_probe_energy_slope=primary_slope,
            )
        )
        observed_later_velocity, _ = fit_observed_later_mean_rms_velocity(
            probe_two_curve,
            total_steps=350,
            probe_steps=200,
            reference_rms=transferred_reference_rms,
            curve_interval=20,
        )

        second_only_plan = estimate_piecewise_training_plan(
            original_steps=4000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=adjusted_reference_rms,
            probe_steps=500,
            dataset_batches_per_epoch=22,
            rms_curve=probe_two_curve,
            current_gradient_accumulation_steps=22,
            probe_gradient_accumulation_steps=22,
            target_microbatches=24000,
            gradient_accumulation_rounding_bias=0.7,
            adjusted_steps_divisible_by=50,
            probe_energy_slope=adjusted_slope,
            later_mean_rms_velocity=old_observed_later_velocity,
            later_mean_rms_velocity_reference_gradient_accumulation_steps=22,
        )
        anchored_plan = estimate_piecewise_training_plan(
            original_steps=4000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=transferred_reference_rms,
            probe_steps=500,
            dataset_batches_per_epoch=22,
            rms_curve=probe_one_curve,
            current_gradient_accumulation_steps=22,
            probe_gradient_accumulation_steps=22,
            target_microbatches=24000,
            gradient_accumulation_rounding_bias=0.7,
            adjusted_steps_divisible_by=50,
            probe_energy_slope=transferred_slope,
            later_mean_rms_velocity=observed_later_velocity,
            later_mean_rms_velocity_reference_gradient_accumulation_steps=22,
        )
        fixed_ga5_plan = estimate_piecewise_training_plan(
            original_steps=4000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=transferred_reference_rms,
            probe_steps=500,
            dataset_batches_per_epoch=22,
            rms_curve=probe_one_curve,
            current_gradient_accumulation_steps=5,
            probe_gradient_accumulation_steps=22,
            target_microbatches=None,
            adjusted_steps_divisible_by=50,
            probe_energy_slope=transferred_slope,
            later_mean_rms_velocity=observed_later_velocity,
            later_mean_rms_velocity_reference_gradient_accumulation_steps=22,
        )

        self.assertEqual(second_only_plan[0], 1435)
        self.assertAlmostEqual(
            transfer_details["rank36_transfer_rms_scale"],
            1.0023300289993526,
        )
        self.assertEqual(transfer_details["rank36_transfer_sample_count"], 3.0)
        self.assertAlmostEqual(observed_later_velocity, 0.6669061655405474)
        self.assertAlmostEqual(
            old_observed_later_velocity * adjusted_reference_rms,
            observed_later_velocity * transferred_reference_rms,
        )
        self.assertEqual(anchored_plan[0], 1185)
        self.assertEqual(anchored_plan[2], 1200)
        self.assertEqual(anchored_plan[4], 22)
        self.assertEqual(
            anchored_plan[5]["gradient_accumulation_is_epoch_compatible"],
            1.0,
        )
        self.assertEqual(fixed_ga5_plan[0], 1827)
        self.assertEqual(fixed_ga5_plan[2], 1850)
        self.assertEqual(fixed_ga5_plan[4], 5)

    def test_rosine_second_plan_no_longer_locks_probe_two_accumulation(self):
        (
            estimated_steps,
            _,
            adjusted_steps,
            uncapped_accumulation,
            adjusted_accumulation,
            _,
        ) = estimate_piecewise_training_plan(
            original_steps=4000,
            final_target_rms=8.425384599385171e-5,
            observed_rms=4.4754658783445186e-5,
            probe_steps=500,
            dataset_batches_per_epoch=95,
            rms_curve=(),
            current_gradient_accumulation_steps=6,
            probe_gradient_accumulation_steps=9,
            target_microbatches=24000,
            gradient_accumulation_rounding_bias=0.7,
            adjusted_steps_divisible_by=100,
            probe_energy_slope=2.216775240715489,
        )

        self.assertEqual(estimated_steps, 3474)
        self.assertEqual(adjusted_steps, 3500)
        self.assertEqual(uncapped_accumulation, 7)
        self.assertEqual(adjusted_accumulation, 7)

    def test_gradient_accumulation_rounding_bias_is_configurable(self):
        self.assertEqual(choose_gradient_accumulation_steps(5500, 24000, 6, rounding_bias=0.5), 4)
        self.assertEqual(choose_gradient_accumulation_steps(5500, 24000, 6, rounding_bias=0.7), 5)
        self.assertEqual(choose_gradient_accumulation_steps(3999, 24000, 6, rounding_bias=0.6), 6)
        self.assertEqual(choose_gradient_accumulation_steps(4000, 24000, 6, rounding_bias=1.0), 6)
        self.assertEqual(choose_gradient_accumulation_steps(3999, 24000, 6, rounding_bias=1.0), 7)

    def test_gradient_accumulation_respects_configured_minimum(self):
        self.assertEqual(choose_gradient_accumulation_steps(12000, 24000, 6, 3), 3)

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

    def test_piecewise_probe_accepts_samples_after_first_original_squeeze(self):
        self.assertTrue(
            validate_rms_probe_configuration(
                make_args(
                    max_train_steps=1600,
                    rms_probe_scaling_policy="piecewise_energy_v1",
                    rms_probe_final_target=8e-5,
                )
            )
        )

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

    def test_forced_adjusted_probe_requires_piecewise_policy(self):
        with self.assertRaisesRegex(ValueError, "requires.*piecewise_energy_v1"):
            validate_rms_probe_configuration(
                make_args(rms_probe_force_adjusted_probe=True)
            )
        self.assertTrue(
            validate_rms_probe_configuration(
                make_args(
                    rms_probe_scaling_policy="piecewise_energy_v1",
                    rms_probe_final_target=8e-5,
                    rms_probe_force_adjusted_probe=True,
                )
            )
        )

    def test_piecewise_policy_accepts_supported_segment_ratios(self):
        self.assertTrue(
            validate_rms_probe_configuration(
                make_args(
                    rms_probe_scaling_policy="piecewise_energy_v1",
                    rms_probe_final_target=8e-5,
                    lora_squeeze_first_segment_ratio=2.0,
                    lora_squeeze_final_segment_ratio=0.75,
                )
            )
        )

    def test_piecewise_policy_allows_ratio_adjusted_squeeze_inside_probe(self):
        self.assertTrue(
            validate_rms_probe_configuration(
                make_args(
                    max_train_steps=4000,
                    rms_probe_scaling_policy="piecewise_energy_v1",
                    rms_probe_final_target=8e-5,
                    lora_squeeze_first_segment_ratio=0.5,
                )
            )
        )

    def test_piecewise_policy_requires_enough_rank36_curve_samples(self):
        with self.assertRaisesRegex(ValueError, "at least two rank-36 RMS curve samples"):
            validate_rms_probe_configuration(
                make_args(
                    rms_probe_scaling_policy="piecewise_energy_v1",
                    rms_probe_final_target=8e-5,
                    rms_probe_steps=100,
                    rms_probe_curve_every_n_steps=50,
                )
            )

    def test_piecewise_policy_rejects_squeeze_too_early_for_rank36_fit(self):
        with self.assertRaisesRegex(ValueError, "before squeeze one"):
            validate_rms_probe_configuration(
                make_args(
                    max_train_steps=500,
                    rms_probe_steps=500,
                    rms_probe_curve_every_n_steps=50,
                    rms_probe_scaling_policy="piecewise_energy_v1",
                    rms_probe_final_target=8e-5,
                )
            )

    def test_piecewise_policy_accepts_non_500_non_interval_aligned_probe(self):
        self.assertTrue(
            validate_rms_probe_configuration(
                make_args(
                    rms_probe_scaling_policy="piecewise_energy_v1",
                    rms_probe_final_target=8e-5,
                    rms_probe_steps=150,
                    rms_probe_curve_every_n_steps=20,
                )
            )
        )

    def test_gradient_accumulation_rounding_bias_must_be_bounded(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            validate_rms_probe_configuration(
                make_args(rms_probe_gradient_accumulation_rounding_bias=1.1)
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

    def test_second_probe_uses_provisional_production_settings(self):
        probe_args = build_rms_probe_args(
            make_args(rms_probe_scaling_policy="piecewise_energy_v1", rms_probe_final_target=8e-5),
            probe_index=2,
            production_steps=700,
            gradient_accumulation_steps=20,
            training_step_limit=240,
        )

        self.assertEqual(probe_args.max_train_steps, 700)
        self.assertEqual(probe_args.gradient_accumulation_steps, 20)
        self.assertEqual(probe_args._training_step_limit, 240)
        self.assertEqual(
            probe_args.output_dir,
            os.path.join("output", "character-rms-probe2-step240-steps700-ga20"),
        )

    def test_second_probe_accepts_production_shorter_than_probe1(self):
        probe_args = build_rms_probe_args(
            make_args(
                rms_probe_scaling_policy="piecewise_energy_v1",
                rms_probe_final_target=8e-5,
            ),
            probe_index=2,
            production_steps=400,
            gradient_accumulation_steps=20,
            training_step_limit=180,
        )

        self.assertEqual(probe_args.max_train_steps, 400)
        self.assertEqual(probe_args._training_step_limit, 180)


class LoglinearPolicyTest(unittest.TestCase):
    # (batches per epoch, GA, step-500 reference RMS, steps, final RMS) of
    # completed Anima 36->9 runs: Rosine_fourth, Izutsumi final, Marcia_second,
    # Ahsoka_second, and the third 001 revision.
    COMPLETED_RUNS = (
        (110, 6, 4.176764188992321e-5, 4000, 8.552275888033521e-5),
        (262, 4, 3.417661355342716e-5, 6400, 8.171637791552906e-5),
        (22, 5, 5.0705232640351776e-5, 2400, 8.776551913866043e-5),
        (243, 4, 3.3271486e-5, 6400, 8.351577e-5),
        (109, 6, 3.9533202e-5, 4600, 9.0057752e-5),
    )

    def test_prediction_matches_completed_runs_within_five_percent(self):
        for batches, accumulation, reference, steps, final in self.COMPLETED_RUNS:
            predicted = predict_loglinear_final_rms(steps, batches, accumulation, reference)
            self.assertLess(abs(predicted / final - 1.0), 0.05)

    def test_step_solver_inverts_the_prediction(self):
        for batches, accumulation, reference, steps, _ in self.COMPLETED_RUNS:
            target = predict_loglinear_final_rms(steps, batches, accumulation, reference)
            self.assertEqual(
                estimate_loglinear_steps(target, batches, accumulation, reference),
                steps,
            )

    def test_more_steps_and_fewer_steps_per_epoch_raise_the_final_rms(self):
        base = predict_loglinear_final_rms(4000, 120, 6, 4e-5)
        self.assertGreater(predict_loglinear_final_rms(5000, 120, 6, 4e-5), base)
        self.assertGreater(predict_loglinear_final_rms(4000, 120, 12, 4e-5), base)
        self.assertGreater(predict_loglinear_final_rms(4000, 60, 6, 4e-5), base)
        self.assertGreater(predict_loglinear_final_rms(4000, 120, 6, 4.4e-5), base)

    def test_plan_without_compute_budget_keeps_gradient_accumulation(self):
        estimated, multiplier, adjusted, uncapped, accumulation, details = (
            estimate_loglinear_training_plan(
                original_steps=4000,
                final_target_rms=8.4e-5,
                reference_rms=3.3271486e-5,
                dataset_batches_per_epoch=243,
                current_gradient_accumulation_steps=4,
                probe_gradient_accumulation_steps=4,
                target_microbatches=None,
                adjusted_steps_divisible_by=100,
            )
        )
        self.assertEqual(accumulation, 4)
        self.assertEqual(uncapped, 4)
        self.assertEqual(adjusted % 100, 0)
        self.assertLessEqual(abs(adjusted - estimated), 50)
        self.assertAlmostEqual(multiplier, estimated / 4000)
        self.assertAlmostEqual(details["production_reference_rms"], 3.3271486e-5)
        self.assertLess(
            abs(details["predicted_final_rms_at_estimated_steps"] / 8.4e-5 - 1.0),
            0.001,
        )
        # the completed run took 6,400 steps to 8.35e-5
        self.assertLess(abs(adjusted / 6400 - 1.0), 0.1)

    def test_plan_with_compute_budget_converges_and_rescales_the_reference(self):
        estimated, _, adjusted, _, accumulation, details = estimate_loglinear_training_plan(
            original_steps=4000,
            final_target_rms=8.425384599385171e-5,
            reference_rms=3.3400877e-5,
            dataset_batches_per_epoch=258,
            current_gradient_accumulation_steps=6,
            probe_gradient_accumulation_steps=6,
            target_microbatches=24000,
            gradient_accumulation_rounding_bias=0.7,
            adjusted_steps_divisible_by=100,
        )
        self.assertEqual(
            accumulation,
            choose_gradient_accumulation_steps(adjusted, 24000, 6, 1, 0.7, 258),
        )
        expected_reference = 3.3400877e-5 * (accumulation / 6) ** 0.13407367061621528
        self.assertAlmostEqual(details["production_reference_rms"], expected_reference)
        self.assertLess(
            abs(
                predict_loglinear_final_rms(estimated, 258, accumulation, expected_reference)
                / 8.425384599385171e-5
                - 1.0
            ),
            0.001,
        )

    def test_validation_accepts_loglinear_and_requires_final_target(self):
        self.assertTrue(
            validate_rms_probe_configuration(
                make_args(rms_probe_scaling_policy="loglinear_v1", rms_probe_final_target=8e-5)
            )
        )
        with self.assertRaisesRegex(ValueError, "rms_probe_final_target"):
            validate_rms_probe_configuration(make_args(rms_probe_scaling_policy="loglinear_v1"))

    def test_validation_rejects_unsupported_loglinear_settings(self):
        with self.assertRaisesRegex(ValueError, "equal squeeze segments"):
            validate_rms_probe_configuration(
                make_args(
                    rms_probe_scaling_policy="loglinear_v1",
                    rms_probe_final_target=8e-5,
                    lora_squeeze_final_segment_ratio=1.5,
                )
            )
        with self.assertRaisesRegex(ValueError, "force_adjusted_probe"):
            validate_rms_probe_configuration(
                make_args(
                    rms_probe_scaling_policy="loglinear_v1",
                    rms_probe_final_target=8e-5,
                    rms_probe_force_adjusted_probe=True,
                )
            )
        with self.assertRaisesRegex(ValueError, "36->9"):
            validate_rms_probe_configuration(
                make_args(
                    rms_probe_scaling_policy="loglinear_v1",
                    rms_probe_final_target=8e-5,
                    lora_squeeze_num_squeezes=2,
                )
            )

    def test_loglinear_never_requests_an_adjusted_probe(self):
        self.assertFalse(should_run_adjusted_probe("loglinear_v1", 1500, 500))


if __name__ == "__main__":
    unittest.main()
