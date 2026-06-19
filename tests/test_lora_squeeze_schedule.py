import math
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train_network import LoRASqueezeSchedule


def make_schedule(**overrides):
    args = {
        "lora_squeeze_start_dim": 41,
        "lora_squeeze_num_squeezes": 4,
        "network_dim": 9,
        "network_alpha": 3,
        "lora_squeeze_train_after_final_squeeze": True,
        "lora_squeeze_step_schedule": None,
        "lora_squeeze_final_segment_ratio": 1.0,
    }
    args.update(overrides)
    return LoRASqueezeSchedule(SimpleNamespace(**args))


def compression_ratio_spread(ranks):
    log_ratios = [math.log(current / following) for current, following in zip(ranks, ranks[1:])]
    return max(log_ratios) - min(log_ratios)


class LoRASqueezeScheduleTest(unittest.TestCase):
    def test_rank_schedule_defaults_to_existing_linear_behavior(self):
        schedule = make_schedule()

        self.assertEqual(schedule.rank_schedule, "linear")
        self.assertEqual(schedule.ranks, [41, 33, 25, 17, 9])

    def test_geometric_rank_schedule_keeps_compression_ratios_nearly_equal(self):
        linear = make_schedule(lora_squeeze_rank_schedule="linear")
        geometric = make_schedule(lora_squeeze_rank_schedule="geometric")

        self.assertEqual(geometric.ranks, [41, 28, 19, 13, 9])
        self.assertLess(compression_ratio_spread(geometric.ranks), compression_ratio_spread(linear.ranks))

    def test_geometric_rank_schedule_remains_strict_when_integer_range_is_tight(self):
        schedule = make_schedule(
            lora_squeeze_start_dim=13,
            network_dim=9,
            lora_squeeze_rank_schedule="geometric",
        )

        self.assertEqual(schedule.ranks, [13, 12, 11, 10, 9])

    def test_explicit_equal_step_schedule_matches_omitted_schedule(self):
        omitted = make_schedule(lora_squeeze_final_segment_ratio=1.5)
        explicit = make_schedule(
            lora_squeeze_step_schedule="equal",
            lora_squeeze_final_segment_ratio=1.5,
        )

        omitted.set_total_steps(5000)
        explicit.set_total_steps(5000)

        self.assertEqual(explicit.segment_weights, omitted.segment_weights)
        self.assertEqual(explicit.squeeze_steps, omitted.squeeze_steps)
        self.assertEqual(explicit.segment_steps, omitted.segment_steps)
        self.assertEqual(
            explicit.step_distribution_text(),
            "equal, final_segment_ratio=1.5; segment steps: 909 / 909 / 909 / 909 / 1364",
        )

    def test_invalid_rank_schedule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "--lora_squeeze_rank_schedule must be one of"):
            make_schedule(lora_squeeze_rank_schedule="quadratic")


if __name__ == "__main__":
    unittest.main()
