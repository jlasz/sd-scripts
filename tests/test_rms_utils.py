import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.rms_utils import compute_total_scaled_lora_rms, validate_rms_log_interval


class DummyLoRA(torch.nn.Module):
    def __init__(self, down, up, scale):
        super().__init__()
        self.lora_down = torch.nn.Linear(down.shape[1], down.shape[0], bias=False)
        self.lora_up = torch.nn.Linear(up.shape[1], up.shape[0], bias=False)
        self.lora_down.weight.data.copy_(down)
        self.lora_up.weight.data.copy_(up)
        self.scale = scale


class RMSUtilsTest(unittest.TestCase):
    def test_total_scaled_lora_rms_matches_materialized_adapter_weights(self):
        first = DummyLoRA(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, -1.0]]),
            scale=0.5,
        )
        second = DummyLoRA(
            torch.tensor([[2.0, -1.0, 0.5]]),
            torch.tensor([[1.5], [-2.0]]),
            scale=2.0,
        )
        network = torch.nn.ModuleList([first, second])

        effective_weights = [
            first.scale * first.lora_up.weight @ first.lora_down.weight,
            second.scale * second.lora_up.weight @ second.lora_down.weight,
        ]
        expected = math.sqrt(
            sum(torch.sum(weight.float() ** 2).item() for weight in effective_weights)
            / sum(weight.numel() for weight in effective_weights)
        )

        self.assertAlmostEqual(compute_total_scaled_lora_rms(network), expected, places=6)

    def test_total_scaled_lora_rms_returns_zero_without_lora_modules(self):
        self.assertEqual(compute_total_scaled_lora_rms(torch.nn.Linear(2, 2)), 0.0)

    def test_rms_log_interval_must_not_be_negative(self):
        validate_rms_log_interval(0)
        validate_rms_log_interval(25)
        with self.assertRaisesRegex(ValueError, "0 or greater"):
            validate_rms_log_interval(-1)


if __name__ == "__main__":
    unittest.main()
