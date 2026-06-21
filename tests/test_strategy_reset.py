import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library import strategy_base


class StrategyResetTest(unittest.TestCase):
    def tearDown(self):
        strategy_base.reset_strategies()

    def test_strategies_can_be_registered_again_after_reset(self):
        first_strategies = [object() for _ in range(4)]
        strategy_classes = [
            strategy_base.TokenizeStrategy,
            strategy_base.TextEncodingStrategy,
            strategy_base.TextEncoderOutputsCachingStrategy,
            strategy_base.LatentsCachingStrategy,
        ]

        for strategy_class, strategy in zip(strategy_classes, first_strategies):
            strategy_class.set_strategy(strategy)

        strategy_base.reset_strategies()

        for strategy_class in strategy_classes:
            self.assertIsNone(strategy_class.get_strategy())
            strategy_class.set_strategy(object())


if __name__ == "__main__":
    unittest.main()
