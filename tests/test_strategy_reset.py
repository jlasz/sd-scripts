import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library import strategy_base


def test_strategies_can_be_registered_again_after_reset():
    strategy_classes = [
        strategy_base.TokenizeStrategy,
        strategy_base.TextEncodingStrategy,
        strategy_base.TextEncoderOutputsCachingStrategy,
        strategy_base.LatentsCachingStrategy,
    ]

    try:
        for strategy_class in strategy_classes:
            strategy_class.set_strategy(object())

        strategy_base.reset_strategies()

        for strategy_class in strategy_classes:
            assert strategy_class.get_strategy() is None
            strategy_class.set_strategy(object())
    finally:
        strategy_base.reset_strategies()
