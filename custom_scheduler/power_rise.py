import math
from torch.optim.lr_scheduler import LambdaLR


class PowerRiseLR(LambdaLR):
    """
    Increasing power scheduler.

    The optimizer's base LR is treated as the final LR.
    The scheduler multiplies it by:

        multiplier = min_lr_ratio + (1 - min_lr_ratio) * progress ** power

    where progress goes from 0 to 1 over total_steps.

    Example:
        base LR = 0.000437
        total_steps = 1500
        power = 0.70

    Then LR rises throughout training and ends at 0.000437.
    """

    def __init__(
        self,
        optimizer,
        total_steps=1500,
        power=0.70,
        min_lr_ratio=0.0,
        last_epoch=-1,
    ):
        self.total_steps = int(total_steps)
        self.power = float(power)
        self.min_lr_ratio = float(min_lr_ratio)

        if self.total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if self.power <= 0:
            raise ValueError("power must be > 0")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be between 0 and 1")

        def lr_lambda(current_step):
            # current_step is 0-based inside PyTorch scheduler.
            # +1 makes the first optimizer step nonzero when min_lr_ratio = 0.
            progress = min((current_step + 1) / self.total_steps, 1.0)
            multiplier = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * math.pow(progress, self.power)
            return multiplier

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)