"""Helpers for training extensions that replace prepared optimizer state."""

from typing import Any, Optional, Tuple

import torch
from torch.optim.lr_scheduler import LRScheduler


class ReplaceableOptimizer(torch.optim.Optimizer):
    """Stable optimizer identity whose implementation can be replaced in place.

    Accelerate registers this object once and wraps it normally. LoRA-Squeeze can
    then rebuild the underlying optimizer without modifying Accelerate's private
    prepared-object registries.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    @property
    def state(self):
        return self.optimizer.state

    @state.setter
    def state(self, state):
        self.optimizer.state = state

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, param_groups):
        self.optimizer.param_groups = param_groups

    @property
    def defaults(self):
        return self.optimizer.defaults

    @defaults.setter
    def defaults(self, defaults):
        self.optimizer.defaults = defaults

    def replace(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def add_param_group(self, param_group):
        return self.optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def zero_grad(self, set_to_none=None):
        if set_to_none is None:
            return self.optimizer.zero_grad()
        return self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        if closure is None:
            return self.optimizer.step()
        return self.optimizer.step(closure)

    def train(self):
        if hasattr(self.optimizer, "train"):
            return self.optimizer.train()

    def eval(self):
        if hasattr(self.optimizer, "eval"):
            return self.optimizer.eval()

    def __getattr__(self, name: str):
        if name == "optimizer":
            raise AttributeError(name)
        return getattr(self.optimizer, name)


class ReplaceableLRScheduler(LRScheduler):
    """Stable scheduler identity paired with :class:`ReplaceableOptimizer`."""

    def __init__(self, scheduler: LRScheduler):
        self.scheduler = scheduler

    @property
    def optimizer(self):
        return self.scheduler.optimizer

    @property
    def _step_count(self):
        return self.scheduler._step_count

    @_step_count.setter
    def _step_count(self, value):
        self.scheduler._step_count = value

    def replace(self, scheduler: LRScheduler):
        self.scheduler = scheduler

    def step(self, *args, **kwargs):
        return self.scheduler.step(*args, **kwargs)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def get_lr(self):
        return self.scheduler.get_lr()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        return self.scheduler.load_state_dict(state_dict)

    def __getattr__(self, name: str):
        if name == "scheduler":
            raise AttributeError(name)
        return getattr(self.scheduler, name)


def make_replaceable_optimizer_scheduler(
    optimizer: torch.optim.Optimizer, scheduler: Any
) -> Tuple[ReplaceableOptimizer, Any]:
    """Wrap dynamic training objects before passing them to Accelerator.prepare."""

    replaceable_optimizer = ReplaceableOptimizer(optimizer)
    replaceable_scheduler = ReplaceableLRScheduler(scheduler) if isinstance(scheduler, LRScheduler) else scheduler
    return replaceable_optimizer, replaceable_scheduler


def find_replaceable_optimizer(optimizer: Any) -> Optional[ReplaceableOptimizer]:
    if isinstance(optimizer, ReplaceableOptimizer):
        return optimizer
    wrapped = getattr(optimizer, "optimizer", None)
    return wrapped if isinstance(wrapped, ReplaceableOptimizer) else None


def find_replaceable_scheduler(scheduler: Any) -> Optional[ReplaceableLRScheduler]:
    if isinstance(scheduler, ReplaceableLRScheduler):
        return scheduler
    wrapped = getattr(scheduler, "scheduler", None)
    return wrapped if isinstance(wrapped, ReplaceableLRScheduler) else None
