from types import SimpleNamespace

import torch
from torch import nn

from library.accelerator_utils import ReplaceableLRScheduler, ReplaceableOptimizer
from library.lora_squeeze_compression import get_lora_factor_matrices
from library.lora_squeeze_network import StandardLoRASqueezeModuleMixin
from library.lora_squeeze_schedule import LoRASqueezeSchedule
from library.lora_squeeze_training import LoRASqueezeTrainingController


def make_schedule_args(**overrides):
    args = {
        "lora_squeeze_start_dim": 41,
        "lora_squeeze_num_squeezes": 4,
        "network_dim": 9,
        "network_alpha": 3,
        "lora_squeeze_train_after_final_squeeze": True,
        "lora_squeeze_rank_schedule": "geometric",
        "lora_squeeze_step_schedule": "equal",
        "lora_squeeze_optimizer_mode": "global",
        "lora_squeeze_scheduler_mode": "global",
        "lora_squeeze_alpha_schedule": "sqrt",
        "lora_squeeze_first_segment_ratio": 1.0,
        "lora_squeeze_final_segment_ratio": 1.0,
        "dim_from_weights": False,
        "deepspeed": False,
        "initial_step": None,
        "initial_epoch": None,
        "torch_compile": False,
        "compile": False,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def make_schedule(**overrides):
    return LoRASqueezeSchedule(make_schedule_args(**overrides))


class FakeLoRAModule(StandardLoRASqueezeModuleMixin, nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, kernel_size=1):
        super().__init__()
        self.lora_name = "test_lora"
        self.lora_dim = rank
        self.scale = alpha / rank
        self.register_buffer("alpha", torch.tensor(alpha))
        if kernel_size == 1:
            self.lora_down = nn.Linear(in_dim, rank, bias=False)
            self.lora_up = nn.Linear(rank, out_dim, bias=False)
        else:
            self.lora_down = nn.Conv2d(in_dim, rank, kernel_size, padding=kernel_size // 2, bias=False)
            self.lora_up = nn.Conv2d(rank, out_dim, 1, bias=False)


class FakeFloatAlphaLoRAModule(StandardLoRASqueezeModuleMixin, nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.lora_name = "float_alpha_lora"
        self.lora_dim = rank
        self.scale = alpha / rank
        self.alpha = alpha
        self.lora_down = nn.Linear(in_dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_dim, bias=False)


class FakeNetwork(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.adapter = module

    def get_lora_squeeze_modules(self):
        return (self.adapter,)


class FakeMultiNetwork(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.adapters = nn.ModuleList(modules)

    def get_lora_squeeze_modules(self):
        return tuple(self.adapters)


class FakeScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loaded_state = None

    def state_dict(self):
        return {"step": 3}

    def load_state_dict(self, state):
        self.loaded_state = state

    def step(self):
        pass

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


class FakeAccelerator:
    num_processes = 1
    device = torch.device("cpu")

    def __init__(self, optimizer, scheduler):
        self._optimizers = [optimizer]
        self._schedulers = [scheduler]
        self.messages = []

    def unwrap_model(self, network):
        return network

    def wait_for_everyone(self):
        pass

    def prepare(self, optimizer, scheduler):
        self._optimizers.append(optimizer)
        self._schedulers.append(scheduler)
        return optimizer, scheduler

    def print(self, message):
        self.messages.append(message)


class FakeOptimizerUtil:
    def __init__(self, optimizer_class=torch.optim.SGD, lr=0.1):
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.training_steps = []

    def get_optimizer(self, args, trainable_params):
        optimizer = self.optimizer_class(trainable_params, lr=self.lr)
        return f"{type(optimizer).__module__}.{type(optimizer).__name__}", "", optimizer

    def get_scheduler_fix(self, args, optimizer, num_processes, training_steps=None):
        self.training_steps.append(training_steps)
        return FakeScheduler(optimizer)

    def get_optimizer_train_eval_fn(self, optimizer, args):
        return lambda: None, lambda: None


def effective_delta(module):
    up, down, _ = get_lora_factor_matrices(module)
    return up @ down * module.scale


def build_controller(schedule, network, optimizer_util=None, args=None):
    args = args or SimpleNamespace()
    raw_optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    optimizer = ReplaceableOptimizer(raw_optimizer)
    scheduler = ReplaceableLRScheduler(FakeScheduler(raw_optimizer))
    accelerator = FakeAccelerator(optimizer, scheduler)
    optimizer_util = optimizer_util or FakeOptimizerUtil()
    grad_refreshes = []

    controller = LoRASqueezeTrainingController(
        args=args,
        accelerator=accelerator,
        schedule=schedule,
        network=network,
        optimizer_util=optimizer_util,
        optimizer_name="torch.optim.SGD",
        optimizer_args="",
        optimizer=optimizer,
        optimizer_train_fn=lambda: None,
        optimizer_eval_fn=lambda: None,
        lr_scheduler=scheduler,
        lr_descriptions=None,
        prepare_optimizer_params_fn=lambda: (list(network.parameters()), None),
        prepare_grad_fn=lambda: grad_refreshes.append(True),
        update_metadata_fn=lambda: None,
        clean_memory_fn=lambda: None,
    )
    return controller, accelerator, optimizer_util, optimizer, scheduler, grad_refreshes
