import torch

from library.train_util import is_adv_optm_optimizer_type, tag_adv_optm_trainable_parameters


class DummyPeftNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_down = torch.nn.Linear(4, 2, bias=False)
        self.lora_up = torch.nn.Linear(2, 4, bias=False)

        self.split = torch.nn.Module()
        self.split.lora_down = torch.nn.ModuleList([torch.nn.Linear(4, 2, bias=False)])
        self.split.lora_up = torch.nn.ModuleList([torch.nn.Linear(2, 4, bias=False)])

        self.lokr_w1_b = torch.nn.Parameter(torch.randn(2, 4))
        self.lokr_w1_a = torch.nn.Parameter(torch.randn(4, 2))
        self.dora_scale = torch.nn.Parameter(torch.ones(4))

        self.frozen_lora_down = torch.nn.Linear(4, 2, bias=False)
        self.frozen_lora_down.weight.requires_grad_(False)


def test_is_adv_optm_optimizer_type():
    assert is_adv_optm_optimizer_type("adv_optm.SinkSGD_adv")
    assert is_adv_optm_optimizer_type("adv_optm.optim.AdamW_adv")
    assert not is_adv_optm_optimizer_type("AdamW")
    assert not is_adv_optm_optimizer_type(None)


def test_tag_adv_optm_trainable_parameters():
    network = DummyPeftNetwork()

    counts = tag_adv_optm_trainable_parameters(network)

    assert counts == {
        "lora_A": 3,
        "lora_B": 3,
        "dora_scale": 1,
        "oft": 0,
    }
    assert network.lora_down.weight._is_lora_A
    assert network.lora_up.weight._is_lora_B
    assert network.split.lora_down[0].weight._is_lora_A
    assert network.split.lora_up[0].weight._is_lora_B
    assert network.lokr_w1_b._is_lora_A
    assert network.lokr_w1_a._is_lora_B
    assert network.dora_scale._is_dora_scale
    assert not hasattr(network.frozen_lora_down.weight, "_is_lora_A")
