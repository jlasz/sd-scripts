import sys
import torch
from safetensors.torch import load_file

lora_path = r"C:\AI\LoRA Trainer Colab\output\Rosine.safetensors"
sd = load_file(lora_path)

n_nan = 0
n_inf = 0
max_abs = 0.0

for k, v in sd.items():
    if not torch.is_tensor(v):
        continue
    if v.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        continue
    nan = torch.isnan(v).sum().item()
    inf = torch.isinf(v).sum().item()
    n_nan += nan
    n_inf += inf
    if v.numel():
        max_abs = max(max_abs, float(v.abs().max().item()))

print("keys:", len(sd))
print("NaNs:", n_nan, "Infs:", n_inf, "max|w|:", max_abs)