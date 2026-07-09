import json
import os
import re
import time
from typing import Optional

import torch
from safetensors.torch import load_file, save_file

from library import sai_model_spec
import library.model_io as model_io
from library.model_io import SS_METADATA_KEY_BASE_MODEL_VERSION, SS_METADATA_KEY_V2


ANIMA_NETWORK_MODULE = "networks.lora_anima"
ANIMA_BASE_MODEL_VERSION = "anima"
ANIMA_MODELSPEC_CONFIG = {"anima": "preview"}
ANIMA_BLOCK_RE = re.compile(r"^lora_unet_blocks_(\d+)_")


def str_to_dtype(value: Optional[str]) -> Optional[torch.dtype]:
    if value is None:
        return None
    if value in ("float", "float32", "fp32"):
        return torch.float
    if value == "fp16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {value}")


def load_state_dict(file_name: str, dtype: Optional[torch.dtype]):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = model_io.load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    if dtype is not None:
        for key in list(sd.keys()):
            value = sd[key]
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                sd[key] = value.to(dtype)

    return sd, metadata


def save_to_file(file_name: str, state_dict: dict, metadata: dict):
    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(state_dict, file_name, metadata=metadata)
    else:
        torch.save(state_dict, file_name)


def cast_state_dict(state_dict: dict, dtype: Optional[torch.dtype]):
    if dtype is None:
        return
    for key in list(state_dict.keys()):
        value = state_dict[key]
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point and value.dtype != dtype:
            state_dict[key] = value.to(dtype)


def tensor_to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


def get_lora_module_name(key: str) -> Optional[str]:
    for suffix in (".lora_down.weight", ".lora_up.weight"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def is_lora_weight_key(key: str) -> bool:
    return key.endswith(".lora_down.weight") or key.endswith(".lora_up.weight")


def collect_alphas_dims(state_dict: dict):
    alphas = {}
    dims = {}
    for key, value in state_dict.items():
        if key.endswith(".alpha"):
            lora_module_name = key[: -len(".alpha")]
            alphas[lora_module_name] = tensor_to_float(value)
        elif key.endswith(".lora_down.weight"):
            lora_module_name = key[: -len(".lora_down.weight")]
            dims[lora_module_name] = value.size(0)

    for lora_module_name, dim in dims.items():
        if lora_module_name not in alphas:
            alphas[lora_module_name] = float(dim)

    return alphas, dims


def get_metadata_model_info(metadata: Optional[dict], v2: Optional[str], base_model: Optional[str]):
    if metadata is None:
        return v2, base_model
    if v2 is None:
        v2 = metadata.get(SS_METADATA_KEY_V2, None)
    if base_model is None:
        base_model = metadata.get(SS_METADATA_KEY_BASE_MODEL_VERSION, None)
    return v2, base_model


def normalize_v2(v2: Optional[str]) -> str:
    return "False" if v2 is None else str(v2)


def v2_to_bool(v2: Optional[str]) -> bool:
    return str(v2).lower() == "true"


def unique_or_dynamic(values) -> str:
    value_set = list({str(v) for v in values})
    if len(value_set) == 1:
        return value_set[0]
    return "Dynamic"


def parse_lbws(lbws: list[str]):
    parsed = []
    for lbw in lbws:
        values = json.loads(lbw)
        if not isinstance(values, list):
            raise ValueError("each --lbws value must be a JSON list")
        if not values:
            raise ValueError("each --lbws list must contain at least one value")
        if not all(isinstance(value, (int, float)) for value in values):
            raise ValueError("all --lbws values must be numbers")
        parsed.append([float(value) for value in values])
    return parsed


def get_lbw_multiplier(key: str, lbw: Optional[list[float]]) -> float:
    if lbw is None:
        return 1.0

    match = ANIMA_BLOCK_RE.match(key)
    if match is None:
        return 1.0

    block_index = int(match.group(1))
    if block_index >= len(lbw):
        raise ValueError(
            f"--lbws list has {len(lbw)} entries, but {key} belongs to Anima block {block_index}"
        )
    return lbw[block_index]


def build_anima_lora_metadata(
    state_dict: dict,
    models: list[str],
    save_to: str,
    dims: str,
    alphas: str,
    v2: Optional[str],
    base_model: Optional[str],
    no_metadata: bool,
    network_args: Optional[dict] = None,
):
    v2 = normalize_v2(v2)
    base_model = base_model or ANIMA_BASE_MODEL_VERSION
    metadata = model_io.build_minimum_network_metadata(
        v2,
        base_model,
        ANIMA_NETWORK_MODULE,
        dims,
        alphas,
        network_args,
    )

    model_hash, legacy_hash = model_io.precalculate_safetensors_hashes(state_dict, metadata)
    metadata["sshs_model_hash"] = model_hash
    metadata["sshs_legacy_hash"] = legacy_hash

    if not no_metadata:
        merged_from = sai_model_spec.build_merged_from(models)
        title = os.path.splitext(os.path.basename(save_to))[0]
        sai_metadata = sai_model_spec.build_metadata(
            state_dict,
            v2_to_bool(v2),
            False,
            False,
            True,
            False,
            time.time(),
            title=title,
            merged_from=merged_from,
            model_config=ANIMA_MODELSPEC_CONFIG,
        )
        metadata.update(sai_metadata)

    return metadata
