import argparse
import json
import math
import os
import time
from typing import Any, Dict, Optional, Tuple, Union

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from library import sai_model_spec, train_util
from library.utils import setup_logging, str_to_dtype

setup_logging()
import logging

logger = logging.getLogger(__name__)


ANIMA_NETWORK_MODULE = "networks.lora_anima"
ANIMA_BASE_MODEL = "anima"
ANIMA_MODEL_CONFIG = {"anima": "preview"}

SD_SCRIPTS_PREFIXES = ("lora_unet_", "lora_te_")
COMFY_PREFIXES = ("diffusion_model.", "text_encoders.qwen3_06b.transformer.model.")

LORA_DOWN_UP = {
    "down_marker": ".lora_down.",
    "up_marker": ".lora_up.",
    "format": "sd-scripts",
}
LORA_A_B = {
    "down_marker": ".lora_A.",
    "up_marker": ".lora_B.",
    "format": "comfy",
}
LORA_MARKER_SETS = (LORA_DOWN_UP, LORA_A_B)


def load_state_dict(file_name: str, dtype: Optional[torch.dtype]):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = train_util.load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    if dtype is not None:
        for key in list(sd.keys()):
            value = sd[key]
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                sd[key] = value.to(dtype)

    return sd, metadata


def save_to_file(file_name: str, state_dict: Dict[str, Union[Any, torch.Tensor]], dtype: Optional[torch.dtype], metadata):
    if dtype is not None:
        logger.info(f"converting to {dtype}...")
        for key in tqdm(list(state_dict.keys())):
            value = state_dict[key]
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                state_dict[key] = value.to(dtype)

    logger.info(f"saving model to: {file_name}")
    save_file(state_dict, file_name, metadata=metadata)


def parse_lora_weight_key(key: str) -> Optional[Tuple[str, str, str, str, str]]:
    for marker_set in LORA_MARKER_SETS:
        down_marker = marker_set["down_marker"]
        up_marker = marker_set["up_marker"]
        lora_format = marker_set["format"]

        if down_marker in key:
            module_name, weight_name = key.split(down_marker, 1)
            return module_name, "down", down_marker, up_marker, lora_format
        if up_marker in key:
            module_name, weight_name = key.split(up_marker, 1)
            return module_name, "up", down_marker, up_marker, lora_format

    return None


def alpha_to_float(alpha) -> float:
    if isinstance(alpha, torch.Tensor):
        return float(alpha.detach().float().cpu().item())
    return float(alpha)


def detect_lora_format(module_name: str) -> str:
    if module_name.startswith(SD_SCRIPTS_PREFIXES):
        return "sd-scripts"
    if module_name.startswith(COMFY_PREFIXES):
        return "comfy"
    return "unknown"


def metadata_is_anima(metadata: dict) -> bool:
    network_module = metadata.get(train_util.SS_METADATA_KEY_NETWORK_MODULE)
    base_model = metadata.get(train_util.SS_METADATA_KEY_BASE_MODEL_VERSION)
    architecture = metadata.get("modelspec.architecture")

    return (
        network_module == ANIMA_NETWORK_MODULE
        or base_model == ANIMA_BASE_MODEL
        or (architecture is not None and architecture.startswith("anima"))
    )


def validate_anima_lora(file_name: str, state_dict: dict, metadata: dict, allow_non_anima: bool = False):
    modules = []
    formats = set()

    for key in state_dict.keys():
        parsed = parse_lora_weight_key(key)
        if parsed is None:
            continue

        module_name, _, _, _, marker_format = parsed
        modules.append(module_name)
        detected_format = detect_lora_format(module_name)
        formats.add(detected_format if detected_format != "unknown" else marker_format)

    if not modules:
        raise ValueError(f"No LoRA up/down tensors were found in {file_name}")

    network_module = metadata.get(train_util.SS_METADATA_KEY_NETWORK_MODULE)
    base_model = metadata.get(train_util.SS_METADATA_KEY_BASE_MODEL_VERSION)
    architecture = metadata.get("modelspec.architecture")
    clearly_other_model = (
        (network_module is not None and network_module != ANIMA_NETWORK_MODULE)
        or (base_model is not None and base_model != ANIMA_BASE_MODEL)
        or (architecture is not None and not architecture.startswith("anima"))
    )

    if clearly_other_model and not allow_non_anima:
        raise ValueError(
            f"{file_name} does not look like an Anima LoRA. "
            f"metadata: ss_network_module={network_module}, ss_base_model_version={base_model}, "
            f"modelspec.architecture={architecture}. Use --allow_non_anima to override."
        )

    if not metadata_is_anima(metadata):
        logger.warning(
            f"{file_name} is not tagged as Anima in metadata. Continuing because the tensor keys look mergeable."
        )

    return formats


def get_common_network_args(metadata_list):
    values = [metadata.get(train_util.SS_METADATA_KEY_NETWORK_ARGS) for metadata in metadata_list]
    values = [value for value in values if value is not None]
    if not values or len(set(values)) != 1:
        return None

    try:
        parsed = json.loads(values[0])
    except Exception:
        return None

    return parsed if isinstance(parsed, dict) else None


def can_concat(existing: torch.Tensor, incoming: torch.Tensor, dim: int) -> bool:
    if len(existing.shape) != len(incoming.shape):
        return False
    for index, (left, right) in enumerate(zip(existing.shape, incoming.shape)):
        if index != dim and left != right:
            return False
    return True


def merge_lora_models(
    models,
    ratios,
    merge_dtype: Optional[torch.dtype],
    concat: bool = False,
    shuffle: bool = False,
    allow_mixed_formats: bool = False,
    allow_non_anima: bool = False,
):
    base_alphas = {}
    base_dims = {}
    base_formats = {}
    base_down_markers = {}
    base_up_markers = {}

    metadata_list = []
    model_formats = []
    merged_sd = {}
    base_model = None

    for model, ratio in zip(models, ratios):
        logger.info(f"loading: {model}")
        lora_sd, lora_metadata = load_state_dict(model, merge_dtype)
        metadata_list.append(lora_metadata)

        formats = validate_anima_lora(model, lora_sd, lora_metadata, allow_non_anima)
        model_formats.append(formats)
        if len(formats) > 1 and not allow_mixed_formats:
            raise ValueError(f"{model} contains mixed LoRA key formats: {sorted(formats)}")

        if lora_metadata is not None and base_model is None:
            base_model = lora_metadata.get(train_util.SS_METADATA_KEY_BASE_MODEL_VERSION, None)

        alphas = {}
        dims = {}

        for key, value in lora_sd.items():
            if key.endswith(".alpha"):
                lora_module_name = key[: key.rfind(".alpha")]
                alpha = alpha_to_float(value)
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha
                continue

            parsed = parse_lora_weight_key(key)
            if parsed is None:
                continue

            lora_module_name, kind, down_marker, up_marker, marker_format = parsed
            if kind != "down":
                continue

            dim = value.size()[0]
            dims[lora_module_name] = dim
            if lora_module_name not in base_dims:
                base_dims[lora_module_name] = dim
                base_formats[lora_module_name] = detect_lora_format(lora_module_name)
                base_down_markers[lora_module_name] = down_marker
                base_up_markers[lora_module_name] = up_marker

        for lora_module_name, dim in dims.items():
            if lora_module_name not in alphas:
                alpha = dim
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha

        logger.info(f"dim: {sorted(set(dims.values()))}, alpha: {sorted(set(alphas.values()))}")

        logger.info("merging...")
        for key in tqdm(list(lora_sd.keys())):
            if key.endswith(".alpha"):
                continue

            parsed = parse_lora_weight_key(key)
            if parsed is None:
                logger.warning(f"Skipping non-LoRA tensor: {key}")
                continue

            lora_module_name, kind, _, _, _ = parsed
            if kind == "up" and concat:
                concat_dim = 1
            elif kind == "down" and concat:
                concat_dim = 0
            else:
                concat_dim = None

            if lora_module_name not in alphas:
                raise ValueError(f"Missing alpha/dim information for {lora_module_name} in {model}")

            base_alpha = base_alphas[lora_module_name]
            alpha = alphas[lora_module_name]

            scale = math.sqrt(alpha / base_alpha) * ratio
            if kind == "up":
                scale = abs(scale)

            weighted_tensor = lora_sd[key] * scale

            if key in merged_sd:
                if concat_dim is not None:
                    if not can_concat(merged_sd[key], weighted_tensor, concat_dim):
                        raise AssertionError(
                            f"weights shape mismatch for {key}: {tuple(merged_sd[key].shape)} vs "
                            f"{tuple(weighted_tensor.shape)}"
                        )
                    merged_sd[key] = torch.cat([merged_sd[key], weighted_tensor], dim=concat_dim)
                else:
                    if merged_sd[key].size() != weighted_tensor.size():
                        raise AssertionError(
                            f"weights shape mismatch for {key}: {tuple(merged_sd[key].shape)} vs "
                            f"{tuple(weighted_tensor.shape)}. Use --concat when merging different ranks."
                        )
                    merged_sd[key] = merged_sd[key] + weighted_tensor
            else:
                merged_sd[key] = weighted_tensor

    if not allow_mixed_formats:
        normalized_formats = [tuple(sorted(formats)) for formats in model_formats]
        if len(set(normalized_formats)) > 1:
            raise ValueError(
                "Input LoRAs use different key formats. Convert them to the same format first, "
                "or pass --allow_mixed_formats if you intentionally want a union of unrelated keys."
            )

    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)

        if shuffle:
            down_key = lora_module_name + base_down_markers[lora_module_name] + "weight"
            up_key = lora_module_name + base_up_markers[lora_module_name] + "weight"
            if down_key not in merged_sd or up_key not in merged_sd:
                logger.warning(f"Cannot shuffle {lora_module_name}; up/down keys are incomplete.")
                continue

            dim = merged_sd[down_key].shape[0]
            perm = torch.randperm(dim)
            merged_sd[down_key] = merged_sd[down_key][perm]
            merged_sd[up_key] = merged_sd[up_key][:, perm]

    if len(base_dims) == 0:
        raise ValueError("No LoRA modules were found to merge.")

    logger.info("merged model")
    logger.info(f"dim: {sorted(set(base_dims.values()))}, alpha: {sorted(set(base_alphas.values()))}")

    dims_list = list(set(base_dims.values()))
    alphas_list = list(set(base_alphas.values()))
    all_same_dims = len(set(dims_list)) == 1
    all_same_alphas = len(set(alphas_list)) == 1

    dims = f"{dims_list[0]}" if all_same_dims else "Dynamic"
    alphas = f"{alphas_list[0]}" if all_same_alphas else "Dynamic"
    network_args = get_common_network_args(metadata_list)
    metadata = train_util.build_minimum_network_metadata(
        str(False), base_model or ANIMA_BASE_MODEL, ANIMA_NETWORK_MODULE, dims, alphas, network_args
    )

    return merged_sd, metadata


def merge(args):
    if args.models is None:
        args.models = []
    if args.ratios is None:
        args.ratios = []

    if len(args.models) != len(args.ratios):
        raise AssertionError("number of models must be equal to number of ratios")
    if len(args.models) == 0:
        raise AssertionError("at least one model must be specified")
    if args.save_to is None:
        raise AssertionError("--save_to must be specified")

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision, merge_dtype)

    dest_dir = os.path.dirname(args.save_to)
    if dest_dir and not os.path.exists(dest_dir):
        logger.info(f"creating directory: {dest_dir}")
        os.makedirs(dest_dir)

    state_dict, metadata = merge_lora_models(
        args.models,
        args.ratios,
        merge_dtype,
        args.concat,
        args.shuffle,
        args.allow_mixed_formats,
        args.allow_non_anima,
    )

    logger.info("calculating hashes and creating metadata...")
    model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
    metadata["sshs_model_hash"] = model_hash
    metadata["sshs_legacy_hash"] = legacy_hash

    if not args.no_metadata:
        merged_from = sai_model_spec.build_merged_from(args.models)
        title = os.path.splitext(os.path.basename(args.save_to))[0]
        sai_metadata = sai_model_spec.build_metadata(
            state_dict,
            False,
            False,
            False,
            True,
            False,
            time.time(),
            title=title,
            merged_from=merged_from,
            model_config=ANIMA_MODEL_CONFIG,
        )
        metadata.update(sai_metadata)

    save_to_file(args.save_to, state_dict, save_dtype, metadata)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge Anima LoRA models.")
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        help="precision for saving; defaults to --precision. Supported: float/float32, fp16, bf16, fp8 variants",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float",
        help="precision for merge calculation. float is recommended",
    )
    parser.add_argument("--save_to", type=str, default=None, help="destination safetensors file")
    parser.add_argument("--models", type=str, nargs="*", help="Anima LoRA models to merge")
    parser.add_argument("--ratios", type=float, nargs="*", help="ratios for each LoRA model")
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save SAI modelspec metadata; minimum ss_network metadata is still saved",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="concat LoRAs instead of summing matching ranks; use when input ranks differ",
    )
    parser.add_argument("--shuffle", action="store_true", help="shuffle LoRA rank order after merging")
    parser.add_argument(
        "--allow_mixed_formats",
        action="store_true",
        help="allow merging sd-scripts and ComfyUI key formats into one file as separate key sets",
    )
    parser.add_argument(
        "--allow_non_anima",
        action="store_true",
        help="allow files without Anima metadata or with conflicting metadata",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    merge(args)
