import argparse
import itertools
import math

import torch
from tqdm import tqdm

from library.utils import setup_logging
from anima_merge_common import (
    build_anima_lora_metadata,
    cast_state_dict,
    collect_alphas_dims,
    get_lbw_multiplier,
    get_lora_module_name,
    get_metadata_model_info,
    is_lora_weight_key,
    load_state_dict,
    parse_lbws,
    save_to_file,
    str_to_dtype,
    unique_or_dynamic,
)

setup_logging()
import logging

logger = logging.getLogger(__name__)


def merge_lora_models(models, ratios, lbws, merge_dtype, concat=False, shuffle=False):
    base_alphas = {}
    base_dims = {}
    merged_sd = {}
    v2 = None
    base_model = None

    parsed_lbws = parse_lbws(lbws) if lbws else []

    for model, ratio, lbw in itertools.zip_longest(models, ratios, parsed_lbws):
        logger.info(f"loading: {model}")
        lora_sd, lora_metadata = load_state_dict(model, merge_dtype)
        v2, base_model = get_metadata_model_info(lora_metadata, v2, base_model)

        alphas, dims = collect_alphas_dims(lora_sd)
        for lora_module_name, alpha in alphas.items():
            if lora_module_name not in base_alphas:
                base_alphas[lora_module_name] = alpha
        for lora_module_name, dim in dims.items():
            if lora_module_name not in base_dims:
                base_dims[lora_module_name] = dim

        logger.info(f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}")

        logger.info("merging...")
        for key in tqdm(lora_sd.keys()):
            if key.endswith(".alpha"):
                continue
            if not is_lora_weight_key(key):
                logger.warning(f"skip non-LoRA weight key: {key}")
                continue

            lora_module_name = get_lora_module_name(key)
            base_alpha = base_alphas[lora_module_name]
            alpha = alphas[lora_module_name]

            scale = math.sqrt(alpha / base_alpha) * ratio
            scale *= get_lbw_multiplier(key, lbw)
            scale = abs(scale) if key.endswith(".lora_up.weight") else scale

            if key.endswith(".lora_up.weight") and concat:
                concat_dim = 1
            elif key.endswith(".lora_down.weight") and concat:
                concat_dim = 0
            else:
                concat_dim = None

            if key in merged_sd:
                assert merged_sd[key].size() == lora_sd[key].size() or concat_dim is not None, (
                    "weights shape mismatch, different dims? Use --concat to combine different ranks."
                )
                if concat_dim is not None:
                    merged_sd[key] = torch.cat([merged_sd[key], lora_sd[key] * scale], dim=concat_dim)
                else:
                    merged_sd[key] = merged_sd[key] + lora_sd[key] * scale
            else:
                merged_sd[key] = lora_sd[key] * scale

    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)

        if shuffle:
            key_down = lora_module_name + ".lora_down.weight"
            key_up = lora_module_name + ".lora_up.weight"
            if key_down in merged_sd and key_up in merged_sd:
                dim = merged_sd[key_down].shape[0]
                perm = torch.randperm(dim)
                merged_sd[key_down] = merged_sd[key_down][perm]
                merged_sd[key_up] = merged_sd[key_up][:, perm]

    output_alphas, output_dims = collect_alphas_dims(merged_sd)
    dims = unique_or_dynamic(output_dims.values())
    alphas = unique_or_dynamic(output_alphas.values())

    logger.info("merged Anima LoRA")
    logger.info(f"dim: {list(set(output_dims.values()))}, alpha: {list(set(output_alphas.values()))}")

    return merged_sd, dims, alphas, v2, base_model


def merge(args):
    assert args.save_to is not None, "save_to must be specified"
    assert args.models is not None and args.ratios is not None, "models and ratios must be specified"
    assert len(args.models) == len(args.ratios), "number of models must be equal to number of ratios"
    if args.lbws:
        assert len(args.models) == len(args.lbws), "number of models must be equal to number of lbws"
    else:
        args.lbws = []

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    state_dict, dims, alphas, v2, base_model = merge_lora_models(
        args.models, args.ratios, args.lbws, merge_dtype, args.concat, args.shuffle
    )

    cast_state_dict(state_dict, save_dtype)

    logger.info("calculating hashes and creating metadata...")
    metadata = build_anima_lora_metadata(
        state_dict,
        args.models,
        args.save_to,
        dims,
        alphas,
        v2,
        base_model,
        args.no_metadata,
    )

    logger.info(f"saving model to: {args.save_to}")
    save_to_file(args.save_to, state_dict, metadata)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_precision", type=str, default=None, choices=[None, "float", "float32", "fp32", "fp16", "bf16"])
    parser.add_argument("--precision", type=str, default="float", choices=["float", "float32", "fp32", "fp16", "bf16"])
    parser.add_argument("--save_to", type=str, default=None, help="destination file name")
    parser.add_argument("--models", type=str, nargs="*", help="Anima LoRA models to merge")
    parser.add_argument("--ratios", type=float, nargs="*", help="ratios for each LoRA model")
    parser.add_argument(
        "--lbws",
        type=str,
        nargs="*",
        help="optional JSON block-weight list for each model; Anima DiT block keys use lora_unet_blocks_<index>_",
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save SAI ModelSpec metadata")
    parser.add_argument(
        "--concat",
        action="store_true",
        help="concat LoRA ranks instead of adding factor tensors; output rank is the sum of input ranks",
    )
    parser.add_argument("--shuffle", action="store_true", help="shuffle LoRA rank channels after merging")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    merge(parser.parse_args())
