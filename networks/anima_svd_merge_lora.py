import argparse
import itertools

import torch
from tqdm import tqdm

from library.utils import setup_logging
from anima_merge_common import (
    build_anima_lora_metadata,
    cast_state_dict,
    get_lbw_multiplier,
    get_lora_module_name,
    get_metadata_model_info,
    load_state_dict,
    parse_lbws,
    save_to_file,
    str_to_dtype,
    tensor_to_float,
)

setup_logging()
import logging

logger = logging.getLogger(__name__)

CLAMP_QUANTILE = 0.99


def merge_lora_models(models, ratios, lbws, new_rank, new_conv_rank, device, merge_dtype):
    logger.info(f"new rank: {new_rank}, new conv rank: {new_conv_rank}")
    merged_weights = {}
    v2 = None
    base_model = None
    parsed_lbws = parse_lbws(lbws) if lbws else []

    for model, ratio, lbw in itertools.zip_longest(models, ratios, parsed_lbws):
        logger.info(f"loading: {model}")
        lora_sd, lora_metadata = load_state_dict(model, merge_dtype)
        v2, base_model = get_metadata_model_info(lora_metadata, v2, base_model)

        logger.info("merging effective deltas...")
        for key in tqdm(list(lora_sd.keys())):
            if not key.endswith(".lora_down.weight"):
                continue

            lora_module_name = get_lora_module_name(key)
            down_weight = lora_sd[key]
            up_weight = lora_sd[lora_module_name + ".lora_up.weight"]
            network_dim = down_weight.size(0)
            alpha = tensor_to_float(lora_sd.get(lora_module_name + ".alpha", network_dim))

            in_dim = down_weight.size(1)
            out_dim = up_weight.size(0)
            conv2d = len(down_weight.size()) == 4
            kernel_size = None if not conv2d else down_weight.size()[2:4]

            if lora_module_name not in merged_weights:
                weight = torch.zeros((out_dim, in_dim, *kernel_size) if conv2d else (out_dim, in_dim), dtype=torch.float)
            else:
                weight = merged_weights[lora_module_name]

            if device:
                weight = weight.to(device)
                up_weight = up_weight.to(device)
                down_weight = down_weight.to(device)
            weight = weight.to(torch.float)
            up_weight = up_weight.to(torch.float)
            down_weight = down_weight.to(torch.float)

            scale = alpha / network_dim
            scale *= get_lbw_multiplier(key, lbw)

            if not conv2d:
                weight = weight + ratio * (up_weight @ down_weight) * scale
            elif kernel_size == (1, 1):
                weight = (
                    weight
                    + ratio
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * scale
                )
            else:
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                weight = weight + ratio * conved * scale

            merged_weights[lora_module_name] = weight.to("cpu")

    logger.info("extracting merged Anima LoRA by SVD...")
    merged_lora_sd = {}
    output_ranks = {}
    with torch.no_grad():
        for lora_module_name, mat in tqdm(list(merged_weights.items())):
            if device:
                mat = mat.to(device)
            mat = mat.to(torch.float)

            conv2d = len(mat.size()) == 4
            kernel_size = None if not conv2d else mat.size()[2:4]
            conv2d_3x3 = conv2d and kernel_size != (1, 1)
            out_dim, in_dim = mat.size()[0:2]

            if conv2d:
                if conv2d_3x3:
                    mat = mat.flatten(start_dim=1)
                else:
                    mat = mat.squeeze()

            module_new_rank = new_conv_rank if conv2d_3x3 else new_rank
            module_new_rank = min(module_new_rank, in_dim, out_dim)
            output_ranks[lora_module_name] = module_new_rank

            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)

            U = U[:, :module_new_rank]
            S = S[:module_new_rank]
            U = U * S.unsqueeze(0)
            Vh = Vh[:module_new_rank, :]

            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, CLAMP_QUANTILE)
            low_val = -hi_val
            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)

            if conv2d:
                U = U.reshape(out_dim, module_new_rank, 1, 1)
                Vh = Vh.reshape(module_new_rank, in_dim, kernel_size[0], kernel_size[1])

            merged_lora_sd[lora_module_name + ".lora_up.weight"] = U.to("cpu").contiguous()
            merged_lora_sd[lora_module_name + ".lora_down.weight"] = Vh.to("cpu").contiguous()
            merged_lora_sd[lora_module_name + ".alpha"] = torch.tensor(module_new_rank, device="cpu")

    dims = str(new_rank) if all(rank == new_rank for rank in output_ranks.values()) else "Dynamic"
    alphas = dims
    return merged_lora_sd, dims, alphas, v2, base_model


def merge(args):
    assert args.save_to is not None, "save_to must be specified"
    assert args.models is not None and args.ratios is not None, "models and ratios must be specified"
    assert len(args.models) == len(args.ratios), "number of models must be equal to number of ratios"
    assert args.new_rank > 0, "new_rank must be positive"
    if args.new_conv_rank is not None:
        assert args.new_conv_rank > 0, "new_conv_rank must be positive"
    if args.lbws:
        assert len(args.models) == len(args.lbws), "number of models must be equal to number of lbws"
    else:
        args.lbws = []

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    new_conv_rank = args.new_conv_rank if args.new_conv_rank is not None else args.new_rank
    state_dict, dims, alphas, v2, base_model = merge_lora_models(
        args.models,
        args.ratios,
        args.lbws,
        args.new_rank,
        new_conv_rank,
        args.device,
        merge_dtype,
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
    parser.add_argument("--new_rank", type=int, default=4, help="rank of output LoRA")
    parser.add_argument(
        "--new_conv_rank",
        type=int,
        default=None,
        help="rank of output LoRA for Conv2d 3x3 modules; defaults to --new_rank",
    )
    parser.add_argument("--device", type=str, default=None, help="device for SVD work, e.g. cuda")
    parser.add_argument("--no_metadata", action="store_true", help="do not save SAI ModelSpec metadata")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    merge(parser.parse_args())
