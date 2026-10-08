# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Rebuild + assemble BF16 deployment weights for Qwen3.6-35B-A3B
quantized by run_qwen36.py (packed 3D expert layout).

Counterpart of rebuild_bf16.py for the qwen3_5_moe checkpoint: the
per-expert QuantizedWeights are reconstructed to 2D, gate|up rows are
concatenated back into `gate_up_proj[e]`, all experts stacked into the
packed 3D tensors, and pass_assemble (imported from rebuild_bf16,
unchanged) swaps them into a checkpoint copy:

    expert_parts/model.language_model.layers.{i}.mlp.experts.gate_up_proj
        .safetensors  {"w": (E, 2*inter, hidden) bf16}
    expert_parts/...mlp.experts.down_proj.safetensors {"w": (E, hidden, inter)}

Two passes, same CLI contract as rebuild_bf16:
    python -m vptq.tools.quantize.rebuild_qwen36 --ckpt ... --quant-dir ... \
        --output-dir ... --layer-range 0:40 --group-num 128 --row-tile 16
    python -m vptq.tools.quantize.rebuild_qwen36 ... --assemble
"""

import argparse
import json
import logging
import os

import torch
from safetensors.torch import save_file

from vptq.tools.quantize.rebuild_bf16 import (
    load_layer_quant,
    pass_assemble,
    reconstruct_weight,
    wmap_shape,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vptq.quantize.rebuild_qwen36")

_PREFIX_CANDIDATES = (
    "model.language_model.layers.{i}.",
    "model.layers.{i}.",
    "layers.{i}.",
)


def detect_layer_prefix(weight_map: dict) -> str:
    for tmpl in _PREFIX_CANDIDATES:
        probe = tmpl.format(i=0) + "mlp.experts.gate_up_proj"
        if probe in weight_map:
            return tmpl
    raise ValueError("packed expert tensors not found in checkpoint index")


def _fold_scales(qw, Q, out_f, in_f):
    """Mirror rebuild_bf16.pass_rebuild: multiply stored scales back."""
    if getattr(qw, "row_scale", None) is not None:
        s = qw.row_scale[:out_f].to(Q.device).float()
        Q = Q.float() * s[:, None]
        if getattr(qw, "col_scale", None) is not None:
            d = qw.col_scale[:in_f].to(Q.device).float()
            Q = Q * d[None, :]
    return Q


@torch.no_grad()
def pass_rebuild_qwen(args):
    with open(os.path.join(args.ckpt, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    prefix = detect_layer_prefix(wmap)
    os.makedirs(args.parts_dir, exist_ok=True)
    device = torch.device(args.device)
    for layer_idx in range(*args.layer_range):
        quant = load_layer_quant(args.quant_dir, layer_idx)
        if not quant:
            logger.warning("layer %d: no quant files, skip", layer_idx)
            continue
        gu_full = prefix.format(i=layer_idx) + "mlp.experts.gate_up_proj"
        dn_full = prefix.format(i=layer_idx) + "mlp.experts.down_proj"
        outs = {}
        for full in (gu_full, dn_full):
            part = os.path.join(args.parts_dir, full + ".safetensors")
            if os.path.exists(part):
                logger.info("exists, skip: %s", part)
            else:
                outs[full] = part
        if not outs:
            continue
        n_experts, gu_rows, hidden = wmap_shape(args.ckpt, wmap, gu_full)
        _, dn_out, inter = wmap_shape(args.ckpt, wmap, dn_full)
        assert gu_rows == 2 * inter, (gu_rows, inter)
        gate_up = torch.empty(n_experts, gu_rows, hidden, dtype=torch.float32)
        down = torch.empty(n_experts, dn_out, inter, dtype=torch.float32)
        for e in range(n_experts):
            qg = reconstruct_weight(
                quant[f"experts.{e}.gate"],
                args.group_num,
                args.row_tile,
                device,
            )
            qu = reconstruct_weight(
                quant[f"experts.{e}.up"], args.group_num, args.row_tile, device
            )
            qd = reconstruct_weight(
                quant[f"experts.{e}.down"],
                args.group_num,
                args.row_tile,
                device,
            )
            qg = _fold_scales(quant[f"experts.{e}.gate"], qg, inter, hidden)
            qu = _fold_scales(quant[f"experts.{e}.up"], qu, inter, hidden)
            qd = _fold_scales(quant[f"experts.{e}.down"], qd, hidden, inter)
            # 行序即 Qwen3_5MoeExperts.forward 的 chunk 顺序：[0:inter]=gate
            gate_up[e] = torch.cat([qg[:inter], qu[:inter]], 0).float()
            down[e] = qd[:hidden].float()
        for full, tensor in ((gu_full, gate_up), (dn_full, down)):
            path = os.path.join(args.parts_dir, full + ".safetensors")
            tmp = path + ".tmp"
            save_file(
                {"w": tensor.to(torch.bfloat16).cpu().contiguous()},
                tmp,
            )
            os.replace(tmp, path)
            logger.info("rebuilt %s %s", tuple(tensor.shape), full)
        del quant, gate_up, down


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quant-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layer-range", default="0:40")
    p.add_argument("--group-num", type=int, default=128)
    p.add_argument("--row-tile", type=int, default=256)
    p.add_argument("--device", default="npu")
    p.add_argument("--assemble", action="store_true")
    args = p.parse_args()
    args.parts_dir = os.path.join(args.output_dir, "expert_parts")
    args.layer_range = tuple(int(x) for x in args.layer_range.split(":"))
    if args.assemble:
        pass_assemble(args)
    else:
        pass_rebuild_qwen(args)


if __name__ == "__main__":
    main()
