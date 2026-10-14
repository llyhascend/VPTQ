# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Rebuild a BF16 checkpoint from tile-quantized experts.

Two passes:

  pass 1 (default): rebuild expert matrices for layers in --layer-range
    from the quant parts into bf16 part files
    (out_dir/expert_parts/<weight_name>.pt). Parallel-safe across
    processes via disjoint layer ranges.

  pass 2 (--assemble): stream the original checkpoint shards, swap in
    rebuilt expert tensors, copy everything else through, and write the
    new BF16 checkpoint (same shard layout; index.json preserved, so
    config/tokenizer just get copied).

Tile layout (from vq.py): codebooks[T=group_num*n_row] with tile index
gi*n_row + t; indices (T, cols, nvec); tile (gi, t) covers rows
[t*row_tile:(t+1)*row_tile], cols [gi*cw:(gi+1)*cw] of the padded weight.

Usage:
  python -m vptq.tools.quantize.rebuild_bf16 --quant-dir Q --ckpt C \
      --output-dir O --layer-range 0:11 --device npu
  python -m vptq.tools.quantize.rebuild_bf16 --quant-dir Q --ckpt C \
      --output-dir O --assemble
"""

import argparse
import glob
import json
import logging
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vptq.rebuild")

EXPERT_RE = re.compile(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight")


def load_rotation(path: str) -> torch.Tensor:
    from safetensors.torch import load_file

    return load_file(path)["global_rotation"].float()


@torch.no_grad()
def reconstruct_weight(qw, group_num: int, row_tile: int, device) -> torch.Tensor:
    """Batched tile reconstruction -> (out_pad, in_f) fp32 on device.

    Vectorized equivalent of the per-tile reference in export_tile.
    """
    def expand(cents, idx):
        cents = cents.float().to(device)
        idx = idx.long().to(device)
        T, cols, nvec = idx.shape
        k, v = cents.shape[1], cents.shape[2]
        flat = cents.reshape(T * k, v)
        off = torch.arange(T, device=device).view(-1, 1, 1) * k
        g = flat[(idx + off).reshape(-1)].view(T, cols, nvec, v)
        g = g.permute(0, 2, 3, 1).reshape(T, nvec * v, cols)  # (T, span, cols)
        n_row = T // group_num
        g = g.view(group_num, n_row, nvec * v, cols)
        g = g.permute(1, 2, 0, 3)  # (n_row, span, group_num, cols)
        return g.reshape(n_row * (nvec * v), group_num * cols)

    Q = expand(qw.centroids, qw.indices)
    if qw.res_centroids is not None:
        Q = Q + expand(qw.res_centroids, qw.res_indices)
    return Q


def load_layer_quant(quant_dir: str, layer_idx: int) -> dict:
    """Merge all expert-range shard files of one layer."""
    merged = {}
    for f in sorted(
        glob.glob(os.path.join(quant_dir, f"quant_layer_{layer_idx:04d}_e*.pt"))
    ):
        merged.update(torch.load(f, weights_only=False))
    if not merged:
        # maybe a single unscoped file
        p = os.path.join(quant_dir, f"quant_layer_{layer_idx:04d}.pt")
        if os.path.exists(p):
            merged = torch.load(p, weights_only=False)
    return merged


def _fallback_sets(path):
    """Parse fallback_list.json -> (full_keys, down_keys).

    full_keys: experts whose w1/w2/w3 all fall back to BF16;
    down_keys: experts whose w2 only falls back.
    Keys match the quant short names, e.g. ffn.experts.6.w2.weight.
    """
    if not path:
        return frozenset(), frozenset()
    with open(path) as f:
        lst = json.load(f)
    full = frozenset(
        (li, f"ffn.experts.{e}.{w}.weight")
        for li, e in lst["full"] for w in ("w1", "w2", "w3")
    )
    down = frozenset(
        (li, f"ffn.experts.{e}.w2.weight") for li, e in lst["down_only"]
    )
    return full, down


def pass_rebuild(args):
    os.makedirs(args.parts_dir, exist_ok=True)
    fb_full, fb_down = _fallback_sets(getattr(args, "fallback_list", None))
    with open(os.path.join(args.ckpt, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    device = torch.device(args.device)
    wrap_R = load_rotation(args.wrap_rotation) if args.wrap_rotation else None
    if wrap_R is not None:
        logger.info("wrap mode: expert parts are Q(W@R); wrapping back "
                    "with @ R from %s", args.wrap_rotation)
    for layer_idx in range(*args.layer_range):
        quant = load_layer_quant(args.quant_dir, layer_idx)
        if not quant:
            logger.warning("layer %d: no quant files, skip", layer_idx)
            continue
        done = 0
        for short, qw in quant.items():
            if (layer_idx, short) in fb_full:
                logger.info("fallback(full): L%d %s", layer_idx, short)
                continue
            if (layer_idx, short) in fb_down:
                logger.info("fallback(down): L%d %s", layer_idx, short)
                continue
            full = f"layers.{layer_idx}.{short}"
            if full not in wmap:
                logger.warning("not in checkpoint index: %s", full)
                continue
            out_path = os.path.join(
                args.parts_dir, full + ".safetensors"
            )
            if os.path.exists(out_path):
                continue
            Q = reconstruct_weight(
                qw, args.group_num, args.row_tile, device
            )
            out_f, in_f = wmap_shape(args.ckpt, wmap, full)
            Q = Q[:out_f]
            # dual-scale normalization: multiply the stored scales back
            # (W_norm = W / s / d was quantized; here Ŵ = s ⊙ Q ⊙ d).
            # Row scale is exact to out_f; col scale may be longer than
            # in_f for padded matrices — slice before broadcasting.
            if getattr(qw, "row_scale", None) is not None:
                s = qw.row_scale[:out_f].to(Q.device).float()
                Q = Q.float() * s[:, None]
                # col_scale 仅 dual-norm 路径存在；row-fp8-only 存 None
                if getattr(qw, "col_scale", None) is not None:
                    d = qw.col_scale[:in_f].to(Q.device).float()
                    Q = Q * d[None, :]
            if wrap_R is not None and EXPERT_RE.fullmatch(full):
                # wrap-around: stored weight is Q(V) @ R^T with V = W @ R
                # (signed Hadamard R is orthogonal but NOT symmetric:
                # R^-1 = R^T); the result is a plain dense weight for
                # the UNROTATED checkpoint. Per-input-dim R (w2: 2048,
                # w1/w3: 4096) matches run.py's maybe_wrap.
                from vptq.tools.quantize.hadamard import wrap_rotation

                R = wrap_rotation(Q.shape[1])
                Q = (Q.float() @ R.t().to(Q.device)).to(Q.dtype)
            save_file(
                {"w": Q.to(torch.bfloat16).cpu().contiguous()},
                out_path,
            )
            done += 1
        logger.info("layer %d: rebuilt %d/%d experts", layer_idx, done,
                    len(quant))
        if getattr(args, "fallback_mxfp4", False):
            fb_done = _rebuild_fallback_mxfp4(
                args, layer_idx, fb_full | fb_down, wmap, device)
            logger.info("layer %d: fallback mxfp4 %d parts", layer_idx,
                        fb_done)
        del quant


def _rebuild_fallback_mxfp4(args, layer_idx, fb_keys, wmap, device):
    """Pre-quantize fallback experts in wrap domain (2026-10-14).

    part = mxfp4_ceilx_roundtrip(W @ R, 7.25) @ R^T, stored BF16;
    assemble swaps it into the checkpoint, and the deploy yaml rule B
    (method scale_one) performs the final e4m3 cast. R is the same
    block-diagonal Hadamard (block=32, seed=0) as VPTQ expert wrap.
    """
    from vptq.tools.quantize.hadamard import wrap_rotation
    from vptq.tools.quantize.mxfp4_ceilx import mxfp4_ceilx_roundtrip

    done = 0
    for li, short in sorted(fb_keys):
        if li != layer_idx:
            continue
        full = f"layers.{li}.{short}"
        if full not in wmap:
            logger.warning("fallback not in checkpoint index: %s", full)
            continue
        out_path = os.path.join(args.parts_dir, full + ".safetensors")
        if os.path.exists(out_path):
            continue
        shard = os.path.join(args.ckpt, wmap[full])
        with safe_open(shard, framework="pt", device="cpu") as f:
            W = f.get_tensor(full).to(device)
        R = wrap_rotation(W.shape[1]).to(device)
        V = W.float() @ R
        Vdq = mxfp4_ceilx_roundtrip(V)
        part = (Vdq @ R.t()).to(torch.bfloat16).cpu().contiguous()
        save_file({"w": part}, out_path)
        done += 1
    return done


def wmap_shape(ckpt, wmap, name):
    # read just the header of the owning shard for the shape
    shard = wmap[name]
    with safe_open(os.path.join(ckpt, shard), framework="pt",
                   device="cpu") as f:
        return tuple(f.get_slice(name).get_shape())


def pass_assemble(args):
    with open(os.path.join(args.ckpt, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    os.makedirs(args.output_dir, exist_ok=True)
    # non-tensor files
    for f in os.listdir(args.ckpt):
        if not f.endswith(".safetensors") and os.path.isfile(
            os.path.join(args.ckpt, f)
        ):
            shutil.copy2(os.path.join(args.ckpt, f), args.output_dir)
    shards = sorted(set(wmap.values()))
    swapped = copied = 0
    for shard in shards:
        src = os.path.join(args.ckpt, shard)
        dst = os.path.join(args.output_dir, shard)
        if os.path.exists(dst):
            # resume: count its contents without rewriting
            with safe_open(dst, framework="pt", device="cpu") as f:
                names = list(f.keys())
            n_swap = sum(
                1 for n in names
                if os.path.exists(
                    os.path.join(args.parts_dir, n + ".safetensors")
                )
            )
            swapped += n_swap
            copied += len(names) - n_swap
            logger.info("shard %s exists, skip (resume)", shard)
            continue
        out = {}
        with safe_open(src, framework="pt", device="cpu") as f:
            for name in f.keys():
                part = os.path.join(args.parts_dir, name + ".safetensors")
                if os.path.exists(part):
                    with safe_open(part, framework="pt", device="cpu") as pf:
                        out[name] = pf.get_tensor("w")
                    swapped += 1
                else:
                    out[name] = f.get_tensor(name)
                    copied += 1
        save_file(out, dst + ".tmp", metadata={"format": "pt"})
        os.replace(dst + ".tmp", dst)
        del out
        logger.info("shard %s done (%d swapped / %d copied so far)",
                    shard, swapped, copied)
    logger.info("assemble complete: %d swapped, %d copied", swapped, copied)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quant-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layer-range", default="0:43")
    p.add_argument("--group-num", type=int, default=128)
    p.add_argument("--row-tile", type=int, default=256)
    p.add_argument("--device", default="npu")
    p.add_argument("--assemble", action="store_true")
    p.add_argument(
        "--fallback-list", default=None,
        help="fallback_list.json (from vptq.tools.quantize.fallback_list): "
        "listed experts keep original BF16 (full: w1+w2+w3; down_only: w2) "
        "and go through msmodelslim's standard W4A8 path instead of VPTQ.",
    )
    p.add_argument(
        "--fallback-mxfp4", action="store_true",
        help="Pre-quantize fallback experts too: W @ R (block-diag Hadamard "
        "block=32, seed=0) -> mxfp4 ceil_x(7.25) roundtrip -> @ R^T, stored "
        "as BF16 parts. Requires the deploy yaml rule B to use scale_one "
        "for the final e4m3 cast.",
    )
    p.add_argument(
        "--wrap-rotation",
        default=None,
        help="Path to rotation.safetensors. Wrap mode: expert parts are "
        "Q(W@R) (from run.py --wrap-rotation); multiply each dense part "
        "by R before swapping into the UNROTATED checkpoint.",
    )
    args = p.parse_args()
    args.parts_dir = os.path.join(args.output_dir, "expert_parts")
    args.layer_range = tuple(int(x) for x in args.layer_range.split(":"))
    if args.assemble:
        pass_assemble(args)
    else:
        pass_rebuild(args)


if __name__ == "__main__":
    main()
