# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Quantization driver for Qwen3.6-35B-A3B routed experts
(qwen3_5_moe packed layout; task-scoped sibling of run.py, which is
DeepSeek-V4 specific).

Checkpoint facts this driver is written against (2026-10-08,
model.safetensors.index.json of Qwen3.6-35B-A3B):
  - routed experts are fused 3D tensors, no per-expert leaves,
    no `.weight` suffix:
      {prefix}mlp.experts.gate_up_proj (E, 2*inter, hidden)  gate|up fused
      {prefix}mlp.experts.down_proj    (E, hidden, inter)
  - prefix `model.language_model.layers.{i}.` (composite multimodal
    model); auto-detected here, `model.layers.{i}.` / `layers.{i}.` also
    accepted
Per expert e (gate|up row split exactly as Qwen3_5MoeExperts.forward
chunks them):
  gate = gate_up[e][:inter], up = gate_up[e][inter:]
      -> vptq_quantize_joint on the shared mlp_in Hessian
         (same amortization as run.py's w1+w3 joint path)
  down = down_proj[e]         -> vptq_quantize on its own Hessian
Hessian groups (from vptq.tools.qwen36.collect_hessian):
  mlp.experts.{e}.mlp_in (hidden^2), mlp.experts.{e}.down_proj (inter^2)

Output mirrors run.py so downstream tooling works unchanged:
  quant_layer_XXXX_e{e0}_{e1}.pt  {short: QuantizedWeight}
    short in {experts.{e}.gate, experts.{e}.up, experts.{e}.down}
  quant_report.json (keyed by full checkpoint names) + .parts resume
Not covered (by design, bring-up scope = experts only): attention /
router / shared expert, dual-norm, row-fp8, rotation wrap; the DSV4
variants of these live in run.py and do not apply to the packed layout.
"""

import argparse
import json
import logging
import os

import torch

from vptq.tools.quantize.config import QuantConfig
from vptq.tools.quantize.hessian_io import HessianStore
from vptq.tools.quantize.run import PartsStaging, WeightStore
from vptq.tools.quantize.vq import (
    vptq_quantize,
    vptq_quantize_joint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vptq.quantize.run_qwen36")

_PREFIX_CANDIDATES = (
    "model.language_model.layers.{i}.",  # composite multimodal (this ckpt)
    "model.layers.{i}.",  # plain HF causal LM
    "layers.{i}.",  # native inference format
)


def detect_layer_prefix(weight_map: dict) -> str:
    for tmpl in _PREFIX_CANDIDATES:
        probe = tmpl.format(i=0) + "mlp.experts.gate_up_proj"
        if probe in weight_map:
            logger.info("layer prefix: %s", tmpl.format(i="<i>"))
            return tmpl
    raise ValueError(
        "no packed expert tensors found; tried "
        f"{[t.format(i=0) + 'mlp.experts.gate_up_proj' for t in _PREFIX_CANDIDATES]}"
    )


def quantize_expert(
    e: int,
    layer_idx: int,
    gate_up_e: torch.Tensor,
    down_e: torch.Tensor,
    store: HessianStore,
    config: QuantConfig,
    device: torch.device,
) -> dict:
    """Quantize one expert -> {short: QuantizedWeight}.

    gate_up_e: (2*inter, hidden); down_e: (hidden, inter).
    """
    inter = gate_up_e.shape[0] // 2
    hess_key_in = f"mlp.experts.{e}.mlp_in"
    hess_key_dn = f"mlp.experts.{e}.down_proj"
    out = {}
    # gate|up 联合量化：共享 mlp_in Hessian，一次批量 kmeans
    # （与 run.py 的 w1+w3 joint 同构，vptq_quantize_joint 要求同形状）
    Wg = gate_up_e[:inter].float()
    Wu = gate_up_e[inter:].float()
    hessian_in, _ = store.get(layer_idx, hess_key_in)
    HinvU_in = store.cholesky_upper(layer_idx, hess_key_in, damp=config.damp)
    qw_g, qw_u = vptq_quantize_joint(
        [Wg, Wu],
        hessian_in.float(),
        config,
        device=device,
        hessian_inv_upper=HinvU_in,
        kmeans_seed=hash(f"layers.{layer_idx}.experts.{e}.gate") & 0xFFFF,
    )
    logger.info(
        "layer %d expert %d gate/up joint -> H[%s] " "proxy_error=%.5f/%.5f",
        layer_idx,
        e,
        hess_key_in,
        qw_g.proxy_error,
        qw_u.proxy_error,
    )
    out[f"experts.{e}.gate"] = qw_g
    out[f"experts.{e}.up"] = qw_u

    Wd = down_e.float()
    hessian_dn, _ = store.get(layer_idx, hess_key_dn)
    HinvU_dn = store.cholesky_upper(layer_idx, hess_key_dn, damp=config.damp)
    qw_d = vptq_quantize(
        Wd,
        hessian_dn.float(),
        config,
        device=device,
        hessian_inv_upper=HinvU_dn,
        kmeans_seed=hash(f"layers.{layer_idx}.experts.{e}.down") & 0xFFFF,
    )
    logger.info(
        "layer %d expert %d down -> H[%s] proxy_error=%.5f mse=%.5f",
        layer_idx,
        e,
        hess_key_dn,
        qw_d.proxy_error,
        qw_d.mse,
    )
    out[f"experts.{e}.down"] = qw_d
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--hessian-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--vector-len", type=int, default=8)
    p.add_argument("--num-centroids", type=int, default=65536)
    p.add_argument("--num-res-centroids", type=int, default=-1)
    p.add_argument("--group-num", type=int, default=1)
    p.add_argument("--kmeans-iters", type=int, default=10)
    p.add_argument("--row-tile", type=int, default=0)
    p.add_argument("--res-vector-len", type=int, default=0)
    p.add_argument("--centroid-fp8", action="store_true")
    p.add_argument("--device", default="npu")
    p.add_argument("--layer-range", default=None, help="start:end")
    p.add_argument("--expert-range", default=None, help="start:end")
    p.add_argument("--n-experts", type=int, default=256)
    p.add_argument(
        "--hidden-size",
        type=int,
        default=2048,
        help="mlp_in identity-fallback dim (zero-hit experts).",
    )
    p.add_argument(
        "--moe-intermediate",
        type=int,
        default=512,
        help="down_proj identity-fallback dim.",
    )
    p.add_argument(
        "--save-staging",
        default=None,
        help="Local NVMe dir to stage quant parts (perf on NFS).",
    )
    args = p.parse_args()

    config = QuantConfig(
        vector_len=args.vector_len,
        num_centroids=args.num_centroids,
        num_res_centroids=args.num_res_centroids,
        group_num=args.group_num,
        kmeans_iters=args.kmeans_iters,
        res_vector_len=args.res_vector_len,
        row_tile=args.row_tile,
        centroid_fp8=args.centroid_fp8,
    )
    logger.info("config: %s (%.3f bits/weight)", config, config.bits_per_weight)
    device = torch.device(args.device)

    def identity_dims(group_key: str) -> int:
        return (
            args.hidden_size
            if group_key.endswith("mlp_in")
            else args.moe_intermediate
        )

    weights = WeightStore(args.ckpt)
    prefix = detect_layer_prefix(weights.weight_map)
    store = HessianStore(args.hessian_dir, identity_dims=identity_dims)
    staging = None
    if args.save_staging:
        staging = PartsStaging(args.save_staging)
        staging.flush_residual(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    layer_range = (0, 40)
    if args.layer_range:
        layer_range = tuple(int(x) for x in args.layer_range.split(":"))
    expert_range = None
    if args.expert_range:
        expert_range = tuple(int(x) for x in args.expert_range.split(":"))
    logger.info("expert range: %s", expert_range or "(all)")

    report = {}
    for layer_idx in range(*layer_range):
        suffix = ""
        if expert_range:
            suffix = f"_e{expert_range[0]}_{expert_range[1]}"
        path = os.path.join(
            args.output_dir, f"quant_layer_{layer_idx:04d}{suffix}.pt"
        )
        parts_dir = path[:-3] + ".parts"
        os.makedirs(parts_dir, exist_ok=True)

        def part_path(short, _d=parts_dir):
            return os.path.join(_d, short + ".pt")

        done_shorts = {
            f[:-3] for f in os.listdir(parts_dir) if f.endswith(".pt")
        }

        def on_weight(short, qw):
            if staging is not None:
                local = os.path.join(
                    staging.dir, os.path.basename(parts_dir), short + ".pt"
                )
                os.makedirs(os.path.dirname(local), exist_ok=True)
                ltmp = local + ".ltmp"
                torch.save(qw, ltmp)
                os.replace(ltmp, local)
                staging.save(local, part_path(short))
            else:
                tmp = part_path(short) + ".tmp"
                torch.save(qw, tmp)
                os.replace(tmp, part_path(short))

        # 打包张量整层读一次（gate_up ~1GB bf16），分片内 64 专家切片复用
        gu_name = prefix.format(i=layer_idx) + "mlp.experts.gate_up_proj"
        dn_name = prefix.format(i=layer_idx) + "mlp.experts.down_proj"
        gate_up = weights.get(gu_name)
        down = weights.get(dn_name)
        e0, e1 = expert_range or (0, args.n_experts)
        out = {}
        for e in range(e0, e1):
            shorts = (
                f"experts.{e}.gate",
                f"experts.{e}.up",
                f"experts.{e}.down",
            )
            if all(s in done_shorts for s in shorts):
                continue
            # per-expert 粒度续跑：半途专家整体重量化，parts 原子覆盖无损
            new = quantize_expert(
                e,
                layer_idx,
                gate_up[e],
                down[e],
                store,
                config,
                device,
            )
            out.update(new)
            for s, qw in new.items():
                on_weight(s, qw)
        del gate_up, down

        merged: dict = {}
        for short in sorted(done_shorts | set(out)):
            merged[short] = torch.load(part_path(short), weights_only=False)
        tmp = path + ".tmp"
        torch.save(merged, tmp)
        os.replace(tmp, path)
        for short, qw in out.items():
            report[f"{prefix.format(i=layer_idx)}mlp.experts.{short}"] = {
                "proxy_error": qw.proxy_error,
                "mse": qw.mse,
            }
        logger.info(
            "saved %s (%d new, %d total experts quantized)",
            path,
            len(out) // 3,
            len(merged) // 3,
        )

    if staging is not None:
        staging.finish()
    report_path = os.path.join(args.output_dir, "quant_report.json")
    existing = {}
    if os.path.exists(report_path):
        existing = json.load(open(report_path))
    existing.update(report)
    existing["_config"] = {
        "vector_len": config.vector_len,
        "num_centroids": config.num_centroids,
        "num_res_centroids": config.num_res_centroids,
        "group_num": config.group_num,
        "bits_per_weight": config.bits_per_weight,
        "model": "qwen3_5_moe-packed-experts",
    }
    with open(report_path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info("report written to %s", report_path)


if __name__ == "__main__":
    main()
