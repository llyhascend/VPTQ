# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Offline per-expert Hessian collection for Qwen3.6-35B-A3B
(HF qwen3_5_moe, packed 3D expert layout).

Why a dedicated collector: the generic vptq.tools.hessian hooks
`nn.Linear` inputs, but this architecture stores the routed experts as
fused 3D parameters (`mlp.experts.gate_up_proj` (E, 2*inter, hidden) and
`.down_proj` (E, hidden, inter)) inside one `Qwen3_5MoeExperts` module —
there are no per-expert Linear leaves to hook, and the `down_proj` input
(the SwiGLU intermediate) is not any module's input at all.

One forward-pre-hook per decoder layer's `mlp.experts` module instead:
the hook receives (hidden_states, top_k_index, top_k_weights), where the
expert input is exactly `hidden_states[rows routed to e]`; the SwiGLU
intermediate is recomputed from the live (unquantized) weights, matching
the activations the real `down_proj` sees. Both Hessian groups per expert
are accumulated from that single hook:
    mlp.experts.{e}.mlp_in    (hidden, hidden)   routed-token E[x x^T]
    mlp.experts.{e}.down_proj (inter, inter)     SwiGLU  E[a a^T]

Output is byte-compatible with the generic collector
(`layer_XXXX.pt` = {group_key: {"hessian", "mean", "n_tokens",
["inv_hessian"]}} + meta.json), so HessianStore / run_qwen36.py consume
it unchanged. Layer indices are positions in `model.language_model.layers`.

Example:
    python -m vptq.tools.qwen36.collect_hessian \
        --model /home/l50062354/models/Qwen3.6-35B-A3B \
        --data-path calib_corpus_R9tau.jsonl \
        --output-dir ./hessians/qwen36 --nsamples 128 --seqlen 4096 \
        --layer-range 0:8 --accum-device npu --model-dtype bf16
"""

import argparse
import json
import logging
import os

import torch

from vptq.tools.hessian.__main__ import build_samples, load_texts
from vptq.tools.hessian.collector import (
    HessianAccumulator,
    damped_inverse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vptq.qwen36.hessian")


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-expert Hessian collection for Qwen3.6-35B-A3B "
        "(qwen3_5_moe packed experts)."
    )
    p.add_argument("--model", required=True, help="HF model name or path.")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--data-path",
        default=None,
        help="Local jsonl (one text per line); overrides --dataset.",
    )
    p.add_argument(
        "--dataset", default="togethercomputer/RedPajama-Data-1T-Sample"
    )
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--layer-range", default=None, help="start:end")
    p.add_argument(
        "--accum-device", default="npu", choices=["cuda", "npu", "cpu"]
    )
    p.add_argument("--accum-dtype", default="fp32", choices=["fp32", "fp64"])
    p.add_argument(
        "--model-dtype", default="bf16", choices=["fp16", "bf16", "fp32"]
    )
    p.add_argument("--device-map", default="auto")
    p.add_argument(
        "--no-inv",
        action="store_true",
        help="Skip inverse Hessians (compute_inverse later).",
    )
    p.add_argument("--damp", type=float, default=0.01)
    return p.parse_args()


def find_text_layers(model):
    """Decoder layer list of the composite multimodal model.

    Qwen3_5MoeForConditionalGeneration.model (Qwen3_5MoeModel) holds
    .language_model (Qwen3_5MoeTextModel) with .layers; be defensive
    about the nesting in case of config-only-text checkpoints.
    """
    inner = getattr(model, "model", model)
    text = getattr(inner, "language_model", inner)
    layers = text.layers
    logger.info("found %d decoder layers", len(layers))
    return layers


def register_expert_hooks(
    model, layer_indices, accumulators, accum_device, accum_dtype
):
    """Forward-pre-hook on each layer's fused experts module.

    Group keys match run_qwen36.py / the qwen mapping in hessian_io:
    `mlp.experts.{e}.mlp_in` and `mlp.experts.{e}.down_proj`.
    """
    layers = find_text_layers(model)
    handles = []

    def get_acc(li, key, dim):
        group = accumulators[li]
        acc = group.get(key)
        if acc is None:
            acc = HessianAccumulator(
                dim, dtype=accum_dtype, device=accum_device
            )
            group[key] = acc
        return acc

    def make_hook(li):
        def hook(module, args):
            hidden, topk_idx = args[0], args[1]
            # hidden: (T, hidden) bf16 on the layer's device;
            # topk_idx: (T, top_k) routing assignment
            for e in torch.unique(topk_idx):
                e = int(e)
                if e >= module.num_experts:
                    continue
                sel = (topk_idx == e).any(-1).nonzero(as_tuple=True)[0]
                x_e = hidden[sel]
                get_acc(li, f"mlp.experts.{e}.mlp_in", x_e.shape[-1]).update(
                    x_e
                )
                # SwiGLU intermediate = the actual down_proj input,
                # recomputed with the live weights (same math as
                # Qwen3_5MoeExperts.forward)
                gate, up = torch.nn.functional.linear(
                    x_e, module.gate_up_proj[e]
                ).chunk(2, dim=-1)
                a_e = module.act_fn(gate) * up
                get_acc(li, f"mlp.experts.{e}.down_proj", a_e.shape[-1]).update(
                    a_e
                )

        return hook

    for idx in layer_indices:
        experts = layers[idx].mlp.experts
        if not hasattr(experts, "gate_up_proj"):
            raise ValueError(
                f"layer {idx}: mlp.experts has no packed gate_up_proj; "
                "is this a qwen3_5_moe checkpoint?"
            )
        accumulators[idx] = {}
        handles.append(experts.register_forward_pre_hook(make_hook(idx)))
        logger.info(
            "layer %d: hooked %s (%d experts)",
            idx,
            type(experts).__name__,
            experts.num_experts,
        )
    return handles


@torch.no_grad()
def collect(
    model,
    samples,
    layer_indices,
    accumulators,
    accum_device,
    accum_dtype,
    batch_size,
):
    was_training = model.training
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    try:
        input_device = model.get_input_embeddings().weight.device
    except AttributeError:
        input_device = next(model.parameters()).device
    for start in range(0, samples.shape[0], batch_size):
        ids = samples[start : start + batch_size].to(input_device)
        model(input_ids=ids)
        done = min(start + batch_size, samples.shape[0])
        logger.info("processed %d/%d samples", done, samples.shape[0])
    if was_training:
        model.train()


def save(
    output_dir,
    layer_indices,
    accumulators,
    save_inv=True,
    damp=0.01,
    model_name="",
    extra_meta=None,
):
    """Write layer_XXXX.pt in the generic collector's exact format."""
    os.makedirs(output_dir, exist_ok=True)
    for idx in layer_indices:
        payload = {}
        for key, acc in sorted(accumulators[idx].items()):
            hessian, mean, n_tokens = acc.finalize()
            hessian = hessian.float().cpu()
            entry = {
                "hessian": hessian,
                "mean": mean.float().cpu(),
                "n_tokens": n_tokens,
            }
            if save_inv:
                entry["inv_hessian"] = damped_inverse(hessian, damp=damp)
            payload[key] = entry
        path = os.path.join(output_dir, f"layer_{idx:04d}.pt")
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        logger.info("saved %s (%d groups)", path, len(payload))
    meta = {
        "model_name": model_name,
        "layer_path": "model.language_model.layers",
        "layer_indices": list(layer_indices),
        "hook_plans": {
            str(i): {
                "experts_module": "mlp.experts",
                "groups_per_expert": ["mlp_in", "down_proj"],
            }
            for i in layer_indices
        },
        "accum_dtype": "torch.float32",
        "center": False,
        "damp": damp,
        "save_inv": save_inv,
        "convention": "hessian = E[x x^T]",
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    args = parse_args()
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    layer_range = None
    if args.layer_range is not None:
        start, end = args.layer_range.split(":")
        layer_range = (int(start), int(end))

    import transformers
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.model_dtype],
        device_map=args.device_map,
    )
    model.eval()

    texts = load_texts(args)
    samples = build_samples(
        tokenizer, texts, args.nsamples, args.seqlen, args.seed
    )

    layers = find_text_layers(model)
    end = min(layer_range[1] if layer_range else len(layers), len(layers))
    start = layer_range[0] if layer_range else 0
    layer_indices = list(range(start, end))

    accumulators = {}
    handles = register_expert_hooks(
        model,
        layer_indices,
        accumulators,
        torch.device(args.accum_device),
        dtype_map[args.accum_dtype],
    )
    try:
        collect(
            model,
            samples,
            layer_indices,
            accumulators,
            torch.device(args.accum_device),
            dtype_map[args.accum_dtype],
            args.batch_size,
        )
    finally:
        for h in handles:
            h.remove()

    save(
        args.output_dir,
        layer_indices,
        accumulators,
        save_inv=not args.no_inv,
        damp=args.damp,
        model_name=args.model,
        extra_meta={
            "dataset": args.data_path or args.dataset,
            "nsamples": int(samples.shape[0]),
            "seqlen": args.seqlen,
            "seed": args.seed,
        },
    )
    logger.info("done. Hessians written to %s", args.output_dir)


if __name__ == "__main__":
    main()
