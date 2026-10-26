#!/usr/bin/env python3
"""fb4 量化前后 MSE 校验（2026-10-20 用户指令）。

对比：expert_parts（VPTQ wrap 重建 + 回退 wrap 域 mxfp4 预量化，未旋转域 BF16）
  vs 原始 ckpt /mnt/share/weight/DeepSeek-V4-Flash-BF16（未旋转域）。
聚合：每层按 router gate 分数加权（L0-2 hash 路由无分数 → 均匀权重），
  分 VPTQ / fallback 两组 + 全体；逐层增量写 jsonl（断点续跑：已有层跳过）。
"""
import argparse
import json
import os

import torch
from safetensors.torch import load_file, safe_open

ORIG = "/mnt/share/weight/DeepSeek-V4-Flash-BF16"
PARTS = ("/mnt/share/w00608002/weights/DeepSeek-V4-Flash-vptq-calibR9-"
         "rf32wfbfb4-bf16/expert_parts")
FB_LIST = ("/mnt/share/w00608002/weights/quant/dsv4-tile32x32-rowfp8-wrap-fb-"
           "calibR9/fallback_list.json")
GATE = ("/mnt/share/w00608002/weights/hessians/"
        "DeepSeek-V4-Flash-BF16-calibR9/gate_scores.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer-range", default="0:43")
    p.add_argument("--out", default="analysis/fb4_mse_by_layer.jsonl")
    p.add_argument("--device", default="npu")
    args = p.parse_args()
    lo, hi = (int(x) for x in args.layer_range.split(":"))
    dev = torch.device(args.device)

    fl = json.load(open(FB_LIST))
    fb_full = {(li, e) for li, e in fl["full"]}
    fb_down = {(li, e) for li, e in fl["down_only"]}
    gate = json.load(open(GATE))
    wmap = json.load(open(os.path.join(ORIG, "model.safetensors.index.json"))
                     )["weight_map"]

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                done.add(json.loads(line)["layer"])
            except Exception:
                pass

    for layer in range(lo, hi):
        if layer in done:
            continue
        # orig 侧：按 shard 分组，同 shard 复用 handle
        by_shard = {}
        for e in range(256):
            for w in ("w1", "w2", "w3"):
                n = f"layers.{layer}.ffn.experts.{e}.{w}.weight"
                by_shard.setdefault(wmap[n], []).append((e, w, n))
        handles = {}
        acc = {"vptq": [0.0, 0], "fb": [0.0, 0]}  # sq_err, numel
        per_expert = []  # (expert, mse, group)  w2 与 w1w3 分开累计
        exp_acc = {}  # expert -> [sq, n]
        rel = {"vptq": [0.0, 0.0], "fb": [0.0, 0.0]}  # ||d||^2, ||W||^2
        for shard, items in by_shard.items():
            f = safe_open(os.path.join(ORIG, shard), framework="pt",
                          device="cpu")
            handles[shard] = f
            for e, w, n in items:
                grp = ("fb" if (layer, e) in fb_full
                       or ((layer, e) in fb_down and w == "w2") else "vptq")
                orig = f.get_tensor(n).to(dev).float()
                q = load_file(os.path.join(PARTS, n + ".safetensors"
                                           ))["w"].to(dev).float()
                d = q - orig
                se = (d * d).sum().item()
                nn = d.numel()
                acc[grp][0] += se
                acc[grp][1] += nn
                rel[grp][0] += se
                rel[grp][1] += (orig * orig).sum().item()
                ea = exp_acc.setdefault(e, [0.0, 0])
                ea[0] += se
                ea[1] += nn
            del f
        handles.clear()

        # gate 加权（专家粒度；L0-2 均匀）
        g = gate.get(str(layer))
        if g is None:
            wts = {e: 1.0 for e in exp_acc}
        else:
            s = sum(g)
            wts = {e: g[e] / s for e in exp_acc}

        def wavg(pred):
            num = den = 0.0
            for e, (se, n) in exp_acc.items():
                if pred(e):
                    mse = se / max(n, 1)
                    num += wts[e] * mse
                    den += wts[e]
            return num / den if den else None

        is_fb = lambda e: ((layer, e) in fb_full or (layer, e) in fb_down)
        rec = {
            "layer": layer,
            "weighted": "gate" if g is not None else "uniform",
            "all_w": wavg(lambda e: True),
            "vptq_w": wavg(lambda e: not is_fb(e)),
            "fb_w": wavg(is_fb),
            "vptq_mse": acc["vptq"][0] / max(acc["vptq"][1], 1),
            "fb_mse": acc["fb"][0] / acc["fb"][1] if acc["fb"][1] else None,
            "vptq_rel": (rel["vptq"][0] / max(rel["vptq"][1], 1e-30)) ** 0.5,
            "fb_rel": (rel["fb"][0] / max(rel["fb"][1], 1e-30)) ** 0.5
                      if rel["fb"][1] > 0 else None,
        }
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        f = None
        print(f"layer {layer}: all_w={rec['all_w']:.3e} "
              f"vptq={rec['vptq_w']:.3e} fb={rec['fb_w']:.3e}", flush=True)


if __name__ == "__main__":
    main()
