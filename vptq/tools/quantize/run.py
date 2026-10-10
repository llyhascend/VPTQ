# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Full-model quantization driver for DeepSeek-V4 (task: apply the VPTQ
quantizer to every quantizable linear using the collected Hessians).

Per decoder layer, quantizes:
  - attention: attn.wq_a / wq_b / wkv / wo_b, compressor.wkv / wgate,
    indexer.wq_b / weights_proj (+ their compressor pair)
  - attn.wo_a as `o_groups` independent slices (per-group Hessians
    attn.wo_a_input.g{i})
  - MoE: ffn.gate (router, reusing the shared-expert Hessian),
    ffn.shared_experts.w1/w2/w3, ffn.experts.N.w1/w2/w3 (per-expert
    Hessians)
Skipped (by convention): embed, head, norms, hc_*, attn_sink, gate
bias / tid2eid, MTP block.

Sharding for throughput: experts of one layer are independent, so run
several processes with --expert-range (and/or --layer-range).

Output per layer: quant_layer_XXXX.pt  {weight_name: QuantizedWeight}
plus a quant_report.json with per-weight proxy_error / mse.
"""

import argparse
import json
import logging
import os
import re
import shutil
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open

from vptq.tools.quantize.config import QuantConfig
from vptq.tools.quantize.hadamard import right_rot, wrap_rotation
from vptq.tools.quantize.hessian_io import HessianStore, hessian_key_of
from vptq.tools.quantize.vq import (
    QuantizedWeight,
    vptq_quantize,
    vptq_quantize_joint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vptq.quantize.run")


class WeightStore:
    """Lazy sharded-safetensors reader with one-shard cache."""

    def __init__(self, ckpt_dir: str):
        import json as _json

        with open(
            os.path.join(ckpt_dir, "model.safetensors.index.json")
        ) as f:
            self.weight_map = _json.load(f)["weight_map"]
        self.ckpt_dir = ckpt_dir
        self._shard = None
        self._fh = None

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        if shard != self._shard:
            if self._fh is not None:
                self._fh.__exit__(None, None, None)
            self._fh = safe_open(
                os.path.join(self.ckpt_dir, shard), framework="pt",
                device="cpu",
            )
            self._shard = shard
        return self._fh.get_tensor(name)


# weights quantized per decoder layer, with their o-group slice count
ATTN_WEIGHTS = (
    "attn.wq_a.weight",
    "attn.wq_b.weight",
    "attn.wkv.weight",
    "attn.wo_b.weight",
    "attn.compressor.wkv.weight",
    "attn.compressor.wgate.weight",
    "attn.indexer.wq_b.weight",
    "attn.indexer.weights_proj.weight",
    "attn.indexer.compressor.wkv.weight",
    "attn.indexer.compressor.wgate.weight",
)
MOE_STATIC = (
    "ffn.gate.weight",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w3.weight",
)


def quantize_one(
    W: torch.Tensor,
    store: HessianStore,
    layer_idx: int,
    hess_key: str,
    config: QuantConfig,
    device: torch.device,
    seed: int,
) -> QuantizedWeight:
    hessian, _ = store.get(layer_idx, hess_key)
    HinvU = store.cholesky_upper(layer_idx, hess_key, damp=config.damp)
    return vptq_quantize(
        W.float(),
        hessian.float(),
        config,
        device=device,
        hessian_inv_upper=HinvU,
        kmeans_seed=seed,
    )


# wrap 模式（--wrap-rotation）下需要输入侧旋转 W -> W @ R 的叶子。
# 路由专家的 w1/w2/w3 全部输入侧（w2 与吸收式不同，见 hessian_io
# is_rotated_group 的 wrap 说明）；R 为带符号 Hadamard，正交但
# 非对称（R^-1 = R^T），rebuild 端以 Q(WR) @ R^T 包装回原坐标。
_WRAP_LEAF_RE = re.compile(
    r"ffn\.experts\.\d+\.(w1|w2|w3)\.weight$"
)


class PartsStaging:
    """NVMe staging for quant parts (perf: NFS small-file writes are
    ~56% of wall time). on_weight writes locally; a background thread
    uploads to the NFS parts dir. Backpressure = queue maxsize bounds
    local usage (~1MB/part * 2000 = ~2GB). Resume semantics unchanged:
    a part is "done" only when visible in the NFS parts dir.
    """

    def __init__(self, staging_dir: str):
        import queue, threading

        self.dir = staging_dir
        os.makedirs(staging_dir, exist_ok=True)
        self.q = queue.Queue(maxsize=2000)
        self.err = []

        def _worker():
            while True:
                item = self.q.get()
                if item is None:
                    return
                local, remote = item
                try:
                    os.makedirs(os.path.dirname(remote), exist_ok=True)
                    tmp = remote + ".tmp"
                    shutil.copyfile(local, tmp)
                    os.replace(tmp, remote)
                    os.remove(local)
                except Exception as e:  # keep uploading the rest
                    self.err.append((remote, str(e)))
                    logger.error("staging upload failed %s: %s", remote, e)

        self.thread = threading.Thread(target=_worker, daemon=True)
        self.thread.start()

    def save(self, local_path, remote_path):
        """Called after the caller wrote local_path atomically."""
        while True:
            try:
                self.q.put((local_path, remote_path), timeout=30)
                return
            except Exception:
                if self.err:
                    raise RuntimeError(f"staging worker died: {self.err[0]}")
                logger.warning("staging queue full, waiting...")

    def finish(self):
        self.q.put(None)
        self.thread.join()
        if self.err:
            raise RuntimeError(f"{len(self.err)} staging uploads failed; "
                               f"first: {self.err[0]}")

    def flush_residual(self, parts_glob_parent):
        """Upload leftover complete parts from a killed run (torn .tmp
        deleted) BEFORE any resume scan reads the parts dirs."""
        import glob as _glob
        import shutil as _shutil
        if not os.path.isdir(self.dir):
            return
        n = 0
        for path in _glob.glob(os.path.join(self.dir, "**", "*.pt"),
                               recursive=True):
            rel = os.path.relpath(path, self.dir)
            remote = os.path.join(parts_glob_parent, rel)
            tmp = remote + ".tmp"
            _shutil.copyfile(path, tmp)
            os.replace(tmp, remote)
            os.remove(path)
            n += 1
        for tmp in _glob.glob(os.path.join(self.dir, "**", "*.tmp"),
                              recursive=True):
            os.remove(tmp)
        if n:
            logger.info("flushed %d residual staged parts", n)


def quantize_layer(
    layer_idx: int,
    weights: WeightStore,
    store: HessianStore,
    config: QuantConfig,
    device: torch.device,
    expert_range: Optional[Tuple[int, int]] = None,
    o_groups: int = 8,
    n_experts: int = 256,
    scopes: Tuple[str, ...] = ("all",),
    skip: Optional[set] = None,
    on_weight=None,
    wrap_R: Optional[torch.Tensor] = None,
    dual_norm: bool = False,
    row_fp8: bool = False,
) -> Dict[str, QuantizedWeight]:
    prefix = f"layers.{layer_idx}."
    out: Dict[str, QuantizedWeight] = {}
    present = set(weights.weight_map)
    want_all = "all" in scopes
    skip = skip or set()

    def maybe_wrap(short: str, W: torch.Tensor) -> torch.Tensor:
        if wrap_R is not None and _WRAP_LEAF_RE.match(short):
            # per-input-dim rotation: w1/w3 in=4096（与全局 R 一致），
            # w2 in=2048（SwiGLU 中间维，独立 Hadamard）
            return right_rot(W, wrap_rotation(W.shape[1]))
        return W

    def dual_hessian(H, inv, d):
        """H̃ = D H D，inv' = D⁻¹ inv D⁻¹（D=diag(d)，x̃=d⊙x）。"""
        if d is None:
            return H, inv
        Hn = d[:, None] * H.float() * d[None, :]
        invn = (inv.float() / d[:, None] / d[None, :]) if inv is not None else None
        return Hn, invn

    def dual_hinvU(layer_idx: int, hess_key: str, d):
        """dual-norm 路径的 H⁻¹ 上 Cholesky（store 原始 H 上做 D 变换）。"""
        H, inv = store.get(layer_idx, hess_key)
        _, invn = dual_hessian(H, inv, d)
        if invn is not None:
            L = torch.linalg.cholesky(invn.double())
            return L.t().float()
        from vptq.tools.quantize.vq import cholesky_upper_of_inv
        Hn, _ = dual_hessian(H, inv, d)
        return cholesky_upper_of_inv(Hn, config.damp)

    def _row_fp8(W):
        """量化前 out-dim 归一化到 FP8 表示域：行峰对齐 e4m3 max=448。

        行对角缩放与 dual-norm 的行 rms 可交换，重建时折回；
        与 dual_norm 叠加时 row_scale 存复合值（fp8_scale × rms_scale）。
        """
        Wf = W.float()
        peak = Wf.abs().amax(dim=1).clamp_min(1e-12)
        scale = 448.0 / peak
        return Wf * scale[:, None], scale

    def _row_norm(W, d):
        """给定共享列尺度 d，行归一化并返回 (W_norm, s)。"""
        Wf = W.float()
        s = Wf.pow(2).mean(1).sqrt().clamp_min(1e-8)
        return (Wf / s[:, None] / d[None, :]), s

    todo = []
    if want_all or "attn" in scopes:
        todo += [w for w in ATTN_WEIGHTS if prefix + w in present]
    if want_all or "router" in scopes:
        todo += [w for w in MOE_STATIC if w.startswith("ffn.gate.")
                 and prefix + w in present]
    if want_all or "shared" in scopes:
        todo += [w for w in MOE_STATIC
                 if w.startswith("ffn.shared_experts.")
                 and prefix + w in present]
    expert_pairs = []
    if want_all or "experts" in scopes:
        e0, e1 = expert_range or (0, n_experts)
        for e in range(e0, e1):
            s1 = f"ffn.experts.{e}.w1.weight"
            s2 = f"ffn.experts.{e}.w2.weight"
            s3 = f"ffn.experts.{e}.w3.weight"
            can_pair = (
                prefix + s1 in present and prefix + s3 in present
                and s1 not in skip and s3 not in skip
            )
            if can_pair:
                expert_pairs.append((e, s1, s3))
                if prefix + s2 in present and s2 not in skip:
                    todo.append(s2)
            else:
                for s in (s1, s2, s3):
                    if prefix + s in present and s not in skip:
                        todo.append(s)
    todo = [w for w in todo if w not in skip]

    for e, s1, s3 in expert_pairs:
        # w1/w3 联合量化：共享 mlp_in Hessian 与同形状，kmeans 一次批量
        hess_key = f"ffn.experts.{e}.mlp_in"
        logger.info(
            "layer %d experts.%d w1+w3 joint -> H[%s]", layer_idx, e, hess_key
        )
        W1 = maybe_wrap(s1, weights.get(prefix + s1))
        W3 = maybe_wrap(s3, weights.get(prefix + s3))
        d_joint = None
        fp1 = fp3 = None
        if row_fp8:  # 第一步：out-dim 峰对齐 448（在 dual-norm 之前）
            W1, fp1 = _row_fp8(W1)
            W3, fp3 = _row_fp8(W3)
        if dual_norm:
            # in-channel 尺度必须对同一输入 x 自洽：w1/w3 拼接计算共享 d
            d_joint = torch.cat([W1.float(), W3.float()], 0) \
                          .pow(2).mean(0).sqrt().clamp_min(1e-8)
            W1n, s1s = _row_norm(W1, d_joint)
            W3n, s3s = _row_norm(W3, d_joint)
            # FP8 归一化是放大（448/peak），折回须除回：row_scale = rms / fp
            s1s = s1s / fp1 if fp1 is not None else None
            s3s = s3s / fp3 if fp3 is not None else None
            hessian, inv = store.get(layer_idx, hess_key)
            hessian, _ = dual_hessian(hessian, inv, d_joint)
            HinvU = dual_hinvU(layer_idx, hess_key, d_joint)
        elif row_fp8:
            W1n, W3n, s1s, s3s = W1, W3, fp1, fp3
            hessian, _ = store.get(layer_idx, hess_key)
            HinvU = store.cholesky_upper(layer_idx, hess_key, damp=config.damp)
        else:
            W1n, W3n, s1s, s3s = W1.float(), W3.float(), None, None
            hessian, _ = store.get(layer_idx, hess_key)
            HinvU = store.cholesky_upper(layer_idx, hess_key, damp=config.damp)
        qw1, qw3 = vptq_quantize_joint(
            [W1n, W3n],
            hessian.float(),
            config,
            device=device,
            hessian_inv_upper=HinvU,
            kmeans_seed=hash(prefix + s1) & 0xFFFF,
        )
        if dual_norm:
            qw1.row_scale, qw1.col_scale = s1s, d_joint
            qw3.row_scale, qw3.col_scale = s3s, d_joint
        elif row_fp8:
            # row-fp8 only: W was quantized in the amplified FP8 domain
            # (448/peak); rebuild must fold back with 1/fp (same as w2).
            qw1.row_scale, qw1.col_scale = 1.0 / fp1, None
            qw3.row_scale, qw3.col_scale = 1.0 / fp3, None
        logger.info(
            "  w1 proxy_error=%.5f  w3 proxy_error=%.5f",
            qw1.proxy_error, qw3.proxy_error,
        )
        out[s1], out[s3] = qw1, qw3
        if on_weight is not None:
            on_weight(s1, qw1)
            on_weight(s3, qw3)

    for short in todo:
        name = prefix + short
        hess_key = hessian_key_of(short)
        W = maybe_wrap(short, weights.get(name))
        logger.info(
            "layer %d %s %s -> H[%s]",
            layer_idx,
            short,
            tuple(W.shape),
            hess_key,
        )
        if (dual_norm or row_fp8) and _WRAP_LEAF_RE.match(short):
            # w2：独立列尺度（输入是 SwiGLU 中间激活，自己一组 H）
            fp = None
            if row_fp8:
                W, fp = _row_fp8(W)
            if dual_norm:
                Wf = W.float()
                d = Wf.pow(2).mean(0).sqrt().clamp_min(1e-8)
                Wn, s_row = _row_norm(W, d)
                s_row = s_row / fp if fp is not None else s_row
                hessian, inv = store.get(layer_idx, hess_key)
                hessian, _ = dual_hessian(hessian, inv, d)
                HinvU = dual_hinvU(layer_idx, hess_key, d)
                out[short] = vptq_quantize(
                    Wn, hessian.float(), config, device=device,
                    hessian_inv_upper=HinvU,
                    kmeans_seed=hash(name) & 0xFFFF,
                )
                out[short].row_scale, out[short].col_scale = s_row, d
            else:  # row_fp8 only：Hessian 不变换，仅行 scale 折回
                out[short] = vptq_quantize(
                    W, store.get(layer_idx, hess_key)[0].float(), config,
                    device=device,
                    hessian_inv_upper=store.cholesky_upper(
                        layer_idx, hess_key, damp=config.damp),
                    kmeans_seed=hash(name) & 0xFFFF,
                )
                out[short].row_scale, out[short].col_scale = 1.0 / fp, None
        else:
            out[short] = quantize_one(
                W, store, layer_idx, hess_key, config, device,
                seed=hash(name) & 0xFFFF,
            )
        logger.info(
            "  proxy_error=%.5f mse=%.5f",
            out[short].proxy_error,
            out[short].mse,
        )
        if on_weight is not None:
            on_weight(short, out[short])

    # wo_a: per o-group slices with per-group Hessians
    wo_a_name = prefix + "attn.wo_a.weight"
    if wo_a_name in present and (want_all or "attn" in scopes):
        W = weights.get(wo_a_name).float()  # (o_groups*r, d)
        r = W.shape[0] // o_groups
        for g in range(o_groups):
            if f"attn.wo_a.g{g}.weight" in skip:
                continue
            hess_key = f"attn.wo_a_input.g{g}"
            logger.info("layer %d wo_a.g%d %s", layer_idx, g, (r, W.shape[1]))
            out[f"attn.wo_a.g{g}.weight"] = quantize_one(
                W[g * r : (g + 1) * r],
                store,
                layer_idx,
                hess_key,
                config,
                device,
                seed=g,
            )
            if on_weight is not None:
                on_weight(
                    f"attn.wo_a.g{g}.weight", out[f"attn.wo_a.g{g}.weight"]
                )
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
    p.add_argument(
        "--row-tile",
        type=int,
        default=0,
        help="0: full-column groups; >0: 2D tiles of "
        "(in/group_num cols) x (row_tile rows), one codebook per tile "
        "(tile scheme, see QuantConfig.row_tile).",
    )
    p.add_argument(
        "--res-vector-len",
        type=int,
        default=0,
        help="0: residual vector length = vector_len; >0: decoupled "
        "residual vector length (tile scheme).",
    )
    p.add_argument(
        "--centroid-fp8",
        action="store_true",
        help="Store centroids as fp8-e4m3 (eval: <=1.01x proxy error, "
        "fully absorbed when residual VQ is enabled).",
    )
    p.add_argument("--device", default="npu")
    p.add_argument("--layer-range", default=None, help="start:end")
    p.add_argument("--expert-range", default=None, help="start:end")
    p.add_argument("--o-groups", type=int, default=8)
    p.add_argument("--n-experts", type=int, default=256)
    p.add_argument(
        "--scope",
        default="all",
        help="Comma-separated subset of: all,attn,router,shared,experts. "
        "E.g. --scope experts quantizes ONLY the routed experts "
        "(w1/w2/w3), leaving attention/router/shared experts in BF16. "
        "Results merge into existing quant_layer files, so different "
        "scopes can use different configs across runs (mixed bitwidth).",
    )
    p.add_argument(
        "--rotation",
        default=None,
        help="Path to rotation.safetensors (from rotate_ckpt). When the "
        "checkpoint was Hadamard-absorbed, pass it so input-side Hessian "
        "groups are rotated consistently (H' = R^T H R).",
    )
    p.add_argument(
        "--save-staging",
        default=None,
        help="Local NVMe dir to stage quant parts; a background thread "
        "uploads them to the NFS parts dirs (perf: NFS small-file "
        "writes are ~56%% of quantization wall time). Resume-safe: "
        "residual staged parts are flushed before the resume scan.",
    )
    p.add_argument(
        "--row-fp8",
        action="store_true",
        help="quantize in FP8 (e4m3) domain: per-row peak aligned to 448 "
        "before (optional) dual-norm; folded back via row_scale. "
        "Composes with --dual-norm.",
        )
    p.add_argument(
        "--dual-norm",
        action="store_true",
        help="Dual-scale normalization for routed experts before VPTQ: "
        "W_norm = W / row_rms / col_rms (joint w1/w3 share the col scale "
        "of the same input; w2 has its own). Hessian is transformed "
        "consistently (H' = D H D). rebuild_bf16 multiplies both scales "
        "back into the dense weights (zero deployment cost). Orthogonal "
        "to --wrap-rotation.",
    )
    p.add_argument(
        "--wrap-rotation",
        default=None,
        help="Wrap mode: quantize V = W @ R from an UNROTATED checkpoint "
        "(routed experts w1/w2/w3, all input-side) with rotated Hessians, "
        "then rebuild_bf16 --wrap-rotation wraps the dense parts back "
        "(Q(WR) @ R) so no other model weight changes. Mutually "
        "exclusive with --rotation.",
    )
    args = p.parse_args()
    if args.rotation and args.wrap_rotation:
        raise ValueError("--rotation and --wrap-rotation are mutually exclusive")

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
    logger.info("config: %s (%.3f bits/weight)", config,
                config.bits_per_weight)
    device = torch.device(args.device)
    weights = WeightStore(args.ckpt)
    rotation_path = args.rotation or args.wrap_rotation
    store = HessianStore(args.hessian_dir, rotation_path=rotation_path,
                         wrap=bool(args.wrap_rotation))
    wrap_R = store.R if args.wrap_rotation else None
    staging = None
    if args.save_staging:
        staging = PartsStaging(args.save_staging)
        # BEFORE any resume scan: upload complete parts left by a
        # killed run, delete torn ones
        staging.flush_residual(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    layer_range = (0, 43)
    if args.layer_range:
        layer_range = tuple(int(x) for x in args.layer_range.split(":"))
    expert_range = None
    if args.expert_range:
        expert_range = tuple(int(x) for x in args.expert_range.split(":"))
    valid_scopes = {"all", "attn", "router", "shared", "experts"}
    scopes = tuple(s.strip() for s in args.scope.split(","))
    bad = set(scopes) - valid_scopes
    if bad:
        raise ValueError(f"unknown scope(s): {bad}; valid: {valid_scopes}")
    logger.info("scopes: %s", scopes)

    report = {}
    for layer_idx in range(*layer_range):
        suffix = ""
        if expert_range:
            suffix = f"_e{expert_range[0]}_{expert_range[1]}"
        path = os.path.join(
            args.output_dir, f"quant_layer_{layer_idx:04d}{suffix}.pt"
        )
        # per-matrix parts dir: this environment kills long-running
        # processes, so every finished matrix is persisted atomically
        # the moment it is done; restarts skip existing parts
        parts_dir = path[:-3] + ".parts"
        os.makedirs(parts_dir, exist_ok=True)

        def part_path(short, _d=parts_dir):
            return os.path.join(_d, short + ".pt")

        done_shorts = {
            f[:-3]
            for f in os.listdir(parts_dir)
            if f.endswith(".pt")
        }

        def on_weight(short, qw):
            if staging is not None:
                local = os.path.join(staging.dir, os.path.basename(parts_dir),
                                     short + ".pt")
                os.makedirs(os.path.dirname(local), exist_ok=True)
                ltmp = local + ".ltmp"
                torch.save(qw, ltmp)
                os.replace(ltmp, local)
                staging.save(local, part_path(short))
            else:
                tmp = part_path(short) + ".tmp"
                torch.save(qw, tmp)
                os.replace(tmp, part_path(short))

        out = quantize_layer(
            layer_idx, weights, store, config, device,
            expert_range=expert_range,
            o_groups=args.o_groups,
            n_experts=args.n_experts,
            scopes=scopes,
            skip=done_shorts,
            on_weight=on_weight,
            wrap_R=wrap_R,
            dual_norm=args.dual_norm,
            row_fp8=args.row_fp8,
        )
        merged: Dict[str, QuantizedWeight] = {}
        for short in sorted(done_shorts | set(out)):
            merged[short] = torch.load(
                part_path(short), weights_only=False
            )
        tmp = path + ".tmp"
        torch.save(merged, tmp)
        os.replace(tmp, path)
        for k, qw in out.items():
            report[f"layers.{layer_idx}.{k}"] = {
                "proxy_error": qw.proxy_error,
                "mse": qw.mse,
            }
        logger.info(
            "saved %s (%d new, %d total weights)",
            path, len(out), len(merged),
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
    }
    with open(report_path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info("report written to %s", report_path)


if __name__ == "__main__":
    main()
