# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Core VPTQ quantization algorithm (tech report §3.1-3.2).

Given a weight matrix W (out_features, in_features) and the Hessian
H = E[x x^T] of the linear's input activations:

1. Hessian-weighted K-means initializes per-group codebooks (the
   diagonal-dominant term of tr(dW^T dW H) makes column i's error weigh
   h_ii, so vectors of column i are weighted by h_ii);
2. Channel-Independent Second-Order Optimization quantizes columns one
   at a time: each column is split into vectors of length `vector_len`,
   each vector snapped to the nearest centroid (H^{-1}_qq is constant
   per column, so plain Euclidean distance is optimal), then the
   quantization error is propagated to the not-yet-quantized columns via
   the Cholesky factor of H^{-1} (GPTQ-style lazy block updates);
3. an optional residual VQ stage repeats the procedure on W - W_hat.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F

from vptq.tools.quantize.config import QuantConfig

logger = logging.getLogger("vptq.quantize")


@dataclass
class QuantizedWeight:
    centroids: torch.Tensor  # (group_num, k, v) fp16
    indices: torch.Tensor  # (group_num, in//group_num, nvec) uint16
    res_centroids: Optional[torch.Tensor]
    res_indices: Optional[torch.Tensor]
    proxy_error: float  # tr(dW^T dW H) / tr(W^T W H)
    mse: float
    # dual-scale normalization (--dual-norm): W was quantized as
    # W_norm = W / row_scale[:,None] / col_scale[None,:]; rebuild
    # multiplies them back (row left, col right). None = not normalized.
    row_scale: Optional[torch.Tensor] = None  # (out,) rms per row
    col_scale: Optional[torch.Tensor] = None  # (in,) rms per column
    # tile geometry this weight was quantized with (self-describing
    # artifact): rebuild reads these per matrix instead of trusting CLI
    # flags, so per-matrix configs (e.g. down's own group_num) and
    # step2/step3 flag mismatches are both handled. Absent in older
    # artifacts — readers fall back to CLI via getattr.
    group_num: int = 1
    row_tile: int = 0
    # wrap-rotation self-description: this weight was quantized as
    # V = W @ R (R = wrap_rotation(in), seed-0 signed Hadamard), so
    # rebuild must rotate back with Q @ R^T (R is orthogonal, not
    # symmetric). False for unwrapped weights; absent in older
    # artifacts — readers fall back to CLI via getattr.
    wrap: bool = False


@torch.no_grad()
def nearest_indices(
    vectors: torch.Tensor,
    centroids: torch.Tensor,
    device: torch.device,
    chunk: int = 16384,
    max_dist_gb: float = 0.0,
) -> torch.Tensor:
    """argmin_k ||v - C_k||^2 via matmul, chunked over vectors.

    vectors: (n, v); centroids: (k, v). Returns (n,) int64 on CPU.
    The (chunk, k) fp32 distance matrix is capped at max_dist_gb —
    with k=65536 the default 16384-row chunk needs 4.3 GiB, which OOMs
    on shared NPUs; shrink the chunk instead. The cap defaults to the
    VPTQ_MAX_DIST_GB env var (fallback 1.0 GiB); raise it on idle cards
    to cut chunk count (and per-chunk launch overhead) 4x.
    """
    import os

    if max_dist_gb <= 0:
        max_dist_gb = float(os.environ.get("VPTQ_MAX_DIST_GB", "1.0"))
    k = centroids.shape[0]
    cap = max(64, int(max_dist_gb * (1 << 30) // (4 * k)))
    chunk = min(chunk, cap)
    centroids = centroids.to(device).float()
    c_sq = centroids.square().sum(-1)
    vectors = vectors.to(device).float()  # hoist: one H2D for all chunks
    out = []
    for i in range(0, vectors.shape[0], chunk):
        v = vectors[i : i + chunk]
        # argmin_k ||v - C||^2 == argmin_k (-2 v·C^T + |C|^2):
        # the per-row |v|^2 term is constant across k, drop it (math
        # identity, saves a full distance-matrix pass). addmm fuses the
        # GEMM + broadcast add into one kernel.
        d2 = torch.addmm(c_sq, v, centroids.t(), alpha=-2.0)
        # stay on device; one host transfer at the end instead of a
        # sync per chunk (chunk count is large when the dist cap
        # shrinks `chunk`, e.g. k=65536 -> 64 chunks)
        out.append(d2.argmin(-1))
    return torch.cat(out).cpu()


@torch.no_grad()
def weighted_kmeans(
    vectors: torch.Tensor,
    weights: torch.Tensor,
    k: int,
    iters: int = 10,
    device: Optional[torch.device] = None,
    seed: int = 0,
    max_points: int = 262144,
    max_dist_gb: float = 1.0,
) -> torch.Tensor:
    """Weighted K-means (Lloyd's) for Hessian-weighted centroid init.

    vectors: (n, v); weights: (n,) — column weight h_ii of each vector.
    Weighted k-means++ seeding; centroid update is the weighted mean
    (assignment is plain nearest — the per-point weight is constant).
    """
    device = device or vectors.device
    n = vectors.shape[0]
    g = torch.Generator().manual_seed(seed)
    if n > max_points:
        sel = torch.randperm(n, generator=g)[:max_points]
        vectors, weights = vectors[sel], weights[sel]
        n = max_points
    vectors = vectors.float()
    weights = weights.float().clamp(min=0)
    # k-means++ seeding is O(k) full-data rounds — fine for small k but
    # infeasible for production codebooks (k=65536). Switch to random
    # seeding for large k; Lloyd iterations fix the rest.
    if k > 2048:
        sel = torch.randperm(n, generator=g)[:k]
        centroids = vectors[sel.to(vectors.device)].clone().to(device)
    else:
        prob = weights / weights.sum()
        # k-means++: first centroid weighted-random; subsequent ones with
        # probability ~ weight * dist_to_nearest
        first = torch.multinomial(prob.cpu(), 1, generator=g).item()
        centroids = [vectors[first]]
        for _ in range(k - 1):
            c = torch.stack(centroids).to(device)
            idx = nearest_indices(vectors, c, device)
            d2 = (
                (vectors.to(device) - c[idx.to(device)])
                .square()
                .sum(-1)
                .cpu()
            )
            p = weights * d2
            if p.sum() <= 0:
                nxt = torch.randint(n, (1,), generator=g).item()
            else:
                nxt = torch.multinomial(p, 1, generator=g).item()
            centroids.append(vectors[nxt])
        centroids = torch.stack(centroids).to(device).float()

    for it in range(iters):
        idx = nearest_indices(
            vectors, centroids, device, max_dist_gb=max_dist_gb
        ).to(device)
        new_c = torch.zeros_like(centroids)
        wsum = torch.zeros(
            centroids.shape[0], device=device, dtype=torch.float32
        )
        w = weights.to(device)
        new_c.index_add_(0, idx, vectors.to(device) * w.unsqueeze(-1))
        wsum.index_add_(0, idx, w)
        # keep empty clusters on their previous centroids
        nonempty = wsum > 0
        new_c[nonempty] = (
            new_c[nonempty] / wsum[nonempty].unsqueeze(-1)
        )
        new_c[~nonempty] = centroids[~nonempty]
        shift = (new_c - centroids).abs().max().item()
        centroids = new_c
        logger.debug("kmeans iter %d shift %.3e", it, shift)
    return centroids


@torch.no_grad()
def weighted_kmeans_batched(
    vectors: torch.Tensor,
    weights: torch.Tensor,
    k: int,
    iters: int = 10,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> torch.Tensor:
    """Batched weighted K-means: same algorithm as `weighted_kmeans`
    run independently over T stacked problems at once.

    vectors: (T, n, v); weights: (T, n). Returns (T, k, v) on `device`.
    Identical math per tile; the batching only removes per-tile kernel
    launch/sync overhead (thousands of tiny tiles otherwise dominate
    runtime).
    """
    device = device or vectors.device
    T, n, v = vectors.shape
    vecs = vectors.to(device).float()
    w = weights.to(device).float().clamp(min=0)
    g = torch.Generator().manual_seed(seed)

    # batched k-means++ (per-row independent sampling), fully on device:
    # Gumbel-max categorical sampling — argmax(log p + G) with
    # G = -log(-log u) is exactly multinomial(p), without per-round
    # D2H transfers / CPU multinomial (the previous bottleneck).
    # Uniforms are drawn on CPU from the seeded generator (cheap) and
    # uploaded once per round; p.clamp handles the psum<=0 case by
    # falling back to (near-)uniform, same as the old torch.where.
    def _sample_cat(p: torch.Tensor) -> torch.Tensor:
        u = torch.rand(p.shape, generator=g).clamp(min=1e-20).to(device)
        score = torch.log(p.clamp(min=1e-20)) - torch.log(-torch.log(u))
        return score.argmax(-1)

    prob = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)
    first = _sample_cat(prob)
    cents = [vecs.gather(1, first.view(T, 1, 1).expand(T, 1, v))]
    for _ in range(k - 1):
        c = torch.cat(cents, dim=1)  # (T, c, v)
        d2 = (
            vecs.square().sum(-1, keepdim=True)
            - 2 * torch.bmm(vecs, c.transpose(1, 2))
            + c.square().sum(-1).unsqueeze(1)
        ).clamp(min=0)  # (T, n, c)
        d2min = d2.min(-1).values  # (T, n)
        nxt = _sample_cat(w * d2min)
        cents.append(
            vecs.gather(1, nxt.view(T, 1, 1).expand(T, 1, v))
        )
    centroids = torch.cat(cents, dim=1)  # (T, k, v)

    offsets = (torch.arange(T, device=device) * k).unsqueeze(1)  # (T,1)
    for it in range(iters):
        # argmin-only: drop row-constant |v|^2, fuse to one baddbmm
        d2 = torch.baddbmm(
            centroids.square().sum(-1).unsqueeze(1),
            vecs,
            centroids.transpose(1, 2),
            alpha=-2.0,
        )  # (T, n, k)
        idx = d2.argmin(-1)  # (T, n)
        flat = (idx + offsets).reshape(-1)  # T*n
        new_c = torch.zeros(T * k, v, device=device)
        wsum = torch.zeros(T * k, device=device)
        new_c.index_add_(0, flat, (vecs * w.unsqueeze(-1)).reshape(-1, v))
        wsum.index_add_(0, flat, w.reshape(-1))
        new_c = new_c.view(T, k, v)
        wsum = wsum.view(T, k)
        nonempty = wsum > 0
        new_c = torch.where(
            nonempty.unsqueeze(-1),
            new_c / wsum.clamp(min=1e-12).unsqueeze(-1),
            centroids,
        )
        shift = (new_c - centroids).abs().max().item()
        centroids = new_c
        logger.debug("batched kmeans iter %d shift %.3e", it, shift)
    return centroids


def cholesky_upper_of_inv(
    hessian: torch.Tensor, damp: float = 0.01
) -> torch.Tensor:
    """Upper Cholesky factor U of (H + lambda I)^{-1}, H^{-1} = U U^T.

    This is the exact operator GPTQ-style error propagation needs.
    Computed in float64, returned float32.
    """
    from vptq.tools.hessian.collector import damped_inverse

    inv = damped_inverse(hessian, damp=damp).double()
    return torch.linalg.cholesky(inv).t().float()


def _pad_out(w: torch.Tensor, v: int) -> torch.Tensor:
    pad = (-w.shape[0]) % v
    return F.pad(w, (0, 0, 0, pad)) if pad else w


class _NpuBlockGraph:
    """NPU-graph capture of the 128-column GPTQ block loop.

    The eager loop issues ~10 tiny kernels per column (~41k launches per
    matrix); capturing the block body once per shape signature and
    replaying it removes the launch gaps (measured 2.6x on the block).
    Static slots hold the block's weight slice, Hessian-Cholesky block
    and per-group codebooks; replay = copy inputs in, g.replay(), copy
    results out. Bitwise identical to the eager loop (same op sequence).
    """

    _cache: dict = {}

    def __init__(self, out_pad, n_row, nvec, v, k, gw, gpb, block_cols,
                 device):
        self.gw, self.gpb = gw, gpb
        self.sW1 = torch.zeros(out_pad, block_cols, device=device)
        self.sH1 = torch.zeros(block_cols, block_cols, device=device)
        self.sCB = torch.zeros(gpb, n_row, k, v, device=device)
        self.sCSQ = torch.zeros(gpb, n_row, k, device=device)
        self.sQ = torch.zeros(out_pad, block_cols, device=device)
        self.sErr = torch.zeros(out_pad, block_cols, device=device)
        self.sIdx = torch.zeros(
            gpb, n_row, min(gw, block_cols), nvec,
            dtype=torch.long, device=device,
        )
        self.graph = None

    def _body(self, n_row, nvec, v):
        gw = self.gw
        for j in range(self.sW1.shape[1]):
            slot = j // gw if gw <= self.sW1.shape[1] else 0
            pos = j % gw if gw <= self.sW1.shape[1] else j
            csb, csq = self.sCB[slot], self.sCSQ[slot]
            w = self.sW1[:, j]
            d = self.sH1[j, j]
            vecs = w.view(n_row, nvec, v)
            d2 = torch.baddbmm(
                csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0
            )
            idx = d2.argmin(-1)
            q = torch.gather(
                csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)
            ).reshape(-1)
            self.sQ[:, j] = q
            self.sIdx[slot][:, pos] = idx
            err = (w - q) / d
            self.sW1[:, j:].addr_(err, self.sH1[j, j:], alpha=-1.0)
            self.sErr[:, j] = err

    def capture(self, n_row, nvec, v):
        self._body(n_row, nvec, v)  # warmup (also validates capturability)
        torch.npu.synchronize()
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            self._body(n_row, nvec, v)
        self.graph = g

    @classmethod
    def get(cls, key, device, factory_args):
        import os

        if os.environ.get("VPTQ_DISABLE_GRAPH"):
            return None
        if key not in cls._cache:
            try:
                inst = cls(*factory_args, device)
                inst.capture(factory_args[1], factory_args[2], factory_args[3])
                cls._cache[key] = inst
            except Exception as err:  # pragma: no cover - env dependent
                logger.warning("NPU graph capture failed (%s); eager", err)
                cls._cache[key] = None
        return cls._cache[key]


@torch.no_grad()
def _quantize_columns(
    W: torch.Tensor,
    HinvU: torch.Tensor,
    codebooks,  # list of (k, v) centroids, one per (col group, row span)
    group_bounds,  # list of (col_start, col_end) per group
    v: int,
    device: torch.device,
    block_cols: int = 128,
    row_spans=None,  # list of (row_start, row_end); default: whole matrix
) -> tuple:
    """Blocked second-order column quantization.

    Codebooks are addressed as codebooks[col_group * len(row_spans) + t];
    with the default single row span this reduces to one codebook per
    column group. Returns (Q, indices) with Q the same shape as W
    (padded) and indices as a list per tile of
    (cols_in_group, span_len // v) int64.
    """
    out_pad, in_f = W.shape
    if row_spans is None:
        row_spans = [(0, out_pad)]
    n_row = len(row_spans)
    group_of_col = [0] * in_f
    for gi, (cs, ce) in enumerate(group_bounds):
        for col in range(cs, ce):
            group_of_col[col] = gi
    col_centroids = [c.to(device).float() for c in codebooks]
    span_lens = [re - rs for rs, re in row_spans]
    uniform = len(set(span_lens)) == 1 and span_lens[0] % v == 0
    if uniform:
        # stack each col-group's span codebooks for one batched assign
        stacked = []
        for gi in range(len(group_bounds)):
            base = gi * n_row
            csb = torch.stack(
                [col_centroids[base + t] for t in range(n_row)]
            )  # (n_row, k, v)
            stacked.append((csb, csb.square().sum(-1)))
        nvec = span_lens[0] // v

    Wwork = W.to(device).float()
    HinvU = HinvU.to(device)
    Q = torch.zeros_like(Wwork)
    if uniform:
        # device-resident index blocks; one host transfer at the end
        # (a .cpu() per column per span would force a device sync each)
        idx_blocks = [
            torch.zeros(
                n_row, ce - cs, span_lens[0] // v,
                dtype=torch.long, device=device,
            )
            for cs, ce in group_bounds
        ]
    else:
        idx_cpu = [
            torch.zeros(ce - cs, (re - rs) // v, dtype=torch.long)
            for cs, ce in group_bounds
            for rs, re in row_spans
        ]
    # NPU-graph fast path: uniform spans, full blocks, equal-width groups
    # aligned to the block grid (covers group_num=1 and tiled schemes)
    graph = None
    if uniform and device.type == "npu" and len(group_bounds) >= 1:
        gw = group_bounds[0][1] - group_bounds[0][0]
        aligned = all(
            (ce - cs) == gw and cs % gw == 0 for cs, ce in group_bounds
        ) and (gw >= block_cols or block_cols % gw == 0)
        if aligned:
            gpb = max(1, block_cols // gw) if gw < block_cols else 1
            k = col_centroids[0].shape[0]
            key = (out_pad, n_row, nvec, v, k, gw, gpb, block_cols)
            graph = _NpuBlockGraph.get(
                key, device, (out_pad, n_row, nvec, v, k, gw, gpb, block_cols)
            )
    for i1 in range(0, in_f, block_cols):
        i2 = min(i1 + block_cols, in_f)
        count = i2 - i1
        if graph is not None and count == block_cols:
            graph.sW1.copy_(Wwork[:, i1:i2])
            graph.sH1.copy_(HinvU[i1:i2, i1:i2])
            gi0 = i1 // gw if gw < block_cols else 0
            for s in range(graph.gpb):
                gi = gi0 + s if gw < block_cols else 0
                graph.sCB[s].copy_(stacked[gi][0])
                graph.sCSQ[s].copy_(stacked[gi][1])
            graph.graph.replay()
            Q[:, i1:i2] = graph.sQ
            Err1 = graph.sErr
            for s in range(graph.gpb):
                if gw < block_cols:
                    idx_blocks[gi0 + s].copy_(graph.sIdx[s])
                else:
                    idx_blocks[0][:, i1:i2].copy_(graph.sIdx[0])
            if i2 < in_f:
                Wwork[:, i2:].addmm_(Err1, HinvU[i1:i2, i2:], alpha=-1.0)
            continue
        W1 = Wwork[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = HinvU[i1:i2, i1:i2]
        for j in range(count):
            col = i1 + j
            w = W1[:, j]
            d = Hinv1[j, j]
            gi = group_of_col[col]
            cs, _ = group_bounds[gi]
            if uniform:
                csb, csq = stacked[gi]
                vecs = w.view(n_row, nvec, v)  # spans contiguous, equal
                # argmin_k ||v-C||^2 == argmin_k (-2 v·C^T + |C|^2)
                # (row-constant |v|^2 dropped); baddbmm fuses to one
                # batched GEMM kernel
                d2 = torch.baddbmm(
                    csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0
                )
                idx = d2.argmin(-1)  # (n_row, nvec)
                q_col = torch.gather(
                    csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)
                ).reshape(-1)
                idx_blocks[gi][:, col - cs] = idx
            else:
                q_col = torch.empty_like(w)
                base = gi * n_row
                for t, (rs, re) in enumerate(row_spans):
                    c = col_centroids[base + t]
                    vecs = w[rs:re].view(-1, v)
                    d2 = torch.addmm(
                        c.square().sum(-1), vecs, c.t(), alpha=-2.0
                    )
                    idx = d2.argmin(-1)
                    q_col[rs:re] = c[idx].view(-1)
                    idx_cpu[base + t][col - cs] = idx.cpu()
            Q[:, col] = q_col
            err1 = (w - q_col) / d
            # rank-1 lazy update fused into one addr_ kernel
            W1[:, j:].addr_(err1, Hinv1[j, j:], alpha=-1.0)
            Err1[:, j] = err1
        if i2 < in_f:
            Wwork[:, i2:].addmm_(Err1, HinvU[i1:i2, i2:], alpha=-1.0)
    if uniform:
        idx_per_group = [
            blk[t].cpu() for blk in idx_blocks for t in range(n_row)
        ]
    else:
        idx_per_group = idx_cpu
    return Q.cpu(), idx_per_group


def _stack_tiles(
    M: torch.Tensor,
    bounds,
    spans,
    diag_h: torch.Tensor,
    v: int,
):
    """Build per-tile vector stacks (T, n, v) + weights (T, n) for kmeans.

    Tile order is group-major: tile = gi * len(spans) + t; within a tile,
    vectors are the tile's columns split along rows into v-length pieces
    (column-major), matching cols.t().reshape(-1, v) from the reference
    loop. Uses view/permute/reshape (one pass) when all tiles are uniform;
    falls back to the per-tile loop otherwise. Returns None in the
    fallback case so callers can use the loop.
    """
    span_lens = {re - rs for rs, re in spans}
    widths = {ce - cs for cs, ce in bounds}
    if len(span_lens) == 1 and len(widths) == 1 and len(spans) > 1:
        S = span_lens.pop()
        B = widths.pop()
        G = len(bounds)
        n_row = len(spans)
        if S % v == 0 and M.shape[0] == n_row * S and M.shape[1] == G * B:
            vecs = (
                M.view(n_row, S, G, B)
                .permute(2, 0, 3, 1)  # (G, n_row, B, S)
                .reshape(G * n_row, B, S // v, v)
                .reshape(G * n_row, B * (S // v), v)
            )
            w = (
                diag_h.view(G, B)
                .unsqueeze(1)
                .unsqueeze(-1)
                .expand(G, n_row, B, S // v)
                .reshape(G * n_row, B * (S // v))
            )
            return vecs.contiguous(), w.contiguous()
    return None


@torch.no_grad()
def _tile_vecs_loop(M, bounds, spans, diag_h, v):
    tile_vecs, tile_w = [], []
    for gi, (cs, ce) in enumerate(bounds):
        for rs, re in spans:
            cols = M[rs:re, cs:ce]
            tile_vecs.append(cols.t().reshape(-1, v))
            tile_w.append(diag_h[cs:ce].repeat_interleave(cols.shape[0] // v))
    return tile_vecs, tile_w


@torch.no_grad()
def _num_col_groups(config, in_f: int) -> int:
    """Column groups at the configured ``col_tile`` width (2026-10-20).

    ``config.col_tile`` is the column-block width itself (not a group
    count): groups = ceil(in_f / col_tile), so every matrix -- w1/w3
    (in_f=4096) and w2 (in_f=2048) alike -- gets the SAME tile geometry
    (col_tile x row_tile, e.g. 32x32). col_tile<=0 means whole-row
    single group (legacy no-grouping semantics).
    """
    if config.col_tile <= 0:
        return 1
    return max(1, -(-in_f // config.col_tile))


def vptq_quantize(
    W: torch.Tensor,
    hessian: torch.Tensor,
    config: QuantConfig,
    device: Optional[torch.device] = None,
    hessian_inv_upper: Optional[torch.Tensor] = None,
    kmeans_seed: int = 0,
) -> QuantizedWeight:
    """Quantize W (out_features, in_features) with VPTQ.

    Args:
        W: weight, any float dtype.
        hessian: E[x x^T] of the input activations (in, in).
        config: QuantConfig.
        device: compute device (NPU/CUDA/CPU).
        hessian_inv_upper: optional precomputed upper Cholesky factor of
            the damped H^{-1} (recomputed from `hessian` otherwise).
    """
    device = device or torch.device("cpu")
    out_f, in_f = W.shape
    v = config.vector_len
    v_res = config.res_vector_len or v
    import math as _math

    Wp = _pad_out(W.float(), v * v_res // _math.gcd(v, v_res))
    row_spans = (
        [(rs, min(rs + config.row_tile, Wp.shape[0]))
         for rs in range(0, Wp.shape[0], config.row_tile)]
        if config.row_tile
        else None
    )

    HinvU = (
        hessian_inv_upper
        if hessian_inv_upper is not None
        else cholesky_upper_of_inv(hessian, config.damp)
    )

    # column groups along in_features (fixed col_tile width, see
    # _num_col_groups: w2 gets in_f//col_tile groups -> uniform 32x32 tiles)
    g_eff = _num_col_groups(config, in_f)
    bounds = []
    for gi in range(g_eff):
        cs = gi * in_f // g_eff
        ce = (gi + 1) * in_f // g_eff
        bounds.append((cs, ce))

    # --- stage 1: weighted k-means init per (col group, row span) tile ---
    diag_h = torch.diagonal(hessian).float().cpu()
    spans = row_spans or [(0, Wp.shape[0])]
    fast = _stack_tiles(Wp, bounds, spans, diag_h, v)
    if fast is not None:
        # vectorized uniform stack: one batched kmeans over all tiles
        vecs_st, w_st = fast
        logger.info(
            "batched kmeans: %d tiles x %d vectors (k=%d)",
            vecs_st.shape[0], vecs_st.shape[1], config.num_centroids,
        )
        codebooks = list(
            weighted_kmeans_batched(
                vecs_st, w_st, config.num_centroids,
                iters=config.kmeans_iters, device=device, seed=kmeans_seed,
            ).unbind(0)
        )
    else:
        tile_vecs, tile_w = _tile_vecs_loop(Wp, bounds, spans, diag_h, v)
        if len({tuple(t.shape) for t in tile_vecs}) == 1 and len(tile_vecs) > 1:
            # uniform tiles: one batched kmeans for all (col group, span)
            # tiles — identical math, removes thousands of tiny launches
            logger.info(
                "batched kmeans: %d tiles x %d vectors (k=%d)",
                len(tile_vecs), tile_vecs[0].shape[0], config.num_centroids,
            )
            codebooks = list(
                weighted_kmeans_batched(
                    torch.stack(tile_vecs),
                    torch.stack(tile_w),
                    config.num_centroids,
                    iters=config.kmeans_iters,
                    device=device,
                    seed=kmeans_seed,
                ).unbind(0)
            )
        else:
            codebooks = [
                weighted_kmeans(
                    vv, ww, config.num_centroids,
                    iters=config.kmeans_iters, device=device,
                    seed=kmeans_seed + ti,
                )
                for ti, (vv, ww) in enumerate(zip(tile_vecs, tile_w))
            ]

    # --- stage 2: second-order column quantization ---
    Q, idx_groups = _quantize_columns(
        Wp,
        HinvU,
        codebooks,
        bounds,
        v,
        device,
        block_cols=config.block_cols,
        row_spans=row_spans,
    )

    # --- stage 3: optional residual VQ ---
    res_cb = res_idx = None
    Qres = torch.zeros_like(Q)
    if config.num_res_centroids > 0:
        R = Wp - Q
        fast_r = _stack_tiles(R, bounds, spans, diag_h, v_res)
        if fast_r is not None:
            res_codebooks = list(
                weighted_kmeans_batched(
                    fast_r[0], fast_r[1], config.num_res_centroids,
                    iters=config.kmeans_iters, device=device,
                    seed=kmeans_seed + 1000,
                ).unbind(0)
            )
        else:
            res_vecs, res_w = _tile_vecs_loop(R, bounds, spans, diag_h, v_res)
            if len({tuple(t.shape) for t in res_vecs}) == 1 and len(res_vecs) > 1:
                res_codebooks = list(
                    weighted_kmeans_batched(
                        torch.stack(res_vecs),
                        torch.stack(res_w),
                        config.num_res_centroids,
                        iters=config.kmeans_iters,
                        device=device,
                        seed=kmeans_seed + 1000,
                    ).unbind(0)
                )
            else:
                res_codebooks = [
                    weighted_kmeans(
                        vv, ww, config.num_res_centroids,
                        iters=config.kmeans_iters, device=device,
                        seed=kmeans_seed + 1000 + ti,
                    )
                    for ti, (vv, ww) in enumerate(zip(res_vecs, res_w))
                ]
        Qres, res_idx = _quantize_columns(
            R, HinvU, res_codebooks, bounds, v_res, device,
            block_cols=config.block_cols,
            row_spans=row_spans,
        )
        res_cb = torch.stack([c.cpu() for c in res_codebooks])
        res_cb = res_cb.to(
            torch.float8_e4m3fn if config.centroid_fp8 else torch.float16
        )

    # --- error metrics against the ORIGINAL (uncompensated) W ---
    What = (Q + Qres)[:out_f].to(device)
    W_orig = W.to(device).float()
    dW = What - W_orig
    H = hessian.to(device).float()
    proxy = (dW @ H * dW).sum() / (W_orig @ H * W_orig).sum().clamp(min=1e-12)
    mse = dW.square().mean() / W_orig.square().mean().clamp(min=1e-12)

    return QuantizedWeight(
        centroids=torch.stack([c.cpu() for c in codebooks]).to(
            torch.float8_e4m3fn if config.centroid_fp8 else torch.float16
        ),
        indices=torch.stack(idx_groups).to(torch.uint16).cpu(),
        res_centroids=res_cb,
        res_indices=(
            torch.stack(res_idx).to(torch.uint16).cpu()
            if res_idx is not None
            else None
        ),
        proxy_error=proxy.item(),
        mse=mse.item(),
        group_num=config.group_num,
        row_tile=config.row_tile,
    )


@torch.no_grad()
def vptq_quantize_joint(
    Ws,
    hessian: torch.Tensor,
    config: QuantConfig,
    device: Optional[torch.device] = None,
    hessian_inv_upper: Optional[torch.Tensor] = None,
    kmeans_seed: int = 0,
):
    """Quantize several same-shape matrices sharing one Hessian, training
    all tile codebooks in ONE batched kmeans (amortizes seeding/launch
    overhead across the batch). Each tile still gets its own codebook —
    results are per-tile identical to separate `vptq_quantize` calls up to
    RNG stream. Requires identical (out, in) shapes (e.g. expert w1+w3).
    """
    assert len({tuple(w.shape) for w in Ws}) == 1, "joint requires same shapes"
    device = device or torch.device("cpu")
    out_f, in_f = Ws[0].shape
    v = config.vector_len
    v_res = config.res_vector_len or v
    import math as _math

    pad_to = v * v_res // _math.gcd(v, v_res)
    Wps = [_pad_out(W.float(), pad_to) for W in Ws]
    row_spans = (
        [(rs, min(rs + config.row_tile, Wps[0].shape[0]))
         for rs in range(0, Wps[0].shape[0], config.row_tile)]
        if config.row_tile else None
    )
    HinvU = (
        hessian_inv_upper
        if hessian_inv_upper is not None
        else cholesky_upper_of_inv(hessian, config.damp)
    )
    g_eff = _num_col_groups(config, in_f)
    bounds = [
        (gi * in_f // g_eff, (gi + 1) * in_f // g_eff)
        for gi in range(g_eff)
    ]
    diag_h = torch.diagonal(hessian).float().cpu()
    spans = row_spans or [(0, Wps[0].shape[0])]

    def kmeans_joint(mats, vv, seed):
        stacks = []
        for M in mats:
            fast = _stack_tiles(M, bounds, spans, diag_h, vv)
            if fast is not None:
                stacks.append(fast)
            else:
                tv, tw = _tile_vecs_loop(M, bounds, spans, diag_h, vv)
                stacks.append((torch.stack(tv), torch.stack(tw)))
        vecs = torch.cat([s[0] for s in stacks], 0)
        w = torch.cat([s[1] for s in stacks], 0)
        logger.info(
            "joint batched kmeans: %d tiles x %d vectors (k=%d, %d mats)",
            vecs.shape[0], vecs.shape[1],
            config.num_centroids if vv == v else config.num_res_centroids,
            len(mats),
        )
        cb = weighted_kmeans_batched(
            vecs, w,
            config.num_centroids if vv == v else config.num_res_centroids,
            iters=config.kmeans_iters, device=device, seed=seed,
        )
        T = vecs.shape[0] // len(mats)
        return [list(cb[i * T : (i + 1) * T].unbind(0)) for i in range(len(mats))]

    cb_per_mat = kmeans_joint(Wps, v, kmeans_seed)
    outs = []
    Rs = []
    for W, Wp, codebooks in zip(Ws, Wps, cb_per_mat):
        Q, idx_groups = _quantize_columns(
            Wp, HinvU, codebooks, bounds, v, device,
            block_cols=config.block_cols, row_spans=row_spans,
        )
        Rs.append((W, Q, idx_groups))

    res_packs = [None] * len(Ws)
    if config.num_res_centroids > 0:
        Rmats = [Wp - Q for (_, Q, _), (_, Wp) in zip(Rs, enumerate(Wps))]
        Rmats = [Wp - Q for (W, Q, _), Wp in zip(Rs, Wps)]
        rcb_per_mat = kmeans_joint(Rmats, v_res, kmeans_seed + 1000)
        res_packs = []
        for (W, Q, _), Wp, res_codebooks in zip(Rs, Wps, rcb_per_mat):
            R = Wp - Q
            Qres, res_idx = _quantize_columns(
                R, HinvU, res_codebooks, bounds, v_res, device,
                block_cols=config.block_cols, row_spans=row_spans,
            )
            res_packs.append((Qres, res_idx, res_codebooks))

    H = hessian.to(device).float()
    for i, ((W, Q, idx_groups), Wp) in enumerate(zip(Rs, Wps)):
        if res_packs[i] is None:
            Qres = torch.zeros_like(Q)
            res_cb = res_idx = None
        else:
            Qres_i, res_idx, res_codebooks = res_packs[i]
            Qres = Qres_i
            res_cb = torch.stack([c.cpu() for c in res_codebooks]).to(
                torch.float8_e4m3fn if config.centroid_fp8 else torch.float16
            )
        What = (Q + Qres)[:out_f].to(device)
        W_orig = Ws[i].to(device).float()
        dW = What - W_orig
        proxy = (dW @ H * dW).sum() / (W_orig @ H * W_orig).sum().clamp(min=1e-12)
        mse = dW.square().mean() / W_orig.square().mean().clamp(min=1e-12)
        outs.append(
            QuantizedWeight(
                centroids=torch.stack([c.cpu() for c in cb_per_mat[i]]).to(
                    torch.float8_e4m3fn if config.centroid_fp8 else torch.float16
                ),
                indices=torch.stack(idx_groups).to(torch.uint16).cpu(),
                res_centroids=res_cb,
                res_indices=(
                    torch.stack(res_idx).to(torch.uint16).cpu()
                    if res_packs[i] is not None else None
                ),
                proxy_error=proxy.item(),
                mse=mse.item(),
                group_num=config.group_num,
                row_tile=config.row_tile,
            )
        )
    return outs
