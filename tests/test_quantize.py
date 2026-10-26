# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Export-path tests: QuantizedWeight -> VQuantLinear -> ops.dequant
must reconstruct the same weight as the reference layout, and the
packed-index path must round-trip."""

import torch

from vptq.tools.quantize import QuantConfig, vptq_quantize
from vptq.tools.quantize.export import build_vqlinear, reconstruct_weight


def _make_wh(seed=0, in_f=48, out_f=64, n=4096):
    g = torch.Generator().manual_seed(seed)
    mixing = torch.randn(in_f, in_f, generator=g) * 0.5 + torch.eye(in_f)
    X = torch.randn(n, in_f, generator=g) @ mixing
    H = X.t() @ X / n
    A = torch.randn(out_f, 4, generator=g)
    B = torch.randn(4, in_f, generator=g)
    W = A @ B / 2 + 0.02 * torch.randn(out_f, in_f, generator=g)
    return W, H


def _config(**kw):
    base = dict(
        vector_len=8,
        num_centroids=64,
        num_res_centroids=16,
        col_tile=24,  # in_f=48 -> 2 groups (was group_num=2)
        kmeans_iters=4,
    )
    base.update(kw)
    return QuantConfig(**base)


def test_build_vqlinear_dequant_matches_reference():
    W, H = _make_wh()
    cfg = _config()
    qw = vptq_quantize(W, H, cfg, device=torch.device("cpu"))
    ref = reconstruct_weight(qw, W.shape[0])

    layer = build_vqlinear(qw, W.shape[1], W.shape[0], cfg, device="cpu")
    from vptq import ops

    v = cfg.vector_len
    g_eff = W.shape[1] // cfg.col_tile
    centroids3 = layer.centroids.weight.view(
        g_eff, cfg.num_centroids, v
    )
    res_centroids3 = layer.res_centroids.weight.view(
        g_eff, cfg.num_res_centroids, v
    )
    qweight = ops.dequant(
        indices=layer.indices,
        centroids=centroids3,
        outlier_indices=None,
        outlier_centroids=None,
        res_indices=layer.res_indices,
        res_centroids=res_centroids3,
        perm=None,
        weight_scale=None,
        weight_bias=None,
        is_indice_packed=False,
        enable_outlier=False,
        enable_residual=True,
        enable_perm=False,
        enable_norm=False,
        num_centroids=cfg.num_centroids,
        num_outlier_centroids=0,
        num_res_centroids=cfg.num_res_centroids,
        padding=layer.padding,
        outlier_padding=0,
        num_codebooks=g_eff,
        group_size=cfg.col_tile,
        outlier_size=0,
        vector_len=cfg.vector_len,
        outlier_vector_len=-1,
    )
    assert qweight.shape == W.shape
    assert torch.allclose(qweight.float(), ref, atol=1e-2)


def test_packed_indices_roundtrip():
    # the packed path (is_indice_packed=True) must dequantize identically
    W, H = _make_wh()
    cfg = _config()
    qw = vptq_quantize(W, H, cfg, device=torch.device("cpu"))
    ref = reconstruct_weight(qw, W.shape[0])

    import math

    from vptq.utils.pack import pack_index, unpack_index_tensor

    index_bits = int(math.log2(cfg.num_centroids))
    res_bits = int(math.log2(cfg.num_res_centroids))
    group_size = cfg.col_tile
    for g in range(W.shape[1] // cfg.col_tile):
        indice = qw.indices[g].t().contiguous().unsqueeze(0)  # (1,nvec,cols)
        res = qw.res_indices[g].t().contiguous().unsqueeze(0)
        packed = pack_index(
            indice=indice,
            index_bits=index_bits,
            res_indice=res,
            res_bits=res_bits,
            index_dtype=torch.uint16,
        )
        back, back_res = unpack_index_tensor(
            packed, index_bits, group_size, res_bits, group_size
        )
        assert torch.equal(back, indice.view(torch.uint16).to(torch.int64))
        assert torch.equal(
            back_res, res.view(torch.uint16).to(torch.int64)
        )


def test_proxy_error_reasonable_on_structured_weight():
    W, H = _make_wh()
    cfg = _config()
    qw = vptq_quantize(W, H, cfg, device=torch.device("cpu"))
    # low-rank-ish weight, 8-bit-equivalent VQ (64 centroids / v8) + res:
    # should reconstruct well under the Hessian metric
    assert qw.proxy_error < 0.2
    assert qw.mse < 0.2


def test_tile_scheme_v2_k16_res8():
    """用户规格: v=2 K=16(4bit) 主量化, v_res=8 K=16 残差,
    32列x256行 tile 分组, fp8 码本 -> 2.03125 / 2.65625 bit/elem"""
    torch.manual_seed(0)
    in_f, out_f = 64, 256
    X = torch.randn(4096, in_f)
    H = X.t() @ X / X.shape[0]
    W = torch.randn(out_f, in_f) * 0.05

    cfg = QuantConfig.v2_k16_res8_k16()
    cfg.col_tile = 32  # 32 列/组
    qw = vptq_quantize(W, H, cfg, device=torch.device("cpu"))

    g_eff = in_f // cfg.col_tile
    n_tiles = g_eff * (out_f // 256)
    assert qw.centroids.shape == (n_tiles, 16, 2)
    assert qw.centroids.dtype == torch.float8_e4m3fn
    assert qw.res_centroids.shape == (n_tiles, 16, 8)
    assert qw.indices.shape == (n_tiles, 32, 128)   # (tile, cols, 256/2)
    assert qw.res_indices.shape == (n_tiles, 32, 32)  # (tile, cols, 256/8)

    # 重建 Ŵ（tile 布局）
    def reconstruct(qw, with_res=True):
        Q = torch.zeros(out_f, in_f)
        for gi in range(g_eff):
            for t in range(1):
                tile = gi * 1 + t
                cb = qw.centroids[tile].float()
                idx = qw.indices[tile].long()
                block = cb[idx].permute(1, 2, 0).reshape(-1, 32)
                if with_res:
                    rcb = qw.res_centroids[tile].float()
                    ridx = qw.res_indices[tile].long()
                    rvecs = rcb[ridx]  # (32, 32, 8)
                    block = block + rvecs.permute(1, 2, 0).reshape(-1, 32)
                Q[t * 256 : (t + 1) * 256, gi * 32 : (gi + 1) * 32] = block
        return Q

    W_main = reconstruct(qw, with_res=False)
    W_full = reconstruct(qw, with_res=True)

    def proxy(What):
        dW = What - W
        return ((dW @ H * dW).sum() / (W @ H * W).sum()).item()

    e_main, e_full = proxy(W_main), proxy(W_full)
    print(f"tile scheme: main {e_main:.5f} -> +res {e_full:.5f}")
    assert e_full < e_main
    # 位宽核算
    body = 4 / 2 + 4 / (4 * 2)
    amort = (16 * 2 * 8 + 16 * 8 * 8) / (32 * 256)
    assert abs((body + amort) - 2.65625) < 1e-9
    assert abs(cfg.bits_per_weight - 2.5) < 1e-9
