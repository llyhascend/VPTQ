"""col_tile 直配列宽（2026-10-20）：groups = in_f // col_tile，
w1/w2/w3 同 tile 几何（32×32）。

验证：① _num_col_groups 语义；② w2 形状（in_f=2048）tile 32×32、
重建 roundtrip 正常；③ w1/w3（in_f=4096）等价旧行为 128 组；
④ 重建对旧 128 组产物（cols=16）自推断兼容。
"""
import torch

from vptq.tools.quantize.config import QuantConfig
from vptq.tools.quantize.rebuild_bf16 import reconstruct_weight
from vptq.tools.quantize.vq import _num_col_groups, vptq_quantize

torch.manual_seed(0)


def _cfg(col_tile=32):
    return QuantConfig(
        vector_len=2, num_centroids=16, num_res_centroids=-1,
        col_tile=col_tile, row_tile=32,
    )


def test_num_col_groups():
    cfg = _cfg()
    # col_tile 直配：in_f 4096 -> 128 组、2048 -> 64 组，同 32 列宽
    assert _num_col_groups(cfg, 4096) == 128
    assert _num_col_groups(cfg, 2048) == 64
    # 关闭：整列一组
    assert _num_col_groups(_cfg(col_tile=0), 4096) == 1


def _roundtrip(out_f, in_f):
    cfg = _cfg()
    W = torch.randn(out_f, in_f) * 0.02
    H = torch.eye(in_f)
    qw = vptq_quantize(W, H, cfg, torch.device("cpu"))
    T, cols, nvec = qw.indices.shape
    g_eff = in_f // 32
    n_row = out_f // cfg.row_tile
    assert qw.centroids.shape[0] == g_eff * n_row  # per-tile codebook (T,k,v)
    assert cols == in_f // g_eff == 32            # 列宽恒 col_tile=32
    assert nvec * cfg.vector_len == cfg.row_tile  # 行高恒 32
    assert T == g_eff * n_row
    Q = reconstruct_weight(qw, g_eff, 32, torch.device("cpu"))
    rel = (Q - W).norm() / W.norm()
    return qw, rel


def test_w2_uniform_tile_and_roundtrip():
    # w2 形状（in_f=2048）：64 组 × 32 列 = tile 32×32（旧代码 128 组 16 列）
    qw, rel = _roundtrip(out_f=128, in_f=2048)
    assert tuple(qw.indices.shape) == (4 * 64, 32, 16)
    assert rel < 0.4  # randn 矩阵 + k=16 粗码本的正常 VQ 误差（实测 0.316）


def test_w1_regression():
    # w1/w3 形状（in_f=4096）：与旧 --group-num 128 行为完全一致
    qw, rel = _roundtrip(out_f=128, in_f=4096)
    assert tuple(qw.indices.shape) == (4 * 128, 32, 16)
    assert rel < 0.4  # 与 w2 同为 randn+k=16 的正常 VQ 误差（实测 0.316）


def test_rebuild_reads_old_128_group_artifact():
    # 旧产物（w2 全局 128 组，cols=16）重建兼容：手工构造旧布局，
    # 调用方按 in_f // cols 推组数（与 pass_rebuild 相同公式）
    in_f, out_f, v, k = 2048, 128, 2, 16
    g_old = 128
    cols, n_row, nvec = in_f // g_old, out_f // 32, 16
    cents = torch.randn(g_old * n_row, k, v)  # per-tile (T, k, v)
    idx = torch.randint(0, k, (g_old * n_row, cols, nvec), dtype=torch.uint16)

    class OldQW:
        pass

    qw = OldQW()
    qw.centroids, qw.indices = cents, idx
    qw.res_centroids = qw.res_indices = None
    gnum = in_f // idx.shape[1]  # 2048 // 16 = 128，新产物则为 2048//32=64
    Q = reconstruct_weight(qw, gnum, 32, torch.device("cpu"))
    assert Q.shape == (out_f, in_f)
