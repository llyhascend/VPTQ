import pytest
import torch

from vptq.tools.quantize.hadamard import block_hadamard_matrix, wrap_rotation
from vptq.tools.quantize.mxfp4_ceilx import mxfp4_ceilx_roundtrip


def test_grid_values():
    """Every output value must sit on the e2m1 grid x 2^s."""
    torch.manual_seed(0)
    w = torch.randn(64, 128) * 0.03
    out = mxfp4_ceilx_roundtrip(w, ceil_x_value=7.25)
    assert out.shape == w.shape and out.dtype == w.dtype
    xb = w.view(64, -1, 32)
    ob = out.view(64, -1, 32)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    s = torch.ceil(torch.log2(amax / 7.25 + 9.6e-7)).clamp(-129, 125)
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    # scale back: out / 2^s must be on the e2m1 grid per block
    scaled = (ob / torch.exp2(s)).flatten()
    mindist = (scaled.abs().unsqueeze(-1) - grid.unsqueeze(0)).abs().min(dim=-1).values
    assert mindist.max() < 1e-6, mindist.max()


def test_zero_blocks_flush():
    w = torch.zeros(4, 64)
    w[0, :3] = torch.tensor([0.001, -0.002, 0.003])  # tiny values in one block
    out = mxfp4_ceilx_roundtrip(w)
    assert torch.equal(out[1:], torch.zeros_like(out[1:]))


def test_determinism():
    torch.manual_seed(1)
    w = torch.randn(32, 96)
    assert torch.equal(mxfp4_ceilx_roundtrip(w), mxfp4_ceilx_roundtrip(w))


def test_wrap_beats_nowrap_same_quantizer():
    """Isolate the rotation benefit: same ceil_x quantizer, with vs without
    wrap. Realistic pattern — channel outliers inside every block (the
    §8A.6b argument: rotation spreads needles so per-block amax drops)."""
    torch.manual_seed(0)
    w = torch.randn(128, 4096) * 0.02
    # systematic channel-wise outliers: a few channels per block carry x50
    chans = torch.arange(3, 4096, 97)
    w[:, chans] *= 50.0

    R = wrap_rotation(w.shape[1])
    assert R.shape == (4096, 4096)

    mse_nowrap = ((mxfp4_ceilx_roundtrip(w) - w) ** 2).mean().item()
    mse_wrap = ((mxfp4_ceilx_roundtrip(w @ R) @ R.t() - w) ** 2).mean().item()
    assert mse_wrap < mse_nowrap, (mse_wrap, mse_nowrap)


def test_block_hadamard_orthogonal():
    R = block_hadamard_matrix(2048, block=32, seed=0)
    I = R @ R.t()
    assert torch.allclose(I, torch.eye(2048), atol=1e-5)
