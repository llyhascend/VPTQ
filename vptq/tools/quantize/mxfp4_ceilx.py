# -------------------------------------------------------------------------
# Standalone MXFP4 ceil_x (c7.25) quantize-dequantize, ported from
# msmodelslim's MXWeightPerBlockCeilX (msmodelslim/core/quantizer/impl/
# ceil_x.py + ir/api/impl/mx_quantization.py). Used by rebuild_bf16.py
# --fallback-mxfp4 to pre-quantize fallback experts in wrap domain:
#   V = W @ R (block-diag Hadamard, block=32, seed=0) ->
#   Vdq = mxfp4_ceilx_roundtrip(V, 7.25) -> part = Vdq @ R^T -> BF16 part.
# Final e4m3 cast stays in msmodelslim via yaml rule B scale_one.
# -------------------------------------------------------------------------

import torch
import torch.nn.functional as F

# MXFP4 finfo (ir/qal/qbase.py): ebits=2, mbits=3, emax=2, scale_bits=8.
_EMAX = 2
_SCALE_EMAX = 2 ** (8 - 1) - 1  # 127
_EPS = 9.6e-7


def mxfp4_ceilx_roundtrip(
    w: torch.Tensor,
    ceil_x_value: float = 7.25,
    block_size: int = 32,
) -> torch.Tensor:
    """Quantize-dequantize ``w`` (float, [out_f, in_f]) as MXFP4 with the
    ceil_x shared-exponent rule; returns a same-shape float tensor.

    shared_exp s = ceil(log2(amax / c + eps)), clipped to
    [-scale_emax - emax, scale_emax - emax] = [-129, 125].

    Grid per 32-block: sign * v * 2^s with v in {0, .5, 1, 1.5, 2, 3, 4, 6}
    (private exp clipped to [0, 2]; mantissa floor(x/2^s/2^pe*2+0.5)/2;
    flush-to-zero at s < -127). Input dim is zero-padded to a multiple of
    block_size and the padding is dropped on return.
    """
    x = w.detach().to(torch.float32)
    n, d = x.shape
    pad = (-d) % block_size
    if pad:
        x = F.pad(x, (0, pad))
    xb = x.view(n, -1, block_size)

    amax = xb.abs().amax(dim=-1, keepdim=True)
    shared_exp = torch.ceil(torch.log2((amax / ceil_x_value).clamp(min=0) + _EPS))
    shared_exp = shared_exp.clip(-_SCALE_EMAX - _EMAX, _SCALE_EMAX - _EMAX)

    x_biased = xb.abs() / torch.exp2(shared_exp)
    # min_exp = -(2**(ebits-1)) + 2 = 0 for MXFP4
    private_exp = torch.clip(
        torch.floor(torch.log2(x_biased + _EPS)), 0, _EMAX
    )
    man_shift = 3 - 2  # mbits - 2
    mant = torch.floor(x_biased / torch.exp2(private_exp) * 2 ** man_shift + 0.5)
    v = mant / 2 ** man_shift * torch.exp2(private_exp)
    v = v.clip(max=2 ** _EMAX * (2 ** (man_shift + 1) - 1) / 2 ** man_shift)
    # flush subnormal blocks
    v = torch.where(shared_exp < -127, torch.zeros_like(v), v)
    out = xb.sign() * v * torch.exp2(shared_exp)

    return out.view(n, -1)[:, :d].to(w.dtype)
