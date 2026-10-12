# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Access layer for collected Hessian checkpoints (layer_XXXX.pt)."""

import logging
import os
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger("vptq.quantize.hessian")

# Hessian group keys whose activations live in the (Hadamard-rotated)
# residual basis and therefore need H' = R^T H R when the checkpoint was
# Hadamard-absorbed (vptq/tools/deepseek_v4/rotate_ckpt.py). These are the
# groups feeding input-side-rotated weights: mlp_in (w1/w3/router) and the
# attention linears that read the normalized residual stream.
_ROTATED_ATTN_KEYS = frozenset({
    "attn.wq_a", "attn.wkv", "attn.compressor.wkv",
    "attn.compressor.wgate", "attn.indexer.weights_proj",
})


def is_rotated_group(group_key: str, wrap: bool = False) -> bool:
    """True if the group needs H' = R^T H R.

    Absorbed mode (rotate_ckpt): only input-side-rotated weights need it
    (w2 was rotated output-side, H unchanged). Wrap mode
    (W -> W @ R quantize -> Q(WR) @ R, all leaves input-side): every
    quantized expert leaf sees rotated inputs, so the per-expert w2
    groups rotate too.
    """
    if group_key.endswith("mlp_in"):
        return True
    if wrap and group_key.startswith("ffn.experts.") \
            and group_key.endswith(".w2"):
        return True
    return group_key in _ROTATED_ATTN_KEYS


# DeepSeek-V4 weight name -> Hessian group key.
# Callable: ffn.experts.{i}.w1 -> ffn.experts.{i}.mlp_in etc.
def hessian_key_of(weight_name: str) -> Optional[str]:
    """Map a model weight name to its Hessian group key in the
    collected layer_XXXX.pt files. Returns None if the weight has no
    Hessian (not meant to be quantized)."""
    n = weight_name
    if n.startswith("mtp."):
        return None
    # strip the trailing ".weight" so rpartition sees the op name
    base = n[: -len(".weight")] if n.endswith(".weight") else n
    head, _, leaf = base.rpartition(".")
    nn = "." + n  # prefix-safe containment ("attn.x" matches ".attn.")
    if ".ffn.experts." in nn:
        # ffn.experts.7.w1/w3 -> ffn.experts.7.mlp_in ; .w2 -> .w2
        if leaf in ("w1", "w3"):
            return f"{head}.mlp_in"
        if leaf == "w2":
            return base
        return None
    if ".ffn.shared_experts." in nn:
        if leaf in ("w1", "w3"):
            return f"{head}.mlp_in"
        if leaf == "w2":
            return base
        return None
    if n.endswith("ffn.gate.weight"):
        # router input == shared expert input
        return n[: -len("ffn.gate.weight")] + "ffn.shared_experts.mlp_in"
    if ".attn." in nn:
        if leaf == "wo_a":
            # grouped: handled separately via attn.wo_a_input.g{g}
            return None
        if leaf in (
            "wq_a",
            "wq_b",
            "wkv",
            "wo_b",
            "wgate",
            "weights_proj",
        ):
            return base
    return None


class HessianStore:
    """Lazy loader for collected Hessian layer files.

    Caches one layer file at a time (files are ~21GB each).
    If `rotation_path` (a rotation.safetensors written by rotate_ckpt) is
    given, input-side groups are returned rotated (H' = R^T H R,
    inv' = R^T inv R) to match the Hadamard-absorbed checkpoint.
    `wrap=True` selects wrap-mode semantics (quantize W @ R from an
    UNROTATED checkpoint, wrap the result back with Q(WR) @ R): the
    per-expert w2 groups are rotated as well (w2 is input-side in this
    scheme, unlike the absorbed pipeline).
    """

    def __init__(self, hessian_dir: str, rotation_path: Optional[str] = None,
                 wrap: bool = False):
        self.dir = hessian_dir
        self.wrap = wrap
        self._cache_idx: Optional[int] = None
        self._cache: Optional[Dict[str, dict]] = None
        self.R: Optional[torch.Tensor] = None
        if rotation_path:
            if self.wrap:
                # 2026-10-12 起 wrap 模式 R 为分块随机 Hadamard（块 32，
                # hadamard.wrap_rotation 派生），与 W 侧 run.py maybe_wrap /
                # rebuild 折回共用同一派生源；rotation 文件仅作为启用旋转的
                # 开关（其 global_rotation 是旧全局版，不再断言一致）。
                from vptq.tools.quantize.hadamard import wrap_rotation

                self.R = wrap_rotation(4096)
                logger.info(
                    "wrap mode: derived block-Hadamard R (block=%d) for "
                    "4096-d groups; rotation file %s used as switch only",
                    32, rotation_path,
                )
            else:
                from safetensors.torch import load_file

                self.R = load_file(rotation_path)["global_rotation"].float()
                logger.info("Hadamard rotation loaded from %s", rotation_path)

    def _load(self, layer_idx: int) -> Dict[str, dict]:
        if self._cache_idx != layer_idx:
            path = os.path.join(self.dir, f"layer_{layer_idx:04d}.pt")
            logger.info("loading %s", path)
            self._cache = torch.load(path, weights_only=False)
            self._cache_idx = layer_idx
        return self._cache

    def get(
        self, layer_idx: int, group_key: str
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (hessian, inv_hessian or None) for a group."""
        entry = self._load(layer_idx).get(group_key)
        if entry is None:
            # 小语料（如 calibR9 49 条）下 0 命中的冷门专家无 Hessian 组：
            # 回退单位阵——加权退化为均匀、误差传播退化为近恒等，
            # 该专家等价朴素 VQ（无校准信息下的最保守行为）
            n = 2048 if group_key.endswith(".w2") else 4096
            logger.warning(
                "group %s missing at layer %d (zero-hit expert); "
                "fall back to identity Hessian (%dx%d)",
                group_key, layer_idx, n, n,
            )
            return torch.eye(n), None
        H, inv = entry["hessian"], entry.get("inv_hessian")
        if self.R is not None and is_rotated_group(group_key, wrap=self.wrap):
            if self.wrap:
                # wrap 模式：按输入维统一派生分块 R（4096 与 w2 的 2048），
                # 与 W 侧同一来源，杜绝两侧错配。
                from vptq.tools.quantize.hadamard import wrap_rotation

                R = wrap_rotation(H.shape[-1])
            else:
                R = self.R
            H = R.t() @ H.float() @ R
            if inv is not None:
                inv = R.t() @ inv.float() @ R
        return H, inv

    def cholesky_upper(
        self, layer_idx: int, group_key: str, damp: float = 0.01
    ) -> torch.Tensor:
        """Upper Cholesky factor of the damped inverse Hessian."""
        hessian, inv = self.get(layer_idx, group_key)
        if inv is not None:
            L = torch.linalg.cholesky(inv.double())
            return L.t().float()
        from vptq.tools.quantize.vq import cholesky_upper_of_inv

        return cholesky_upper_of_inv(hessian, damp)

    def group_keys(self, layer_idx: int):
        return sorted(self._load(layer_idx).keys())
