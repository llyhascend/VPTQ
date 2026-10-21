# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Configuration for the VPTQ quantizer."""

from dataclasses import dataclass


@dataclass
class QuantConfig:
    """Vector-quantization hyperparameters for one linear weight.

    Mirrors the VQuantLinear constructor semantics:
      - weight columns (input channels) are quantized one at a time,
        each column split into vectors of `vector_len`;
      - columns are cut into `col_tile`-wide blocks along in_features
        (number of groups = in_features // col_tile; 0 = whole-row single
        group), each block owning a codebook of `num_centroids` centroids;
      - optional residual VQ stage with `num_res_centroids` centroids
        (-1 disables);
      - optional outlier separation / act-order permutation / per-channel
        normalization (all implemented in vq.quantize_weight).
    """

    vector_len: int = 8
    num_centroids: int = 65536
    num_res_centroids: int = -1  # -1: no residual VQ
    col_tile: int = 0  # column-block width; groups = in_f // col_tile
    outlier_size: int = 0  # input channels pulled out by top diag(H)
    outlier_vector_len: int = 4
    num_outlier_centroids: int = 256
    enable_perm: bool = False  # act-order channel permutation
    enable_norm: bool = False  # per-channel scale/bias
    kmeans_iters: int = 10
    kmeans_sample: int = 262144  # max vectors per Lloyd iteration
    seed: int = 0
    block_cols: int = 128  # GPTQ lazy-update block width (reserved)
    damp: float = 0.01  # used only when H must be re-inverted
    # --- extended scheme (tile grouping / decoupled residual vector) ---
    res_vector_len: int = 0  # 0: same as vector_len; >0: residual v
    row_tile: int = 0  # 0: full-column groups; >0: 2D tiles of
    # col_tile cols x row_tile rows, one codebook per tile
    centroid_fp8: bool = False  # store centroids as fp8 (8bit/elem)

    @property
    def index_bits(self) -> int:
        import math

        return int(math.log2(self.num_centroids))

    @property
    def res_index_bits(self) -> int:
        import math

        return (
            int(math.log2(self.num_res_centroids))
            if self.enable_residual
            else 0
        )

    @property
    def enable_residual(self) -> bool:
        return self.num_res_centroids is not None and self.num_res_centroids > 0

    @property
    def bits_per_weight(self) -> float:
        main = self.index_bits / self.vector_len
        if self.enable_residual:
            rv = self.res_vector_len or self.vector_len
            return main + self.res_index_bits / rv
        return main

    # presets close to the official VPTQ checkpoints
    @staticmethod
    def v8_k65536_65536() -> "QuantConfig":
        """~2.06 bit: v8, 16bit main + 16bit residual."""
        return QuantConfig(
            vector_len=8, num_centroids=65536, num_res_centroids=65536
        )

    @staticmethod
    def v8_k65536_256() -> "QuantConfig":
        """~3 bit: v8, 16bit main + 8bit residual."""
        return QuantConfig(
            vector_len=8, num_centroids=65536, num_res_centroids=256
        )

    @staticmethod
    def v2_k16_res8_k16() -> "QuantConfig":
        """2.03125 bit/elem (2.65625 with residual): v=2, K=16 (4bit),
        residual vectors span 4 main vectors (v_res=8), K_res=16,
        meant for 32-col x 256-row tiles with fp8 centroids.
        Set col_tile = 32 and row_tile=256 at call site."""
        return QuantConfig(
            vector_len=2,
            num_centroids=16,
            num_res_centroids=16,
            res_vector_len=8,
            row_tile=256,
            centroid_fp8=True,
        )

    @staticmethod
    def v16_k65536_65536() -> "QuantConfig":
        """2 bit: v16, 16bit main + 16bit residual."""
        return QuantConfig(
            vector_len=16, num_centroids=65536, num_res_centroids=65536
        )
