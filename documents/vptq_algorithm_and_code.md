# VPTQ 算法流程与核心代码详解

> 本文档对应代码库 `/mnt/share/rr08002/work/VPTQ` 的当前状态（2026-09），
> 覆盖：算法原理 → Hessian 采集 → 向量量化器 → 打包导出 → CUDA/NPU 推理，
> 以及 DeepSeek-V4-Flash-BF16（272B MoE）的落地适配。
> 文中所有 `path:line` 引用均可在库内点击定位。

---

## 0. 全景

VPTQ（Vector Post-Training Quantization）用**向量量化（VQ）**替代逐元素标量量化：权重被切成长度为 `v` 的小向量，每个向量只存一个指向共享码本（codebook/centroids）的索引。配合 Hessian 二阶优化与残差 VQ，在 2bit 等效位宽下仍能保持模型精度。

```
校准数据                量化器（本库 vptq/tools）              推理（vptq.layers / csrc）
─────────────    ────────────────────────────────    ───────────────────────────
RedPajama  ──► ① Hessian 采集  H=E[xxᵀ], H⁻¹         ┌─ VQuantLinear（压缩格式定义）
128×4096        ② 加权K-means 码本初始化              ├─ pack/unpack（索引位打包）
                ③ 逐列VQ + H⁻¹误差传播  ────────────► ├─ dequant / quant_gemm
                ③b 残差VQ（RVQ 二级补偿）             │    （torch 回退，NPU 已验证）
                ④ (可选) proxy error 码本微调        └─ csrc/ quant_gemv CUDA kernel
```

---

## 1. 数学基础

### 1.1 PTQ 的二阶优化目标

量化前后模型 loss 的变化按 Taylor 展开（假设已收敛，一阶项≈0）：

```
arg min_ΔW  ΔWᵀ · H(W) · ΔW          ΔW = Ŵ − W
```

对线性层 `y = Wx`，Hessian 即输入激活的二阶矩（Gauss-Newton 近似）：

```
H = E[x xᵀ] ≈ (1/N) Σᵢ xᵢ xᵢᵀ      (in_features × in_features)
```

**H 只依赖校准数据的前向激活，无需标签、无需反向传播**——可离线采集。

### 1.2 逐列量化与误差传播（Channel-Independent Second-Order Opt.）

用拉格朗日法解上述目标（技术报告 §3.1），得到两条结论：

1. **选质心无需考虑 H**：逐列量化时 `H⁻¹_qq` 为常数，`min Σ‖v−C‖²/H⁻¹_qq` 退化为普通欧氏距离最近质心；
2. **量化完第 q 列后，误差按 H⁻¹ 结构补偿给未量化列**（GPTQ/OBQ 更新）：

```
err = (w_q − ŵ_q) / U_qq
W[:, q+1:] −= err ⊗ U[q, q+1:]        U = chol(H⁻¹) 的上三角因子
```

VPTQ 与 GPTVQ 的关键差异：VPTQ **逐列**传播（每量化一列立即补偿），GPTVQ 要等 v 列成组量化完才传播，长向量下误差累积更大——这是 VPTQ 能用 v=8/16 长向量的原因。

### 1.3 Hessian 加权的码本初始化

利用迹的循环性质展开代理误差：

```
tr(ΔWᵀΔW ⊙ H) = Σ_j h_jj‖ΔW:,j‖² + 交叉项（H 近似对角占优，忽略）
```

对角主导项 ⇒ **加权 K-means**：列 j 的向量以 `h_jj` 为权。激活能量大的通道在聚类中话语权更大（`vptq/tools/quantize/vq.py: weighted_kmeans`）。

### 1.4 残差向量量化（RVQ）

主量化后残差 `R = W − Ŵ_main` 仍含结构，用第二个码本再做一遍同样的 VPTQ：`v ≈ C_main[i] + C_res[j]`。等效位宽：

```
bits/weight = (log2 K_main + log2 K_res) / v
例：v16, K=65536+65536 → (16+16)/16 = 2 bit
```

### 1.5 Outlier 与置换/归一化

- `diag(H)` 特别大的输入通道即 outlier，切出用独立小码本高精度表示；
- 按 `diag(H)` 排序重排通道（act-order perm），使同组/同向量通道能量相近；
- 每通道仿射 `W' = (W−b)/s` 吸收幅度差异。（当前核心量化器默认未启用这三项，见 §3 说明。）

---

## 2. 阶段①：Hessian 采集（`vptq/tools/hessian/`）

仿 QuIP# `hessian_offline_llama.py` 的离线采集（官方 VPTQ 同样使用预计算 Hessian checkpoint，采集于 RedPajama-Data-1T-Sample）。

### 2.1 算法

```
校准语料前向（FP 模型）
  └─ 每个 Linear 注册 forward hook，捕获输入 x
       └─ 流式累积: sum_xxT += xᵀx;  sum_x += Σx;  n += rows
            └─ 保存: H = sum_xxT / n;  mean;  n_tokens
                 └─ H⁻¹: dead-diag→1, H += 0.01·mean(diag)·I,
                       fp64 Cholesky 求逆（GPTQ 阻尼）
```

核心代码：

| 功能 | 位置 |
|---|---|
| 流式累加器 `HessianAccumulator`（addmm_ 在线累积） | `vptq/tools/hessian/collector.py: HessianAccumulator` |
| 共享输入去重分组（q/k/v→attn_in、w1/w3→mlp_in，**按父模块作用域**） | `collector.py: build_hook_plan` |
| 阻尼 Cholesky 求逆（fp64→fp32，对称化） | `collector.py: damped_inverse` |
| 逐层挂钩/保存驱动 | `collector.py: HessianCollector` |
| 通用 CLI（HF 模型） | `vptq/tools/hessian/__main__.py` |
| 离线补算 H⁻¹（4 worker 分片） | `vptq/tools/hessian/compute_inverse.py` |

### 2.2 MoE 与 DeepSeek-V4 专项

- **逐专家 Hessian**：父模块作用域分组保证 `ffn.experts.N.w1/w3` 各专家独立成组（不同专家看到的 token 子集不同，绝不能混合）；
- `wo_a` 经 einsum 消费权重、无模块边界 → 插入 `nn.Identity` 钩子点 `attn.wo_a_input`，按 `_vptq_group_dim` 拆成 **8 组独立 Hessian**（`attn.wo_a_input.g0~g7`）；
- router（`ffn.gate`）为函数式 linear 无钩子，其输入 ≡ 共享专家输入，复用 `ffn.shared_experts.mlp_in`；
- 前 3 层 hash 路由（`tid2eid` 查表）与校准数据无关；第 4 层起 score 路由由校准数据驱动——**校准分布必须接近真实使用分布**（故按 RedPajama 官方 6-slice 配比采样：cc 73%/c4 15%/github 5%/arxiv/wiki/se）。score 路由的内容特化同时造成专家 token 供给偏斜（每层最差专家中位数仅 ~209 token），n_tokens ↔ MSE 交叉验证与补采论证见 §8A.7。

### 2.3 产物格式

```
layer_XXXX.pt = { group_key: {hessian (n,n) fp32,
                              inv_hessian (n,n) fp32,
                              mean (n,), n_tokens int} }
meta.json     = 模型/数据/分组方案/约定
```

DeepSeek-V4-Flash 实测：43 层 × 518~532 组 ≈ 22784 组，H+H⁻¹ 共 ~1.96TB；校验项：finite / 对称 / token 计数 /`(H+λI)⁻¹` 精确比对（rel err ~1e-9）。

---

## 3. 阶段②③：VPTQ 量化器（`vptq/tools/quantize/`）

官方未开源部分的本库实现。逐层、逐线性权重执行：

### 3.1 主流程（`vq.py: vptq_quantize`）

```
输入: W (out×in), H (in×in), [H⁻¹ 的上 Cholesky 因子 U]
 1. U ← cholesky(damped_inverse(H)).t()          # 若未预计算
 2. 按 in 维均分 group_num 个列组
 3. 每组: 加权K-means(diag(H) 加权) → 主码本 C_main
 4. _quantize_columns:                          # GPTQ 式分块逐列
      for 每个块 (block_cols 列):
        for 块内每列 q:
          向量切分 w_q → 最近质心分配 → 记录索引
          err = (w_q − ŵ_q)/U_qq
          块内后续列 −= err ⊗ U[q, :]
        块外后续列 −= Err_block @ U[块, 外]
 5. (可选) 残差 VQ:
      R = W − Q_main
      每组: 加权K-means(R) → 残差码本 C_res
      _quantize_columns(R, 同一 U) → 残差索引
      Ŵ = Q_main + Q_res
 6. 指标: proxy_error = tr(ΔWᵀΔW⊙H)/tr(WᵀW⊙H), mse
输出: QuantizedWeight(centroids fp16, indices uint16
      (g, cols, nvec), res_*, proxy_error, mse)
```

### 3.2 加权 K-means（`vq.py: weighted_kmeans`）

- **播种**：k-means++（按 h_jj×最近距离² 概率逐个点选）；**k>2048 自动切随机播种**——k-means++ 是 O(k) 轮全数据扫描，k=65536 时不可行（此为本库实测后加的保护）；
- **E 步**：`nearest_indices` 分块 GEMM 计算最近质心（16384 行/块，防 k=65536 时距离矩阵爆显存）；
- **M 步**：`index_add_` 加权均值（h_jj 为逐点权重），空簇保留旧质心。

### 3.3 残差 VQ 的语义要点

- 残差码本训练**沿用同一 h_jj 列权**（代理目标不变）；
- 残差列循环**复用同一 U**（误差补偿结构由输入几何决定）；
- 传播发生在 R 空间内部（函数内 clone），等价于技术报告 Algorithm 2 的 `W″ ← VPTQ(W − W′, C_res)`；
- outlier 分支无残差（`outlier_num_res_centroids = -1` 的设计限制）。

### 3.4 与官方实现对齐的说明

当前核心版实现的是官方算法的**主干**（加权初始化 + 逐列二阶优化 +RVQ）。官方另有的 outlier 切分 / act-order perm / 通道仿射三项在 `QuantConfig` 中留有开关（`outlier_size/enable_perm/enable_norm`），默认关闭——它们对 2bit 极限位宽帮助更大，可按需补全。

### 3.5 扩展方案：Tile 分组方案 / 残差向量长解耦 / 8bit 码本

在主干之上，`QuantConfig` 支持一套自定义细粒度方案（预设 `QuantConfig.v2_k16_res8_k16()`）：

| 扩展项 | 配置字段 | 说明 |
|---|---|---|
| 2D tile 分组 | `row_tile` | 码本按 (in/group_num 列 × row_tile 行) 的 2D tile 训练与寻址；H⁻¹ 逐列误差传播仍在全列空间进行，与 tile 划分正交 |
| 残差向量长解耦 | `res_vector_len` | 残差码本/分配按 v_res 进行（如 v_main=2、v_res=8 ⇒ 每 4 个主 vector 共用 1 个残差索引）；0 = 与主一致 |
| 8bit 码本存储 | `centroid_fp8` | 码本以 float8_e4m3fn 存储（摊销减半） |

位宽核算（v=2, K=16, v_res=8, K_res=16, tile 32×256, 测试断言锁定）：

```
主体:      4bit/2elem = 2.0      残差主体: 4bit/4vec = +0.5
主码本摊销: (16×2×8)/8192 = 0.03125
残码本摊销: (16×8×8)/8192 = 0.125
不带残差: 2.03125 bit/elem      带残差: 2.65625 bit/elem
```

注意：该组合超出标准 VQuantLinear 推理格式（要求主/残差同 v、整列分组、fp16 码本），属量化器侧研究配置，部署需配套 tile 寻址 dequantkernel。

**向量量化在干什么**：权重 `W`（如 256 行 × 64 列）每 2 个相邻元素划成一个"小向量"(v=2)，在码本（K=16 个质心）里找最像的代表，存 4bit 编号代替原来的 2 个 fp16（32bit）→ 2 bit/元素。还原时查表。

**为什么需要分组**：全矩阵数千万个向量共用 16 个质心，质心只能取"平均长相"，误差大。把矩阵切成小块、每块训练自己的质心——块越小质心越贴合，但每块都要存一份码本。tile 就是精度与开销的权衡点。

**Tile 的几何：两个维度同时切**（老方案只切列方向）：

```
in_features（列，输入通道）
              ← 32列 →← 32列 →← 32列 →
        ┌───────────┬───────────┬───────────┐
   256行 │  tile 0    │  tile 1   │  tile 2   │   每个 tile 有自己
        ├───────────┼───────────┼───────────┤   的码本（16个质心）
   256行 │  tile 3    │  tile 4   │  tile 5   │
        ├───────────┼───────────┼───────────┤
   256行 │  tile 6    │  tile 7   │  tile 8   │
        └───────────┴───────────┴───────────┘
              out_features（行，输出神经元）向下
```

一个 tile = 32 列 × 256 行 = 8192 元素 = 4096 个向量。列方向按输入通道切（激活统计由 `diag(H)` 刻画，能量相近的通道共用码本才合理）；行方向按输出神经元切，码本进一步贴合。

**量化一个 tile 的三步**：

1. **训练码本**：tile 内 4096 个向量做加权 K-means（权重 = 向量所在列的 h_jj，激活强的列误差更"贵"），得 16 个质心，fp8 存 32B；
2. **逐列分配 + 误差传播**：tile 的 32 列一列一列处理——该列 128 个向量各找最近质心记 4bit 编号，然后算本列误差并经 H⁻¹ 摊给右边所有未量化列。**误差传播在整列维度进行，行方向的 tile 切分只影响"用哪个码本查表"，不影响误差怎么传**；
3. **全矩阵重复**：每个 tile 独立训练码本；列走到哪里就用哪个 tile 的码本。

**残差阶段（同结构、1/4 粒度）**：主量化后 `R = W − Ŵ`，每个 tile 再训练 16 个残差质心（向量长 v_res=8），每 4 个主 vector（8 元素）找一次，记 4bit：

```
主量化：  [· ·][· ·][· ·][· ·]   每 [2元素] 一个 4bit 编号
残差：    [· ·  · ·  · ·  · ·]   每 [8元素] 一个 4bit 编号
```

残差是"零头"，幅度小、结构少，1/4 索引率就够——这是残差只加 0.5bit 的原因。

**每个 tile 的存储**：

| 内容 | 大小 |
|---|---|
| 主索引：32 列 × 128 个 × 4bit | 2048B |
| 主码本：16 质心 × 2 × fp8 | 32B |
| 残差索引：32 列 × 32 个 × 4bit | 512B |
| 残差码本：16 质心 × 8 × fp8 | 128B |

摊到 8192 元素：索引 2.5 bit + 码本 0.156 bit = 2.65625 bit/元素。

**还原（dequant）**：每 2 元素查主码本[编号]，每 8 元素再加残差码本[编号]，`Ŵ = 主重建 + 残差重建 `。

**一图总结**：

```
W ──切 32列×256行 tile──► 每 tile: 4096个向量 ──K-means(16)──► 主码本
                              │
              逐列: 找最近质心 → 4bit 索引（误差经 H⁻¹ 传给右列）
                              │
                    R = W − Ŵ ──K-means(16, v=8)──► 残差码本
                              │
                    每 4 vector: 找最近质心 → 4bit 残差索引
                              │
        存储 = 主索引(2bit/元素) + 残差索引(0.5bit) + 码本摊销(0.156bit)
             = 2.65625 bit/元素
```

一句话：tile 分组是"把矩阵切成小块、每块配一套更贴合的码本"，用少量码本存储换误差下降；残差阶段用 1/4 粒度把主量化损失的精度低价赎回。

### 3.6 驱动与映射（`run.py`, `hessian_io.py`）

- `hessian_key_of`：权重名 → Hessian 组（w1/w3→mlp_in；gate→shared_experts.mlp_in；wo_a→None 走逐组路径）；
- `run.py`：逐层量化 DeepSeek 全部线性（注意力/router/共享专家/256 路由专家/wo_a 逐 8 组），`--layer-range/--expert-range` 多进程分片、`--scope`（all/attn/router/shared/experts）支持混合位宽（如专家 2bit + 注意力 4bit 分开跑，结果合并、断点续跑）；
- 实测（k=65536, v8, NPU）：**5.7s/矩阵**，proxy_error ≈ 0.2%，残差再降约一半；proxy_error ≪ mse 证明二阶优化把误差集中到了 Hessian 认为不重要的方向。

---

## 4. 阶段④：层微调（可选）

`VQuantLinear.centroids` 是 `nn.Embedding`，索引冻结、码本可开梯度（`vqlinear.py: set_centroids_grad`）。微调目标为 Hessian 加权逐层误差：

```python
# vptq/layers/vqlinear.py: proxy_error_forward
diff = dequant(...) − W
proxy_error = diff.T @ diff * H        # (Ŵ−W)ᵀ(Ŵ−W) ⊙ H
```

官方模型名 `-woft`（without finetune）即跳过此步；极限位宽建议开启。

---

## 5. 阶段⑤：打包导出（`vptq/utils/pack.py`, `tools/quantize/export.py`）

### 5.1 VQuantLinear 存储布局

| 参数 | 形状 | 说明 |
|---|---|---|
| `centroids` | (num_codebooks, K·v) Embedding | key 0 为 outlier 槽（无 outlier 也需占位） |
| `indices` | (cb, out/v, group_size) uint16/int32 | 主索引（可与残差索引打包） |
| `res_centroids/indices` | 同上 | 残差码本/索引 |
| `outlier_centroids/indices` | (1, K_o·v_o) / (1, out/v_o, outlier_size) | outlier 块 |
| `perm` | (in_features,) | 列置换（推理时 argsort 逆置换） |
| `weight_scale/bias` | (in_features,) | 通道仿射 |

导出注意（`export.py: build_vqlinear`）：量化器内部索引布局为 `(g, cols, nvec)`，装配时转置为 `(g, nvec, cols)`。

### 5.2 索引位打包（`pack.py: pack_index / unpack_index_tensor`）

```
merged = (res_index << index_bits) | main_index     # 残差在高、主在低
位流切段塞入 int32（末位补零对齐）
```

2bit 配置（16+16=32bit）下每向量恰好一个 int32——GEMV kernel 因此可走 `WBITS==32` 的单次对齐加载快路径（`csrc/util/cuda_utils.cuh:116`）。打包自带 round-trip 断言。`absorb_perm` 可把 perm 离线吸收进索引，省掉推理时逆置换。

---

## 6. 阶段⑥：推理

### 6.1 解量化语义（`vptq/ops/quant_gemm.py: dequant`）

```
解包索引 → gather 主码本 → (+ gather 残差码本) → 拼 outlier 块
→ 逆 perm → × weight_scale + weight_bias → Ŵ
```

### 6.2 分发（`quant_gemm`）

| 场景 | 路径 |
|---|---|
| decode（token<3） | `vptq_ops.quant_gemv` 融合 CUDA kernel：位解包→查双码本→half2 相加→FMA→warp 归约，权重按压缩格式只读一次 |
| prefill/批≥3 | `vptq_ops.dequant` 物化 Ŵ → `F.linear` |
| 无 CUDA（NPU/CPU） | torch 回退 dequant + F.linear（**NPU 与 CPU 已验证逐 bit 一致**） |

GEMV kernel 按 `IDXBITS/ResidualBits/GROUPSIZE` 模板特化（`csrc/quant_gemv.cu: DispatchWqA16Kernel`），激活侧吸收 norm/perm：`x·W = (x∘s)·Ŵ_perm + x·b`（`csrc/kernels/quant_gemv.cuh:53-56`）。

### 6.3 NPU 适配（本库新增）

- `vptq/utils/device.py`：CUDA>NPU>CPU 设备抽象、`empty_cache` 兼容；
- 修复：int16 arange 不上 NPU（perm 改 CPU 创建）、`enbale_perm` 拼写、残差+非打包构造 `num_indices` 未定义、`torch.where` 标量设备问题；
- torch 回退路径全程 NPU 可用（无原生 kernel 的位算子自动 CPU 回退，正确性不变）。

---

## 7. DeepSeek-V4-Flash-BF16 落地（`vptq/tools/deepseek_v4/`）

272B 新架构（MLA + DSA indexer + Hyper-Connections + 256 专家 MoE +前 3 层 hash 路由），公开 transformers 不支持，官方推理代码依赖 tilelang(CUDA)。适配内容：

1. **`kernels_torch.py`**：4 个 tilelang kernel 的纯 torch 等价（sparse_attn 索引注意力+attention sink、hc_split_sinkhorn、act_quant/fp4_act_quant QAT 模拟、hadamard），全部与朴素参考实现对拍；
2. **`model.py`**：vendor 官方实现 + 设备安全补丁 + `wo_a_input` 钩子点；
3. **`loader.py`**：543GB 分片权重加载（dtype 转换）+ 三种驻留模式（cpu / stream 逐层流式 / map 多卡分布）；
4. **`collect_hessian.py`**：逐层挂钩采集、断点续跑（有效 layer 文件自动跳过）、官方 RedPajama 配比采样；
5. **集群韧性**（本环境"空闲加速器回收 + 随机杀进程"必需）：
   - 加载期 NPU keep-alive（每 shard 一个小算子，防止驱动会话被回收）；
   - 权重本地 NVMe 副本（`/root/dsv4-weights`），脱离 NFS 慢读/掉线；
   - 看门狗脚本自动重试（`run_collect_rpmix.sh`，本地副本就绪后自动切换 `--ckpt`），配合 resume 逻辑使重启代价 ≈ 3~5 分钟。

---

## 8. 关键实测数据

| 项 | 数据 |
|---|---|
| Hessian 采集（CPU，32×4096） | 43 层 ~5.1h |
| Hessian 采集（8×NPU map，128×4096） | ~7min/层 |
| H⁻¹ 计算（22784 组，4 worker CPU） | ~4.5h |
| 专家 token 供给（rpmix 逐专家 n_tokens） | topk=6、均匀期望 12288；score 层 min=3、每层最差中位 209（§8A.7） |
| 量化速度（k=65536, v8, NPU） | 5.7s/矩阵（2048×4096） |
| 量化质量 | proxy_error ≈0.2%（主）→ 减半（+RVQ）；mse ≈14% |
| tile 方案（v2-K16 + res8-K16, fp8 码本） | 2.03125 / 2.65625 bit/elem（核算见 §3.5），残差单调降低 proxy_error |
| 推理一致性 | NPU vs CPU dequant+linear：max diff = 0.0 |
| 测试覆盖 | 22 项单测：hessian collector / deepseek_v4 移植 / 量化器 / tile 方案，全绿 |

## 8A. 生产全链路实验报告（2026-09，DeepSeek-V4-Flash 272B）

完整链路：VPTQ 量化路由专家 → BF16 重建 → ModelSlim（attnW8A8 + moeW4A8）→ vLLM → aisbench（GPQA / aime2024）。

### 8A.1 性能优化历程（Tile 方案，单矩阵 2048×4096）

```
64.4s（初版）→ 7.1s（批量 kmeans + 去同步）→ 5.4s（addmm/addr_ 融合）
→ 3.5s（k-means++ 设备端 Gumbel-max 采样）→ 1.2s（NPU 图捕获逐列循环）
```

累计 **53.7x**；图模式与 eager 逐位一致（bit-exact 对拍验证）。全模型量化（33,024 矩阵，8 卡）实测 **2h15m**。

### 8A.2 方案对照（aisbench，attnW8A8+moeW4A8 同配方）

| 组别 | 专家位宽（VPTQ 侧） | GPQA 各轮（均值） | aime2024 各轮（均值） |
|---|---|---|---|
| attnw8a8ceil+moeW4A8（floor→ceil 修复，基线） | ~4.25 bit（无 VPTQ） | 75.25 / 71.72 / 75.76 / 73.23（**73.99**） | 73.33 / 70.00（**71.67**） |
| VPTQ 32×256 + 残差 | 2.66 bit | 65.66 / 66.16 / 65.15 / 67.68 / 67.68 / 63.13 / 61.62（**65.30**） | 66.67 / 63.33 / 70.00 / 70.00（**67.50**） |
| VPTQ 32×256 + 残差+混合消融（top20→MXFP4） | ~2.78 bit（92.2%×2.66 + 7.8%×4.25） | 66.67 / 68.18 / 63.13 / 67.68 / 60.61 / 65.15 / 65.66（**65.30**） | 60.00 / 66.67 / 76.67 / 70.00 / 80.00（**70.67**） |
| VPTQ 32×256 无残差 | 2.03 bit | 60.10 / 67.17 / 60.61（**62.63**） | 53.33 / 76.67 / 63.33（**64.44**） |
| VPTQ 32×16 无残差 | 2.50 bit | 64.14 / 68.18 / 64.65 / 65.15（**65.53**） | 70.00 / 80.00 / 63.33（**71.11**） |
| VPTQ 32×16 无残差 + Hadamard（rot2 修复版） | 2.50 bit（旋转零开销吸收） | 68.18 / 64.65 / 60.61 / 65.66 / 65.66 / 60.61 / 63.64 / 65.15 / 63.64 / 64.14 / 66.67 / 65.15 / 64.65 / 65.15（**64.54**，σ=2.04） | 63.33 / 73.33 / 63.33 / 76.67 / 66.67 / 50.00（**65.56**，σ=9.35） |
| VPTQ 32×16 无残差 + wrap Hadamard（只转专家，Q(WR)·Rᵀ） | 2.50 bit（R 折入权重，部署零改动） | 63.13 / 65.15 / 61.11 / 68.18 / 70.71 / 68.69 / 66.67 / 67.17 / 71.21 / 64.14 / 66.16 / 67.17 / 64.65 / 68.18 / 64.14（**66.43**，σ=2.78） | 73.33 / 73.33 / 80.00 / 66.67（**73.33**，σ=5.44） |
| VPTQ 32×16 无残差 + 补采 Hessian（floor=1024，未旋转） | 2.50 bit | 67.17 / 63.64 / 69.70 / 62.63 / 66.16 / 66.16 / 65.66（**65.87**） | 73.33 / 70.00 / 80.00（**74.44**，σ=5.09） |
| VPTQ 32×16 无残差 + dual-norm 双尺度归一化 | 2.50 bit | 63.64 / 66.16 / 68.69 / 63.64 / 64.65 / 67.68（**65.75**） | 73.33 / 70.00（**71.67**） |

注：位宽为 VPTQ 预处理侧的专家表示位宽（索引+码本摊销）；各组最终部署格式统一为 attnW8A8 + moeW4A8(mxfp4)，注意力均为 W8A8，差异只在专家路径。评测期间 vLLM 被环境进程杀手 SIGKILL 十余次，各组成绩系"接力"凑齐；aime（30 题）单题 3.33 分，轮间波动天然较大（如混合组 60.00~80.00）。

**单变量对照：Hessian 数据源 / tile 尺寸 / gate 分数回退 / FP8 域归一化（2026-09-23~29）**

| 组别 | Hessian 数据源 | 专家位宽 | GPQA 各轮（均值） | aime2024 各轮（均值） |
|---|---|---|---|---|
| attnw8a8ceil+moeW4A8（floor→ceil 修复，基线） | —（ModelSlim 直量化，无 VPTQ） | ~4.25 bit | 75.25 / 71.72 / 75.76 / 73.23（**73.99**） | 73.33 / 70.00（**71.67**） |
| VPTQ 32×16 无残差 | RedPajama rpmix（官方 6-slice 配比） | 2.50 bit | 64.14 / 68.18 / 64.65 / 65.15（**65.53**） | 70.00 / 80.00 / 63.33（**71.11**） |
| VPTQ 32×16 无残差 + calibR9 | **R9tau 评测对齐语料（203k token / 49 条）** | 2.50 bit | 72.73 / 70.20 / 70.71 / 73.23 / 71.72 / 73.74 / 73.23（**72.22**，σ=1.3） | 66.67 / 76.67 / 73.33（**72.22**，σ=4.2） |
| VPTQ 32×32 无残差 + calibR9 | R9tau（同上） | **2.25 bit** | 68.18 / 70.20 / 68.69 / 66.16（**68.31**，σ=1.6） | 60.00 / 66.67（**63.33**） |
| VPTQ 32×32 + calibR9 + 12.5% 专家回退 | R9tau（同上） | ~2.50 bit（2.25×87.5% + 4.25×12.5%） | 73.23 / 68.69 / 70.20 / 69.70 / 70.20 / 71.21（**70.54**，6 轮，σ≈1.5） | 70.00 / 56.67 / 56.67（**61.11**，σ=7.7） |
| VPTQ 32×32 + calibR9 + row-fp8 + 12.5% 专家回退（rf32fb） | R9tau（同上） | ~2.50 bit（2.25×87.5% + 4.25×12.5%） | 75.25 / 69.70 / 70.71（**71.89**，σ≈2.4） | 73.33 / 60.00 / 60.00（**64.44**，σ=6.1） |
| VPTQ 32×16 + calibR9 + dual-norm + row-fp8 | R9tau（同上） | 2.50 bit | 74.24 / 71.21 / 69.70（**71.72**，σ≈2.3） | 73.33 / 66.67 / 73.33（**71.11**，σ≈3.9） |

calibR9 采集为严格基线口径（无 token-floor、无 save-inv、128×4096，仅数据源换为 `calib_corpus_R9tau.jsonl`）；6 个 0 命中专家无 Hessian 组，量化时单位阵回退。

**读表三轴结论**：

1. **Hessian 数据源轴**（前三行，同 tile32×16）：仅换数据源 → GPQA **+6.7**（65.53→72.22），与基线差距从 8.5 分缩至 1.77 分，aime 反超基线——**Hessian 分布与评测分布的失配是 GPQA 损伤主因**（详见结论⑨）；
2. **tile 尺寸轴**（第 3 vs 4 行）：行高 16→32（码本摊销减半，2.50→2.25bit）GPQA **−3.9**（72.22→68.31）——码本密度是 2bit 级 VPTQ 的第一精度杠杆；
3. **回退轴**（第 4 vs 5 行）：12.5% gate 分数回退（专家分数 = sum(激活时 gate 分数) 降序：688 整专家 + 2064 down 层 = 4128 矩阵；`gate_scores.py`/`fallback_list.py`/`rebuild_bf16.py --fallback-list`，回退专家走 msmodelslim 标准 W4A8）→ GPQA **+2.4**（68.31→70.71）——排序依据从权重域 MSE（② 的 top20 无效）换成路由域 gate 分数后回退首次兑现；但同位宽下仍不敌 tile 密度（70.71 < 72.22），且对 aime 无增益（gate 高分偏高频通用专家，数学冷门专家未覆盖；混合准则 gate×冷门度是改进方向）。
4. **FP8 域归一化 + 双尺度叠加轴**（第 3 vs 6 行，同 tile32×16）：量化前 out-dim 行峰对齐 e4m3 满量程 448（`run.py --row-fp8`，复合 row_scale=rms/fp8 折回，恒等性 1.5e-08）叠加 dual-norm → GPQA 71.72 / aime 71.11，与纯 tile16+calibR9（72.22/72.22）统计持平——**无损叠加但无净增益**：单轮 74.24 为 VPTQ 组历史首个超基线（73.99）轮次，证明 FP8 域对动态范围利用的改善真实存在，却被 dual-norm 的白化中性化抵消（⑤ 的教训再现）。**纯 row-fp8（不带 dual-norm）的隔离数据点已补**（rf32fb，第 5b 行，t32×32 基座 + 12.5% 回退同配置对照）：GPQA 71.89（单轮 75.25 与基线最高轮持平）/ aime 64.44——vs 同基座无 row-fp8（70.54/61.11）**GPQA +1.35、aime +3.33，row-fp8 单独贡献为正**；同位宽 GPQA 追平 32×16 系（72.22/71.72）但 aime 仍落后 7~8 分，tile 密度代价在 aime 上依旧成立。

报告：`ais_bench_logs/vptq-calibR9-tile16-attnw8a8-moew4a8-20260923/`、`vptq-calibR9-t32x32fb-20260925/`、`vptq-calibR9-t32x32fb-extra7-20260928/`（回退组 GPQA 并至 6 轮 70.54）、`vptq-calibR9-dnrowfp8-20260929/`、`vptq-calibR9-rf32fb-attnw8a8-moew4a8-20261009/`（rf32fb 修复折回 bug 后重跑，见 §8A.7b'）。

**结论**：① 残差贡献约 +3 分（65.30 vs 62.63）；② top20 定点修补无效（65.30 持平）——损伤是弥漫性的（k=16 码本整体粗糙），非尾部问题；③ **细块 32×16 是 VPTQ 各 tile 形态中最优**（GPQA 唯一超有残差组、aime 71.11 ≈ 基线 71.67）——局部码本密度 > 残差补偿，Tile 演进方向应继续缩小 tile 或增大 k（终局最优组 wrap 亦为 32×16 基座 + 旋转，见⑦）；④ 所有 VPTQ 预处理组 GPQA 均落后基线（73.99）约 8.5~11.4 分，k=16 级 VPTQ 预处理在该部署配方下整体为负收益，其正确形态是 Hadamard 旋转前置或端到端部署；⑤ 双尺度归一化被数据否决（通道模量 p99/中位 ≈1.1，分布本已均匀，且与 MSE 相关性 |r|≤0.17）；09-22 全量实测印证——上表 dual-norm 行 GPQA 65.75 / aime 71.67，与未归一化组 65.53/71.11 统计持平（重建 MSE −0.6pp 但 proxy_error +0.15pp，白化抹平 H⁻¹ 利用的通道结构），该线关闭；⑥ **Hadamard 旋转端到端未兑现权重域收益**：rot2 修复版 GPQA 64.54 vs 未旋转 65.53（−1.0，σ=2.04 内）、aime 65.56 vs 71.11（−5.6，但 aime σ=9.35 覆盖）——尽管量化侧 proxy 误差降 35-40%（§8A.4/8A.5 的 860 实例结论在全模型 33,024 矩阵复现）、重建态 MSE 同步下降，任务域成绩却持平甚至略降。教训有二：(a) 权重域 MSE 改善 ≠ 任务域收益，Hadamard 把误差"均摊"到所有通道的同时，也抹平了 H⁻¹ 误差传播原本利用的通道间误差结构（ proxy 目标的优化方向与下游任务损失并不对齐）；(b) **漏转 42 个张量的代价**——首版旋转（未修复 indexer.compressor 内层 wkv/wgate，详见 §8A.5 排障）同配方仅得 GPQA 63.13 / aime 55.56，修复后回升至 64.54/65.56：一个张量类别的旋转遗漏造成 ~1.4 GPQA + ~~~10 aime 的损失，V4-Flash 的 indexer 压缩注意力对输入一致性极度敏感，旋转类方案必须以"残差流全读取点"清单为准（本例共 82 个读取点张量）；⑦ **wrap-around Hadamard（只转专家）是终局答案**：Ŵ=Q(W·R)·Rᵀ（w1/w2/w3 全输入侧，w2 用独立 2048 维 R，Hessian 同步 RᵀHR），模型其余部分逐位不动——GPQA 66.43（15 轮，超未旋转组 +0.90、超 rot2 +1.89）、**aime 73.33（4 轮，超基线 71.67）**，VPTQ 各方案最优。归因链闭合：wrap（只动专家）≫ rot2（专家+注意力全转）⇒ **rot2 的损失来自注意力/残差侧旋转而非专家侧**；专家侧旋转正收益（权重域 rel-MSE −3~9%~~ 在任务域兑现为 GPQA +0.9 / aime +2.2），且部署零开销（R 折入权重内部，推理图与未旋转一致）。实现：`run.py --wrap-rotation` / `rebuild_bf16.py --wrap-rotation` / `hadamard.py: wrap_rotation()`；带符号 Hadamard R 非对称（R⁻¹=Rᵀ），包装必须右乘 Rᵀ。评测侧教训：vLLM 起服初期（health 200 但 worker 未就绪）请求会整轮 500，编排器需真实请求预热探测；⑧ **补采线（floor=1024）收益确认且与旋转线正交**：aime **74.44**（3 轮，超基线 +2.78、超旧 Hessian 组 +3.33、超 wrap 组 +1.11）为全场最高——冷门专家 Hessian 质量是数学推理误差的主杠杆（兑现 §8A.7 的 Spearman -0.914 预测）；GPQA 65.87 与旧组 65.53 持平（补采只改 sub-1024 专家的 H，GPQA 损伤不在该路径）。两条线作用域不同（补采管 Hessian 质量、wrap 管权重几何），**补采+wrap 组合**为下一步（在补采 Hessian 上 `--wrap-rotation` 重跑量化即可）；⑨ **calibR9 线（Hessian-语料对齐）是 GPQA 的决定性杠杆**（上节单变量对照表）：同一 tile32×16 方案仅换 Hessian 数据源（RedPajama→R9tau 评测对齐语料）GPQA 65.53→72.22（**+6.7**，σ=1.3 历史最稳），距基线仅 1.77 分，aime 72.22 反超基线 71.67——补采线（Hessian 质量）、旋转线（权重几何）、双尺度线对 GPQA 均无实质改善，**GPQA 损伤的主因是 Hessian 分布与评测分布失配**；且对齐收益的语料门槛极低（203k token / 49 条即兑现，6 个 0 命中专家单位阵回退亦无碍）。与补采/wrap 的正交叠加（R9tau + 补采 + wrap）为终局组合方向。

### 8A.3 逐专家 MSE 分析（33,024 矩阵对实测）

- 全局 rel-MSE：mean 14.0% / median 12.9% / p95 21.8% / max 25.8%
- layer 0 最难（17.8%），w2 系统性低于 w1/w3（12.5% vs 14.5%）
- **相邻层 MSE 相关性 ≈ -0.009**：量化难度是每层局部权重结构的独立事件，不存在"天然难量化的专家 ID"；定点修补应以（层， 专家）为粒度
- 产物：`dsv4-tile-v2k16-fp8/expert_mse_analysis.png`、`expert_mse_perlayer.png`、`top20_experts_per_layer.json`
  expert_mse_analysis.png
  
  ![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090912777162/50931252/97517250a29c4a6d8c5f7a7c23e73fc0.png)
  
  ![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090912777162/50931368/bcfa66be9dca4861a740e32d9bd7cb8b.png)

**图解读（expert_mse_perlayer.png，两联）**：
![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090912777162/50938706/99bf6ea6d5d34485b9d6b1d4395dfbe5.png)

- 上联（跨层均值最差 10 个 ID 的逐层曲线）：10 条曲线在 10%-23% 区间**全程剧烈震荡并频繁交叉**——没有任何一个 ID 能连续 3 层以上停留在最差位置；最高尖峰为 E102 在 layer-29、E77 在 layer-39 的 -23.5%，但同一 ID 在相邻层立刻回落到 12-14%。震荡幅度（±4~6%）远大于 ID 间均值差（最差 ID 16.0% vs 全局 14.0%），直观证明"层内运气 ≫ ID 属性"。
- 下联（高频上榜 10 个 ID 的 层×MSE 热力图）：深红斑块**弥散分布、无连续条带**；E72 在 layer 4/7-8/11/20/32/34/39-40 反复出现深斑（上榜 9 层最多），E238/E246 类似但位置互不重合；layer 0 整列普遍偏深（与"layer 0 最难 17.8%"相互印证），layer 39-42 区域整体略深于中部。结论：高频 ID 有**弱倾向性**（比随机上榜率约高 2 倍），但远不足以按 ID 全局换配置。

### 8A.3b 部署态 A-vs-B 交叉验证（mxfp4 反量化口径，2026-09-17）

§8A.3 是**重建态**（VPTQ 重建 BF16 vs 原始 BF16）的误差；本节把两端的**最终部署产物**直接对拍：
A = `DeepSeek-V4-Flash-attnw8a8ceil-moew4a8`（原始 BF16 → ModelSlim 直量化，参照系），
B = `DeepSeek-V4-Flash-vptq-tile16-attnw8a8-moew4a8`（VPTQ tile32×16 无残差 → BF16 重建 → ModelSlim）。
双侧 mxfp4 反量化后逐矩阵 rel-MSE（A 为参照），33,024 矩阵 → 11,008 路由专家（w1/w2/w3 按元素数加权）。

- **mxfp4 格式实测确认**：`weight` uint8 每字节 2 个 e2m1 码（**lo-nibble 在前**），`weight_scale` uint8 为 **e8m0**（2^(x−127)）、32 元素/块；A 反量化与原始 BF16 相对误差 ≈ 0（格式校验通过）。注意 modelslim 产物的 index 中 `weight` 与 `weight_scale` 可能分布在**不同物理分片**，必须按张量独立解析 shard（本次踩坑）。
- **总量**：全局均值 0.1718，p50 0.161 / p90 0.232 / p99 0.251 / max 0.259；逐叶 w1 0.185 / w2 0.146 / w3 0.184；逐层均值最差 L0 0.204 > L42 0.180 > L39 0.179；全局最差专家 L21E187 0.2594。
- **口径自洽（与量化报告交叉验证，层 13-42 共 7,447 专家）**：Spearman 秩相关 **1.000**（max rank diff 333/7447，Pearson 0.99997），比值恒 1.111±0.011——部署态比重建态高出的 ~11% 即 mxfp4 重量化附加噪声；每层 top20 重叠率（覆盖层）95%+。整条"VPTQ → 重建 → ModelSlim"链路无隐藏误差源。
- **归因**：A-vs-B MSE 与专家校准 token 数 Spearman **-0.914**；分桶均值 >16k → 0.138 / 8-16k → 0.156 / 4-8k → 0.181 / 1-4k → **0.233**（最差）/ <1k → 0.198（秩亏时 damping 主导、量化趋于保守，非单调）。tile16 无残差的部署态误差几乎完全由冷门专家的 Hessian 质量驱动；继续压误差的最优路径是冷门专家补采（预计全局 0.172 → ~0.15），与 §8A.7 结论互证。
- 产物：`ab_mxfp4_mse/`（REPORT.md、top20_experts_per_layer_ab.json、crosscheck_vs_quant_report.json、ntokens_corr.json、ab_mxfp4_mse_analysis.png、ab_mse_vs_ntokens.png、ab_mse_by_leaf.png）

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090912777162/51134443/f0480b7c1ee04d738f6bea0388d47fe6.png)

**图解读（ab_mxfp4_mse_analysis.png，四联）**：

- 左上（11,008 专家 MSE 排序曲线，log 轴）：曲线平滑、尾部无断崖——最差专家（0.259）与 p99（0.251）几乎持平，不存在极端离群专家；误差是"普涨"而非"点爆"，定点修补价值低。
- 右上（逐层均值/最大）：L0 均值 0.204 显著高于全层（印证 §8A.3 "layer 0 最难"），L1-L2 骤降后中段平稳（0.165-0.175），L39-42 尾部回升至 0.178-0.180——首尾两端偏难，与 hash/score 路由切换（L3 起 score 路由冷门专家涌现）及深层表示集中化相关。
- 左下（MSE 直方图）：明显**双峰**——0.14 峰为 token 充足专家、0.24 峰为 1-4k token 冷门专家（w2 整体左移 0.146，w1/w3 双峰来自专家间差异而非叶间差异），直观展示误差按 token 供给分层。
- 右下（每层 top20 与量化报告重叠率）：层 13-42 覆盖区间几乎全程 ~100%（个别 90%），证明"量化报告里最差的专家"与"部署后实际差最多的专家"是同一批——量化阶段的 top20 清单可直接作为部署态修补靶点；层 0-12 为 0 系报告未覆盖（该报告仅含层 13-42），非真实分歧。
  
  ![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090912777162/51134551/f60ceab64c854f5c8bbeb328eee962de.png)

**图解读（ab_mse_vs_ntokens.png，双联）**：

- 左联（11,008 专家散点 + 桶均值折线，x 轴 log token 数）：散点云整体右下倾斜，ρ=-0.914 肉眼可辨；灰色散点呈多条"斜带"——每层一条带，带内斜率同为负，说明 token→误差的机制在层内成立而非层间混淆。蓝色桶均值线从 1-4k 桶的 0.233 一路单调降至 >16k 桶的 0.138，唯 <1k 桶（n=317）反转至 0.198：H 秩亏到极点时 damping 项主导 H⁻¹，量化退化为保守策略（箭头注释处）。
- 右联（桶均值柱状图，柱内白字为桶内专家数）：0.233 → 0.181 → 0.156 → 0.138 的严格单调剂量响应；误差最大的 1-4k 桶有 1,874 个专家（占 17%）——这正是补采收益最大的目标人群，把它续采到 8k+ 预计可压全局均值 ~0.02。

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090912777162/51134598/c0183643f40d40df8c941de303b3a993.png)

**图解读（ab_mse_by_leaf.png，单联）**：w2（橙，mean 0.146）整体左移且近似单峰——down_proj 的 4096×2048 形状在 tile32×16 下码本摊销更优、输入侧经 SwiGLU 门控后分布更规整；w1/w3（蓝/青，mean 0.185/0.184）高度重合且呈**双峰**：0.14 主峰是 token 充足专家、0.23-0.25 次峰是 1-4k 冷门专家——证实直方图的双峰结构来自**专家间 token 供给差异**而非叶类型差异。蓝青两系列几乎完全重叠也印证 w1/w3 输入同分布（共享 mlp_in 分组）的对称性。

### 8A.4 随机 Hadamard 旋转前置实验（QuaRot 思路）

对每层 top20 专家的 w2（860 实例）做 W·R / RᵀHR 联合旋转后量化：

| 指标 | 基线 | Hadamard | 变化 |
|---|---|---|---|
| MSE 均值 | 0.2066 | 0.1734 | **-16.0%（100% 实例改善）** |
| proxy 均值 | 0.0411 | 0.0363 | -11.5% |
| 最差专家（L41/E84） | 0.379 | 0.146 | -61% |

**越难量化的专家收益越大**。与双尺度归一化互补：scale 管模长（此处无用），rotation 管方向结构（此处有效）。部署：w1/w3 的旋转可折入前级线性（零开销）；w2 因 SwiGLU 逐元素乘需在线 Hadamard kernel（<1% 开销）。旋转 Hessian 无需重采（RᵀHR 直接变换现有产物）。
机理可视化：最差专家 L41/E84 的向量分布在旋转前呈"针形"（单 bin 峰值 14k、峰度 23.88——质心全部冗余堆在针尖，重尾无人覆盖），旋转后变为规整二维高斯钟（峰度 4.05、双维 std 拉平）——MSE 37.8%→14.6%（-61%）。
数据与图：`dsv4-tile-v2k16-fp8/hadamard_top20_w2.json`、`hadamard_top20_w2.png`（三联曲线）、`hadamard_3d_dist_L41_E84.png`（3D 分布对比）

**图解读（hadamard_top20_w2.png，三联）**：

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090912777162/50941616/5b65777689c14e3f94f8d2e92236fb0a.png)

- 上联（配对排序曲线）：860 个专家按基线 MSE 降序，红线（基线）全程压过蓝线（旋转后），绿色改善带无一处断裂——100% 实例改善的直观证明；改善带在最差专家端（左侧）最宽（38%→15%，-60%+），尾部收窄至 ~1 个百分点——收益与难度单调相关。
- 中联（逐实例散点）：MSE（红）与 proxy（蓝）点云全部位于 y=x 虚线下方，无例外；云带右下倾斜，基线越差偏离对角线越远。
- 下联（降幅直方图）：近似正态右拖尾，均值 -15.5% / 中位 -13.5% / 最小 +5.3%（无一例为负）/ 最大 -61.5%；>25% 降幅的右尾约占 15%，正是最难量化的那批专家。

**图解读（hadamard_3d_dist_L41_E84.png，3D 分布对比）**：

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090912777162/50940865/1f282142d5654c569e6228e6ca583dcb.png)

- 左（旋转前）：向量分布塌缩成中心一根极尖的"针"（单 bin 计数 ~14,000，峰度 23.88）。对 k=16 VQ 的致命性在于：K-means 被针尖高密度吸引，多数质心冗余堆在原点附近，而真正决定精度的针外重尾向量离所有质心都极远——这是该专家成为全场最差（MSE 37.8%）的结构原因。
- 右（旋转后）：针的能量被均匀涂抹成规整二维高斯钟（单 bin 峰值 ~700，峰度 4.05≈高斯分布的 3，双维 std 拉平 0.0491/0.0491）。总方差不变（旋转保范数），但 16 个质心可以均匀覆盖钟面——kurtosis 23.88→4.05 就是 MSE 37.8%→14.6%（-61%）的全部机制。

### 8A.5 Hadamard 吸收：把旋转折进权重，而不是加进计算图

8A.4 的旋转实验是在量化器内部做的：W·R 量化完就结束。但生产部署时，如果按 QuaRot 原版做法把 R 当作运行时算子（在 SwiGLU 之后对每个 token 做一次在线 Hadamard），每次前向都要持续付费。本节走另一条路：**把 R 永久折进权重存储，推理图上不新增任何算子，模型输入输出完全不变**——而量化器看到的权重，已经是旋转打散后的平滑分布。

先固定符号，一个 decoder 层的 MoE 前向（pre-norm，注意力略写）：

```
x̃  = RMSNorm(s)                         s ∈ R^d 为残差流
r   = softmax(x̃ · W_gᵀ)                 router，r ∈ R^256
h_e = silu(x̃ · W1_eᵀ) ⊙ (x̃ · W3_eᵀ)     专家 e 的 SwiGLU 中间激活
o_e = r_e · (h_e · W2_eᵀ)                专家 e 输出（topk 内）
o_s = silu(x̃·W1sᵀ) ⊙ (x̃·W3sᵀ) · W2sᵀ     共享专家输出（与专家同构）
s'  = s + attn_out + Σ_{e∈topK} o_e + o_s
```

推导只用到 R 的两条性质：正交（RRᵀ = RᵀR = I）与保范数（‖vR‖ = ‖v‖，后者是前者的推论）。

**核心想法是整体换基，而不是在某处插入旋转。** 取固定正交矩阵 R，让全模型在 R-基下运行：残差流处处从 s 变成 s·R。直觉上，旋转不改变向量本身（模型函数不变），只改变每个坐标的读数——8A.4 里"针形"分布被摊成高斯钟，正是坐标读数重新分配的结果。只要每层满足"输入差 R、输出也恰好差 R"，L 层堆叠后残差流就是 s_L·R，输出 head 再做一次输入侧吸收把 R 收掉，logits 与原模型逐位一致；而所有存储权重都带上了 R，量化器吃到的就是平滑分布。逐层检查部件，它们整齐地分成三类。

**唯一的非线性障碍是 RMSNorm，用增益折叠扫清。** RMSNorm 带逐通道增益 g，定义为 (s/‖s‖)⊙g，而逐通道乘法 ⊙g 与 R 不对易：

```
RMSNorm(sR) = (sR/‖sR‖)⊙g ≠ ((s/‖s‖)⊙g)·R = RMSNorm(s)·R
```

处理：把 g 折叠进所有读 x̃ 的线性权重（W1/W3/W_g/共享专家一律 W ← W·diag(g)，此后 g 置 1），归一化退化为纯缩放 RMSNorm₀(s) = s/‖s‖。由保范数性：

```
RMSNorm₀(sR) = (sR)/‖sR‖ = (s/‖s‖)·R = x̃·R      ✓
```

换基后的归一化输出恰好是原输出乘 R——不改任何计算，只换权重存储。

**读残差流的线性（router、w1/w3）做输入侧吸收：W' = W·R，吃掉输入里的 R。** 预激活完全复原：

```
x̃R·(W·R)ᵀ = x̃·(R·Rᵀ)·Wᵀ = x̃·Wᵀ               ✓
```

由此得到两个关键事实。其一，router 的 logits、topk 选择、token 分发逐位不变——逐专家 Hessian 按路由后的 token 子集统计，旋转前后统计的是**同一批 token**，语义不变、可直接复用（怎么变换见下文量化器视角）。其二，SwiGLU 中间激活 h = silu(x̃W1ᵀ)⊙(x̃W3ᵀ) 逐位不变，而 h 正是 w2 的输入——**w2 的输入分布完全没动，因此不需要任何在线 Hadamard kernel**。这也是与 QuaRot 的分野所在：QuaRot 对 w2 也做输入侧旋转（W2 ← W2·R），就必须在 SwiGLU 后插一个在线算子把 h 变成 hR；本方案把 w2 的旋转挪到输出侧，恰好绕开。SwiGLU 自身的非线性则因两侧预激活复原而自动通过，无需单独处理。

**写残差流的线性（w2、注意力 W_o）做输出侧吸收：W' = Rᵀ·W，让输出带上 R。**

```
h·(RᵀW2)ᵀ = h·W2ᵀ·R = o_e·R                    ✓
```

专家输出恰好差一个 R，正好落回 R-基残差流。共享专家同构（W1s' = W1s·R，W2s' = RᵀW2s），注意力同样 W_o' = RᵀW_o。再把 embedding 表右乘 R（s₀' = s₀·R）、head 权重做输入侧吸收（W_head' = W_head·R，logits 完全复原），每层即有

```
s'·R = (s + attn_out + Σ_{e∈topK} o_e + o_s)·R   ✓
```

闭环成立：输入差 R、输出差 R，模型函数逐位不变。

**量化器视角只回答一个问题：每种权重配哪本 Hessian。** VPTQ 需要 H = E[zzᵀ]（z 为该线性的输入激活）：

- **w1/w3/router**：输入从 x̃ 变为 x̃R，故 H' = RᵀHR——对现有 Hessian 做一次相似变换即可，**免重采**；Cholesky 因子随之变换 chol(H'⁻¹) = Rᵀ·chol(H⁻¹)。
- **w2**：输入是 h，h 逐位不变，**原样复用**。w2 的旋转发生在权重行侧（RᵀW2 的行是 W2 行的正交混合），行内切分的 VQ 向量坐标被直接打散。

| 权重 | 存储形式 | 量化用 Hessian | 实测收益（8A.4/8A.6） |
|---|---|---|---|
| w1 / w3 / router | `W·R`（含 diag(g) 折叠） | **RᵀHR** | 输入侧，-16.0% |
| w2（down） | `Rᵀ·W2` | **原始 H** | 输出侧，-4.8% |
| shared 全部 | 同上 | 同上 | 同构 |

**R 的构造要处理 d=7168 非 2 幂。** 标准 Hadamard 只存在于 2^k 阶，做法是分块对角：取 stride 为 2 的幂（如 1024），把 7168 = 7×1024 摆成 7 个对角块 R = blockdiag(H₁₀₂₄ × 7)，再做随机 ±1 行/列符号翻转。分块对角阵仍是正交阵，符号翻转保证方向充分打散；8A.4 验证实验用的等价形式是 R = H·diag(±1)/√n（kurtosis 23.88→4.05 的那次）。

**最后声明与验证实验的差异。** 8A.4 的 860 实例对 w2 做的是**输入侧**旋转（W2·R + RᵀHR），证明的是"旋转能救 k=16 码本"的收益上界（MSE -16%）；生产吸收形态 w2 只能走**输出侧**（RᵀW2 + 原 H），实测 -4.8%。机理相同（正交混合打散坐标），量级差异的原因：tile 行向量在 out 方向本已相对平滑（vec_cv≈0.53），输出侧可收割的畸变有限；列间的"针"（kurtosis 23.88）只有输入侧能及。生产路径的取舍（哪侧必做、哪侧留作二期）见 8A.6 末尾的裁定。

### 8A.6 输出侧旋转验证（生产吸收形态，860 实例）

对每层 top20 专家的 w2 做输出侧旋转（Rᵀ·W2，Hessian 原样），与 8A.4 的输入侧实验完全同构：

| 指标 | 基线 | 输出侧旋转 | 输入侧旋转（8A.4） |
|---|---|---|---|
| MSE 均值 | 0.2066 | 0.1967（**-4.8%**） | 0.1734（**-16.0%**） |
| proxy 均值 | 0.0411 | 0.0392（-4.6%） | 0.0363（-11.5%） |
| 改善覆盖率 | — | 100% | 100% |
| 降幅分布 | — | 窄钟 3.0~7.8%（均值 4.7%） | 右拖尾（最差 -61%） |

**两条路径的裁定**：

| | 输入侧（W·R + RᵀHR） | 输出侧（Rᵀ·W + 原 H，生产形态） |
|---|---|---|
| MSE 改善 | **-16.0%** | -4.8% |
| 部署代价 | w2 需在线 Hadamard kernel | **零运行时开销** |
| Hessian | 需 RᵀHR 变换 | 原样复用 |

机理：输出侧混合的是"行"，而 tile 行向量在列内 out 方向本已相对平滑（vec_cv≈0.53 实测），可收割畸变有限；输入侧攻击的列间峰度（kurtosis 23.88 的"针"）才是主战场。**生产建议**：w1/w3 输入侧吸收（零开销、-16% 级收益）必做；w2 输出侧吸收（零开销、~5%）顺手做；w2 追加输入侧旋转（在线 kernel，再 ~11%）作为二期可选项。

产物：`dsv4-tile-v2k16-fp8/hadamard_outside_top20_w2.json` / `hadamard_outside_top20_w2.png`（与 8A.4 同构三联图）。

### 8A.6b wrap-around Hadamard：把旋转封进每个专家矩阵内部（2026-09-18，Hadamard 系列终章）

吸收式（§8A.5）把 R 折进全模型 82 个残差流读取点——收益被注意力侧旋转的损伤抵消（§8A.2 结论⑥）。wrap 式是第三形态：**每个专家矩阵自包含地完成"旋转→量化→转回"**，模型其余部分逐位不动：

```
量化阶段:  V = W·R  →  VPTQ 量化得 Q(V)        （w1/w2/w3 全输入侧）
重建阶段:  Ŵ = Q(W·R)·Rᵀ                       （换回未旋转 checkpoint）
部署:      Ŵ·x = Q(WR)·(Rᵀx) ≡ 吸收式输出      （数学逐位等价）
Hessian:   输入侧旋转的叶同步 H' = RᵀHR、H'⁻¹ = RᵀH⁻¹R（零重采，秒级矩阵乘）
```

与吸收式的关键差异：(a) **w2 也可输入侧旋转**——吸收式里 SwiGLU 逐元素乘阻挡 R 吸收被迫输出侧（§8A.6），wrap 的 R 在矩阵内部闭合，w2 用**独立 2048 维 Hadamard**（SwiGLU 中间维），三叶统一 `Q(W·R)·Rᵀ` 形式，避开了在线 kernel；(b) 每输入维独立 R（4096 维与 rot2 全局 R 逐位一致，seed-0 带符号 Hadamard）；(c) 不动注意力/router/残差流——82 读取点清单与 indexer.compressor 漏转 bug（§8A.5 排障，−1.4 GPQA/−10 aime）整类风险消失。

**实现**（4 文件）：`hadamard.py: wrap_rotation(dim)` 按输入维派生并缓存 R；`hessian_io.py: HessianStore(wrap=True)` wrap 下 w2 组 Hessian 同步旋转（吸收式不转）+ seed-0 一致性断言防 W/H 两侧错配；`run.py --wrap-rotation`（与 `--rotation` 互斥）量化前 `W@R`；`rebuild_bf16.py --wrap-rotation` parts 重建后 `@Rᵀ` 包装。驱动 `quantize_dsv4_experts_tile16_wrap.sh`。

**验证期抓出的 3 个真 bug**：① 带符号 Hadamard R=H·D **非对称**（R⁻¹=Rᵀ≠R，包装必须右乘 Rᵀ 而非 R——首轮冒烟即暴露，E·R 与 E·Rᵀ 差一个错误转置）；② w2 输入维 2048≠4096（集成测试 shape 报错暴露，w2 需独立维度的 R）；③ `is_rotated_group` 前导点漏判（`ffn.experts.` 匹配不到 `.ffn.experts.` 键，单测暴露）。单矩阵对拍（layer0/expert0，重建态 rel-MSE）：wrap 全胜 nores（w1 −3.2% / w2 −9.3% / w3 −3.1%），wrap-w2 也胜 rot 输出侧（−8.6%）——w2 收益最大印证 §8A.6 "输入侧攻击列间峰度才是主战场"的机理判断。

**全链路时序**（2026-09-18，8 卡无人值守）：量化 33,024 矩阵 4.3h（IO 瓶颈，见下）→ 重建+assemble（×Rᵀ 包装，33,024 swapped）→ ModelSlim 30min（EXIT=0）→ 中间 BF16 自动清理（1.1TB，39 分片校验通过后删）→ 评测 19 轮。

**端到端结果**（§8A.2 表格 wrap 行）：GPQA **66.43**（15 轮 σ=2.78，超未旋转 +0.90、超 rot2 +1.89）；aime **73.33**（4 轮 σ=5.44，**超基线 71.67**）。**归因链闭合**：wrap（只动专家）≫ rot2（专家+注意力全转）⇒ rot2 的损失来自注意力/残差侧旋转而非专家侧；专家侧旋转正收益且部署零开销。这是 VPTQ 各方案的最优成绩，2.5bit 专家在数学推理上追平 4.25bit 直量化基线。

**运维记录**：① NFS 高负载下 `ls dir/*/*.pt` 大 glob 间歇静默失败返回 0（假零），统计/等待循环必须用 `find`；② vLLM 起服初期 health 200 但 worker 未就绪，aime 整轮 Internal Server Error（9-13s 内 30 题全拒，连续 3 轮），编排器加真实 chat 请求预热探测（WARMED 标记）修复——GPQA 短输出不受影响；③ 量化耗时分解（运行日志实测）：存盘段（3×torch.save+os.replace/专家，33k 小文件写 NFS）占 56%、w1w3 传播 35%（已有 NPU graph）、Hessian 层加载 8.5%、kmeans 仅 4%；wrap 的 RᵀHR 变换开销≈0（页缓存+NPU 吸收）——下一轮优化：parts 本地 NVMe staging + 每专家 3 文件合并（33k→11k），预计 3h→1.5h。

产物：量化 `weights/quant/dsv4-tile32x16-nores-wrap/`（33,024 矩阵 + quant_report.json）、部署 `weights/DeepSeek-V4-Flash-vptq-wrap-tile16-attnw8a8-moew4a8/`、归档 `ais_bench_logs/vptq-wrap-tile16-attnw8a8-moew4a8-20260918/`（19 轮 outputs + REPORT.md + 全链日志）。

### 8A.7 校准 token 供给分析与补采论证（n_tokens ↔ MSE 交叉验证）

背景：EAQuant（`/mnt/share/rr08002/work/EAQuant`，Expert-Aware MoE PTQ）指出 MoE 校准的病根是路由偏斜——热门专家 token 爆多、冷门专家稀少甚至零命中，其对策是主动的 token 供给管理（均值序列 + 每专家 token 预算封顶 + 绕过 router 重放缓存 token + 孤儿专家强制喂数）。本库 Hessian 采集是**被动收集**（hook 累加路由子集、`n_tokens` 如实记录，仅靠 `damped_inverse` 兜底），本节检查实际产物的供给状况，并与 8A.3 的逐专家 MSE 交叉验证，论证补采是否有收益。

**供给分布**（rpmix Hessian 43 层逐专家 `n_tokens`；守恒验证 `sum(experts)/524288 ≡ 6.000` → topk=6，均匀期望 12288 token/专家）：

| 统计 | score 路由层 3–42（10,240 专家） | hash 路由层 0–2（768 专家） |
|---|---|---|
| min | **3**（L39/E122） | 4702 |
| p1 / p5 | 461 / 1403 | 6105 / 6741 |
| median | 8598 | 10076 |
| max | 121976（≈10× 均值） | 47215 |
| <1000 token | **3.1%（317 个）** | 0 |
| <2000 token | 8.3% | 0 |

两类路由的机制差异（`model.py: Gate`）解释了分布形状：hash 层（`layer_id < n_hash_layers`，DSV4-V4-Flash 前 3 层）走 `tid2eid[input_ids]` 固定查表，供给由 **Zipf token 频率**决定（跨领域稳定，下限稳、对校准数据选择不敏感）；score 层走 `scores.topk(6)`，供给反映**学习到的内容特化**——校准语料未覆盖的领域对应专家近乎零命中（min=3）。这也反证 8A.3 的"layer 0 最难（17.8%）"与供给无关（layer 0 是 hash 层、供给完全充足），来自权重结构本身。

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090912777162/51143360/2a9154537f0f40fea49d2c6464b48287.png)

**图解读（expert_ntokens_analysis.png，两联）**：

先说图上画的是什么。上联是分布直方图：**横轴 = 每个专家分到的校准 token 数**（对数刻度，越靠右越多），**纵轴 = 密度**（两组专家数量相差 13 倍，用密度而非计数才能直接比形状）。蓝色填充是 score 路由层（3–42，10,240 个专家），橙色阶梯线是 hash 路由层（0–2，768 个专家）；灰色点线是**均匀期望 12,288**（校准集每层 524,288 个 token、每 token 激活 6 个专家、摊给 256 个专家——若路由绝对公平，每个专家应得之数），红色虚线是 500 这条"严重不足"参考线。下联是逐层统计：横轴为层号（3–42），纵轴（对数）为该层专家的 token 数，**三条线分别是每层最热的专家（max）、中位专家（median）、最冷的专家（min）**，灰点线同样是均匀期望。

这张图回答一个前置问题：**实际的 token 分配离"人人 12,288"的理想有多远？**分三步读：

1. **上联比形状（蓝 vs 橙）**：橙色（hash 层）是一个收窄的钟形，落在 4.7k–47k 之间、峰值紧贴均匀期望线——查表式路由把 token 摊得相当公平，最冷也有 4,702。蓝色（score 层）则被拉成一个**从个位数一直铺到 12 万的宽分布**：最左端 L39/E122 只有 **3 个 token**（箭头标注），红虚线左侧聚集 1.08% 的专家，<2,000 的占 8.3%（左上文字框）。机理在 §2.2 已述：score 路由按内容选专家，校准语料没覆盖到的领域，对应专家就几乎无人问津；hash 路由只看 token id，与内容无关，天然摊平。
2. **下联看"是不是每层都这样"**：蓝色 min 线（每层最冷专家）全程贴着横轴底部——40 层里**每层的最冷专家中位数只有 209 个 token**（最好一层 869，最差就是那个 3）。对比平稳的 median 线（6.5k–9.6k）和起伏的 max 线（4k–122k），结论是：冷门专家不是某几层的坏运气，而是**每一层的常态**。
3. **两张图的分工**：本图只建立了"供给偏斜存在、且在所有 score 层系统性存在"；它是否真的伤害量化质量（还是被 Hessian 阻尼兜住了），由下一张图（n_tokens vs MSE）回答。补采策略的含义在第 2 步已经浮现——既然层层都有冷门专家，补采必须是覆盖全部 score 层的全局动作，挑几层修补没有意义。

**交叉验证**（`quant_report.json` 的 33,024 矩阵 MSE 聚合到专家级，与 `n_tokens` 对齐 11,008 对）：

| 口径 | n_tokens ↔ MSE |
|---|---|
| score 层聚合 | Spearman **−0.946**，Pearson(log n) −0.826 |
| 层内控制（40 层逐层） | **40/40 全负**，mean ρ=−0.953，最弱 −0.792 |
| hash 层（对照） | −0.407（供给本就均匀，弱相关符合预期） |
| proxy_error（对照） | **+0.43（与 MSE 方向相反）** |

| token 供给桶 | <0.5k | 0.5–1k | 1–2k | 2–4k | 4–8k | 8–16k | >16k |
|---|---|---|---|---|---|---|---|
| 专家数 | 111 | 206 | 535 | 1339 | 2601 | 2757 | 2691 |
| mean rel-MSE | 0.130 | 0.190 | **0.218** | 0.203 | 0.148 | 0.116 | 0.099 |

三个机理发现：

1. **非单调"倒钟"**（w1/w2/w3 三矩阵形状一致）：最差区在 0.5–2k 而非 <0.5k。假设：完全秩亏的专家（n≪维度）其 Hessian 被 damping 项主导，误差传播退化为近似无补偿——平庸但稳；而"半盲"区（几百到几千 token）的 Hessian 带大噪声参与误差传播，**主动误导比不作为更伤**。若走"专家自身质量"混淆通路（冷门=训练不充分）则 <0.5k 应最差，与数据相反。
2. **量化器自评失真**：proxy_error 与供给**正相关**（+0.43）而真实 MSE 强负相关——低 token 专家的 H 低估输入能量，proxy 被系统性低估。任何基于 proxy 的筛选/早停在该人群上都会被骗。
3. **40/40 层内全负**排除聚合假象与层间混杂（如 layer 0 效应）。

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090912777162/51143310/271397de2885456faca14251059e5b62.png)

**图解读（expert_ntokens_vs_mse.png，两联）**：

先说图上画的是什么。上联是散点图：**横轴 = 每个专家拿到的校准 token 数**（对数刻度，越靠右越多），**纵轴 = 该专家量化后的相对 MSE**（越低越好）；**每个点 = 一个（层, 专家）组合**——蓝点是 score 路由层（3–42）的 10,240 个专家，橙点是 hash 路由层（0–2）的 768 个专家，青色阶梯线（带数字）是把横轴切成 7 个桶后每桶的平均 MSE。下联是同一条阶梯线单独放大成的条形图：每桶一根条，条顶数字为桶均值、条内 `n=` 为桶中专家数，灰色点线是全部 score 层专家的总平均（0.138），作对照基准。

这张图回答一个问题：**给专家的校准 token 越多，量化误差是否越小？**分三步读：

1. **整体趋势（上联从左往右扫一遍）**：蓝点整体向下倾斜——token 少的专家（左）普遍误差高，token 多的（右）普遍低。青色均值线从左侧爬到 1–2k 桶的峰值 0.218，随后一路降到 >16k 桶的 0.099，落差超过一倍。这个趋势不是个别层造成的：40 个 score 层逐层单独算相关系数，**全部为负（平均 −0.95）**。橙点（hash 层）挤在右侧 4.7k–47k 的窄带里、高低与供给关系很弱（−0.41）——这批供给本就充足的专家不呈现强倾斜，反过来说明蓝点的倾斜确实来自"供给不足"，而非其他混杂因素。
2. **唯一的例外（最左那桶）**：<0.5k 桶的均值 0.130 反而低于 0.5–2k 桶的 0.19–0.22，阶梯线左端翘起一个"倒钩"。解释假设：token 极少（几十到几百）的专家其 Hessian 几乎是空矩阵，阻尼项在求逆中完全主导，误差补偿退化为"近似不做"——平庸但稳；而 0.5–2k 的专家拿到的是**带大噪声的** Hessian，量化器自以为在做误差补偿，传播方向却是错的——"半盲"比"全盲"更糟。该形状在 w1/w2/w3 三个矩阵上一致，不是偶然。
3. **落差有多大（下联对照灰线看）**：0.5–4k 的三根条全部高出全局均值 40–58%（覆盖 2,080 个专家，约 20%），>16k 桶则低于均值 28%。把这批冷门专家的供给抬到 4–8k 水平（即补采的 floor 目标），全局平均 MSE 即可从 0.138 压到 0.120–0.127，相当于 **−8~−13%**——这就是补采收益的图形化依据。

**结论与建议**：

- **补采明确有收益**：floor=4000 下全局 rel-MSE −8~13%（与 Hadamard 输入侧 −16% 同量级），0.5–2k 人群自身 −45% 级；且与 Hadamard **正交可叠加**（补采修 Hessian 统计质量、旋转修权重分布形态）。floor=8000 收益趋缓（4–8k→8–16k 仅再降 0.03），不建议。
- **成本**：token 缺口 floor 2000/4000/8000 = 0.7M/3.7M/17.7M expert-token，按均匀上界折 29/152/722 条序列，计入偏斜分配效率（~3–5×）后 floor=4000 约 500 条序列量级——采集时间 7→~14min/层。
- **实现要点**：`HessianCollector` 增加"专家下限检查 + 继续消费样本直到达标或样本耗尽"循环（与 EAQuant 的上限截断对称：已达标专家跳过累积，节省累积算力）；低层（hash 0–2）无需求，可跳过。原产物 43 层无需重采——补采是增量更新（`HessianAccumulator` 状态可从已存 `hessian/mean/n_tokens` 恢复续加）。

### 8A.7b dual-norm 双尺度归一化算法（2026-09-20/22，`--dual-norm`）

**动机**：若权重通道模量不均（部分通道能量远高于其他），VQ 码本被大能量通道主导，小能量通道量化粗糙。对策是先把行/列能量归一化、在"白化"空间做 VQ，重建时把尺度折回。§8A.2⑤ 的先验分析显示本模型权重分布本已均匀（通道模量 p99/中位 ≈1.1），实测该消融与未归一化组统计持平（结果见 §8A.2 表 dual-norm 行）——算法本身正确且已落地，记录于此备其他模型复用。

**数学**：变量代换 `x̃ = d ⊙ x`（D=diag(d)）下的共轭 Hessian

```
H̃ = E[x̃x̃ᵀ] = D H D
(H̃)⁻¹ = D⁻¹ H⁻¹ D⁻¹
```

误差补偿（Cholesky 上三角 U）在共轭空间进行：`HinvU = chol((H̃)⁻¹).t()`——等价于对归一化后的权重 `W_norm` 做标准 GPTQ 式逐列传播，几何自洽。

**尺度计算**（`run.py`）：

| 尺度 | 定义 | 代码 |
|---|---|---|
| 列尺度 d_j | `rms over rows(cat(W1,W3))[:, j]`（w1/w3 共享）；w2 独立按自身列 | `_row_norm` / 专家对分支 |
| 行尺度 s_i | `rms(W_i, :) = sqrt(mean_j W_ij²)`，clamp 1e-8 | `_row_norm` |

归一化与重建：

```
W_norm = W / (s ⊗ d)          # 量化在归一化空间
Ŵ ≈ (s ⊗ d) ⊙ Q(W_norm)       # 重建时折回（左乘 s、右乘 d）
```

**关键正确性约束——列尺度对同一输入 x 自洽**：w1/w3 消费同一 `mlp_in` 激活，若各自取列尺度则 x̃ 的代换在两个矩阵间不一致；必须拼接两矩阵联合计算共享 `d_joint`（`run.py: dual_norm 专家对分支`）。w2 的输入是 SwiGLU 中间激活（独立一组 Hessian），故独立取 d。共轭变换本身在 HessianStore 的原始 H 上进行（`dual_hessian`），无需重采。

**产物与折回链**：`QuantizedWeight.row_scale (out,) / col_scale (in,)`（`vq.py`，None=未归一化）→ `rebuild_bf16` 按 `row ⊗ col` 折回 BF16；H⁻¹ 缺失时 `dual_hinvU` 从共轭 H 现算（`cholesky_upper_of_inv`）。

**量化侧观测**（quant_report 均值）：重建域 MSE 略降（15.38%→14.80%），但 Hessian 加权 proxy_error 反升（3.45%→3.60%）——双尺度把能量均匀化的同时抹平了 H⁻¹ 误差传播原本利用的通道间结构（与 §8A.2⑥(a) 的 Hadamard 教训同构：权重域/白化空间的改善不必然传导到任务域）。

### 8A.7b' row-fp8：FP8 表示域量化（2026-10-09 记录；dnrowfp8 / rf32fb 线已用，`--row-fp8`）

**动机**：把 VQ 码本约束到 FP8 表示域——每行峰值对齐 e4m3 max=448 后再量化，配合 `--centroid-fp8`（centroids 落盘 cast e4m3fn，§3.5），产物天然落在 FP8 网格上，与部署侧 ModelSlim MX 格式（E8M0 块缩放）换算近自洽。本身不改变位宽（码本尺寸不变），是轴4（存储/表示域）的消融变量。

**实现**（`run.py`）：

```
_row_fp8(W):  peak = |W|.amax(dim=1).clamp_min(1e-12)   # (out,) 行峰
              scale = 448.0 / peak                        # per-row 放大（非 2 幂）
              return W·scale, scale
```

| 路径 | 流程 | row_scale 落盘 |
|---|---|---|
| w1/w3 joint 分支 | 先 `_row_fp8` 再 dual-norm（可叠加）；纯 row-fp8 时 Hessian **不变换**（与 dual-norm 的 D H D 不同） | `1.0/fp`（纯）；`rms/fp`（复合，FP8 放大须除回） |
| w2 独立分支 | `_row_fp8` 后量化 | `1.0/fp` |

rebuild 折回：`Ŵ = Q ⊙ row_scale`（`rebuild_bf16.py`，col_scale=None 容错）。

**踩坑史（2026-10-09 修复，rf16fb/rf32fb 双线中招）**：旧代码仅在 `dual_norm` 分支把 row_scale 写入 `QuantizedWeight`，纯 row-fp8 分支 w1/w3 的 fp 被丢弃（w2 分支正确）→ rebuild 后 w1/w3 放大 ~4755×（=448/peak），模型输出乱码；此前记录的"row-fp8 系慢生成+多超长题"（30 分钟 2 题）即此 bug 症状而非性能特性。修复：`elif row_fp8:` 分支落盘 `1.0/fp`。判别方法：抽查 `.pt` 中 w1/w3 的 `row_scale` 是否为 None；或重建后与原始权重逐行比 rms（比值≈448/peak 即中招）。

**数值特性与部署交互**（10-09 实测，rf32fb 产物）：

- 反量化值 = E4M3 网格 × `1/fp`：**FP8 尾数精度，但不在标准 E4M3 网格**（1/fp 非 2 幂，网格被"斜"）。
- E4M3 → MXFP8（E8M0 块缩放）：2 的幂缩放下网格自洽，实测 rel<1e-6 近零损失（attn/shared 列）。
- E4M3 → MXFP4（E2M1）：3 位尾数→1 位截断是**格式固有**舍入（~10% rel，实测 10.4%），2 幂 scale 技巧只救 MXFP8、救不了 MXFP4。
- 误差链：VPTQ 2bit 环节占总误差能量 97.3%，MXFP4 部署环节 4.8%——部署舍入非精度瓶颈。
- 基线零损失同源：原始 ckpt 权重值本身就在 E2M1 网格（"BF16"实为低精度放大存储），故 ModelSlim 直量化 W4 零舍入。

**数据点**（§8A.2 表）：dnrowfp8（32×16+dual-norm+row-fp8）GPQA 71.72 / aime 71.11；rf32fb（32×32+纯 row-fp8+12.5% 回退）GPQA 71.89 / aime 64.44——同位宽下 GPQA 追平 32×16 系，row-fp8 自身正贡献（vs 无 row-fp8 同基座 +1.35 / +3.33），无损叠加（FP8 域增益部分被 dual-norm 白化抵消，见 §8A.2 轴4）。

### 8A.7c 专家回退策略与实现（2026-09-24，gate 分数排序，`--fallback-list`）

**策略定义**（以矩阵计的总预算 43×256×3×12.5% = 4128，对半分配）：

| 类别 | 数量 | 公式 | 回退范围 |
|---|---|---|---|
| 整专家回退 | **688** 个专家 | 43×256×3×12.5%×0.5/**3** | w1+w2+w3 全部 |
| down 层回退 | **2064** 个矩阵 | 43×256×3×12.5%×0.5 | 仅 w2 |

**排序依据**：专家分数 = `sum(专家被激活时的 gate 分数)` —— score 路由层每个
(token, topk) 命中按 routing weight 累加，**降序**回退高分者（每次被选中时路由
权重大 → 量化损伤的输出代价高）。hash 路由层（L0-2 查表，无 gate 分数）不参与
排序。

**回退语义**：名单内专家**不走 VPTQ**，重建时保持原始 BF16，经 msmodelslim 走
标准 W4A8(mxfp4) 路径（与基线处理一致）——即"高分专家花更多 bit 买精度"。
位宽核算：2.25bit×87.5% + ~4.25bit×12.5% ≈ 2.50bit（t32×32 基座）。

**代码实现**（三组件流水线）：

```
vptq/tools/deepseek_v4/gate_scores.py      组件①：Gate forward hook 累加
                                           (weights, indices) → 每 (layer, expert)
                                           的 gate 分数和 → gate_scores.json
                                           （hash 层记 null；与 Hessian 同源样本：
                                           同 data_file/seed/round 逐位一致）
        ↓
vptq/tools/quantize/fallback_list.py       组件②：按分数降序取前 688 → full、
                                           接 2064 → down_only（跳过已 full 的
                                           专家防重叠）→ fallback_list.json
                                           （meta 含预算核对字段）
        ↓
rebuild_bf16.py --fallback-list <json>     组件③：pass_rebuild 循环内按
                                           (layer_idx, short) 二元组查名单跳过
                                           生成 expert_parts → pass_assemble 时
                                           copy 原始 BF16 换入
```

关键实现细节：① 名单 key 必须带 **layer 维度**（short 名如
`ffn.experts.6.w2.weight` 不含层前缀，同专家号跨 43 层同名）；② 组件① 的
`build_samples` 走 `--data-file` 分支时返回 list，需自行 `torch.tensor`（踩坑
记录）；③ 采集约 25 分钟（单卡 stream，43 层前向 + hook 累加）。

**实测**（§8A.2 单变量表第 4/5 行）：gate 分数回退 GPQA **+2.4**（68.31→70.54，
6 轮）—— 排序依据从权重域 MSE（② 的 top20 修补无效）换成路由域 gate 分数后
回退首次兑现增益；但同位宽下仍不敌 tile 密度（70.54 < 72.22），且对 aime 无
增益（gate 高分偏高频通用专家，数学冷门专家未覆盖；混合准则 gate×冷门度是
改进方向）。

### 8A.7e weights 目录用途与策略总表（活文档：新产物须同步更新本表）

磁盘：`/mnt/share/w00608002/weights`（≈19T）。命名约定：`-bf16` 后缀 = VPTQ 重建中间产物
（543GB 级，部署后可删）；`-attnw8a8-moew4a8` 后缀 = msmodelslim 部署权重（153GB，vllm serve 用）。

**A. Hessian 与量化中间产物（非权重）**

| 目录 | 用途 | 策略/口径 |
|---|---|---|
| `hessians/DeepSeek-V4-Flash-BF16-rpmix` | 基线 Hessian（8.8T） | RedPajama 6-slice、128×4096、含 topup1024/4k 增量 |
| `hessians/DeepSeek-V4-Flash-BF16-calibR9` | **calibR9 Hessian（现行主用）** | R9tau 评测对齐语料（49 条 203k token）、严格基线口径、含 inv 与 gate_scores |
| `quant/dsv4-*` | 各线量化产物 + quant_report.json | 命名含方案（tile 规格/回退/归一化）；t32x32 系含 fallback_list.json |
| `calib_data/` | R9tau 校准语料（jsonl） | calibR9 线数据源 |

**B. VPTQ 重建 BF16（中间产物，评测完成且无复用计划可删）**

| 目录 | 策略 | 对应 §8A.2 数据点 |
|---|---|---|
| `vptq-calibR9-tile16-bf16` | 32×16 + calibR9 | 72.22（最优基座） |
| `vptq-calibR9-t32x32-bf16` | 32×32 + calibR9 | 68.31 |
| `vptq-calibR9-t32x32fb-bf16` | 32×32 + 12.5% gate 回退 | 70.54 |
| `vptq-calibR9-dnrowfp8-bf16` | 32×16 + dual-norm + row-fp8 | 71.72 |
| `vptq-calibR9-rf16fb-bf16` | **32×16 + row-fp8 + gate 回退（09-29 进行中）** | 待评测 |

**C. 部署权重（vllm serve 入口，153GB each）**

| 目录 | 策略 | 状态 |
|---|---|---|
| `DeepSeek-V4-Flash-attnw8a8ceil-moew4a8` | 无 VPTQ 基线（73.99/71.67） | 保留（对照锚点） |
| `vptq-tile16-attnw8a8-moew4a8` | 32×16+rpmix（65.53） | 已评测 |
| `vptq-topup1024-...` | 补采 floor1024（65.87/74.44） | 已评测 |
| `vptq-wrap-tile16-...` | wrap Hadamard（66.43/73.33） | 已评测 |
| `vptq-dualnorm-tile16-...` | dual-norm（65.75） | 已评测（线关闭） |
| `vptq-rot/rot2-tile16-...`、`vptq-tile/nores/hyb-*` | §8A.2 历史消融 | 已评测 |
| `vptq-calibR9-tile16-...` | 32×16+calibR9（72.22） | 已评测（主结果） |
| `vptq-calibR9-t32x32[-fb]-...` | 32×32 系 | 已评测 |
| `vptq-calibR9-dnrowfp8-...` | dual-norm+row-fp8 | 已评测 |
| `vptq-calibR9-rf16fb-...` | **row-fp8+回退（S5 进行中）** | 待评测 |
| `w4a4-*`、`attnNative-*`、`BF16-rot2`、`vbench` | 早期/其他线产物 | 历史存档 |

**D. 隔离区**：`_OVERWRITTEN_*` 两个污染部署目录已于 2026-09-29 确认删除（原数据均在事故前完成评测，无损失，释放 ~306G）。

**维护规则**：每次新线产出（quant/BF16/部署）落地时，在本表追加行（策略 + 数据点 + 状态）；被删目录同步标注。

### 8A.8 资产索引

```
weights/quant/dsv4-tile-v2k16-fp8/          Tile+残差量化产物 + quant_report.json + MSE/Hadamard/n_tokens 分析图
weights/quant/dsv4-tile32x16-nores/         细块 32×16 无残差量化产物（未旋转组）
weights/quant/dsv4-tile32x16-nores-rot/     同上 + Hadamard 吸收式（rot2 组）
weights/quant/dsv4-tile32x16-nores-wrap/    同上 + wrap-around（终局最优组，§8A.6b）
weights/quant/ab_mxfp4_mse/                 部署态 A-vs-B 专家 MSE 分析（REPORT.md + top20 + 交叉验证，§8A.3b）
weights/DeepSeek-V4-Flash-vptq-tile-bf16/   VPTQ 重建 BF16（543GB）
weights/DeepSeek-V4-Flash-vptq-tile-attnw8a8-moew4a8/  部署权重（153GB）
weights/DeepSeek-V4-Flash-vptq-wrap-tile16-attnw8a8-moew4a8/  wrap 组部署权重（§8A.6b）
work/vllm-ascend/ais_bench_logs/            七组评测归档（vptq-tile / hyb / nores / tile16 / rot2 / wrap）
quantize_dsv4_experts.sh                    一键量化（8 卡分片 + 看门狗）
quantize_dsv4_experts_tile16_wrap.sh        wrap 组一键量化（--wrap-rotation）
quantize_dsv4_experts_tile16_dualnorm.sh    dual-norm 组一键量化（--dual-norm）
calib_collect_par.sh                         calibR9 采集 4 分片并行版（stream 模式，绕单实例锁）
weights/hessians/DeepSeek-V4-Flash-BF16-calibR9/   calibR9 Hessian（R9tau 语料，43 层）
weights/quant/dsv4-tile32x16-nores-calibR9/  calibR9 量化产物 + quant_report.json
weights/DeepSeek-V4-Flash-vptq-calibR9-tile16-attnw8a8-moew4a8/  calibR9 部署权重（§8A.2⑨）
vptq/tools/deepseek_v4/gate_scores.py         gate 分数采集（Gate hook，R9tau 语料）
vptq/tools/quantize/fallback_list.py          回退名单生成（gate 分数降序，688+2064）
vptq/tools/quantize/rebuild_bf16.py           --fallback-list 回退支持
weights/DeepSeek-V4-Flash-vptq-calibR9-t32x32fb-attnw8a8-moew4a8/  B 组（12.5% 回退）部署权重
documents/imgs/                             本文档引用的分析图（expert_ntokens_*.png / ab_mse_*.png 等，本地相对路径）
```

## 8B.1 vptq-tile16 vs attnw8a8ceil：MoE 路由分布对比实验（2026-09-17）

**目的**：vptq-tile16 与 attnw8a8ceil 的量化描述标签完全一致（attn/共享专家 W8A8_MXFP8、MoE 专家 W4A8_MXFP），差异只在 fp4 权重值的来源（msmodelslim minmax+ceil vs VPTQ tile16 码本量化）。本实验验证这层权重值差异是否传导到 MoE 的专家选择（routing）。

**方法**（同 vllm-ascend 侧 4.5 节实验）：服务加 `--enable-return-routed-experts`，GPQA 198 题（与精度评测完全相同的 prompt 构造，含 ABCD 轮转选项），相同采样参数（T=1.0/top_k=20/top_p=1.0），从响应解析 base64 numpy 数组 `[num_tokens, 43层, topk6]`，聚合成每层×每专家的路由计数。

| 组 | 请求数 | 生成 token |
|---|---|---|
| attnw8a8ceil | 198 | 112,568 |
| vptq-tile16 | 198 | **142,117（+26%）** |

### 结果

| 指标 | ceil vs vptq-tile16 | 历史对照（attn 量化差异 vs 原生） |
|---|---|---|
| 全局 JS 散度 | **0.000166** | 0.000010 |
| 每层 JS（均值/最大） | **0.00394 / 0.00672（L40）** | 0.00037 / 0.00064~71 |
| 每层 top-6 集合 Jaccard | **均值 0.811（最低 0.50 @L12）** | 0.947~0.953 |
| 单专家份额最大偏移 | **±0.061%** | ±0.012% |

单专家份额偏移最大的 6 个：expert 113（-0.061%）、97（-0.051%）、142（-0.042%）、246（+0.042%）、123（-0.036%）、183（-0.033%）。

### 图 1：每层路由分布 JS 散度条形图

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090912777162/51136028/0c5c72de81f542b688a70d4a86e8acfd.png)

- **是什么**：对每一层，把该层 256 个专家被选中次数归一化成概率分布，计算两组 ckpt 分布之间的 JS 散度（0 = 完全相同，ln2≈0.693 = 完全不相关）；虚线为均值
- **怎么读**：43 层的柱子在 4e-3~7e-3 量级——比历史对照（~4e-4，几乎贴零）高一个数量级，但仍比"完全不相关"小两个数量级；深层（L32+，峰值 L40）抬升明显，与历史模式一致（深层权重更敏感）
- **若权重差异严重破坏路由，该图会显示**：部分层柱子达到 0.01+ 甚至接近 ln2

### 图 2：专家份额一致性散点图（log-log）

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090912777162/51136629/9d25fb2985ff40f6a90951f583063160.png)

- **是什么**：每个点是一个专家；x = 该专家在 attnw8a8ceil 下的使用份额，y = 在 vptq-tile16 下的使用份额；虚线为 y=x（完全一致线）
- **怎么读**：256 个点整体紧贴对角线，但相比历史对照（全部紧贴、偏差 ±0.012%）出现轻微散开——最大偏移 ±0.061%，方向上既有 vptq 变冷的热专家（113/97/142），也有变热的（246）。没有出现离群点（无专家被弃用/暴增）
- **若路由被显著扰动，该图会显示**：点云明显散开、出现远离对角线的离群专家

### 图 3：专家使用份额排序曲线

![image](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026090912777162/51136299/d4af4f8db59a4739892cdd33039e81bd.png)

- **是什么**：256 个专家按使用份额从大到小排序后画曲线（y 对数轴），两组各一条；比较分布的"形状"（头部集中度、长尾平坦度），不关心具体是哪个专家
- **怎么读**：两条曲线几乎完全重合——头部热专家的集中度和长尾形态一致，**专家使用的"贫富结构"没有被改变**；说明路由的宏观形态稳定，差异只在个体层面的微调
- **若路由结构改变，该图会显示**：两曲线分离（一边头部更陡 = 更集中，或尾部更高 = 更均匀）

### 结论

1. **有差别，且比"attn 量化策略差异"大一个数量级**：JS ~10×、top-6 Jaccard 0.811 vs 0.95（约 19% 的层 top-6 专家集合有变化，L12 最明显——6 个换了 3 个）、份额偏移 5×。VPTQ tile16 的 fp4 权重值对 hidden states（router 输入）的扰动明显大于 attn 量化差异。
2. **绝对量级仍小**：每层 JS ~0.004 ≪ ln2；专家贫富结构（排序曲线）两边重合；无离群专家。路由对权重扰动仍然鲁棒，只是不如历史对照那么"纹丝不动"。
3. **混杂因素（必须交代）**：vptq-tile16 生成了多 26% 的 token（142k vs 113k，推理链更长），两边推理内容不同——路由差异同时包含"权重差异"与"内容差异"，不能全归因于权重。历史对照两组 token 数几乎相同（±0.7%），更干净。**如需纯净的权重效应**，建议 `max_tokens=64` + temperature=0 重采一轮对齐内容。

### 产物

```
vllm-ascend/ais_bench_logs/routed_experts/
  attnw8a8ceil.npz / vptqtile16.npz      每层×每专家路由计数(198 请求完整)
  moe_routing_{js_per_layer,expert_scatter,sorted_share}_vptqtile16.png
  attnw8a8ceil.log / vptqtile16.log / resilient_collect.log
弹性采集脚本: /tmp/resilient_collect.sh(服务被杀自动重启重采)
```

## 8B.2 rebuild BF16 vs msmodelslim 部署权重 MSE 对比与误差链分解（2026-10-09，rf32fb 产物）

**目的**：分解"VPTQ 重建 → ModelSlim W4/W8 部署"两环节的权重误差贡献，验证部署量化（MXFP4/MXFP8）是否是评测精度的瓶颈（§8A.7b' 轴4 疑点：expert 非回退存在 ~10% rel 损失）。

**方法**：对部署权重手工 dequant 后与 rebuild BF16 逐矩阵对比（抽样 51 矩阵，L0/10/21/30/42 五层 × 五类）。部署存储格式：**W8A8_MXFP8** = E4M3 原生值 × E8M0 块缩放（`2^(u8-127)`，per-32 输入元素）；**W4A8_MXFP** = E2M1 nibble 打包（**低 4 位在前**，数值判定：低前 rel 0.106 vs 高前 1.40）× E8M0。脚本 `/tmp/verify_mslim_mse.py`（环境重置丢失后按本节格式可重写）。

### 分类 MSE（rebuild BF16 vs 部署 dequant）

| 类别 | MSE 中位 | 相对误差 | 机理 |
|---|---|---|---|
| attn W8A8_MXFP8（原始→MXFP8） | 2.4e-18 | <1e-6 | 值已在 E4M3 网格，2 的幂缩放网格自洽 |
| **expert 非回退**（VPTQ 2bit 重建→MXFP4） | **6.9e-6** | **10.4%** | 重建值 = E4M3 码本 × `1/fp`（非 2 幂）离开网格，且 E4M3(3 位尾数)→E2M1(1 位) 截断格式固有 |
| expert 回退（原始 BF16→MXFP4） | **精确 0** | 0 | 原始值恰在 E2M1 网格（见下） |
| shared_experts（E4M3 存储） | 1.4e-18 | <1e-6 | 同 attn |
| gate FLOAT（对照组） | 0 | 0 | — |

### 误差链分解（同一批非回退专家矩阵）

| 环节 | rel_err | mse | 误差能量占比 |
|---|---|---|---|
| ① VPTQ 2bit（原始→rebuild） | 47.5% | 1.35e-4 | **97.3%** |
| ② MXFP4（rebuild→部署） | 10.4% | 6.7e-6 | **4.8%** |
| ③ 总链（原始→部署） | 48.1% | ≈√(①²+②²) | 正交叠加验证 ✓ |

（非回退即冷门专家：原始权重 rms≈0.025；②的 10.4% 是小基数上的相对值，绝对量级比①小 20 倍。）

### 结论

1. **部署环节非精度瓶颈**：MXFP4 舍入仅占部署前总误差能量 4.8%，VPTQ 2bit 本身占 97.3%；同位数下该 4.8% 无法消除（E4M3→E2M1 尾数截断是格式固有，2 幂 row_scale 只救 MXFP8 打包、救不了 MXFP4），若要消除须非回退专家升 W8 部署（位宽翻倍，非同位数实验，可作上界探针——已提议未执行）。
2. **原始 ckpt 权重本身在 E2M1/E4M3 网格上**（抽查值如 `-0.0234375=-3×2⁻⁷`，"DeepSeek-V4-Flash-BF16"实为低精度格式放大存储）——回退专家 MXFP4 精确零损失、基线 ModelSlim 4.25bit 直量化高保真均同源于此；**任何"真 FP8 尾数精度"的值（如 VPTQ 重建值）进 E2M1 都要付 ~10% rel 舍入**。
3. 评测不达标（rf32fb aime 64.44 vs 32×16 系 72.22）主因证据指向 t32×32 基座 tile 密度（§8A.2：t32×32 全系 aime 61~65），而非部署舍入。

## 9. 关键文件索引

```
vptq/tools/hessian/collector.py        Hessian 采集核心
vptq/tools/hessian/compute_inverse.py  H⁻¹ 离线计算
vptq/tools/deepseek_v4/                DeepSeek-V4 移植与采集驱动
vptq/tools/quantize/vq.py              VPTQ 量化核心（vptq_quantize）
vptq/tools/quantize/run.py             全模型量化驱动
vptq/tools/quantize/export.py          QuantizedWeight→VQuantLinear
vptq/layers/vqlinear.py                压缩格式层定义
vptq/ops/quant_gemm.py                 dequant / 算子分发
vptq/utils/pack.py                     索引位打包
csrc/                                  CUDA dequant / quant_gemv kernel
```
