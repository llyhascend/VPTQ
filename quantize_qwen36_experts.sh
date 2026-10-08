#!/bin/bash
# =============================================================================
# Qwen3.6-35B-A3B 路由专家 VPTQ 量化流水线（前三步：Hessian 采集 → 量化 → 装配）
#
# 分支：qwen36-packed（VPTQ-qwen36 本地 clone，root 上游仓库只读）
# 无看门狗 / 无 cron / 无 staging / 无重试 —— 顺序执行 + fail-fast
# （set -euo pipefail），任一步失败人工排查后从该步重跑；各步产物幂等
# （已存在的层文件 / .parts 自动跳过），重跑只补缺不重做。
#
# checkpoint 实测事实（2026-10-08，model.safetensors.index.json）：
#   - Qwen3_5MoeForConditionalGeneration / qwen3_5_moe，混合注意力
#     （3 线性 : 1 全注意力）—— 仅量化专家，注意力不触碰，无影响
#   - 路由专家打包 3D 张量：mlp.experts.gate_up_proj (256,1024,2048)
#     （gate|up 融合各 512 行）、mlp.experts.down_proj (256,2048,512)
#   - 前缀 model.language_model.layers.{i}；hidden=2048, inter=512,
#     40 层, 256 专家, top-8, bf16
#
# 三步对应的适配代码（本分支新增，CPU 冒烟通过 /tmp/smoke_qwen36.py）：
#   Step1 vptq/tools/qwen36/collect_hessian.py —— 专家输入/SwiGLU 中间激活
#         逐专家 H（融合 experts 模块无 per-expert Linear 可 hook，单 hook
#         双组）；输出 layer_XXXX.pt 与通用采集器逐字节同格式
#   Step2 vptq/tools/quantize/run_qwen36.py —— [e] 切片 + gate|up 行拆分
#         联合量化（共享 mlp_in H），.parts/quant_report.json 语义同 run.py
#   Step3 vptq/tools/quantize/rebuild_qwen36.py —— 2D 重建 + 256 专家重打包
#         3D，assemble 复用 rebuild_bf16.pass_assemble 原样换入 shard
#
# 用法: bash quantize_qwen36_experts.sh
#       单独重跑某步：将对应 DO_STEPx 置 1、其余置 0
# =============================================================================
set -euo pipefail
cd /home/l50062354/VPTQ-qwen36   # python -m vptq 依赖仓库根目录

# ----------------------------- 配置 -----------------------------------------
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3   # 4 卡（与 unisvq_qwen3.6 脚本一致）
export MALLOC_ARENA_MAX=4

CKPT=/home/l50062354/models/Qwen3.6-35B-A3B
CALIB_DATA=/home/l50062354/datasets/calib_corpus_R9tau.jsonl
BASE=/home/l50062354/vptq_qwen36
HESS_DIR=$BASE/hessians            # Step1 产物: layer_XXXX.pt
QUANT_DIR=$BASE/quant-tile32x16    # Step2 产物: quant_layer_XXXX_e*_e*.pt
OUT_DIR=$BASE/hf-vptq-tile32x16-bf16  # Step3 产物: 可部署 checkpoint 副本

N_LAYERS=40                        # text_config.num_hidden_layers
N_EXPERTS=256                      # text_config.num_experts
N_CARDS=4
SHARD=$((N_EXPERTS / N_CARDS))     # 64 专家/卡
HESS_SHARD=8                       # 每片层数：累加器 ~4.3GB/层
                                   # (256×(2048²+512²)fp32)，8 层 ≈ 35GB/片

DO_STEP1=1
DO_STEP2=1
DO_STEP3=1

# ----------------------------- 环境自检 -------------------------------------
python3 -c "import torch_npu" 2>/dev/null || {
    echo "ERROR: torch_npu 不可用（先 source CANN set_env.sh / 进入 NPU 环境）"
    exit 1
}
[ -f "$CKPT/model.safetensors.index.json" ] || { echo "ERROR: 缺 $CKPT index"; exit 1; }
[ -f "$CALIB_DATA" ] || { echo "ERROR: 缺校准语料 $CALIB_DATA"; exit 1; }

# =============================================================================
# Step 1: Hessian 采集（按层分片，控累加器显存）
#   128×4096 校准语料过一遍前向，逐专家累计路由 token 的 E[xxᵀ]（mlp_in）
#   与 SwiGLU 中间激活的 E[aaᵀ]（down_proj），每层一个 layer_XXXX.pt。
#   已齐的片直接跳过。
# =============================================================================
if [ "$DO_STEP1" -eq 1 ]; then
    mkdir -p "$HESS_DIR"
    for ((s = 0; s < N_LAYERS; s += HESS_SHARD)); do
        e=$((s + HESS_SHARD)); [ $e -gt $N_LAYERS ] && e=$N_LAYERS
        have=0
        for ((l = s; l < e; l++)); do
            [ -f "$HESS_DIR/layer_$(printf '%04d' "$l").pt" ] && have=$((have+1))
        done
        if [ "$have" -eq $((e - s)) ]; then
            echo "[step1] hessian $s:$e 已完成，跳过"
            continue
        fi
        echo "[step1] hessian $s:$e 采集中..."
        python3 -m vptq.tools.qwen36.collect_hessian \
            --model "$CKPT" \
            --data-path "$CALIB_DATA" \
            --output-dir "$HESS_DIR" \
            --nsamples 128 --seqlen 4096 --batch-size 4 \
            --model-dtype bf16 --device-map auto \
            --accum-device npu \
            --layer-range "$s:$e"
    done
    echo "[step1] 全部完成: $HESS_DIR ($(ls "$HESS_DIR"/layer_*.pt | wc -l)/$N_LAYERS 层)"
fi

# =============================================================================
# Step 2: 量化（4 卡并行，每卡 64 专家 × 40 层）
#   tile 方案 ≈2.66bit/权重；产物 quant_layer_XXXX_e{e0}_{e1}.pt +
#   quant_report.json（proxy_error 为质量首要指标）
# =============================================================================
if [ "$DO_STEP2" -eq 1 ]; then
    mkdir -p "$QUANT_DIR"
    pids=()
    for i in $(seq 0 $((N_CARDS - 1))); do
        e0=$((i * SHARD)); e1=$((e0 + SHARD))
        echo "[step2] shard $i: npu:$i experts $e0:$e1 -> $QUANT_DIR/shard$i.log"
        ASCEND_RT_VISIBLE_DEVICES=$i python3 -m vptq.tools.quantize.run_qwen36 \
            --ckpt "$CKPT" \
            --hessian-dir "$HESS_DIR" \
            --output-dir "$QUANT_DIR" \
            --vector-len 2 --num-centroids 16 --num-res-centroids -1 \
            --row-tile 16 --centroid-fp8 --group-num 128 --kmeans-iters 5 \
            --device npu --layer-range "0:$N_LAYERS" \
            --expert-range "$e0:$e1" \
            --hidden-size 2048 --moe-intermediate 512 \
            > "$QUANT_DIR/shard$i.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    echo "[step2] 全部完成，质量报告: $QUANT_DIR/quant_report.json"
fi

# =============================================================================
# Step 3: rebuild + 装配（两个 pass，产出可部署 checkpoint 副本）
#   pass1: per-expert 2D 重建（scale 折回）→ 256 专家重打包 3D parts
#   pass2: --assemble 复用 rebuild_bf16.pass_assemble 换入 shard 副本，
#          其余权重（attn/router/shared/mtp/embed）原样拷贝
# =============================================================================
if [ "$DO_STEP3" -eq 1 ]; then
    mkdir -p "$OUT_DIR"
    echo "[step3] pass1: rebuild 打包 3D 权重..."
    python3 -m vptq.tools.quantize.rebuild_qwen36 \
        --ckpt "$CKPT" --quant-dir "$QUANT_DIR" --output-dir "$OUT_DIR" \
        --layer-range "0:$N_LAYERS" --group-num 128 --row-tile 16 --device npu
    echo "[step3] pass2: assemble 装配部署权重..."
    python3 -m vptq.tools.quantize.rebuild_qwen36 \
        --ckpt "$CKPT" --quant-dir "$QUANT_DIR" --output-dir "$OUT_DIR" \
        --layer-range "0:$N_LAYERS" --group-num 128 --row-tile 16 --device npu \
        --assemble
    echo "[step3] 全部完成: $OUT_DIR"
fi

echo "PIPELINE DONE: $OUT_DIR"
