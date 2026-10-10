#!/bin/bash
# =============================================================================
# DeepSeek-V4-Flash-BF16 路由专家一键量化（Tile 方案 v2-k16-res8-fp8）
#
# - 仅量化 MoE 路由专家（--scope experts --row-fp8；共享专家与注意力保持 BF16）
# - 8 卡并行：每卡一个进程，各负责 32 个专家 × 43 层
# - 每进程带看门狗：环境会 ~20-25 分钟杀进程，被杀后自动重启，
#   逐矩阵落盘（.parts）保证重启零损失
# - 用法:  bash quantize_dsv4_experts.sh [start|status|stop]
# =============================================================================
set -u
# 保证无论从哪里调用都以仓库根目录为工作目录（python -m 依赖）
cd "$(dirname "$0")"

# ------------------------------ 配置 ----------------------------------------
CKPT=/mnt/share/weight/DeepSeek-V4-Flash-BF16
HESSIAN_DIR=/mnt/share/w00608002/weights/hessians/DeepSeek-V4-Flash-BF16-calibR9
# 09-20: floor=4000 补采后(below 4382→430)的 Hessian 量化对照;
# topup1024 版(劣化,+38% 坑底)保留勿动;原始基线在 dsv4-tile32x16-nores
OUTPUT_DIR=/mnt/share/w00608002/weights/quant/dsv4-tile32x32-rowfp8-fb-calibR9
LOG_DIR=$OUTPUT_DIR/logs

# Tile 方案参数（实测 1.2s/矩阵，proxy_error ~1.5%，2.66 bit/权重）
V_LEN=2               # 主量化向量长
K=16                  # 主码本大小（4bit 索引）
K_RES=16              # 残差码本大小
V_RES=8               # 残差向量长（解耦）
ROW_TILE=32          # tile 行高（32x32 基座）
GROUP_NUM=128         # 列组数（in_features/32）
KMEANS_ITERS=5

N_LAYERS=43
N_EXPERTS=256
EXPERTS_PER_SHARD=32  # 256 / 8 卡

# --------------------------- 看门狗函数 -------------------------------------
# 重启直到 run.py 打出 "report written"（其 layer-range 全部完成）
watchdog() {
    local dev=$1 log=$2; shift 2
    for attempt in $(seq 1 500); do
        if grep -aq "report written" "$log" 2>/dev/null; then
            echo "[watchdog] shard complete $(date)" >> "$log"; return 0
        fi
        echo "[watchdog] attempt $attempt $(date)" >> "$log"
        ASCEND_RT_VISIBLE_DEVICES=$dev python3 -m vptq.tools.quantize.run "$@" \
            >> "$log" 2>&1
        sleep 10
    done
    echo "[watchdog] ERROR: exhausted attempts" >> "$log"; return 1
}

# ----------------------------- 子命令 ---------------------------------------
start() {
    mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
    for i in 0 1 2 3 4 5 6 7; do
        e0=$((i * EXPERTS_PER_SHARD)); e1=$((e0 + EXPERTS_PER_SHARD))
        setsid nohup bash -c "$(declare -f watchdog); watchdog $i $LOG_DIR/shard$i.log \
            --ckpt $CKPT \
            --hessian-dir $HESSIAN_DIR \
            --output-dir $OUTPUT_DIR \
            --vector-len $V_LEN --num-centroids $K \
            --num-res-centroids -1 \
            --row-tile $ROW_TILE --centroid-fp8 \
            --group-num $GROUP_NUM --kmeans-iters $KMEANS_ITERS \
            --device npu --layer-range 0:$N_LAYERS \
            --expert-range $e0:$e1 --scope experts --row-fp8" \
            > /dev/null 2>&1 &
        echo "shard $i: npu:$i experts $e0:$e1 -> $LOG_DIR/shard$i.log (pid $!)"
    done
    echo "8 shards launched. ETA ~2h (33024 matrices, ~1.2s each per card)."
}

status() {
    local total=0
    for i in 0 1 2 3 4 5 6 7; do
        d=$OUTPUT_DIR/quant_layer_*.parts
        n=$(cat $OUTPUT_DIR/quant_layer_*_e$((i*32))_$((i*32+32)).parts/*.pt 2>/dev/null | wc -c)
        parts=$(ls $OUTPUT_DIR/quant_layer_*_e$((i*32))_$((i*32+32)).parts 2>/dev/null | grep -c '\.pt$' || true)
        done_layers=$(ls $OUTPUT_DIR/quant_layer_*_e$((i*32))_$((i*32+32)).pt 2>/dev/null | wc -l)
        echo "shard $i: $parts matrices saved, $done_layers/$N_LAYERS layers merged"
        total=$((total + parts))
    done
    echo "TOTAL: $total / $((N_LAYERS * N_EXPERTS * 3)) matrices"
}

stop() {
    pkill -f "vptq.tools.quantize.run" ; pkill -f "watchdog $"
    echo "stopped (progress is saved in .parts dirs; 'start' resumes)"
}

case "${1:-start}" in
    start)  start ;;
    status) status ;;
    stop)   stop ;;
    *) echo "usage: $0 [start|status|stop]"; exit 1 ;;
esac
