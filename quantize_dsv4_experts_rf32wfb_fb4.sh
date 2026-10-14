#!/bin/bash
# rf32wfb-fb4：回退专家 wrap(block32)+mxfp4 c7.25 预量化 → scale_one 部署
# 2026-10-14。S3 复用 rf32wfb 产物；S4' 在独立 BF16 目录重跑 rebuild（含
# fallback parts）；S5' 用 gen_yaml_rf32wfb.py 新 yaml（规则 B=scale_one）。
set -u
cd "$(dirname "$0")"

QUANT=/mnt/share/w00608002/weights/quant/dsv4-tile32x32-rowfp8-fb-calibR9
CKPT=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-vptq-calibR9-rf32wfb-bf16
ROTATION=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-BF16-rot2/rotation.safetensors
BF16=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-vptq-calibR9-rf32wfbfb4-bf16
DEPLOY=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-vptq-calibR9-rf32wfbfb4-attnw8a8-moew8a8
YAML=/mnt/share/w00608002/msmodelslim/deepseek-v4-flash-attenw8a8-vptqw8a8scale1-moew8a8.yaml
STATE=/tmp/pipeline_rf32wfbfb4
mkdir -p "$STATE"
log(){ echo "[fb4 $(date '+%m-%d %H:%M:%S')] $*" >> /tmp/pipeline_rf32wfbfb4.log; }

# S4'：专家 pass（VPTQ 28896 + fallback mxfp4 4128 = 33024 parts）+ assemble
# 2026-10-14：8 进程分层并行（disjoint layer-range，进程级安全），watchdog
# 统一判 parts 总数；某进程被 killer 杀 → 全组重启，resume 秒过已完成层。
s4(){
  [ -f "$STATE/s4.done" ] && return 0
  local a=0 n
  local ranges=(0:6 6:12 12:18 18:24 24:30 30:35 35:39 39:43)
  while true; do
    n=$(find "$BF16/expert_parts" -name "*.safetensors" 2>/dev/null | wc -l)
    [ "$n" -ge 33024 ] && break
    if ! pgrep -f "rebuild_bf1[6]" >/dev/null; then
      a=$((a+1)); [ $a -gt 40 ] && { log "S4 重建重试耗尽"; return 1; }
      log "S4 rebuild(8p×8卡) parts=$n/33024 attempt=$a"
      local i=0
      for r in "${ranges[@]}"; do
        setsid nohup python3 -m vptq.tools.quantize.rebuild_bf16 \
          --quant-dir "$QUANT" --ckpt "$CKPT" --output-dir "$BF16" \
          --group-num 128 --row-tile 32 --device "npu:$i" \
          --layer-range "$r" \
          --fallback-list "$QUANT/fallback_list.json" --fallback-mxfp4 \
          --wrap-rotation "$ROTATION" \
          >> /tmp/rebuild_rf32wfbfb4_expert.log 2>&1 &
        i=$((i+1))
      done
    fi
    sleep 120
  done
  a=0
  while true; do
    [ -f "$BF16/model.safetensors.index.json" ] && break
    if ! pgrep -f "rebuild_bf1[6]" >/dev/null; then
      a=$((a+1)); [ $a -gt 40 ] && { log "S4 组装重试耗尽"; return 1; }
      log "S4 assemble attempt=$a"
      setsid nohup python3 -m vptq.tools.quantize.rebuild_bf16 \
        --quant-dir "$QUANT" --ckpt "$CKPT" --output-dir "$BF16" \
        --assemble >> /tmp/rebuild_rf32wfbfb4_assemble.log 2>&1 &
    fi
    sleep 300
  done
  touch "$STATE/s4.done"; log "S4 done"
}

# S5'：msmodelslim 部署（新 yaml，规则 B=scale_one；CLI 同 modelslim_rf32wfb.sh）
s5(){
  [ -f "$STATE/s5.done" ] && return 0
  local a=0
  export HF_DATASETS_OFFLINE=1
  export ASCEND_RT_VISIBLE_DEVICES=2
  export ASCEND_SLOG_PRINT_TO_STDOUT=0
  export ASCEND_GLOBAL_LOG_LEVEL=2
  export ASCEND_HOST_LOG_FILE_NUM=1000
  mkdir -p "$DEPLOY/log/"
  export ASCEND_PROCESS_LOG_PATH=$DEPLOY/log/
  while true; do
    [ -f "$DEPLOY/quant_model_weights.safetensors.index.json" ] && break
    if ! pgrep -f "msmodelslim quan[t]" >/dev/null; then
      a=$((a+1)); [ $a -gt 40 ] && { log "S5 重试耗尽"; return 1; }
      log "S5 msmodelslim quant attempt=$a"
      setsid nohup msmodelslim quant \
        --model_path "$BF16" --save_path "$DEPLOY" --device npu \
        --model_type DeepSeek-V4-Flash --trust_remote_code True \
        --config "$YAML" \
        >> /tmp/msmodelslim_rf32wfbfb4.log 2>&1 &
    fi
    sleep 120
  done
  touch "$STATE/s5.done"; log "S5 done"
}

s4 && s5
log "pipeline fb4 complete"
