#!/bin/bash
# 夜间链（2026-10-14 用户指令）：02:00 且 NPU 空闲时依次执行
#   A. fb4 S5' 收尾（msmodelslim，killer 自愈）
#   B. fb4 评测：gpqa×5 + aime×5（vllm 9001）
#   C. 基线2 部署：原始 ckpt + attnw8a8ceil + moe 专家 cast e4m3 scale=1
#   D. 基线2 评测：gpqa×5 + aime×5
# 幂等：STATE done 标记 + 轮数按 outputs/default 新增 csv 事实判定；
# 全程 flock 防重入，被 killer 杀掉后由 cron 重新拉起续跑。
set -u
exec 9>/tmp/night_fb4_base.lock
flock -n 9 || { echo "[night] 已有实例在跑，退出"; exit 0; }

STATE=/tmp/night_fb4_base
LOG=/tmp/night_fb4_base.log
BENCH=/mnt/share/w00608002/benchmark
OUTBASE=$BENCH/outputs/default
FB4_DEPLOY=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-vptq-calibR9-rf32wfbfb4-attnw8a8-moew8a8
FB4_BF16=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-vptq-calibR9-rf32wfbfb4-bf16
FB4_YAML=/mnt/share/w00608002/msmodelslim/deepseek-v4-flash-attenw8a8-vptqw8a8scale1-moew8a8.yaml
ORIG_CKPT=/mnt/share/weight/DeepSeek-V4-Flash-BF16
BASE2_DEPLOY=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-attnw8a8ceil-moescale1
BASE2_YAML=/mnt/share/w00608002/msmodelslim/deepseek-v4-flash-attnw8a8ceil-moescale1.yaml
VLLM_FB4=/mnt/share/w00608002/work/vllm-ascend/vllm_start_rf32wfbfb4.sh
VLLM_BASE2=/mnt/share/w00608002/work/vllm-ascend/vllm_start_base2.sh
mkdir -p "$STATE"
log(){ echo "[night $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# ---------- 全链（含 msmodelslim）等下一个 02:00（今天未到等今天，已过等明天）----------
wait_0200(){
  local now=$(date +%s)
  local t0=$(date -d "today 02:00" +%s)
  [ "$now" -ge "$t0" ] && t0=$(date -d "tomorrow 02:00" +%s)
  if [ "$now" -lt "$t0" ]; then
    log "等待 02:00 启动全链（$(( (t0-now)/60 )) 分钟后）"
    sleep $((t0-now))
  fi
}

# ---------- 等待 NPU 空闲（02:00 后最多等 2 小时，仍忙则退出由看门狗重试）----------
wait_idle(){
  local deadline=$(( $(date +%s) + 7200 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    local busy=$(npu-smi info 2>/dev/null | grep -c "No running processes")
    [ "$busy" -ge 8 ] && return 0
    sleep 600
  done
  log "02:00+2h 仍不空闲，本轮放弃（看门狗重试）"; return 1
}

# ---- 全部阶段（A/B/C/D）统一：02:00 + NPU 空闲后才开跑（用户 10-14 修正指令）----
wait_0200
wait_idle || exit 0
log "02:00+空闲，夜间链开跑"

# ---------- 通用：vllm 保活 ----------
ensure_vllm(){
  local sh=$1 tag=$2 wv=0
  curl -s -m 5 http://127.0.0.1:9001/v1/models 2>/dev/null | grep -q '"id"' && return 0
  pkill -f "vllm serv[e]" 2>/dev/null; sleep 20
  ( cd /mnt/share/w00608002/work/vllm-ascend && setsid nohup bash "$sh" > /tmp/vllm_night_$tag.log 2>&1 & )
  until curl -s -m 5 http://127.0.0.1:9001/v1/models 2>/dev/null | grep -q '"id"'; do
    sleep 60; wv=$((wv+60)); [ $wv -gt 2400 ] && { log "vllm($tag) 起服超时"; return 1; }
  done
  # 真实请求预热（health 200 != worker 就绪）
  until curl -s -m 300 http://127.0.0.1:9001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"dsv","messages":[{"role":"user","content":"ping"}],"max_tokens":8}' \
    2>/dev/null | grep -q '"finish_reason"'; do
    sleep 60; wv=$((wv+60)); [ $wv -gt 3000 ] && { log "vllm($tag) 预热超时"; return 1; }
  done
  log "vllm($tag) 就绪+预热完成"
}

stop_vllm(){
  pkill -f "vllm serv[e]" 2>/dev/null
  local t=0
  while [ "$(npu-smi info 2>/dev/null | grep -c 'No running processes')" -lt 8 ]; do
    sleep 30; t=$((t+30)); [ $t -gt 900 ] && { log "vllm 停止超时（HBM 未释放）"; break; }
  done
  log "vllm 已停（HBM 释放）"
}

# ---------- 通用：跑满 n 轮 ----------
run_rounds(){
  local ds=$1 n=$2 t0file=$3
  local dsarg
  [ "$ds" = gpqa ] && dsarg=gpqa_gen_0_shot_cot_chat_prompt.py || dsarg=aime2024_gen_0_shot_chat_prompt.py
  [ -f "$t0file" ] || touch "$t0file"
  local t0=$(stat -c %Y "$t0file")
  while true; do
    local cnt=0 c
    for c in "$OUTBASE"/*/summary/summary_*.csv; do
      [ -f "$c" ] || continue
      [ "$(stat -c %Y "$c")" -lt "$t0" ] && continue
      grep -qi "$ds" "$c" 2>/dev/null && grep -q "accuracy,gen,[0-9]" "$c" 2>/dev/null && cnt=$((cnt+1))
    done
    [ "$cnt" -ge "$n" ] && { log "$ds 完成 $cnt/$n"; return 0; }
    ensure_vllm "$( [ "${4:-}" = base2 ] && echo "$VLLM_BASE2" || echo "$VLLM_FB4" )" "${4:-fb4}" || return 1
    log "$ds 启动一轮（已有 $cnt/$n）"
    ( cd "$BENCH" && timeout 2400 ais_bench --models vllm_api_general_chat \
      --datasets "$dsarg" --mode all --dump-eval-details --merge-ds \
      >> /tmp/aisbench_night.log 2>&1 )
    local rc=$?
    log "$ds 轮结束 rc=$rc（kill/超时则不计，自动重试）"
    sleep 30
  done
}

# ---------- msmodelslim 通用（killer 自愈直至 index 落盘） ----------
run_slim(){
  local model=$1 out=$2 yaml=$3 tag=$4 a=0
  export HF_DATASETS_OFFLINE=1
  export ASCEND_RT_VISIBLE_DEVICES=2
  export ASCEND_SLOG_PRINT_TO_STDOUT=0 ASCEND_GLOBAL_LOG_LEVEL=2 ASCEND_HOST_LOG_FILE_NUM=1000
  mkdir -p "$out/log/"; export ASCEND_PROCESS_LOG_PATH=$out/log/
  while true; do
    [ -f "$out/quant_model_weights.safetensors.index.json" ] && { log "slim($tag) 完成"; return 0; }
    if ! pgrep -f "msmodelslim quan[t]" >/dev/null; then
      a=$((a+1)); [ $a -gt 60 ] && { log "slim($tag) 重试耗尽"; return 1; }
      log "slim($tag) attempt=$a"
      setsid nohup msmodelslim quant --model_path "$model" --save_path "$out" \
        --device npu --model_type DeepSeek-V4-Flash --trust_remote_code True \
        --config "$yaml" >> /tmp/msmodelslim_$tag.log 2>&1 &
    fi
    sleep 120
  done
}

# ================= Phase A：fb4 S5' =================
if [ ! -f "$STATE/a.done" ]; then
  run_slim "$FB4_BF16" "$FB4_DEPLOY" "$FB4_YAML" fb4 && touch "$STATE/a.done" || exit 1
fi

# ================= Phase B：fb4 评测 =================
if [ ! -f "$STATE/b.done" ]; then
  run_rounds gpqa 5 "$STATE/fb4_t0" fb4 && run_rounds aime 5 "$STATE/fb4_t0" fb4 \
    && touch "$STATE/b.done" || exit 1
  stop_vllm
fi

# ================= Phase C：基线2 部署 =================
if [ ! -f "$STATE/c.done" ]; then
  run_slim "$ORIG_CKPT" "$BASE2_DEPLOY" "$BASE2_YAML" base2 && touch "$STATE/c.done" || exit 1
fi

# ================= Phase D：基线2 评测 =================
if [ ! -f "$STATE/d.done" ]; then
  run_rounds gpqa 5 "$STATE/base2_t0" base2 && run_rounds aime 5 "$STATE/base2_t0" base2 \
    && touch "$STATE/d.done" || exit 1
  stop_vllm
fi

touch "$STATE/FINISHED"
log "夜间链全部完成：fb4(5+5) + 基线2 attnw8a8ceil+moe-scale1(5+5)"
