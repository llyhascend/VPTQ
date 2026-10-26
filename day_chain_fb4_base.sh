#!/bin/bash
# 日间链（2026-10-15 用户指令：8 卡空闲即跑，不等整点）：
#   1. assemble 补齐 29-46（判据=46 shards，防 index.json 误判）
#   2. fb4 S5' 部署（killer 自愈）
#   3. vllm + 单请求真实性验证（乱码则停链报告）
#   4. 验证通过 → fb4 评测 gpqa×5 + aime×5 → 停 vllm
#   5. 基线2 部署（原始 ckpt + attnw8a8ceil + moe 专家 scale_one）
#   6. 基线2 评测 gpqa×5 + aime×5 → 停 vllm → FINISHED
# 幂等 done 标记 + 轮数按 outputs 事实判定；flock 防重入；watchdog 拉活。
set -u
exec 9>/tmp/day_chain_fb4.lock
flock -n 9 || { echo "[day] 已有实例，退出"; exit 0; }

STATE=/tmp/day_chain_fb4
LOG=/tmp/day_chain_fb4.log
BENCH=/mnt/share/w00608002/benchmark
OUTBASE=$BENCH/outputs/default
QUANT=/mnt/share/w00608002/weights/quant/dsv4-tile32x32-rowfp8-wrap-fb-calibR9
CKPT=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-vptq-calibR9-rf32wfb-bf16
FB4_BF16=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-vptq-calibR9-rf32wfbfb4-bf16
FB4_DEPLOY=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-vptq-calibR9-rf32wfbfb4-attnw8a8-moew8a8
FB4_YAML=/mnt/share/w00608002/msmodelslim/deepseek-v4-flash-attenw8a8-vptqw8a8scale1-moew8a8.yaml
ROT=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-BF16-rot2/rotation.safetensors
ORIG_CKPT=/mnt/share/weight/DeepSeek-V4-Flash-BF16
BASE2_DEPLOY=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-attnw8a8ceil-moescale1
BASE2_YAML=/mnt/share/w00608002/msmodelslim/deepseek-v4-flash-attnw8a8ceil-moescale1.yaml
VLLM_FB4=/mnt/share/w00608002/work/vllm-ascend/vllm_start_rf32wfbfb4.sh
VLLM_BASE2=/mnt/share/w00608002/work/vllm-ascend/vllm_start_base2.sh
mkdir -p "$STATE"
log(){ echo "[day $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

wait_idle(){
  while true; do
    local free=$(npu-smi info 2>/dev/null | grep -c "No running processes")
    [ "$free" -ge 8 ] && return 0
    sleep 120
  done
}
ensure_vllm(){
  local sh=$1 tag=$2 wv=0
  curl -s -m 5 http://127.0.0.1:9001/v1/models 2>/dev/null | grep -q '"id"' && return 0
  pkill -9 -f "vllm serv[e]" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null; sleep 20
  # 10-17 事故修复：killer 杀 vllm 后 psm_* shm 段泄漏（64MB tmpfs 满载 →
  # 后续实例并发请求全 500 Internal Server Error）。进程死透后清死段。
  find /dev/shm -name 'psm_*' -mmin +2 -delete 2>/dev/null; sync
  ( cd /mnt/share/w00608002/work/vllm-ascend && setsid nohup bash "$sh" > /tmp/vllm_day_$tag.log 2>&1 & )
  until curl -s -m 5 http://127.0.0.1:9001/v1/models 2>/dev/null | grep -q '"id"'; do
    sleep 60; wv=$((wv+60)); [ $wv -gt 2400 ] && { log "vllm($tag) 起服超时"; return 1; }
  done
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
    sleep 30; t=$((t+30)); [ $t -gt 900 ] && { log "vllm 停止超时"; break; }
  done
  log "vllm 已停"
}
run_rounds(){
  local ds=$1 n=$2 t0file=$3 tag=$4
  local dsarg
  [ "$ds" = gpqa ] && dsarg=gpqa_gen_0_shot_cot_chat_prompt.py || dsarg=aime2024_gen_0_shot_chat_prompt.py
  [ -f "$t0file" ] || touch "$t0file"
  local t0=$(stat -c %Y "$t0file") sh
  [ "$tag" = base2 ] && sh=$VLLM_BASE2 || sh=$VLLM_FB4
  while true; do
    local cnt=0 c
    for c in "$OUTBASE"/*/summary/summary_*.csv; do
      [ -f "$c" ] || continue
      [ "$(stat -c %Y "$c")" -lt "$t0" ] && continue
      grep -qi "$ds" "$c" 2>/dev/null && grep -q "accuracy,gen,[0-9]" "$c" 2>/dev/null && cnt=$((cnt+1))
    done
    [ "$cnt" -ge "$n" ] && { log "$ds($tag) 完成 $cnt/$n"; return 0; }
    ensure_vllm "$sh" "$tag" || return 1
    log "$ds($tag) 启动一轮（已有 $cnt/$n）"
    ( cd "$BENCH" && timeout 2400 ais_bench --models vllm_api_general_chat \
      --datasets "$dsarg" --mode all --dump-eval-details --merge-ds \
      >> /tmp/aisbench_day.log 2>&1 )
    log "$ds($tag) 轮结束 rc=$?"
    sleep 30
  done
}
run_slim(){
  local model=$1 out=$2 yaml=$3 tag=$4 a=0
  export HF_DATASETS_OFFLINE=1 ASCEND_RT_VISIBLE_DEVICES=2
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

log "=== 日间链开跑（等 8 卡空闲）==="
wait_idle
log "NPU 空闲"

# Stage 1：assemble 补齐（判据 46 shards——10-15 事故加固点）
if [ ! -f "$STATE/s1.done" ]; then
  a=0
  while true; do
    n_shards=$(ls "$FB4_BF16"/model-*.safetensors 2>/dev/null | wc -l)
    [ "$n_shards" -ge 46 ] && break
    if ! pgrep -f "rebuild_bf1[6].*assembl[e]" >/dev/null; then
      a=$((a+1)); [ $a -gt 30 ] && { log "assemble 重试耗尽"; exit 1; }
      log "assemble 补齐 shards=$n_shards/46 attempt=$a"
      setsid nohup python3 -m vptq.tools.quantize.rebuild_bf16 \
        --quant-dir "$QUANT" --ckpt "$CKPT" --output-dir "$FB4_BF16" \
        --assemble >> /tmp/rebuild_day_assemble.log 2>&1 &
    fi
    sleep 120
  done
  touch "$STATE/s1.done"; log "assemble 46/46 完成"
fi

# Stage 2：fb4 S5'
[ -f "$STATE/s2.done" ] || { run_slim "$FB4_BF16" "$FB4_DEPLOY" "$FB4_YAML" fb4 && touch "$STATE/s2.done" || exit 1; }

# Stage 3：真实性验证（乱码即停链）
if [ ! -f "$STATE/s3.done" ]; then
  ensure_vllm "$VLLM_FB4" fb4 || exit 1
  curl -s -m 600 http://127.0.0.1:9001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"dsv","messages":[{"role":"user","content":"一个笼子里有鸡和兔共 35 个头、94 只脚，问鸡兔各几只？请一步步推理后给出答案。"}],"max_tokens":512,"temperature":0}' \
    > /tmp/fb4_verify_out.json 2>&1
  verdict=$(python3 - <<'PYEOF'
import json
try:
    d=json.load(open('/tmp/fb4_verify_out.json'))
    txt=d['choices'][0]['message']['content']
    open('/tmp/fb4_verify_out.txt','w').write(txt)
    fr=d['choices'][0].get('finish_reason')
    uniq=len(set(txt))/max(len(txt),1)
    print(f"OK finish={fr} len={len(txt)} uniq={uniq:.3f}" if (fr=='stop' and uniq>0.10) else f"BAD finish={fr} len={len(txt)} uniq={uniq:.3f}")
except Exception as e:
    print(f"BAD parse:{e}")
PYEOF
)
  log "验证结果：$verdict"
  case "$verdict" in OK*) touch "$STATE/s3.done";; *) log "输出异常，链停在评测前——人工复核 /tmp/fb4_verify_out.txt"; exit 1;; esac
fi

# Stage 4：fb4 评测 5+5
if [ ! -f "$STATE/s4.done" ]; then
  run_rounds gpqa 5 "$STATE/fb4_t0" fb4 && run_rounds aime 5 "$STATE/fb4_t0" fb4 \
    && touch "$STATE/s4.done" || exit 1
  stop_vllm
fi

# Stage 5：基线2 部署
[ -f "$STATE/s5.done" ] || { run_slim "$ORIG_CKPT" "$BASE2_DEPLOY" "$BASE2_YAML" base2 && touch "$STATE/s5.done" || exit 1; }

# Stage 6：基线2 评测 5+5
if [ ! -f "$STATE/s6.done" ]; then
  run_rounds gpqa 5 "$STATE/base2_t0" base2 && run_rounds aime 5 "$STATE/base2_t0" base2 \
    && touch "$STATE/s6.done" || exit 1
  stop_vllm
fi

touch "$STATE/FINISHED"
log "=== 日间链全部完成：fb4(5+5) + 基线2 moe-scale1(5+5) ==="
