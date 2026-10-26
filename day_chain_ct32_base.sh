#!/bin/bash
# ct32 端到端链（2026-10-20 用户指令：col_tile 统一 32×32 tile 后按 fb4 配置重跑）：
#   0. 等量化完成（33024 parts，quantize_dsv4_experts_rf32wfbct32.sh 自愈中）
#   1. rebuild parts + assemble（判据=46 shards）
#   2. msmodelslim 部署（fb4 同 yaml：vptq 直通 scale_one + moeW8A8）
#   3. vllm + 单请求真实性验证（乱码则停链报告）
#   4. 验证通过 → GPQA×5 + aime×5 → 停 vllm → FINISHED
# 幂等 done 标记 + 轮数按 outputs 事实判定；flock 防重入；watchdog 拉活。
set -u
exec 9>/tmp/day_chain_ct32.lock
flock -n 9 || { echo "[ct] 已有实例，退出"; exit 0; }

STATE=/tmp/day_chain_ct32
LOG=/tmp/day_chain_ct32.log
BENCH=/mnt/share/w00608002/benchmark
OUTBASE=$BENCH/outputs/default
QUANT=/mnt/share/w00608002/weights/quant/dsv4-ct32x32-rowfp8-wrap-fb-calibR9
CKPT=/mnt/share/weight/DeepSeek-V4-Flash-BF16
CT32_BF16=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-vptq-calibR9-rf32wfbct32-bf16
CT32_DEPLOY=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-vptq-calibR9-rf32wfbct32-attnw8a8-moew8a8
CT32_YAML=/mnt/share/w00608002/msmodelslim/deepseek-v4-flash-attenw8a8-vptqw8a8scale1-moew8a8.yaml
FALLBACK_LIST=/mnt/share/w00608002/weights/quant/dsv4-tile32x32-rowfp8-wrap-fb-calibR9/fallback_list.json
ROTATION=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-BF16-rot2/rotation.safetensors
VLLM_CT32=/mnt/share/w00608002/work/vllm-ascend/vllm_start_rf32wfbct32.sh
mkdir -p "$STATE"
log(){ echo "[ct $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

wait_quant(){
  # 等量化 33024 parts + 43 层合并文件齐（quantize 脚本自身带看门狗自愈）
  local a=0
  while true; do
    local parts=$(find "$QUANT" -name "*.pt" -path "*parts*" 2>/dev/null | wc -l)
    local merged=$(ls "$QUANT"/quant_layer_*_e*.pt 2>/dev/null | wc -l)
    [ "$parts" -ge 33024 ] && [ "$merged" -ge 344 ] && return 0
    if ! pgrep -f "quantize_dsv4_experts_rf32wfbct3[2]" >/dev/null \
       && ! pgrep -f -- "--output-dir $QUANT" >/dev/null; then
      a=$((a+1)); [ $a -gt 10 ] && { log "量化驱动死亡且未完成 ($parts/33024)"; return 1; }
      log "量化驱动不在，重启 ($parts/33024)"
      ( cd /mnt/share/w00608002/work/VPTQ && setsid nohup bash \
        quantize_dsv4_experts_rf32wfbct32.sh start >> /tmp/quant_ct32_start.log 2>&1 & )
    fi
    sleep 300
  done
}

ensure_vllm(){
  local sh=$1 tag=$2 wv=0
  curl -s -m 5 http://127.0.0.1:9001/v1/models 2>/dev/null | grep -q '"id"' && return 0
  pkill -9 -f "vllm serv[e]" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null; sleep 20
  # 10-17 事故修复：killer 杀 vllm 后 psm_* shm 段泄漏致并发全 500
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
  pkill -9 -f "vllm serv[e]" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null
  sleep 10; find /dev/shm -name 'psm_*' -mmin +2 -delete 2>/dev/null
  local t=0
  while [ "$(npu-smi info 2>/dev/null | grep -c 'No running processes')" -lt 8 ]; do
    sleep 30; t=$((t+30)); [ $t -gt 900 ] && { log "vllm 停止超时"; break; }
  done
  log "vllm 已停"
}
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
    [ "$cnt" -ge "$n" ] && { log "$ds(ct32) 完成 $cnt/$n"; return 0; }
    ensure_vllm "$VLLM_CT32" ct32 || return 1
    log "$ds(ct32) 启动一轮（已有 $cnt/$n）"
    ( cd "$BENCH" && timeout 2400 ais_bench --models vllm_api_general_chat \
      --datasets "$dsarg" --mode all --dump-eval-details --merge-ds \
      >> /tmp/aisbench_ct32.log 2>&1 )
    log "$ds(ct32) 轮结束 rc=$?"
    sleep 30
  done
}
run_slim(){
  local model=$1 out=$2 yaml=$3 a=0
  export HF_DATASETS_OFFLINE=1 ASCEND_RT_VISIBLE_DEVICES=2
  export ASCEND_SLOG_PRINT_TO_STDOUT=0 ASCEND_GLOBAL_LOG_LEVEL=2 ASCEND_HOST_LOG_FILE_NUM=1000
  mkdir -p "$out/log/"; export ASCEND_PROCESS_LOG_PATH=$out/log/
  while true; do
    [ -f "$out/quant_model_weights.safetensors.index.json" ] && { log "slim(ct32) 完成"; return 0; }
    if ! pgrep -f "msmodelslim quan[t]" >/dev/null; then
      a=$((a+1)); [ $a -gt 60 ] && { log "slim(ct32) 重试耗尽"; return 1; }
      log "slim(ct32) attempt=$a"
      setsid nohup msmodelslim quant --model_path "$model" --save_path "$out" \
        --device npu --model_type DeepSeek-V4-Flash --trust_remote_code True \
        --config "$yaml" >> /tmp/msmodelslim_ct32.log 2>&1 &
    fi
    sleep 120
  done
}

log "=== ct32 链开跑（先等量化，全程 killer 自愈）==="
# Stage 0：等量化（判据 33024 parts + 344 合并文件——数文件不数 index）
[ -f "$STATE/s0.done" ] || { wait_quant && touch "$STATE/s0.done" || exit 1; }

# Stage 1：rebuild parts（VPTQ 反量化 wrap 回 + fallback wrap 域 mxfp4）+ assemble
if [ ! -f "$STATE/s1.done" ]; then
  a=0
  while true; do
    n_parts=$(find "$CT32_BF16/expert_parts" -name "*.safetensors" 2>/dev/null | wc -l)
    [ "$n_parts" -ge 33024 ] && break
    if ! pgrep -f "rebuild_bf1[6].*ct3[2]" >/dev/null; then
      a=$((a+1)); [ $a -gt 30 ] && { log "rebuild 重试耗尽"; exit 1; }
      log "rebuild parts=$n_parts/33024 attempt=$a"
      local_i=0
      for r in 0:6 6:12 12:18 18:24 24:30 30:35 35:39 39:43; do
        setsid nohup python3 -m vptq.tools.quantize.rebuild_bf16 \
          --quant-dir "$QUANT" --ckpt "$CKPT" --output-dir "$CT32_BF16" \
          --device "npu:$local_i" --layer-range "$r" \
          --fallback-list "$FALLBACK_LIST" --fallback-mxfp4 \
          --wrap-rotation "$ROTATION" \
          >> /tmp/rebuild_ct32.log 2>&1 &
        local_i=$((local_i+1))
      done
    fi
    sleep 120
  done
  touch "$STATE/re.done"
  # assemble（判据 46 shards——10-15 事故加固点）
  a=0
  while true; do
    n_shards=$(ls "$CT32_BF16"/model-*.safetensors 2>/dev/null | wc -l)
    [ "$n_shards" -ge 46 ] && break
    if ! pgrep -f "rebuild_bf1[6].*assembl[e]" >/dev/null; then
      a=$((a+1)); [ $a -gt 30 ] && { log "assemble 重试耗尽"; exit 1; }
      log "assemble 补齐 shards=$n_shards/46 attempt=$a"
      setsid nohup python3 -m vptq.tools.quantize.rebuild_bf16 \
        --quant-dir "$QUANT" --ckpt "$CKPT" --output-dir "$CT32_BF16" \
        --assemble >> /tmp/rebuild_ct32_assemble.log 2>&1 &
    fi
    sleep 120
  done
  touch "$STATE/s1.done"; log "rebuild+assemble 完成"
fi

# Stage 2：部署（fb4 同 yaml：VPTQ 重建值 scale_one 直通 + moeW8A8）
[ -f "$STATE/s2.done" ] || { run_slim "$CT32_BF16" "$CT32_DEPLOY" "$CT32_YAML" && touch "$STATE/s2.done" || exit 1; }

# Stage 3：真实性验证（乱码即停链）
if [ ! -f "$STATE/s3.done" ]; then
  ensure_vllm "$VLLM_CT32" ct32 || exit 1
  curl -s -m 600 http://127.0.0.1:9001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"dsv","messages":[{"role":"user","content":"一个笼子里有鸡和兔共 35 个头、94 只脚，问鸡兔各几只？请一步步推理后给出答案。"}],"max_tokens":512,"temperature":0}' \
    > /tmp/ct32_verify_out.json 2>&1
  verdict=$(python3 - <<'PYEOF'
import json
try:
    d=json.load(open('/tmp/ct32_verify_out.json'))
    txt=d['choices'][0]['message']['content']
    open('/tmp/ct32_verify_out.txt','w').write(txt)
    fr=d['choices'][0].get('finish_reason')
    uniq=len(set(txt))/max(len(txt),1)
    print(f"OK finish={fr} len={len(txt)} uniq={uniq:.3f}" if (fr=='stop' and uniq>0.10) else f"BAD finish={fr} len={len(txt)} uniq={uniq:.3f}")
except Exception as e:
    print(f"BAD parse:{e}")
PYEOF
)
  log "验证结果：$verdict"
  case "$verdict" in OK*) touch "$STATE/s3.done";; *) log "输出异常，链停在评测前——人工复核 /tmp/ct32_verify_out.txt"; exit 1;; esac
fi

# Stage 4：评测 GPQA×5 + aime×5
if [ ! -f "$STATE/s4.done" ]; then
  run_rounds gpqa 5 "$STATE/ct32_t0" && run_rounds aime 5 "$STATE/ct32_t0" \
    && touch "$STATE/s4.done" || exit 1
  stop_vllm
fi

touch "$STATE/FINISHED"
log "=== ct32 链全部完成：GPQA×5 + aime×5 ==="
