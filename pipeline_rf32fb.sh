#!/bin/bash
# =============================================================================
# calibR9 全管线看门狗（2026-09-21）
#   dualnorm 评测完成 → 停 vllm → calibR9 Hessian 基线采集 → inv 补全
#   → tile32×16 无旋转量化（calibR9 Hessian）→ rebuild_bf16 → msmodelslim
#   → vllm(calibR9 部署权重) → aisbench 6 轮 → 汇总 REPORT
#
# 原则：
#   - 阶段状态落盘 $STATE/*.done，幂等续跑（被杀后重启脚本从断点继续）
#   - 关键交接点保守冷却（用户 2026-09-21 指示）：
#     静默 20min → 等环境任务结束 → 连续空闲 20min → 才启动下一阶段
#   - 用法: setsid nohup bash pipeline_calibR9.sh &
#   - 停止: touch /tmp/pipeline_calibR9.STOP
# =============================================================================
set -u
cd /mnt/share/w00608002/work/VPTQ

# ------------------------------ 路径常量 ------------------------------------
CKPT=/mnt/share/weight/DeepSeek-V4-Flash-BF16
HESS=/mnt/share/w00608002/weights/hessians/DeepSeek-V4-Flash-BF16-calibR9
QUANT=/mnt/share/w00608002/weights/quant/dsv4-tile32x32-rowfp8-fb-calibR9
BF16=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-vptq-calibR9-rf32fb-bf16
DEPLOY=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-vptq-calibR9-rf32fb-attnw8a8-moew4a8
QUANT_SH=quantize_dsv4_experts_tile16_calibR9.sh          # 由 tile16.sh 生成
SLIM_SH=/mnt/share/w00608002/msmodelslim/modelslim_rf32fb.sh
VLLM_SH=/mnt/share/w00608002/work/vllm-ascend/vllm_start_rf32fb.sh
VLLM_DUALNORM=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-vptq-dualnorm-tile16-attnw8a8-moew4a8

STATE=/tmp/pipeline_rf32fb
LOG=/tmp/pipeline_rf32fb.log
BENCH=/mnt/share/w00608002/benchmark
OUTBASE=$BENCH/outputs/default
STOP=/tmp/pipeline_rf32fb.STOP
mkdir -p "$STATE"

log() { echo "[pipe $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
check_stop() { [ -f "$STOP" ] && { log "STOP 信号，退出"; exit 0; }; }

# ------------------------------ 环境判忙 ------------------------------------
env_has_tasks() {
    # 特征长任务进程（正则加字符类避免 pgrep 自匹配）
    if pgrep -f "collect_hessia[n]|vptq.tools.quantiz[e]|rebuild_bf1[6]|compute_invers[e]|calib_collec[t]|msmodelsli[m] quant" >/dev/null 2>&1; then
        log "  忙: 特征任务进程"
        return 0
    fi
    # ais_bench 评测进程
    if pgrep -f "bin/ais_bench --model[s]" >/dev/null 2>&1; then
        log "  忙: ais_bench 在跑"
        return 0
    fi
    # 任意 NPU 卡 util>0（vllm 空闲时 8 卡 util 全 0）
    if npu-smi info 2>/dev/null | awk -F'|' '/NA/ {split($5,a," "); if (a[1]+0>0) f=1} END{exit f?0:1}'; then
        log "  忙: NPU util>0"
        return 0
    fi
    return 1
}

# 保守冷却：静默 20min → 等任务结束 → 连续空闲 20min
cooldown() {
    log "冷却开始: 静默 20 分钟"
    sleep 1200
    local idle=0
    while [ $idle -lt 1200 ]; do
        check_stop
        if env_has_tasks; then
            idle=0; sleep 300
        else
            sleep 60; idle=$((idle+60))
        fi
    done
    log "冷却完成: 环境已连续空闲 20 分钟"
}

# 所有 NPU 卡 HBM 回落（vllm 停止后判断显存释放）
hbm_released() {
    local busy
    busy=$(npu-smi info 2>/dev/null | grep -oE "[0-9]+ */ *98304" | \
           awk '{gsub(/ /,"",$1); if ($1+0 > 10000) c++} END{print c+0}')
    [ "$busy" -eq 0 ]
}

# =============================================================================
# S0: 等 dualnorm 评测 6 轮完成 → 冷却 → 停 vllm → 等 HBM 释放
# =============================================================================
s0_wait_eval() {
    [ -f "$STATE/s0.done" ] && return 0
    # 独立锁存评测基线（不依赖看门狗日志文本——旧日志含错误行）
    local bg=0 ba=0 csv
    for csv in "$OUTBASE"/*/summary/summary_*.csv; do
        [ -f "$csv" ] || continue
        if grep -qi gpqa "$csv" 2>/dev/null && grep -q "accuracy,gen,[0-9]" "$csv" 2>/dev/null; then bg=$((bg+1)); fi
        if grep -qi aime "$csv" 2>/dev/null && grep -q "accuracy,gen,[0-9]" "$csv" 2>/dev/null; then ba=$((ba+1)); fi
    done
    echo "$bg" > "$STATE/s0_base_g"; echo "$ba" > "$STATE/s0_base_a"
    log "S0: 等待 dualnorm 评测（基线 g=$bg a=$ba，目标各新增 3）"
    local ng na
    while true; do
        check_stop
        ng=0; na=0
        for csv in "$OUTBASE"/*/summary/summary_*.csv; do
            [ -f "$csv" ] || continue
            if grep -qi gpqa "$csv" 2>/dev/null && grep -q "accuracy,gen,[0-9]" "$csv" 2>/dev/null; then ng=$((ng+1)); fi
            if grep -qi aime "$csv" 2>/dev/null && grep -q "accuracy,gen,[0-9]" "$csv" 2>/dev/null; then na=$((na+1)); fi
        done
        if [ $((ng-bg)) -ge 3 ] && [ $((na-ba)) -ge 3 ] && \
           ! pgrep -f "bin/ais_bench --model[s]" >/dev/null 2>&1; then
            break
        fi
        # 自愈：评测看门狗死了且未达标 → 重启它（幂等补缺）
        if ! pgrep -f "watchdog_eval_dualnorm" >/dev/null 2>&1 && \
           ! pgrep -f "bin/ais_bench --model[s]" >/dev/null 2>&1; then
            log "S0: 评测看门狗不在且未达标（新增 g=$((ng-bg)) a=$((na-ba))），重启"
            ( cd "$BENCH" && setsid nohup bash watchdog_eval_dualnorm.sh > /dev/null 2>&1 & )
            sleep 120
        fi
        sleep 300
    done
    log "S0: dualnorm 评测 6 轮完成（独立计数 g=+$((ng-bg)) a=+$((na-ba))）"
    cooldown                      # 冷却后停 vllm（不再有人用服务）
    log "S0: 停止 vllm（dualnorm 服务）"
    pkill -f "vllm serve" 2>/dev/null; sleep 10
    pkill -9 -f "VLLM:[:]:" 2>/dev/null
    pkill -9 -f "EngineCor[e]" 2>/dev/null
    pkill -9 -f "vllm serve" 2>/dev/null
    local n=0
    until hbm_released; do
        sleep 30; n=$((n+30))
        [ $n -gt 1800 ] && { log "S0: 警告 HBM 30min 未释放"; break; }
    done
    log "S0: HBM 已释放（或超时）"
    touch "$STATE/s0.done"
}

# =============================================================================
# S1: calibR9 Hessian 基线采集（calib_collect.sh 自带 attempt+NPU 空闲检查；
#     外层再加保守冷却兜底）
# =============================================================================
s1_calib() {
    [ -f "$STATE/s1.done" ] && return 0
    local attempt=0
    while true; do
        check_stop
        if grep -aq "CALIB COLLECTION DONE" "$HESS/collect.log" 2>/dev/null; then
            break
        fi
        if ! pgrep -f "calib_collec[t].sh" >/dev/null 2>&1; then
            attempt=$((attempt+1))
            [ $attempt -gt 60 ] && { log "S1: 重试耗尽"; return 1; }
            log "S1: 启动 calib_collect.sh（attempt $attempt）"
            setsid nohup bash calib_collect.sh > /dev/null 2>&1 &
            sleep 120
        fi
        sleep 300
        # calib_collect.sh 自身退出且未完成 → 保守冷却后由下轮循环重启
        if ! pgrep -f "calib_collec[t].sh" >/dev/null 2>&1 && \
           ! grep -aq "CALIB COLLECTION DONE" "$HESS/collect.log" 2>/dev/null; then
            log "S1: calib_collect.sh 死亡且未完成，进入保守冷却"
            cooldown
        fi
    done
    log "S1: calibR9 Hessian 采集完成"
    touch "$STATE/s1.done"
}

# =============================================================================
# S2: compute_inverse 补 inv（4 分片并行，CPU，幂等）
# =============================================================================
s2_inv() {
    [ -f "$STATE/s2.done" ] && return 0
    # 11 分片（3-4 层/片）：网络盘单流读 ~30MB/s 是瓶颈，提读并发；
    # 每进程限 24 核（11×24=264<384），Cholesky 计算被 I/O 等待完全遮盖
    local ranges="0:4 4:8 8:12 12:16 16:19 19:23 23:27 27:31 31:35 35:39 39:43"
    local n_ranges=11
    local r done_n attempt=0
    while true; do
        check_stop
        done_n=0
        for r in $ranges; do
            grep -aq "all done" "$HESS/inv_$r.log" 2>/dev/null && done_n=$((done_n+1))
        done
        [ $done_n -eq $n_ranges ] && break
        if ! pgrep -f "compute_invers[e]" >/dev/null 2>&1; then
            attempt=$((attempt+1))
            [ $attempt -gt 40 ] && { log "S2: 重试耗尽"; return 1; }
            log "S2: 启动缺失 inv 分片（$done_n/$n_ranges 已完成）"
            for r in $ranges; do
                grep -aq "all done" "$HESS/inv_$r.log" 2>/dev/null && continue
                OMP_NUM_THREADS=24 setsid nohup python3 -m vptq.tools.hessian.compute_inverse \
                    --input-dir "$HESS" --layer-range "$r" --device cpu \
                    >> "$HESS/inv_$r.log" 2>&1 &
            done
        fi
        sleep 300
    done
    log "S2: compute_inverse $n_ranges/$n_ranges 分片 all done"
    touch "$STATE/s2.done"
}

# =============================================================================
# S3: tile32×16 无旋转量化（calibR9 Hessian，8 shard + 内置看门狗）
# =============================================================================
s3_quant() {
    [ -f "$STATE/s3.done" ] && return 0
    # 生成 calibR9 专用量化脚本（仅改 HESSIAN_DIR / OUTPUT_DIR）
    if [ ! -f "$QUANT_SH" ]; then
        sed -e "s|^HESSIAN_DIR=.*|HESSIAN_DIR=$HESS|" \
            -e "s|^OUTPUT_DIR=.*|OUTPUT_DIR=$QUANT|" \
            -e "s|--scope experts|--scope experts --row-fp8|" \
            -e "s|^ROW_TILE=16.*|ROW_TILE=32          # tile 行高（32x32 基座）|" \
            quantize_dsv4_experts_tile16.sh > "$QUANT_SH"
        log "S3: 已生成 $QUANT_SH"
    fi
    local attempt=0
    while true; do
        check_stop
        local done_shards=0
        for i in 0 1 2 3 4 5 6 7; do
            grep -aq "report written\|shard complete" "$QUANT/logs/shard$i.log" 2>/dev/null \
                && done_shards=$((done_shards+1))
        done
        if [ $done_shards -eq 8 ]; then
            break
        fi
        if ! pgrep -f "vptq.tools.quantiz[e].run" >/dev/null 2>&1; then
            attempt=$((attempt+1))
            [ $attempt -gt 40 ] && { log "S3: 重试耗尽（$done_shards/8 shard 完成）"; return 1; }
            log "S3: 量化 shard 不在（$done_shards/8 完成），保守冷却后 start"
            cooldown
            log "S3: bash $QUANT_SH start（幂等续跑）"
            bash "$QUANT_SH" start >> "$LOG" 2>&1
            sleep 120
        fi
        sleep 300
    done
    log "S3: 量化 8/8 shard 完成"
    touch "$STATE/s3.done"
}

# =============================================================================
# S4: rebuild_bf16（覆盖式重跑，attempt + 冷却）
# =============================================================================
s4_bf16() {
    [ -f "$STATE/s4.done" ] && return 0
    local attempt=0 n_parts
    # 阶段 1：专家重建 pass（--assemble 是互斥开关：不传则只跑本 pass）
    # 完成判据：expert_parts 33024 个文件齐全（256 专家 × 3 权重 × 43 层）
    while true; do
        check_stop
        n_parts=$(ls "$BF16/expert_parts" 2>/dev/null | wc -l)
        [ "$n_parts" -ge 28896 ] && break
        if ! pgrep -f "rebuild_bf1[6]" >/dev/null 2>&1; then
            attempt=$((attempt+1))
            [ $attempt -gt 40 ] && { log "S4: 重建重试耗尽"; return 1; }
            log "S4: 启动 rebuild 专家 pass（attempt $attempt，parts=$n_parts/28896）"
            setsid nohup python3 -m vptq.tools.quantize.rebuild_bf16 \
                --quant-dir "$QUANT" --ckpt "$CKPT" --output-dir "$BF16" \
                --group-num 128 --row-tile 32 --device npu \
                --fallback-list "$QUANT/fallback_list.json" \
                >> /tmp/rebuild_rf16fb.log 2>&1 &
            sleep 300
        else
            sleep 300
        fi
    done
    log "S4: 专家重建 pass 完成（28896 parts = 33024 - 4128 回退）"
    # 阶段 2：组装 pass（--assemble）：流式过 46 个源 shard 换入专家 parts
    # 并写最终模型 + index.json。网络盘读写 bound，预计数小时；断点续跑
    # （已写 shard 直接跳过）。
    attempt=0
    while true; do
        check_stop
        [ -f "$BF16/model.safetensors.index.json" ] && break
        if ! pgrep -f "rebuild_bf1[6]" >/dev/null 2>&1; then
            attempt=$((attempt+1))
            [ $attempt -gt 40 ] && { log "S4: 组装重试耗尽"; return 1; }
            log "S4: 启动 rebuild 组装 pass --assemble（attempt $attempt）"
            setsid nohup python3 -m vptq.tools.quantize.rebuild_bf16 \
                --quant-dir "$QUANT" --ckpt "$CKPT" --output-dir "$BF16" \
                --assemble \
                >> /tmp/rebuild_calibR9.log 2>&1 &
            sleep 300
        else
            sleep 300
        fi
    done
    log "S4: BF16 重建+组装完成"
    touch "$STATE/s4.done"
}

# =============================================================================
# S5: msmodelslim 部署（attnW8A8ceil+moeW4A8，单卡 npu:2）
# =============================================================================
s5_deploy() {
    [ -f "$STATE/s5.done" ] && return 0
    if [ ! -f "$SLIM_SH" ]; then
        sed -e "s|^MODEL_PATH=.*|MODEL_PATH=$BF16|" \
            -e "s|^SAVE_PATH=.*|SAVE_PATH=$DEPLOY|" \
            /mnt/share/w00608002/msmodelslim/modelslim.sh > "$SLIM_SH"
        log "S5: 已生成 $SLIM_SH"
    fi
    local attempt=0
    while true; do
        check_stop
        if [ -f "$DEPLOY/quant_model_weights.safetensors.index.json" ] && ! pgrep -f "msmodelsli[m] quant" >/dev/null 2>&1; then
            break
        fi
        if ! pgrep -f "msmodelsli[m] quant" >/dev/null 2>&1; then
            attempt=$((attempt+1))
            [ $attempt -gt 40 ] && { log "S5: 重试耗尽"; return 1; }
            log "S5: 启动 msmodelslim（attempt $attempt）"
            setsid nohup bash "$SLIM_SH" >> /tmp/modelslim_calibR9.log 2>&1 &
            sleep 300
        else
            sleep 300
        fi
        if ! pgrep -f "msmodelsli[m] quant" >/dev/null 2>&1 && [ ! -f "$DEPLOY/index.json" ]; then
            log "S5: msmodelslim 死亡未完成，保守冷却"
            cooldown
        fi
    done
    log "S5: 部署权重就绪"
    touch "$STATE/s5.done"
}

# =============================================================================
# S6: 起 vllm（calibR9 部署权重）+ 真实请求预热
# =============================================================================
s6_vllm() {
    [ -f "$STATE/s6.done" ] && return 0
    if [ ! -f "$VLLM_SH" ]; then
        sed "s|$VLLM_DUALNORM|$DEPLOY|" \
            /mnt/share/w00608002/work/vllm-ascend/vllm_start_dp.sh > "$VLLM_SH"
        log "S6: 已生成 $VLLM_SH"
    fi
    local attempt=0
    while true; do
        check_stop
        if curl -s -m 10 http://127.0.0.1:9001/v1/models 2>/dev/null | grep -q "$DEPLOY"; then
            break
        fi
        if ! pgrep -f "vllm serv[e]" >/dev/null 2>&1; then
            attempt=$((attempt+1))
            [ $attempt -gt 40 ] && { log "S6: 重试耗尽"; return 1; }
            log "S6: 启动 vllm（attempt $attempt）"
            setsid nohup bash "$VLLM_SH" > /tmp/vllm_calibR9.log 2>&1 &
            sleep 300
        else
            sleep 300
        fi
    done
    # 真实请求预热（§8A.2 教训：health 200 ≠ worker 就绪，未预热整轮 500）
    log "S6: 服务就绪，预热请求"
    local warm=0 try=0
    until [ $warm -ge 2 ]; do
        try=$((try+1)); [ $try -gt 60 ] && { log "S6: 预热超时"; return 1; }
        if curl -s -m 300 http://127.0.0.1:9001/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{"model":"dsv","messages":[{"role":"user","content":"ping"}],"max_tokens":8}' \
            | grep -q '"finish_reason"'; then
            warm=$((warm+1)); log "S6: 预热成功 $warm/2"
        else
            warm=0; sleep 60
        fi
    done
    log "S6: vllm calibR9 服务就绪且预热完成"
    touch "$STATE/s6.done"
}

# =============================================================================
# S7: aisbench 6 轮（gpqa/aime 各 3）+ 汇总
# =============================================================================
s7_bench() {
    [ -f "$STATE/s7.done" ] && return 0
    # 锁基线：启动时刻已有轮数
    local base_g=0 base_a=0 csv
    for csv in "$OUTBASE"/*/summary/summary_*.csv; do
        [ -f "$csv" ] || continue
        if grep -qi gpqa "$csv" 2>/dev/null && grep -q "accuracy,gen,[0-9]" "$csv" 2>/dev/null; then base_g=$((base_g+1)); fi
        if grep -qi aime "$csv" 2>/dev/null && grep -q "accuracy,gen,[0-9]" "$csv" 2>/dev/null; then base_a=$((base_a+1)); fi
    done
    log "S7: aisbench 基线 gpqa=$base_g aime=$base_a"
    echo "$base_g" > "$STATE/base_g"; echo "$base_a" > "$STATE/base_a"

    count_new() {
        local n=0 c
        for c in "$OUTBASE"/*/summary/summary_*.csv; do
            [ -f "$c" ] || continue
            if grep -qi "$1" "$c" 2>/dev/null && grep -q "accuracy,gen,[0-9]" "$c" 2>/dev/null; then n=$((n+1)); fi
        done
        echo $(( n - $([ "$1" = gpqa ] && cat "$STATE/base_g" || cat "$STATE/base_a") ))
    }

    local ds dsarg
    while true; do
        check_stop
        [ "$(count_new gpqa)" -ge 3 ] && [ "$(count_new aime)" -ge 3 ] && break
        if pgrep -f "bin/ais_bench --model[s]" >/dev/null 2>&1; then
            sleep 300; continue
        fi
        if [ "$(count_new gpqa)" -lt 3 ]; then ds=gpqa; dsarg=gpqa_gen_0_shot_cot_chat_prompt.py
        else ds=aime; dsarg=aime2024_gen_0_shot_chat_prompt.py; fi
        # vllm 看护（killer 会周期性杀服务；死则重启+预热再开轮）
        if ! curl -s -m 5 http://127.0.0.1:9001/v1/models 2>/dev/null | grep -q '"id"'; then
            log "S7: vllm 不在，重启服务"
            ( cd /mnt/share/w00608002/work/vllm-ascend && setsid nohup bash "$VLLM_SH" > /tmp/vllm_rowfp8.log 2>&1 & )
            local wv=0
            until curl -s -m 5 http://127.0.0.1:9001/v1/models 2>/dev/null | grep -q '"id"'; do
                sleep 60; wv=$((wv+60)); [ $wv -gt 2400 ] && break
            done
            # 真实请求预热验证（API 通 != worker 就绪，失败开轮=空轮）
            until curl -s -m 300 http://127.0.0.1:9001/v1/chat/completions \
                -H "Content-Type: application/json" \
                -d '{"model":"dsv","messages":[{"role":"user","content":"ping"}],"max_tokens":4}' \
                2>/dev/null | grep -q '"finish_reason"'; do
                sleep 60; wv=$((wv+60)); [ $wv -gt 3000 ] && break
            done
        fi
        log "S7: 启动 $ds 轮（新增 gpqa=$(count_new gpqa) aime=$(count_new aime)）"
        ( cd "$BENCH" && setsid nohup ais_bench --models vllm_api_general_chat \
            --datasets "$dsarg" --mode all --dump-eval-details --merge-ds \
            >> /tmp/aisbench_calibR9.log 2>&1 & )
        sleep 120
        # 轮间不冷却（用户 09-23 指令跳过）；被杀由主循环 5min 内重启续跑
    done
    log "S7: 6 轮全部完成（gpqa=$(count_new gpqa) aime=$(count_new aime)）"
    touch "$STATE/s7.done"
}

# =============================================================================
# S8: 汇总 REPORT
# =============================================================================
s8_report() {
    [ -f "$STATE/s8.done" ] && return 0
    local dest=/mnt/share/w00608002/work/vllm-ascend/ais_bench_logs/vptq-calibR9-rf32fb-attnw8a8-moew4a8-$(date +%Y%m%d)
    mkdir -p "$dest/bench_outputs"
    local base_g=$(cat "$STATE/base_g") base_a=$(cat "$STATE/base_a")
    local g=0 a=0 csv
    for csv in "$OUTBASE"/*/summary/summary_*.csv; do
        [ -f "$csv" ] || continue
        if grep -qi gpqa "$csv" 2>/dev/null && grep -q "accuracy,gen,[0-9]" "$csv" 2>/dev/null; then
            g=$((g+1))
            if [ $g -gt $base_g ] && [ $((g - base_g)) -le 3 ]; then cp "$csv" "$dest/bench_outputs/"; fi
        fi
        if grep -qi aime "$csv" 2>/dev/null && grep -q "accuracy,gen,[0-9]" "$csv" 2>/dev/null; then
            a=$((a+1))
            if [ $a -gt $base_a ] && [ $((a - base_a)) -le 3 ]; then cp "$csv" "$dest/bench_outputs/"; fi
        fi
    done
    {
        echo "# rf32fb 线评测报告（tile32×32 + 纯 row-fp8 + 12.5% gate 回退 + calibR9 Hessian）"
        echo
        echo "- 生成: $(date '+%F %T')"
        echo "- Hessian: $HESS（基线方式，calib_corpus_R9tau.jsonl）"
        echo "- 量化: $QUANT"
        echo
        echo "## 各轮成绩"
        for csv in "$dest"/bench_outputs/*.csv; do
            [ -f "$csv" ] && tail -n +2 "$csv" | awk -F, '{print $1, $5}'
        done
        echo
        echo "## 对照（历史）"
        echo "- 无 VPTQ 基线: GPQA 73.99 / aime 71.67"
        echo "- tile16+旧Hessian: GPQA 65.53 / aime 71.11"
        echo "- topup1024: GPQA 65.87 / aime 74.44"
        echo "- wrap+rpmix: GPQA 66.43 / aime 73.33"
        echo "- dualnorm: 见 dualnorm 评测看门狗日志（本次同时段产出）"
    } > "$dest/REPORT.md"
    log "S8: 报告已写 $dest/REPORT.md"
    touch "$STATE/s8.done"
}

# ------------------------------ 主流程 --------------------------------------
# batch7 等待段已删除（2026-10-09：batch7 线早已结束，其 log 随环境重置丢失，
# 且 s0~s3 已 touch done，直接从 S4 起跑）
log "===== rf32fb 管线启动（pid $$）：t32×32 + 纯 row-fp8 + 12.5% 回退 + calibR9 ====="
s0_wait_eval || exit 1
s1_calib     || exit 1
s2_inv       || exit 1
s3_quant     || exit 1
s4_bf16      || exit 1
s5_deploy    || exit 1
s6_vllm      || exit 1
s7_bench     || exit 1
s8_report
log "===== 全管线完成 ====="
