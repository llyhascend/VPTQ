#!/bin/bash
# fb4 MSE 校验看门狗：8 进程绑 8 卡分片跑 fb4_mse_check.py，
# killer 杀掉后自动重启（脚本幂等跳过已完成层）；全部完成即退出。
cd /mnt/share/w00608002/work/VPTQ
RANGES=("0:6" "6:11" "11:17" "17:22" "22:28" "28:33" "33:39" "39:43")
while true; do
  alldone=1
  for i in "${!RANGES[@]}"; do
    lo=${RANGES[$i]%%:*}; hi=${RANGES[$i]##*:}
    n_done=$(grep -c . analysis/fb4_mse_p$i.jsonl 2>/dev/null || echo 0)
    need=$((hi - lo))
    if [ "$n_done" -lt "$need" ]; then
      alldone=0
      if ! pgrep -f "fb4_mse_check.py --layer-range ${RANGES[$i]//:/ } " >/dev/null \
         && ! pgrep -f "fb4_mse_check.py --layer-range ${RANGES[$i]} " >/dev/null; then
        echo "[wd $(date '+%H:%M:%S')] 重启分片 $i (${RANGES[$i]}, 已 $n_done/$need)"
        ASCEND_RT_VISIBLE_DEVICES=$i setsid nohup python3 analysis/fb4_mse_check.py \
          --layer-range "${RANGES[$i]}" --out analysis/fb4_mse_p$i.jsonl \
          >> analysis/fb4_mse_p$i.log 2>&1 &
      fi
    fi
  done
  [ "$alldone" = "1" ] && { echo "[wd $(date '+%H:%M:%S')] 43 层全部完成"; break; }
  sleep 120
done
cat analysis/fb4_mse_p*.jsonl | sort -t'"' -k4 -n > analysis/fb4_mse_by_layer.jsonl
echo "[wd] 已合并 analysis/fb4_mse_by_layer.jsonl"
