#!/bin/bash
# 日间链看门狗：每 5 分钟拉活 day_chain（flock 防重入，幂等续跑）；
# FINISHED 或 STOP 标记出现即自退。
while true; do
  [ -f /tmp/day_chain_ct32/FINISHED ] && { echo "[dwd $(date '+%m-%d %H:%M:%S')] FINISHED，退出" >> /tmp/day_chain_ct32.log; exit 0; }
  [ -f /tmp/day_chain_ct32.STOP ] && { echo "[dwd $(date '+%m-%d %H:%M:%S')] STOP 信号，退出" >> /tmp/day_chain_ct32.log; exit 0; }
  if ! pgrep -f "day_chain_ct32_bas[e].sh" >/dev/null; then
    echo "[dwd $(date '+%m-%d %H:%M:%S')] 拉起 day_chain" >> /tmp/day_chain_ct32.log
    setsid nohup bash /mnt/share/w00608002/work/VPTQ/day_chain_ct32_base.sh >> /tmp/day_chain_ct32.log 2>&1 &
  fi
  sleep 300
done
