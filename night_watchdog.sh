#!/bin/bash
# 夜间链看门狗：每 5 分钟检查，night_pipeline 死了就重新拉起（幂等续跑）；
# 检测到 FINISHED 后自动退出。轻量 bash 循环，killer 主要杀重计算进程。
while [ ! -f /tmp/night_fb4_base/FINISHED ]; do
  if ! pgrep -f "night_pipeline_fb4_bas[e].sh" >/dev/null; then
    echo "[wdog $(date '+%m-%d %H:%M:%S')] 拉起 night_pipeline" >> /tmp/night_fb4_base.log
    setsid nohup bash /mnt/share/w00608002/work/VPTQ/night_pipeline_fb4_base.sh \
      >> /tmp/night_fb4_base.log 2>&1 &
  fi
  sleep 300
done
echo "[wdog $(date '+%m-%d %H:%M:%S')] FINISHED，看门狗退出" >> /tmp/night_fb4_base.log
