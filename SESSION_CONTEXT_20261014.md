# 会话上下文交接（2026-10-15 上午更新：夜间事故复盘与修复就绪）

> 接续同日版本。本文档覆盖：夜间链执行结果、**QUANT 目录事故根因**、
> 修复状态（文件已改好，等待用户指令重跑）。

## ⓪ 10-15 上午最新状态（先读这里）

- **夜间链失败根因已定位并修复文件**：fb4 驱动 QUANT 误用 rf32fb 的
  `dsv4-tile32x32-rowfp8-fb-calibR9`（无 wrap），rebuild `@Rᵀ` 后
  28896 个 VPTQ parts 全部污染（实测 rel=1.42、97.6% 元素错位、
  与生产 rf32wfb deploy DIFFER）；fallback 4128 parts 不读 quant 目录、
  **无辜且正确**（实测 rel=11.7%≈wrap+cast 预期、输出 cos=0.993、
  scale≡127 全对）。
- **已改**：`quantize_dsv4_experts_rf32wfb_fb4.sh` QUANT 指向
  `...rowfp8-wrap-fb-calibR9` + 启动 guard（目录名不含 wrap 即中止）。
- **已备好未运行**：`reset_and_rerun_fb4.sh`（清 28896 污染 parts 保留
  fallback、删污染 BF16/DEPLOY、重跑 S4'+S5'，内含数量断言 4128/28896）。
  **用户指令触发才跑**。
- 夜间链（night_pipeline/night_watchdog）已全部停止；其 STATE 的 a.done
  与 fb4 deploy index 均属污染版，reset 脚本会清。
- 重跑验收判据：S4 后 VPTQ part 与生产 rf32wfb deploy **逐位一致**
  （10-15 检查 3 复跑应 IDENTICAL）。

## ⓪' 夜间链执行事实（10-15 02:00-09:10）

1. 02:00 准时起跑 ✓（全链等 02:00 版本生效）
2. Phase A fb4 S5'：05:38 完成（3.6h，killer 反复杀→看门狗自愈，最终
   72/72 + index ✓）
3. Phase B：vllm(fb4) 05:45 就绪+预热 ✓ → gpqa 前 4 轮每轮 40min 超时
   **0 预测**（576 行请求无一个 finish_reason——权重乱码不自停）；
   第 5 轮 warmup 全 500；08:48 重启 vllm 4 分钟后再崩（HCCL/AICPU
   kernel exception，libccl_kernel.so RunAicpuIndOpCommInit）
4. Phase C/D 未开始；09:10 我停掉全部组件止住空转
5. 09:3x 低成本权重验证（用户指令）：三层检查定位根因（见 ⓪）


## 一、夜间任务（用户指令原文要点）

> 凌晨2点，如果卡空闲再启动任务：S5 跑完之后评测 fb4（GPQA×5 + aime×5）；
> 之后重跑基线 attnw8a8ceil+moeW4A8（floor→ceil 修复），**moe 路由专家权重
> 改为反量化后 cast FP8 scale=1**，再评测 GPQA×5 + aime×5。

## 二、fb4 线状态（当日）

- S4' ✅：33024/33024 parts + assemble（8 进程×`--device npu:N` 绑卡）
- S5' 🔄：第一次 14:06-14:53 被 killer 杀于 63/72；17:19 看门狗接管重跑
  （msmodelslim 无断点、从头覆写；**ETA ~18:25**）
- yaml 已简化：单规则 `*ffn.experts*` → scale_one（76 行版，glob 语义已验证
  不误伤 shared_experts/attn；旧名单版备份 `.bak-1014`）

## 三、夜间链组件（全部在位）

| 文件 | 说明 |
|---|---|
| `msmodelslim/deepseek-v4-flash-attnw8a8ceil-moescale1.yaml` | 基线2：attn W8A8(ceil) + 路由专家 33024 全 scale_one，shared/attn 照基线 |
| `vllm-ascend/vllm_start_rf32wfbfb4.sh` | fb4 起服（端口 9001） |
| `vllm-ascend/vllm_start_base2.sh` | 基线2 起服 |
| `night_pipeline_fb4_base.sh` | 编排器（见下） |
| `night_watchdog.sh` | 每 5 分钟拉活编排器，FINISHED 自退 |

**编排器逻辑**（PID 见 pgrep night_pipeline）：
- Phase A（立即）：S5' 自愈跑完（判据 fb4 deploy index.json）→ a.done
- **wait 下一个 02:00 + NPU 空闲**（最多等 2h，忙则退给看门狗重试）
- Phase B：fb4 起服+预热 → gpqa×5 → aime×5 → 停 vllm
- Phase C：基线2 = 原始 ckpt `/mnt/share/weight/DeepSeek-V4-Flash-BF16` + 上述 yaml
  （msmodelslim 自愈直至 index.json）→ c.done
- Phase D：基线2 起服+预热 → gpqa×5 + aime×5 → 停 vllm → FINISHED
- 幂等：`/tmp/night_fb4_base/{a,b,c,d}.done` + 轮数按 outputs/default 新增 csv
  事实判定（t0 文件时间戳）；flock 防重入；ais_bench 单轮 40min 超时不计自动重试
- 日志：`/tmp/night_fb4_base.log`（链路）、`/tmp/aisbench_night.log`（评测）、
  `/tmp/msmodelslim_{fb4,base2}.log`

**修正记录**：初版 `today 02:00` 判断在下午会立即放行（17:19 曾提前开跑，已叫停
msmodelslim 保留）——已改为「S5 立即收尾、评测链等下一个 02:00（今天未到等今天，
已过等明天）」。

## 四、crontab 不可用

容器内无 crontab 命令（历史 cron 在宿主机）。看门狗 = 轻量 bash 循环
（项目实践：S3 量化 4.3h 期间 8 个看门狗循环全程存活，killer 主杀重计算进程）。

## 五、当日全部代码改动（早间部分，已完成）

| 文件 | 内容 |
|---|---|
| `vptq/tools/quantize/mxfp4_ceilx.py` | 独立 c7.25 量化-反量化模块 |
| `tests/test_mxfp4_ceilx.py` | 5 项单测全过 |
| `vptq/tools/quantize/rebuild_bf16.py` | `--fallback-mxfp4` 开关 + wrap 链 |
| `msmodelslim/gen_yaml_rf32wfb.py` | 简化为单 `*ffn.experts*` 规则 |
| `quantize_dsv4_experts_rf32wfb_fb4.sh` | fb4 S4'+S5' 驱动（8 进程绑卡） |

关键实测：fallback 专家 wrap rel≈11.3% vs nowrap 0%（原权重在 mxfp4 网格上），
用户知情后选定 wrap 方案。

## 六、踩坑（当日新增）

7. `today 02:00` 语义：下午启动时 now>target 立即放行——夜任务必须
   「now>=today02 则等 tomorrow02」
8. msmodelslim CLI 误写 `python3 -m msmodelslim.py` → 13 轮空转（已修，见早间版）
9. 会话死亡会带走 harness 后台任务（setsid nohup 的独立进程存活）——
   夜间链因此全部用 setsid 独立进程 + 看门狗，不依赖会话
10. S5' 一次被杀后 63/72 无断点，重跑覆写（正常行为，排程时按全程估时）

## 七、明早验收清单

1. `tail /tmp/night_fb4_base.log`：应见 FINISHED
2. fb4 评测 5+5：outputs/default 夜间新增 gpqa/aime csv 各 5 → 均值算出
3. 基线2 产物：`weights/DeepSeek-V4-Flash-attnw8a8ceil-moescale1/`（72 shards）
   + 自检路由专家 scale≡127
4. 基线2 评测 5+5 同上
5. 对照锚点：fb4 旧点 71.38/68.89（3 轮）；rf32fb=71.89/64.44；
   rf32fb-moeW8A8=69.39/73.34；旧基线（moeW4A8 未 scale1）=73.99/71.67
6. 若 FINISHED 未到：看 /tmp/night_fb4_base/ 缺哪个 done，看门狗是否存活
   （`pgrep -f night_watchdog`），必要时 `setsid nohup bash
   /mnt/share/w00608002/work/VPTQ/night_watchdog.sh &` 重拉

## 八、文档待办

- fb4 + 基线2 结果入 §8A.2 表；8A.7d 更新（wrap 11.3%/nowrap 0% 数据点与决策）
- 位宽口径：fb4 = VPTQ 专家 2bit→FP8 直通 + fallback mxfp4 预量化→FP8 直通
  （路由专家统一 8bit 存储）；基线2 = 路由专家 8bit 直通（无 VPTQ）
