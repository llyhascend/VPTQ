# 会话上下文交接（2026-10-17 15:20 保存：fb4 + 基线2 全链完成，结果已入文档）

> 接续 `SESSION_CONTEXT_20261015.md`。本日：容器重启环境恢复、日间链跑完
> fb4(5+5) + 基线2(5+5)、两次排障（shm 泄漏 / 验证阈值误报）。

## 一、最终结果（已写入 documents/vptq_algorithm_and_code.md §8A.2 单变量对照表）

| 配置 | GPQA | aime |
|---|---|---|
| 旧基线（moeW4A8） | 73.99 | 71.67 |
| 基线2（原始 ckpt + attnW8A8ceil + moe scale1 W8A8 直通） | **72.32**（σ1.8） | 68.00（σ3.0） |
| **fb4**（wrap VPTQ + 回退 wrap 域 mxfp4 预量化，W8A8 直通） | 71.62（σ2.7） | **75.33**（σ6.1，全表最高） |
| rf32wfb（3 轮） | 71.38 | 68.89 |

核心结论（表第 5 轴）：VPTQ 2bit 重建专家任务域追平无 VPTQ 直通基线；
aime 75.33 全场最高但 σ 大；结论④"VPTQ 预处理整体负收益"在
wrap+calibR9+回退+W8A8 直通组合下不再成立。

## 二、本日验证事实

1. **回退专家 Hadamard 融合对拍**（用户质询点）：磁盘 fallback part 与
   `mxfp4_ceilx_roundtrip(W@R)@Rᵀ` 重算 **rel=0.000000 逐位一致**
   （nowrap/直通假说均 0.1137 排除）。生成参数齐：S4' 驱动带
   `--fallback-list --fallback-mxfp4 --wrap-rotation`。
2. **真实性验证误报**：uniq>0.25 阈值对中文+LaTeX 过严（实测好输出 0.179），
   已改 0.10（防乱码语义不变：乱码 uniq≈1.0）。

## 三、事故与修复（新增踩坑）

15. **/dev/shm psm_* 泄漏**（重大，已修）：killer 杀 vllm 后泄漏 ~250MB
    psm 段打满 64MB tmpfs → 后续 vllm 实例"预热过、并发全 500"死循环
    （failed.jsonl 198 请求全 Internal Server Error）。修复：手清 + 
    `day_chain ensure_vllm` 加 `find /dev/shm -name 'psm_*' -mmin +2 -delete`。
    诊断要点：vllm 日志每次重启被覆盖，证据在 outputs/*/predictions/*_failed.jsonl。
16. 容器重启后环境恢复按持久记忆 dsv4-env-restore-checklist（vllm_ascend
    editable 构建 ~40 分钟；vptq 不 pip 装、靠 cwd=VPTQ）。
17. day_chain Stage1 的 rebuild 依赖 cwd 导入 vptq——watchdog 必须从 VPTQ
    目录启动。

## 四、链路资产

- `day_chain_fb4_base.sh` / `day_watchdog.sh`：全链跑通（FINISHED 15:12），
  含 s1-s6 全部 done；/tmp/day_chain_fb4.STOP 可停
- fb4 BF16：46/46 + expert_parts 33024；fb4 deploy 72 shards
- 基线2 deploy：`weights/DeepSeek-V4-Flash-attnw8a8ceil-moescale1`（72 shards）
- 评测产物：`benchmark/outputs/default/2026101{6,7}_*`

## 五、后续方向（表内结论延伸）

- fb4 的 aime 优势（+7.33 vs 基线2）σ 偏大，可加轮次收敛
- R9tau + 补采 + wrap 终局组合（结论⑨）；回退准则 gate×冷门度混合（结论 3）
- VPTQ 侧下一步若继续压位宽：tile 密度仍是第一杠杆（结论 2/3）
