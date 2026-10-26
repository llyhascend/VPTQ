# 会话上下文交接（2026-10-08 11:40）

> 接续 `SESSION_CONTEXT_20260921.md`。覆盖 09-21 晚 ~ 10-08 的全部实验：
> calibR9 系六条线、回退策略、FP8 域量化、两次批量评测、文档体系。

## 一、实验结论总表（全部已入 §8A.2 大表，2026-09-23~29）

| 线 | 配置 | GPQA | aime | 状态 |
|---|---|---|---|---|
| 基线 | ModelSlim 直量化 4.25bit | 73.99 | 71.67 | 锚点 |
| **t32×16+calibR9** | R9tau Hessian | **72.22**（7轮 σ=1.3） | 72.22 | ✅ 主结果（⑨：Hessian-语料对齐 +6.7 是 GPQA 决定性杠杆） |
| t32×32 纯 | 同上 | 68.31 | 63.33 | ✅（tile 密度 −3.9；calibR9 揭示码本密度真实代价） |
| t32×32+12.5%回退 | gate 分数排序 | 70.54（6轮） | 61.11 | ✅（回退 +2.4；同位宽不敌 tile 密度） |
| t32×16+dual-norm+row-fp8 | | 71.72（含 74.24 破基线单轮） | 71.11 | ✅（无损叠加，FP8 域增益被 dual-norm 白化抵消） |
| **rf32fb（进行中）** | **t32×32+row-fp8+回退** | S3 量化 54% | — | ▶️ 见下 |

**注意**：10-08 上午曾误跑 rf16fb（t32×16+row-fp8+回退）—— 用户澄清要的是
**t32×32** 基座。rf16fb 产物完整（量化+部署）但评测被叫停，保留作备用数据点。

## 二、当前运行中：rf32fb（tile32×32 + calibR9 + row-fp8 + 12.5% gate 回退）

- 管线 `pipeline_rf32fb.sh`（10-08 09:58 起，STATE=/tmp/pipeline_rf32fb，s0-s2/s3b 预标）
- S3 量化 10:00 手动启动（跳冷却），**11:35 时 185/344 层**（~2.2 分/层，ETA ~12:30）
  - 量化脚本 `quantize_dsv4_experts_tile16_calibR9.sh`（**共用名**，当前内容为 rf32fb 版：
    ROW_TILE=32、纯 --row-fp8、OUTPUT_DIR=dsv4-tile32x32-rowfp8-fb-calibR9）—— 已五点验证
- 后续：S4 回退重建（28896 parts，`--row-tile 32 --fallback-list`）→ S5 部署
  （modelslim_rf32fb.sh）→ S6 vllm（vllm_start_rf32fb.sh）→ S7 评测 6 轮 → S8 REPORT
- **评测预警**：row-fp8 系权重有慢生成+多超长题并发特性（09-29 r1 曾 67 分钟仅
  12/198 题；10-08 上午 r1 30 分钟 2 题）—— 60 分钟单轮干预线 + 残轮隔离规则在位
- 监控 cron `:19/:49`（session-only）

## 三、关键实现资产（全部有文档章节）

| 组件 | 位置 | 文档 |
|---|---|---|
| gate 分数采集 | `vptq/tools/deepseek_v4/gate_scores.py` | §8A.7c |
| 回退名单生成 | `vptq/tools/quantize/fallback_list.py`（688+2064） | §8A.7c |
| rebuild 回退 | `rebuild_bf16.py --fallback-list`（(layer,short) 键） | §8A.7c |
| row-fp8（行峰对齐 448） | `run.py --row-fp8`（复合 row_scale=rms/fp8） | §8A.2 轴4 |
| dual-norm 算法 | `run.py --dual-norm` | §8A.7b |
| centroid-fp8（存储格式） | `vq.py` 落盘 cast e4m3fn | §3.5 |
| 并行采集 | `calib_collect_par.sh`（4分片 stream 绕锁） | 资产索引 |
| 批量评测 | `batch_eval_extra7.sh`（timeout 3600+残轮甄别） | — |

## 四、文档体系（documents/vptq_algorithm_and_code.md）

- **§8A.2 六行统一大表**（数据源/tile/回退/FP8 四轴结论）—— 新实验先入此表
- **§8A.7b/7c/7e**：dual-norm 算法 / 回退策略实现 / **weights 目录活文档总表**（新产物必须更新 §8A.7e！）
- 各线 REPORT 在 `ais_bench_logs/vptq-calibR9-*/`

## 五、踩坑全集（复用管线前必读）

1. **脚本复用名是系统性坑**（SLIM_SH/QUANT_SH/VLLM_SH 三处各踩过）：新管线变体必须独立命名；生成逻辑 `if [ ! -f ]` 会复用旧内容 → 两次污染事故（已隔离删除）
2. **S4/S5 index 判据竞态**（index 文件在组装开头被 copy）：处方 = 等真完成（46/46 或 39/39）→ `touch s4/s5.done` + kill 重启管线跳冷却
3. **pgrep 自匹配**：`bash -c` 包装进程 cmdline 含特征词 → 用字符类 `[e]` 或 `bin/` 前缀；heredoc 代码文本同样会污染
4. **修复 bug 后重启量化**：先 `pkill -9 -f "watchdog [0-9] .*expert-range"` 防双组
5. **run.py 纯 row-fp8 分支**：UnboundLocal（已修）+ rebuild col_scale=None 容错（已修）
6. **ais_bench status 文件 WARNING**：Monitor 读空文件竞态，**良性**（2105 次/轮的常态噪声）
7. vllm 环境坑已根治进模板 `vllm_start_dp.sh`：GLOO_SOCKET_IFNAME=lo（fe80:: 坑）
8. killer 频率不定（20min~数小时）：所有看门狗 + 残轮隔离 + 超时干预线是标准配置

## 六、下一步

1. rf32fb 全链完成 → REPORT → **更新 §8A.2 表 + §8A.7e 权重表**（轴4 结论补 t32×32+row-fp8+回退数据点）
2. 待定：rf16fb 备用线评测（32×16 vs 32×32 基座对比）；row-fp8 单独贡献隔离实验
3. 长期方向（§8A.2⑧⑨）：R9tau + 补采 + wrap 终局组合
