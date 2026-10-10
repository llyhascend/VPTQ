# 会话上下文交接（2026-10-09 14:00）

> 接续 `SESSION_CONTEXT_20261008.md`。覆盖：新环境恢复 + **row-fp8 折回 bug 的发现/修复/重跑**。

## 一、环境恢复（新环境，/tmp 与 site-packages 被重置）

- **vptq**：`pip install -e .` 在无 CUDA 机器必然失败（`nvcc_threads()` Python 层 TypeError，CMakeLists 无跳过开关）→ **改用 .pth**：`echo /mnt/share/w00608002/work/VPTQ > $(python3 -c "import site;print(site.getsitepackages()[0])")/vptq_dev.pth`，功能等效已验证
- **msmodelslim** 26.1.0 editable 重装 ✅；**vllm_ascend** `0.23.1.dev15+g31491834e` editable 重装 ✅（预装版不含本地 DeepSeek-V4 适配提交，必须重装；先 `rm -rf csrc/build`）；vllm 0.23.0+empty 沿用预装
- **HCCL 8 卡 ALL PASS**（`/tmp/hccl_sanity_check.py` 已重建，torchrun 8 卡 allreduce/allgather/broadcast）——新环境无 09-21 的 EI0014 故障

## 二、重大 bug：纯 row-fp8 分支 w1/w3 折回丢失（已修 + 重跑）

**现象**（数值铁证，experts.0 非回退矩阵）：

| 矩阵 | row_scale 落盘 | Q/W 实测 rms 比 | 理论放大 448/peak |
|---|---|---|---|
| w1/w3 | None | **4748~4755** | 4779 |
| w2 | 1/fp | 0.975 | — |

**根因**：`run.py` 旧 L345 只在 `dual_norm` 时把 row_scale 写入 QuantizedWeight；纯 `--row-fp8` 分支的 fp1/fp3 被丢弃（w2 分支 L395 正确存了 1/fp）。rebuild（`rebuild_bf16.py:160`）只认 `qw.row_scale` → w1/w3 重建值大 ~4755 倍，模型输出乱码。

**影响面**：
- **rf32fb**（S3 产物 344/344 层，10-08 记录"完成"）——坏，已重跑
- **rf16fb**（产物完整+已部署）——**同样是废数据**；若评测会接近崩溃
- **10-08 记录的"row-fp8 系慢生成+多超长题"（30 分钟 2 题）极可能就是这个 bug 的症状，不是特性**——乱码不自停。dnrowfp8 线走 dual_norm 复合路径不受影响（71.72 分佐证）
- 量化码本/索引本身在放大域正确执行（Q/(448/peak)≈0.997），proxy_error 口径自洽——bug 只在落盘折回因子

**修复**（run.py 已改）：
```python
if dual_norm:
    qw1.row_scale, qw1.col_scale = s1s, d_joint
    ...
elif row_fp8:                      # 新增：与 w2 分支同语义
    qw1.row_scale, qw1.col_scale = 1.0 / fp1, None
    qw3.row_scale, qw3.col_scale = 1.0 / fp3, None
```
冒烟（1 层 2 专家）+ 端到端验证：折回后 Q/W rms 比中位 0.997/0.975/0.996 ✅

**产物处置**：
- 旧 bug 产物整体 mv 到 `weights/quant/dsv4-tile32x32-rowfp8-fb-calibR9-bugged-1010/`（538G，**新产物验证 OK 后删除**）
- `fallback_list.json`（12.5%，688 full + 2064 down_only）与量化无关，保留复用
- 旧 `logs/` 一并 mv 走（**S3 判据 grep shard log，旧 log 会让管线误判完成**）

## 三、当前运行中：rf32fb 管线（S4 起，18:19 启动）

- **S3 重跑已完成**（13:44→16:27，344/344 层，8/8 shard）：
  - row_scale 全验证：w1/w2/w3 三类矩阵 × 3 分片抽查，`row_scale`=理论 peak/448（相对差 7e-8）、centroids 峰值恰好 448（FP8 放大域证据）
  - proxy_error 中位 0.01672 / 均值 0.01960，与 bug 前旧产物**完全一致**（量化等价性证明）
  - 538G bug 备份已删（`quant_report.bugged-1010.ref.json` 留档在 QUANT 目录）
- **管线重启要点（本次实际操作）**：
  - 主流程原有 batch7 等待段（`/tmp/batch7.log` 含"完成"）——**log 已随环境重置丢失会死等，已删除该段**（用户指示直接从 S4 起）
  - `touch /tmp/pipeline_rf32fb/{s0,s1,s2,s3}.done` 后 `setsid nohup bash pipeline_rf32fb.sh &`（pid 79505）
  - S8 陈旧文案已顺手修正：REPORT 目录改为 `vptq-calibR9-rf32fb-attnw8a8-moew4a8-$(date)`（原 tile16 命名会污染 §8A.7e 归档）
- **S4 已完成**（专家 pass 18:19→18:54 28896/28896；组装 18:54→19:29 46/46，28896 swapped / 6124 copied）
- **S4/S5 index 判据竞态复现**（踩坑 #2）：管线 18:59 误判 S4 完成 → S5 attempt 1 读缺 shard 的 BF16 崩溃（零污染）→ 保守冷却。处方执行：等 46/46 真完成 → kill 管线重启跳冷却 → **S5 msmodelslim 19:31 正常起跑**。附教训：pkill 与含目标字符串的命令复合会自匹配自杀，pkill 须单独执行
- **S5 已完成**（attempt 2 于 21:29:49 SUCCESS，43/43 层，39 shards 163.6GB，index 零缺失）。
  踩坑 #4：**S5 冷却分支判据文件名错误**——`$DEPLOY/index.json` 恒不存在（真名 `quant_model_weights.safetensors.index.json`）→ msmodelslim 成功退出后仍被判"死亡未完成"进冷却。处置：touch s5.done + 重启跳过。**复用管线前须修此判据**
- **S6-S8 全部完成（22:58:57 全管线收官）**：
  - vllm 22:10 就绪（起服 10 分钟），S7 六轮 22:10-22:58（~7-8 分/轮，**评测速度恢复正常=慢生成确系折回 bug 症状**）
  - **rf32fb 最终成绩：GPQA 均值 71.89（75.25/69.70/70.71，单轮 75.25 破基线 73.99）；aime 均值 64.44（73.33/60/60）**
  - REPORT：`ais_bench_logs/vptq-calibR9-rf32fb-attnw8a8-moew4a8-20261009/`
  - 结论要点：t32×32 基座上 row-fp8+回退 GPQA +3.6（vs t32×32 纯 68.31）但仍不敌 t32×16 密度（72.22）；vs t32×32+回退（无 row-fp8）GPQA +1.35 / aime +3.33 = row-fp8 小幅正贡献；aime 3 轮两轮 60 偏弱
  - **待办**：§8A.2 大表 + §8A.7e 权重表更新（轴4 数据点）；vllm 服务仍在跑（8 卡占用，是否停待用户定）

## 四、后续注意（S4 运行后阅读）

1. S6 vllm 起服务前 HCCL 已验证 PASS（8 卡 allreduce/allgather/broadcast），无需再测
2. **修正 10-08 预警**：row-fp8"慢生成"极可能是折回 bug 症状，修复后评测速度应恢复正常——若仍慢再查真性能问题
3. 若再次环境重置（/tmp 清空）：状态目录丢失需按磁盘判据重建 `touch sX.done`；batch7 等待段已删不再挡路
4. S7 评测 60 分钟单轮干预线仍在管线设计中（残轮甄别规则）

## 五、文档体系待办

- rf32fb 全链完成后：REPORT → 更新 §8A.2 表 + §8A.7e 权重表（轴4 数据点）
- §8A.7c/文档中补记本 bug（纯 row-fp8 折回丢失）与修复
- rf16fb 若要作对比数据点，需用修复后代码重跑（旧产物在 `dsv4-tile32x16-rowfp8-fb-calibR9`，同样是坏的）
