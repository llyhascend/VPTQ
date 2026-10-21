# 会话上下文交接（2026-10-21 06:10 保存：col_tile 统一 tile 几何 + ct32 端到端完成）

> 接续 `SESSION_CONTEXT_20261017.md`。覆盖：group_num→col_tile 重构、
> fb4 MSE 校验与每层趋势图、ct32 全链（量化→部署→评测 5+5）完成。

## 一、ct32 最终结果（已入 documents §8A.2 表第 12 行 + 结论 6）

| 配置 | GPQA | aime |
|---|---|---|
| fb4（全局 group_num=128，w2 tile 32×16） | 71.62（σ2.7） | 75.33（σ6.1） |
| **ct32（col_tile=32 统一，w2 tile 32×32）** | **70.51（σ1.4）** | **74.67（σ6.1）** |

结论：统计持平（差异在 σ 内，ct32 GPQA σ 最小）；w2 码本摊销 0.5→0.25bit
（参数加权位宽 2.33→2.25）无任务域代价。ct32 产物见权重表 #12。

## 二、代码重构（group_num → col_tile，已验证）

- `config.py`：字段 group_num → col_tile（列宽直配，组数=in_f//col_tile）
- `vq.py: _num_col_groups`：两处 bounds 切分
- `run.py`：CLI --col-tile；--group-num 留 deprecated 别名（换算 4096/group）
- `rebuild_bf16.py`：调用方按 `in_f // indices.shape[1]` 自推断组数，
  **新旧产物（w2 cols=16/32）双兼容**
- 测试 8/8 绿：tests/test_uniform_tile.py（4）+ test_quantize.py（4）

## 三、fb4 MSE 校验（10-20，analysis/）

- 逐层 gate 加权 MSE：VPTQ 1.33e-4 / 回退 mxfp4 4.4e-5 / 全体 7.6e-5；
  尾层 L40-42 最高；L0 rel² 与 §8A.3 交叉验证吻合
- 产物：fb4_mse_by_layer.jsonl / .png（英文标签，容器无 CJK 字体）/
  fb4_mse_check.py（8 卡并行 killer 自愈）

## 四、链路资产（新增）

- `quantize_dsv4_experts_rf32wfbct32.sh`：ct32 量化（--col-tile 32）
- `day_chain_ct32_base.sh` + `day_watchdog_ct32.sh`：Stage0 等量化→
  rebuild→assemble→部署→验证→评测全自动；含 shm 清理防护
- `vllm_start_rf32wfbct32.sh`（9001）
- 30 分钟防死锁检查机制：CronCreate/23deed0e（已完成删除）
- 实测节奏：量化 33024 parts 5h（killer 干扰下）、rebuild 14min、
  assemble 30min、部署 36min、评测 5+5 约 50min

## 五、踩坑追加

18. tile 模式下 `centroids.shape[0]` 是 T（tile 数）而非组数——组数只能
    由 in_f//indices.shape[1] 推断（export.py 一直如此）
19. day_chain 里 pgrep 防自匹配的 `$VAR[T]` 写法与 set -u 冲突
    （unbound variable），用 `pgrep -f -- "--flag $VAR"` 代替
20. matplotlib 中文方框：容器仅 DejaVu 无 CJK；装 fonts-noto-cjk 可解

## 六、下一步候选

- ct32 σ 收敛（GPQA 加轮）或 fb4/ct32 二选一作为主线
- R9tau + 补采 + wrap 终局组合（结论⑨方向）
- 回退准则 gate×冷门度混合（结论 3 方向）
