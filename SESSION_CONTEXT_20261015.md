# 会话上下文交接（2026-10-15 11:40 保存，全面待命——用户指令：不启动服务，听指令再跑）

> 接续 `SESSION_CONTEXT_20261014.md`。覆盖：夜间评测事故复盘、QUANT 路径修复、
> 重建验证通过、assemble 半途中断、**全部自动化已停**。

## ⚠️ 当前状态（重要）

**一切自动化进程已停（11:35 用户指令）**：verify_launch 监控脚本已删除、fb4 驱动
与 S5 attempt 循环已杀、无 vllm/msmodelslim/rebuild。NPU 设备侧有一个残留 PID
2271006（宿主机侧，容器不可杀，HBM ~1.2G，等驱动回收即可，不阻塞）。
**后续任何启动都等用户明确指令。**

## 一、夜间事故复盘（10-14 夜 → 10-15 晨）

**根因**：fb4 驱动最初 QUANT 误指 rf32fb 无 wrap 的
`dsv4-tile32x32-rowfp8-fb-calibR9` → rebuild --wrap-rotation 给 Q(W) 乘 Rᵀ
→ 28896 VPTQ 矩阵全乱码 → 生成不可自停（4 轮 40min 超时 0 预测）→ vllm 后被
HCCL AICPU 异常拖崩（warmup 全 500）。修复：驱动改指
`dsv4-tile32x32-rowfp8-wrap-fb-calibR9` + `case *wrap*` 防再犯护栏（脚本头注释）。

**连带发现**：assemble 判据缺陷——rebuild 先拷 config/index.json 再流式写 46
shards，killer 杀于 28/46 时 index.json 已在 → 驱动误判 S4 完成 → S5 读缺失
shard 崩（attempt 1-29 全同因）。**已修**：s4 判据加固为「46 个 model-*.safetensors
齐全」（10-15 11:0x 改入驱动）。

## 二、fb4 线资产状态（磁盘事实）

| 项 | 状态 |
|---|---|
| S4' expert parts | ✅ 33024/33024（**正确 wrap QUANT，数值对拍验证 rel=0.0017 vs 错误源 1.418**） |
| S4' assemble | ⚠️ **28/46 shards**（10:34 killer 杀断，缺 29-46；resume 补齐 ~20 分钟） |
| S5' deploy | ❌ 空（0/72；assemble 补完后跑，~50 分钟） |
| 验证 | 未做（原计划：单请求鸡兔同笼题检查输出连贯性再评测） |

BF16: `weights/DeepSeek-V4-Flash-vptq-calibR9-rf32wfbfb4-bf16`（28 shards + parts）
DEPLOY: `weights/DeepSeek-V4-Flash-vptq-calibR9-rf32wfbfb4-attnw8a8-moew8a8`（空）

## 三、夜间链资产（全部未运行）

- `night_pipeline_fb4_base.sh` / `night_watchdog.sh`：保留文件，**进程已停**——
  其 Phase A 判据还是旧 index.json 版且与 fb4 驱动重叠，复用前需 review
- 基线2 yaml：`deepseek-v4-flash-attnw8a8ceil-moescale1.yaml`（attn W8A8 ceil +
  路由专家 33024 全 scale_one，原始 ckpt）——**尚未部署过**
- 起服脚本：`vllm_start_rf32wfbfb4.sh` / `vllm_start_base2.sh`（均 9001）
- 夜间链日志：`/tmp/night_fb4_base.log`、`/tmp/aisbench_night.log`、
  `/tmp/msmodelslim_{fb4,base2}.log`

## 四、待用户指令的任务清单（按序）

1. assemble 补齐 29-46（~20 分钟）：`python3 -m vptq.tools.quantize.rebuild_bf16
   --quant-dir <wrap QUANT> --ckpt <rf32wfb-bf16> --output-dir <fb4-bf16> --assemble`
   （或直接重启 fb4 驱动，s4 新判据生效，幂等续跑）
2. S5' msmodelslim（~50 分钟）
3. 单请求真实性验证（鸡兔同笼，检查连贯性——夜间教训：先验证再评测）
4. fb4 评测 GPQA×5 + aime×5
5. 基线2 部署 + 评测 GPQA×5 + aime×5（原始 ckpt + moescale1 yaml）
6. 评测锚点：rf32wfb(3轮)=71.38/68.89；rf32fb+moeW4A8=71.89/64.44；
   rf32fb+moeW8A8=69.39/73.34；旧基线=73.99/71.67

## 五、10-15 踩坑追加

11. fb4 QUANT 路径错配（重大事故，见上；护栏已加）
12. assemble 判据：index.json 先于 shards 存在——完成判据必须数 shard 文件
13. killer 环境下 8 进程全启会秒杀 7 个——手动补区间时要按 layer-range 去重
    （pgrep 提取多行变量的陷阱：R 变量多行致条件永真，启动了重复进程）
14. 会话终止会带走 harness 后台任务与驱动进程——夜间链必须 setsid 独立
    （已验证 setsid 进程存活，但本次用户要求全部停止待命）

## 六、环境

- vllm_ascend 0.23.1.dev15+g31491834e.d20261013 editable 在位；msmodelslim 26.1.0
- 持久记忆 dsv4-env-restore-checklist 在
- fb4 代码资产（10-14 完成）：mxfp4_ceilx.py + 单测、rebuild --fallback-mxfp4、
  gen_yaml 简化版（`*ffn.experts*` 单规则 scale_one）、fb4 驱动（判据已加固）
