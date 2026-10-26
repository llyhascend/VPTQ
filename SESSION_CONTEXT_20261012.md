# 会话上下文交接(2026-10-12 14:10 保存,S3 量化运行中)

> 接续 `SESSION_CONTEXT_20261009.md`。覆盖:新环境恢复、rf32fb 文档收官核对、
> **rf32wfb 线设计与启动**(rf32fb + 分块 wrap Hadamard + VPTQ 专家 FP8 直通部署)。

## 一、环境恢复(当日完成,新容器 /tmp 与 site-packages 重置)

- 三步恢复全验证(细节见持久记忆 dsv4-env-restore-checklist):vptq `.pth` 注入、
  msmodelslim 26.1.0 editable、vllm_ascend `0.23.1.dev15+g31491834e` editable
  (编译 ~20min,`rm -rf csrc/build` 必做);vllm 0.23.0+empty / torch 2.10 沿用预装
- **原始 BF16 权重确认在 `/mnt/share/weight/DeepSeek-V4-Flash-BF16`**(543G/46 分片,
  所有活跃脚本 CKPT= 早已统一指向);documents §7 过时描述已更正
- 重要发现:**原始 ckpt 权重本身是 FP8(128×128 块+scale)解码的网格值**
  (`msmodelslim/model/deepseek_v4/convert_fp8_to_bf16.py`)——抽查值全在 e4m3 网格上

## 二、rf32fb 收官核对(用户已补 §8A.2/8A.7b'/8B.2)

- 用户侧已补:rf32fb 行(71.89/64.44)、**rf32fb+moeW8A8 行(69.39/73.34,aime 高点)**、
  轴4 隔离结论、§8B.2 误差链分解(aime 落后主因=t32×32 tile 密度非部署舍入)
- 我补:§8A.7e 权重表 rf32fb 三行 + rf16fb 两行状态改"折回 bug 废数据(勿评测)"

## 三、rf32wfb 线(当前主线):rf32fb 全配置 + wrap + VPTQ 专家 FP8 直通

### 3.1 需求演进(用户指令链)

1. rf32fb(t32×32+calibR9+row-fp8+12.5% gate 回退)+ moeW8A8 基础上加 wrap Hadamard
2. 回退名单:**纯 gate 复用 rf32fb 现名单**(用户确认;gate 是路由统计不受旋转影响)
3. **R 改为分块随机 Hadamard,块大小 32**(用户指令,推翻全局版)
4. 部署:VPTQ 专家**直接 cast FP8 + scale=1(原始幅度字面方案,用户在知情实测数据下选定)**;
   回退专家/attention/shared 与 attnw8a8ceil+moeW4A8 基线逐条一致

### 3.2 代码改动(4 处,全部冒烟验证)

| 文件 | 改动 |
|---|---|
| `vptq/tools/quantize/hadamard.py` | 新增 `_sylvester`/`block_hadamard_matrix`;`wrap_rotation(dim)` 默认 **WRAP_BLOCK=32 分块**(4096→128 块,w2 2048→64 块,seed-0 顺序生成可复现);`block=None` 回旧全局版 |
| `vptq/tools/quantize/hessian_io.py` | wrap 模式 R 统一由 `wrap_rotation` 派生(W/H 两侧同源),rotation 文件降级为开关,移除旧全局断言 |
| `msmodelslim/.../impl/minmax.py` | 新增 `MXWeightPerBlockScaleOne`(method="scale_one"):shared_exp 全 0,权重直接 RTN e4m3(=cast FP8+scale=1),A8 不变 |
| `gen_yaml_rf32wfb.py` + 新 yaml | **互斥规则对**(不依赖规则顺序语义):规则 A include 28896 VPTQ 精确名→W8A8 scale_one;规则 B `*ffn*` W4A8 + exclude 同名单→回退专家;attn/wo/shared 照基线。ConfigSet 精确名分流双向验证 PASS |

run.py / rebuild_bf16.py 零改动(经 wrap_rotation 自动继承)。

### 3.3 冒烟验证(L3 experts 0:2,分块版全过)

- row_scale 理论差 **1.2e-7**(模拟 right_rot 的 `.to(bf16)` cast 后精确匹配;
  此前 3e-3 偏差=bf16 精度非 bug)
- centroids 峰 6/6=448.0;折回 rms 比(row_scale→R_blkᵀ)0.982~0.989
- proxy_error 6/6 低于 rf32fb,中位 -2.9%(远小于历史 wrap 线 35-40%:row-fp8 白化
  增益重叠 + 跨域口径,真正判据是端到端)

### 3.4 scale=1 的关键实测(用户知情)

- 原始 ckpt 权重=FP8 网格值 → scale=1 cast **零损伤**(方案直觉来源)
- 但 **VPTQ 重建值是任意实数**:幅度 rms≈0.025,实测 **36% <2^-6(subnormal,绝对步长
  2^-9)、11.8% <2^-9(近零)**——cast 损伤真实存在,**S4 后将逐矩阵量化留档**
- 手动验证:_quant(0.001)=0.00195(subnormal 大舍入实证);minmax 对照正常

## 四、当前运行中:S3 量化(8 shard)

- 14:01 快照:**7614/33024 矩阵(23%),72/344 层**,8/8 shard 存活
- **wrap 开销使 ~6.3 分/层(rf32fb 是 2.2)——S3 ETA ~17:30**,全链 ETA ~22:00
- 管线 `pipeline_rf32wfb.sh`(STATE=/tmp/pipeline_rf32wfb,监控 cron :07/:47 在位)
- 脚本四件套(全部独立命名):`quantize_dsv4_experts_rf32wfb.sh` /
  `pipeline_rf32wfb.sh` / `msmodelslim/modelslim_rf32wfb.sh`(指向新 yaml)/
  `vllm-ascend/vllm_start_rf32wfb.sh`
- 产物:QUANT=`dsv4-tile32x32-rowfp8-wrap-fb-calibR9`、
  BF16=`...-rf32wfb-bf16`、DEPLOY=`...-rf32wfb-attnw8a8-moew8a8`
  (注意 DEPLOY 目录名保留 moew8a8 字样但**实际部署配置已是 vptqw8a8scale1+moew4a8**)

## 五、后续把关点(S4 之后逐项)

1. **S4 后 cast 损伤留档**:真实重建权重逐矩阵相对误差(36% subnormal 区的实际代价)
2. **部署产物自检**(S5 完成后必做):VPTQ 专家 `weight_scale` 应**全 0**、weight=e4m3;
   回退专家 scale 非零(minmax);错分流则查规则命中重跑 S5
3. S4 组装 index 判据竞态(10-09 踩坑#2):等 46/46 真完成再放行 S5
4. S6 起 vllm 前:重建 /tmp/hccl_sanity_check.py 跑 8 卡 PASS
5. 评测对照锚点:rf32fb+moeW4A8=71.89/64.44;**rf32fb+moeW8A8=69.39/73.34**(同基座
   无 wrap 的直接对照);基线 73.99/71.67
6. 收官后更新 §8A.2(注:位宽口径=VPTQ 专家 FP8 直通 8bit×87.5% + 回退 W4A8 4.25bit×
   12.5% ≈ 7.5bit?——部署混合口径,入表时与用户确认表述)+ §8A.7e 三行 + 分块 R 章节

## 六、本会话问答结论(用户三问,已答复)

1. **回退专家是否需要 W·R**:不需要。wrap 是逐矩阵自包含,折回后 VPTQ 专家与原始权重
   同坐标系;W@R@Rᵀ=I 恒等;若不折回则违背 wrap"部署零改动"设计
2. **全局 vs 分块 Hadamard**:改造前是全局(4096 单矩阵 w1/w3 共享 + w2 独立 2048);
   **现已按指令改分块 32**(见 3.2)
3. **为什么回退专家用原始 BF16 无影响**:wrap 的"换基"未跨出矩阵边界(R 只在量化中间
   表示中存在);误差 ‖ΔW‖=‖Q(WR)−WR‖·Rᵀ 等范数搬回;专家间误差独立可加;
   多精度拼接唯一前提=接口一致,折回保证之。rot2/吸收式(真整体换基)才需全员旋转

## 七、踩坑与新知识

- `quantize_*.sh status` 的 NFS 大 glob 慢到超时——用 `find` 统计(CLAUDE.md 已有类似记录)
- wrap 量化 3 倍耗时(W@R + RᵀHR + 分块 matmul),排 ETA 用它
- msmodelslim `ConfigSet` 精确名优先于 fnmatch → yaml 可枚举 28896 名单分流专家
- `QStorage.to()` 只改 dtype 标签不动 value;编码发生在 `mxfp_per_block_quantize._quant`
- msmodelslim 规则顺序语义未定论(文档称"按顺序执行")→ **互斥规则对**绕开
- gen_yaml 脚本保留在 msmodelslim 根(`gen_yaml_rf32wfb.py`),fallback_list 变更时重跑
