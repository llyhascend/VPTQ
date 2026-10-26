#!/bin/bash
# fb4 重置+重跑（2026-10-15，修复 QUANT 目录事故后；用户指令触发才运行）
#
# 背景：10-14 fb4 驱动误用 rf32fb 的无 wrap 量化产物，rebuild @Rᵀ 后
# 28896 个 VPTQ parts 全部污染（fallback 4128 parts 无辜且正确，保留）。
# 本脚本：清污染 → 重跑 S4'(VPTQ parts+assemble) → S5'(部署)。
# 评测链（fb4 5+5 与基线2 5+5）另见 night_pipeline_fb4_base.sh。
set -eu
cd "$(dirname "$0")"

BF16=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-vptq-calibR9-rf32wfbfb4-bf16
DEPLOY=/mnt/share/w00608002/weights/DeepSeek-V4-Flash-vptq-calibR9-rf32wfbfb4-attnw8a8-moew8a8
FB=/mnt/share/w00608002/weights/quant/dsv4-tile32x32-rowfp8-wrap-fb-calibR9/fallback_list.json

echo "== 1. 清状态标记 =="
rm -f /tmp/pipeline_rf32wfbfb4/*.done 2>/dev/null || true
rm -f /tmp/night_fb4_base/*.done /tmp/night_fb4_base/fb4_t0 2>/dev/null || true

echo "== 2. 删污染的 VPTQ parts（保留 fallback 4128 个）=="
python3 - "$FB" "$BF16/expert_parts" <<'EOF'
import json, os, sys
fb, parts = sys.argv[1], sys.argv[2]
lst = json.load(open(fb))
keep = set()
for li, e in lst["full"]:
    for w in ("w1", "w2", "w3"):
        keep.add(f"layers.{li}.ffn.experts.{e}.{w}.weight.safetensors")
for li, e in lst["down_only"]:
    keep.add(f"layers.{li}.ffn.experts.{e}.w2.weight.safetensors")
removed = kept = 0
for f in os.listdir(parts):
    if f.endswith(".safetensors"):
        if f in keep: kept += 1
        else: os.remove(os.path.join(parts, f)); removed += 1
print(f"保留 fallback parts: {kept}（应 4128），删除污染 VPTQ parts: {removed}（应 28896）")
assert kept == 4128 and removed == 28896, "数量不符，中止"
EOF

echo "== 3. 删污染的 BF16 模型与部署产物 =="
rm -f "$BF16"/model-*.safetensors "$BF16"/model.safetensors.index.json 2>/dev/null || true
rm -rf "$DEPLOY" 2>/dev/null || true

echo "== 4. 启动 fb4 驱动（S4'+S5'，含看门狗自愈）=="
setsid nohup bash "$(dirname "$0")/quantize_dsv4_experts_rf32wfb_fb4.sh" \
  >> /tmp/fb4_driver3.log 2>&1 &
echo "已启动，日志：/tmp/pipeline_rf32wfbfb4.log（驱动）/tmp/rebuild_rf32wfbfb4_expert.log"
echo "验收：S4 完成后抽查 VPTQ part 与生产 rf32wfb 逐位一致（检查 3 应 IDENTICAL）"
