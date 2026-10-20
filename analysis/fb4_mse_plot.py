#!/usr/bin/env python3
"""fb4 量化前后 MSE 每层趋势图（dataviz 规范：折线 3 系列、2px 线、
直接标签 + 图例、recessive 网格、L0-2 hash 路由段标注）。"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

C_ALL, C_VPTQ, C_FB = "#2a78d6", "#eb6834", "#1baf7a"  # categorical 1/2/3
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e4e3de"

recs = [json.loads(l) for l in open("analysis/fb4_mse_by_layer.jsonl")]
recs.sort(key=lambda r: r["layer"])
L = [r["layer"] for r in recs]

fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=160)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

# L0-2 hash 路由段（无 gate 分数，均匀权重 + 无 fallback）
ax.add_patch(Rectangle((-0.6, 0), 3.0, 1, transform=ax.get_xaxis_transform(),
                       facecolor="#f1f0ec", edgecolor="none", zorder=0))
ax.text(1.0, 0.965, "L0-2 hash 路由（均匀权重，无回退）", ha="center",
        va="top", fontsize=7.5, color=TXT2, transform=ax.get_xaxis_transform())

for key, color, label in [
        ("vptq_w", C_VPTQ, "VPTQ 专家（2.25 bit）"),
        ("fb_w", C_FB, "回退专家 mxfp4（4.25 bit）"),
        ("all_w", C_ALL, "全体（gate 加权）")]:
    ys = [r[key] for r in recs]
    ax.plot(L, ys, color=color, lw=2, label=label, solid_capstyle="round",
            zorder=3)
# 直接标签（系列末端）
for key, color, txt in [("vptq_w", C_VPTQ, "VPTQ"), ("all_w", C_ALL, "全体"),
                        ("fb_w", C_FB, "回退")]:
    ax.annotate(txt, (L[-1], recs[-1][key]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=8.5,
                color=TXT, fontweight="bold")

ax.set_yscale("log")
ax.set_xlim(-0.6, 45.6)
ax.set_xticks(range(0, 43, 4))
ax.set_xlabel("层索引", fontsize=9, color=TXT)
ax.set_ylabel("MSE（log 轴）", fontsize=9, color=TXT)
ax.grid(True, which="major", color=GRID, lw=0.7, zorder=1)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=TXT2, labelsize=8)
ax.legend(loc="upper left", frameon=False, fontsize=8.5)
ax.set_title("fb4 量化前后每层专家 MSE（router gate 分数加权）",
             fontsize=11, color=TXT, loc="left", pad=10)

fig.tight_layout()
fig.savefig("analysis/fb4_mse_by_layer.png", bbox_inches="tight",
            facecolor="#fcfcfb")
print("saved analysis/fb4_mse_by_layer.png")

# 汇总数值
import math
for k in ("all_w", "vptq_w", "fb_w"):
    vs = [r[k] for r in recs if r[k]]
    print(f"{k}: min={min(vs):.3e} max={max(vs):.3e} "
          f"mean={sum(vs)/len(vs):.3e}")
