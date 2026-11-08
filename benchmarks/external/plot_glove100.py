"""README plot and dark cover image for glove-100-angular from an ann-benchmarks export.

Draws each library's Pareto frontier (qann-ivf-fastscan, hnswlib, faiss-ivfpqfs) with
ann-benchmarks' own recall ("k-nn" column), recall 0.8 and up on a -log10(1 - recall)
axis, lines labelled directly. "light" is the README image (docs/images/ann-glove-100-angular.png
on main), "dark" the cover. Recheck the dark title's speedup against each new run:
it is the largest qann / hnswlib best-QPS ratio over hnswlib's recall range.

    # on the VM: export only the three plotted algorithms (data_export.py reads results/)
    mkdir -p p/results/glove-100-angular/10 && cd p
    for x in ann_benchmarks data venv logging.conf data_export.py; do ln -s ~/qann-bench/ann-benchmarks/$x; done
    for a in qann-ivf-fastscan hnswlib faiss-ivfpqfs; do
        cp -r ~/qann-bench/ann-benchmarks/results/glove-100-angular/10/$a results/glove-100-angular/10/; done
    venv/bin/python data_export.py --output glove100.csv

    # locally: Inter from https://github.com/rsms/inter/releases (v4.0 zip, unpacked to inter/)
    python plot_glove100.py glove100.csv inter readme.png light
    python plot_glove100.py glove100.csv inter cover.png dark
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

csv, font_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
THEME = sys.argv[4] if len(sys.argv) > 4 else "dark"   # dark: social cover; light: README
for w in ("Regular", "Medium", "SemiBold", "Bold"):
    font_manager.fontManager.addfont(f"{font_dir}/extras/ttf/Inter-{w}.ttf")
plt.rcParams["font.family"] = "Inter"

if THEME == "dark":
    BG, GRID, INK, MUTED, OTHER, ACCENT = "#0a0a0a", "#262626", "#ededed", "#a3a3a3", "#8a8a8a", "#8b5cf6"
else:
    BG, GRID, INK, MUTED, OTHER, ACCENT = "#ffffff", "#e5e5e5", "#171717", "#525252", "#8a8a8a", "#7c3aed"
SERIES = [  # drawn back to front
    ("faiss-ivfpqfs", "FAISS fast-scan", OTHER, (0, (6, 4)), 3.0),
    ("hnswlib", "hnswlib", OTHER, "-", 3.0),
    ("qann-ivf-fastscan", "QaNN", ACCENT, "-", 5.0),
]
X0, X1 = 0.8, 0.999995
tx = lambda r: -np.log10(1 - np.asarray(r))


def frontier(g):
    g = g.sort_values("k-nn", ascending=False)
    pts, best = [], 0
    for r, q in zip(g["k-nn"], g["qps"]):
        if q > best:
            best = q
            pts.append((r, q))
    pts.sort()
    # keep one point left of the crop so the line enters from the edge
    first = max(i for i, (r, _) in enumerate(pts) if r < X0) if pts[0][0] < X0 else 0
    return np.array(pts[first:])


d = pd.read_csv(csv)
fig, ax = plt.subplots(figsize=(19, 10.6875), dpi=180)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

for alg, label, color, style, lw in SERIES:
    p = frontier(d[d.algorithm == alg])
    x, y = tx(p[:, 0]), p[:, 1]
    big = alg == "qann-ivf-fastscan"
    ax.plot(x, y, color=color, lw=lw, ls=style, zorder=3 if big else 2, solid_capstyle="round", dash_capstyle="round")
    if big:
        ax.plot([x[-1]], [y[-1]], "o", color=color, ms=12, markeredgecolor=BG, markeredgewidth=3, zorder=4)
    ex, ey, er = x[-1], y[-1], p[-1, 0]
    if big:
        ax.annotate(f"{label}\nreaches {er:.5f}", (ex, ey), xytext=(0, 30), textcoords="offset points",
                    ha="right", va="bottom", color=color, fontsize=30, fontweight="semibold", linespacing=1.25)
    else:
        # FAISS's label sits above its end, in the gap under QaNN; hnswlib's beside its end, below
        above = alg == "faiss-ivfpqfs"
        ax.annotate(label, (ex, ey), xytext=(12, 12) if above else (16, 0),
                    textcoords="offset points", ha="left", va="bottom" if above else "center",
                    color=MUTED, fontsize=26, fontweight="medium")

ax.set_xlim(tx(X0), tx(X1))
ax.set_yscale("log")
ax.set_ylim(140, 20000)
xt = [0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999]
ax.set_xticks(tx(xt), [f"{v:g}" for v in xt])
yt = [300, 1000, 3000, 10000]
ax.set_yticks(yt, ["300", "1K", "3K", "10K"])
ax.minorticks_off()
ax.tick_params(colors=MUTED, labelsize=24, length=0, pad=12)
ax.grid(True, color=GRID, lw=1.2)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlabel("Recall@10", color=MUTED, fontsize=26, labelpad=16)
ax.set_ylabel("Queries / sec", color=MUTED, fontsize=26, labelpad=16)

if THEME == "dark":
    title, sub = "Up to 3.6x faster than hnswlib on GloVe-100", "single thread · ann-benchmarks"
else:
    title, sub = "glove-100-angular: recall vs queries per second", "ann-benchmarks · one query at a time on one thread · AMD EPYC (Zen 4)"
fig.text(0.065, 0.955, title, color=INK, fontsize=44, fontweight="bold", va="top")
fig.text(0.065, 0.875, sub, color=MUTED, fontsize=28, va="top")
fig.subplots_adjust(left=0.085, right=0.965, top=0.79, bottom=0.12)
fig.savefig(out, dpi=180, facecolor=BG)
