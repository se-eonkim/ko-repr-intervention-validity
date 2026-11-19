"""Figure 3: LEACE before/after gender BAcc for each split.

Open circle = before erasure, filled circle = after, grey tick = the split's
permutation q95 for the erased vectors, green = the split met the criterion.

Input : results/o2_leace_repeated_splits.json
Output: figures/figure3_leace_repeated_splits.{png,pdf}

Run   : python O2_intervention_validity/leace/plot_figure3.py
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

for cand in ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.family"] = cand
        break
    except Exception:
        continue
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FIGSIZE = (88.6 / 25.4, 50.0 / 25.4)
FS, FS_LEG = 9, 9

C_PASS = "#1B7F3B"
C_FAIL = "#4D5156"
C_NULL = "#9AA0A6"
MODEL = [("kcbert", "KcBERT", "#D81B60"), ("klue", "KLUE-BERT", "#0072B2")]
GAP = 1.6


def load():
    d = json.load(open(ROOT / "results" / "o2_leace_repeated_splits.json",
                       encoding="utf-8"))
    return d["rows"]


def main():
    rows = load()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    xticks, xlabels, group_x = [], [], {}
    x = 0.0
    for key, name, col in MODEL:
        rs = sorted([r for r in rows if r["model"] == key], key=lambda r: r["seed"])
        xs = []
        for r in rs:
            ok = r["pass"]
            c = C_PASS if ok else C_FAIL
            o, e, q = r["original_bacc"], r["erased_bacc"], r["null_q95_erased"]

            ax.plot([x - 0.34, x + 0.34], [q, q], color=C_NULL, lw=1.3,
                    solid_capstyle="butt", zorder=2)
            ax.plot([x, x], [o, e], "-", color=c, lw=1.3, zorder=3)
            ax.plot([x], [o], "o", ms=3.6, mfc="white", mec=c, mew=1.2, zorder=4)
            ax.plot([x], [e], "o", ms=3.8, color=c, zorder=4)

            xticks.append(x); xlabels.append(str(r["seed"])); xs.append(x)
            x += 1.0
        group_x[name] = (np.mean(xs), col)
        x += GAP

    ax.set_xlim(-0.8, x - GAP - 0.2)
    ax.set_ylim(0.45, 0.95)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=FS)
    ax.tick_params(labelsize=FS, length=2.5, pad=1.5)
    ax.grid(True, axis="y", alpha=0.15, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_ylabel("Gender BAcc", fontsize=FS, labelpad=2.0)
    for name, (xc, _col) in group_x.items():
        ax.text(xc, -0.165, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=FS, color="black")

    handles = [
        Line2D([], [], color=C_PASS, lw=1.4, marker="o", ms=3.4, label="pass"),
        Line2D([], [], color=C_FAIL, lw=1.4, marker="o", ms=3.4, label="fail"),
        Line2D([], [], color=C_NULL, lw=1.3, label="null threshold"),
    ]
    ax.legend(handles=handles, fontsize=FS_LEG, ncol=3, loc="upper right",
              frameon=True, facecolor="white", edgecolor="none", framealpha=0.85,
              handlelength=1.5, handletextpad=0.35, columnspacing=0.9,
              borderaxespad=0.25, borderpad=0.25).set_zorder(10)

    fig.subplots_adjust(left=0.145, right=0.99, top=0.97, bottom=0.185)
    for ext in ("png", "pdf"):
        p = FIG / f"figure3_leace_repeated_splits.{ext}"
        fig.savefig(p, dpi=600)
        print("saved", p.relative_to(ROOT))
    plt.close(fig)


if __name__ == "__main__":
    main()
