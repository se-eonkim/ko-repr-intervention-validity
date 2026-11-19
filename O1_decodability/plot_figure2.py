"""Figure 2: layer-wise gender decoding (real labels vs permutation control).

Input : results/o1_layerwise_probe.csv
Output: figures/figure2_gender_decoding.{png,pdf}

Run   : python O1_decodability/plot_figure2.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patheffects as pe

for cand in ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.family"] = cand
        break
    except Exception:
        continue
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "o1_layerwise_probe.csv"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PANELS = [("kcbert", "(a) KcBERT", "#D81B60"),
          ("klue",   "(b) KLUE-BERT", "#0072B2")]

# one column: 88.6 mm x 85 mm
FIGSIZE = (88.6 / 25.4, 85.0 / 25.4)
FS = 9
FS_LEG = 9

C_CTRL = "#808080"


def main():
    df = pd.read_csv(CSV)
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

    for ax, (model, label, c_real) in zip(axes, PANELS):
        r = df[df.model == model].groupby("layer").real_f1.agg(["mean", "std"])
        c = df[df.model == model].groupby("layer").ctrl_f1.mean()
        L = r.index.values
        m, s = r["mean"].values, np.nan_to_num(r["std"].values)

        # shaded area between the two lines = selectivity
        ax.fill_between(L, c.values, m, color="#9AA0A6", alpha=0.13, lw=0, zorder=1)

        ax.plot(L, c.values, ls=(0, (3, 2)), color=C_CTRL, lw=0.95,
                label="permutation control", zorder=2)
        ax.fill_between(L, m - s, m + s, color=c_real, alpha=0.15, lw=0, zorder=3)
        ax.plot(L, m, "-o", color=c_real, ms=2.2, lw=1.2,
                label="real", zorder=4)
        # star = layer with the highest mean F1
        pk = int(np.argmax(m))
        ax.plot(pk, m[pk], marker="*", color=c_real, ms=12.5,
                mec="white", mew=0.9, zorder=6)
        ax.annotate(f"L{pk}", (pk, m[pk]), textcoords="offset points",
                    xytext=(0.0, -8.5), fontsize=FS_LEG, color=c_real,
                    ha="center", va="top",
                    path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])

        for xv in range(1, 13):
            ax.axvline(xv, color="0.88", lw=0.4, ls="-", zorder=0)
        ax.set_ylim(0.35, 1.0)
        ax.set_xlim(-0.4, 12.4)
        ax.set_yticks([0.4, 0.6, 0.8, 1.0])
        ax.tick_params(labelsize=FS, length=2.5, pad=1.5)
        ax.grid(True, axis="y", alpha=0.15, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    for ax in axes:
        ax.set_xticks(range(0, 13, 2))
        ax.tick_params(labelbottom=True)

    ROW_Y = -0.22
    axes[1].set_xlabel("Layer (hidden state)", fontsize=FS, labelpad=2.0)
    for i, (ax, (_, label, _)) in enumerate(zip(axes, PANELS)):
        y = ROW_Y - (0.17 if i == 1 else 0.0)
        ax.text(0.5, y, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=FS)
    fig.supylabel("Gender macro-F1", fontsize=FS, x=0.035)
    def ordered(ax):
        """List 'real' first in the legend."""
        h, l = ax.get_legend_handles_labels()
        order = sorted(range(len(l)), key=lambda i: 0 if l[i] == "real" else 1)
        return [h[i] for i in order], [l[i] for i in order]

    for i, ax in enumerate(axes):
        anchor = (1.0, 1.035) if i == 1 else (1.0, 1.0)
        ax.legend(*ordered(ax), fontsize=FS_LEG, ncol=2, loc="upper right",
                  bbox_to_anchor=anchor, frameon=True, facecolor="white",
                  edgecolor="none", framealpha=0.85, handlelength=1.4,
                  handletextpad=0.3, columnspacing=0.7, borderaxespad=0.25,
                  borderpad=0.25).set_zorder(10)

    fig.subplots_adjust(left=0.205, right=0.99, top=0.968,
                        bottom=0.175, hspace=0.33)
    for ext, dpi in (("png", 600), ("pdf", 600)):
        p = FIG / f"figure2_gender_decoding.{ext}"
        fig.savefig(p, dpi=dpi)
        print("saved", p.relative_to(ROOT))
    plt.close(fig)


if __name__ == "__main__":
    main()
