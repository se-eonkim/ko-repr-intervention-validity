"""Figure 4: which AlterRep validity gates each setting met.

Each setting (model, subspace size m, edit strength alpha) shows three cells:
C = completeness, S = target-free survival, R = restricted-X. A filled cell
means the gate was met. A setting is valid only if all three are filled.

Input : results/o2_alterrep_validity.json
Output: figures/figure4_alterrep_gates.{png,pdf}

Run   : python O2_intervention_validity/alterrep/plot_figure4.py
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

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

FIGSIZE = (88.6 / 25.4, 48.0 / 25.4)
FS, FS_SM, FS_XS, FS_LAMP = 9, 9, 9, 9
ALPHAS = [0.5, 1, 2, 4]
GATES = ["C", "S", "R"]
ROWS = [("kcbert", 4), ("kcbert", 8), ("klue", 4), ("klue", 8)]
MODEL_NAME = {"kcbert": "KcBERT", "klue": "KLUE-BERT"}

ON_FC, ON_TC = "#3C4043", "#FFFFFF"
OFF_FC, OFF_TC = "#EFF1F2", "#A8ADB2"
C_RULE, C_GRAY = "#202124", "#5F6368"


def main():
    data = json.load(open(ROOT / "results" / "o2_alterrep_validity.json",
                          encoding="utf-8"))
    g = {(k, c["m"], c["alpha"]): c["gates"]
         for k in ("kcbert", "klue")
         for c in data["results"][k]["candidates"]}

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_axis_off()

    LW, LH, LGAP = 0.30, 0.56, 0.035
    CW = 3 * LW + 2 * LGAP
    CGAP, RGAP, GROUP = 0.42, 0.26, 0.24

    ytop, y = {}, 0.0
    for i, (k, m) in enumerate(ROWS):
        if i == 2:
            y -= GROUP
        ytop[(k, m)] = y
        y -= LH + RGAP

    for (k, m), yy in ytop.items():
        for j, a in enumerate(ALPHAS):
            mg = g[(k, m, a)]
            x0 = j * (CW + CGAP)
            for t, gate in enumerate(GATES):
                on = bool(mg[gate])
                xx = x0 + t * (LW + LGAP)
                ax.add_patch(FancyBboxPatch(
                    (xx, yy - LH), LW, LH,
                    boxstyle="round,pad=0,rounding_size=0.045",
                    fc=ON_FC if on else OFF_FC, ec="none", zorder=2))
                ax.text(xx + LW / 2, yy - LH / 2, gate, fontsize=FS_LAMP,
                        color=ON_TC if on else OFF_TC,
                        ha="center", va="center", zorder=3)

    for (k, m), yy in ytop.items():
        ax.text(-0.20, yy - LH / 2, f"{MODEL_NAME[k]}  $m{{=}}{m}$", fontsize=FS_SM,
                color=C_RULE, ha="right", va="center")

    for j, a in enumerate(ALPHAS):
        ax.text(j * (CW + CGAP) + CW / 2, 0.22,
                "0.5" if a == 0.5 else str(int(a)),
                fontsize=FS, color=C_RULE, ha="center", va="center")
    xr = len(ALPHAS) * (CW + CGAP) - CGAP
    ax.text(xr / 2, 0.62, r"edit strength  $\alpha$", fontsize=FS,
            color=C_RULE, ha="center", va="center")

    ax.set_xlim(-2.58, xr + 0.03)
    ax.set_ylim(ytop[ROWS[-1]] - LH - 0.06, 0.88)

    fig.subplots_adjust(left=0.012, right=0.995, top=0.995, bottom=0.300)

    fig.text(0.988, 0.225,
             "C  Completeness      S  Target-free survival      R  Restricted-X",
             fontsize=FS_XS, color=C_GRAY, ha="right", va="center")

    t = fig.text(0.988, 0.140, "=  requirement met",
                 fontsize=FS_XS, color=C_GRAY, ha="right", va="center")
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(fig.transFigure.inverted())
    sw, sh = 2.7 / 88.6, 3.4 / 48.0
    fig.patches.append(FancyBboxPatch(
        (bb.x0 - 0.016 - sw, 0.140 - sh / 2), sw, sh,
        boxstyle="round,pad=0,rounding_size=0.006",
        fc=ON_FC, ec="none", transform=fig.transFigure, figure=fig))

    for ext in ("png", "pdf"):
        p = FIG / f"figure4_alterrep_gates.{ext}"
        fig.savefig(p, dpi=600)
        print("saved", p.relative_to(ROOT))
    plt.close(fig)


if __name__ == "__main__":
    main()
