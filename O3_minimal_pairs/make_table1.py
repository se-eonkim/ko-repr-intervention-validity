"""Table 1: output difference on the 39 gender minimal pairs.

Rows: mean delta with 95% bootstrap CI, effect size d_z, and the number of
pairs with delta > 0 / delta < 0. delta = logit(female) - logit(male); the two
models are reported separately (n = 39 each).

Input : results/o3_minimal_pairs_summary.json
Output: results/table1_minimal_pairs.csv (and a printed Markdown table)

Run   : python O3_minimal_pairs/make_table1.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MODELS, RESULTS, read_json  # noqa: E402

NAMES = {"kcbert": "KcBERT", "klue": "KLUE-BERT"}
OUT = RESULTS / "table1_minimal_pairs.csv"


def fmt(x, nd=3):
    return f"{x:+.{nd}f}".replace("0.", ".", 1)


def main():
    s = read_json(RESULTS / "o3_minimal_pairs_summary.json")["models"]
    rows = [
        ["", *(NAMES[m] for m in MODELS)],
        ["mean delta [95% CI]", *(f"{fmt(s[m]['mean_delta_logit'])} "
                                  f"[{fmt(s[m]['bootstrap_95ci'][0])}, {fmt(s[m]['bootstrap_95ci'][1])}]"
                                  for m in MODELS)],
        ["effect size d_z", *(fmt(s[m]["dz"], 2) for m in MODELS)],
        ["delta>0 / delta<0", *(f"{s[m]['direction']['female_higher']} / {s[m]['direction']['male_higher']}"
                                for m in MODELS)],
    ]
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    print("| " + " | ".join(rows[0]) + " |")
    print("|" + "---|" * len(rows[0]))
    for r in rows[1:]:
        print("| " + " | ".join(r) + " |")


if __name__ == "__main__":
    main()
