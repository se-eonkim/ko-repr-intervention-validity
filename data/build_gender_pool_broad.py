"""Build the broader gender pool used by the O2 interventions.

Starts from every offensive KOLD comment with a single female or male target
and assigns a tier:
  strict  target span and offensive span both present and not overlapping
          (identical to the gender members of the O1 probe splits)
  A       offensive span present; target span missing or overlapping
  B       everything else
Rows carry only GUIDs and flags (no comment text), so the output is tracked.

Input : data/raw/kold_v1.json, data/splits/probe_splits.json,
        data/splits/excluded_eval_pool.json, data/splits/minimal_pairs.json
Output: data/splits/gender_pool.csv

Run   : python data/build_gender_pool_broad.py
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import KOLD, SPLITS, read_json  # noqa: E402

GENDER = {"gender-female": 0, "gender-male": 1}
OUT = SPLITS / "gender_pool.csv"


def single_group(grp) -> bool:
    return isinstance(grp, str) and bool(grp) and "&" not in grp


def span_chars(raw_labels, start_key, end_key, thresh=2) -> frozenset:
    votes = Counter()
    for lab in raw_labels:
        marked = set()
        for s, e in zip(lab.get(start_key) or [], lab.get(end_key) or []):
            marked.update(range(s, e))
        for c in marked:
            votes[c] += 1
    return frozenset(c for c, v in votes.items() if v >= thresh)


def main():
    df = pd.DataFrame(read_json(KOLD))
    strict_members = set(read_json(SPLITS / "probe_splits.json")["axes"]["gender"]["members"])
    excluded = set(read_json(SPLITS / "excluded_eval_pool.json")["guids"])
    pair_guids = {r["guid"] for r in read_json(SPLITS / "minimal_pairs.json")["pairs"]}

    cand = df[(df["OFF"] == True) & (df["TGT"] == "group")  # noqa: E712
              & df["GRP"].apply(single_group) & df["GRP"].isin(GENDER)]
    rows = []
    for _, r in cand.iterrows():
        off = span_chars(r["raw_labels"], "off_start_idx", "off_end_idx")
        tgt = span_chars(r["raw_labels"], "tgt_start_idx", "tgt_end_idx")
        inter = off & tgt
        if not inter:
            overlap = "none"
        elif tgt and tgt <= off:
            overlap = "full"
        else:
            overlap = "partial"
        strict = bool(off) and bool(tgt) and not inter
        rows.append({
            "guid": r["guid"],
            "Y_group": r["GRP"],
            "Y_label": GENDER[r["GRP"]],
            "tier": "strict" if strict else ("A" if off else "B"),
            "overlap_degree": overlap,
            "in_strict": strict,
            "in_excluded_eval_pool": r["guid"] in excluded,
            "in_minimal_pairs": r["guid"] in pair_guids,
        })
    pool = pd.DataFrame(rows).sort_values("guid").reset_index(drop=True)

    got = set(pool.loc[pool.tier.eq("strict"), "guid"])
    assert got == strict_members, "strict tier must equal the O1 gender members"
    assert not pool.guid.duplicated().any()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(OUT, index=False, encoding="utf-8")
    print(f"saved {OUT.relative_to(OUT.parents[2])} ({len(pool)} rows)")
    print(pool.tier.value_counts().to_string())


if __name__ == "__main__":
    main()
