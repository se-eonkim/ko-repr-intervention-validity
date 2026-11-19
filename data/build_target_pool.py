"""Build the explicit-target pool from KOLD.

Keeps offensive comments that attack a single target group, where both the
target span and the offensive span are present and do not overlap. Spans are
the KOLD gold definition: characters marked by at least 2 of 3 annotators,
aggregated from the raw character offsets.

Input : data/raw/kold_v1.json
Output: data/processed/target_pool.csv   (contains comment text; not tracked)

Run   : python data/build_target_pool.py
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KOLD_PATH = ROOT / "data" / "raw" / "kold_v1.json"
OUT_PATH = ROOT / "data" / "processed" / "target_pool.csv"


def load_kold() -> pd.DataFrame:
    return pd.DataFrame(json.loads(KOLD_PATH.read_text(encoding="utf-8")))


def single_group(grp) -> bool:
    """True if exactly one target group is annotated."""
    if not isinstance(grp, str) or not grp:
        return False
    return "&" not in grp


def span_chars(raw_labels, start_key, end_key, thresh=2) -> frozenset:
    """Character indices marked by at least `thresh` annotators."""
    votes = Counter()
    for lab in raw_labels:
        starts = lab.get(start_key) or []
        ends = lab.get(end_key) or []
        marked = set()
        for s, e in zip(starts, ends):
            marked.update(range(s, e))
        for c in marked:
            votes[c] += 1
    return frozenset(c for c, v in votes.items() if v >= thresh)


def spans_text(comment: str, chars) -> list:
    """Character index set -> list of contiguous text pieces."""
    if not chars:
        return []
    cs = sorted(chars)
    out, start, prev = [], cs[0], cs[0]
    for c in cs[1:]:
        if c == prev + 1:
            prev = c
        else:
            out.append(comment[start:prev + 1])
            start = prev = c
    out.append(comment[start:prev + 1])
    return out


if __name__ == "__main__":
    df = load_kold()
    N = len(df)
    df["off_chars"] = df["raw_labels"].apply(
        lambda rl: span_chars(rl, "off_start_idx", "off_end_idx"))
    df["tgt_chars"] = df["raw_labels"].apply(
        lambda rl: span_chars(rl, "tgt_start_idx", "tgt_end_idx"))

    def line(label, mask):
        n = int(mask.sum())
        print(f"  {label:32s} {n:>7,}  ({n / N * 100:5.1f}%)")
        return mask

    print(f"KOLD records: {N:,}")
    m_off = line("offensive", df["OFF"] == True)  # noqa: E712
    m_grp = line("  & group target", m_off & (df["TGT"] == "group"))
    m_single = line("  & single group", m_grp & df["GRP"].apply(single_group))
    m_tgt = line("  & target span", m_single & (df["tgt_chars"].apply(len) > 0))
    m_both = line("  & offensive span", m_tgt & (df["off_chars"].apply(len) > 0))

    cand = df[m_both].copy()
    cand["overlap"] = cand.apply(lambda r: len(r["off_chars"] & r["tgt_chars"]) > 0, axis=1)
    pool = cand[~cand["overlap"]].copy()
    print(f"  & spans do not overlap           {len(pool):>7,}")

    out = pool[["guid", "source", "comment", "GRP"]].rename(columns={"GRP": "Y_group"})
    out["X_offensive"] = 1
    out["off_span"] = pool.apply(lambda r: " | ".join(spans_text(r["comment"], r["off_chars"])), axis=1)
    out["tgt_span"] = pool.apply(lambda r: " | ".join(spans_text(r["comment"], r["tgt_chars"])), axis=1)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"saved {OUT_PATH.relative_to(ROOT)} ({len(out):,} rows)")
    print(out["Y_group"].value_counts().head(10).to_string())
