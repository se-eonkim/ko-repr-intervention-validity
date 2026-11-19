"""Freeze the repeated train/test splits used by the layer-wise probes (O1).

Two label sets are built:
  gender     female (0) vs male (1) targets from the explicit-target pool
  offensive  offensive (1) vs non-offensive (0), class-balanced from the
             classifier's test split (used as the non-target probe in O2)

Each is split into stratified GUID-level train/test sets for five seeds, so
every layer and model is probed on the same samples.

Input : data/processed/target_pool.csv, data/raw/kold_v1.json,
        data/splits/p2_split.json
Output: data/splits/probe_splits.json   (GUIDs and labels only)

Run   : python data/build_probe_splits.py
"""
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "data.json").read_text(encoding="utf-8"))["probe_splits"]
TARGET_POOL = ROOT / "data" / "processed" / "target_pool.csv"
KOLD = ROOT / "data" / "raw" / "kold_v1.json"
P2_SPLIT = ROOT / "data" / "splits" / "p2_split.json"
OUT = ROOT / "data" / "splits" / "probe_splits.json"

GENDER_LABELS = {"gender-female": 0, "gender-male": 1}


def gender_members() -> dict:
    df = pd.read_csv(TARGET_POOL)
    sub = df[df["Y_group"].isin(GENDER_LABELS)]
    return {str(r.guid): int(GENDER_LABELS[r.Y_group]) for r in sub.itertuples()}


def offensive_members() -> dict:
    kold = json.loads(KOLD.read_text(encoding="utf-8"))
    off = {r["guid"]: bool(r["OFF"]) for r in kold}
    test = [g for g in json.loads(P2_SPLIT.read_text(encoding="utf-8"))["test_guids"] if g in off]
    pos = [g for g in test if off[g]]
    neg = [g for g in test if not off[g]]
    rng = random.Random(CFG["offensive_sample_seed"])
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(CFG["offensive_n_per_class"], len(pos), len(neg))
    return {str(g): int(off[g]) for g in pos[:k] + neg[:k]}


def make_splits(members: dict) -> dict:
    guids = list(members)
    labels = [members[g] for g in guids]
    splits = {}
    for s in CFG["seeds"]:
        tr, te = train_test_split(guids, test_size=CFG["test_size"], random_state=s,
                                  stratify=labels, shuffle=True)
        splits[str(s)] = {"train": sorted(tr), "test": sorted(te)}
    return splits


def main():
    axes = {}
    for name, members, label_map in (
        ("gender", gender_members(), GENDER_LABELS),
        ("offensive", offensive_members(), {"non_offensive": 0, "offensive": 1}),
    ):
        axes[name] = {"label_map": label_map, "n": len(members),
                      "members": members, "splits": make_splits(members)}
        cc = Counter(members.values())
        print(f"{name:9s} n={len(members)}  class0={cc[0]} class1={cc[1]}")
    out = {"meta": {"seeds": CFG["seeds"], "test_size": CFG["test_size"]}, "axes": axes}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
