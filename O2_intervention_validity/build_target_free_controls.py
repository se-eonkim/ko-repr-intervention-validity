"""Select the target-free control comments used by the AlterRep survival gate (S).

Offensive comments whose target is not a group (untargeted, individual,
other) carry no gender target, so a gender edit should not change their
offensiveness. 250 comments per subtype are drawn from comments the
classifier did not see in training and that are not in the probe splits.

Input : data/raw/kold_v1.json, data/splits/p2_split.json,
        data/splits/probe_splits.json
Output: data/splits/target_free_controls.json   (GUIDs only)

Run   : python O2_intervention_validity/build_target_free_controls.py
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import KOLD, SPLITS, config, read_json, write_json  # noqa: E402

CFG = config("data")["target_free_controls"]
OUT = SPLITS / "target_free_controls.json"


def main():
    meta = {r["guid"]: (r.get("OFF"), r.get("TGT")) for r in read_json(KOLD)}
    train = set(read_json(SPLITS / "p2_split.json")["train_guids"])
    probe_offensive = set(read_json(SPLITS / "probe_splits.json")["axes"]["offensive"]["members"])

    target_free = {g for g, (off, tgt) in meta.items() if off is True and tgt != "group"}
    eligible = target_free - train - probe_offensive
    by_subtype = defaultdict(list)
    for g in eligible:
        by_subtype[meta[g][1]].append(g)

    rng = np.random.default_rng(CFG["seed"])
    picked = []
    for st in CFG["subtypes"]:
        pool = sorted(by_subtype[st])
        assert len(pool) >= CFG["per_subtype"], st
        picked += rng.choice(pool, size=CFG["per_subtype"], replace=False).tolist()

    write_json(OUT, {"n": len(picked),
                     "subtypes": dict(Counter(meta[g][1] for g in picked)),
                     "guids": picked})
    print(f"saved {OUT.name}: {len(picked)} comments from {len(eligible)} eligible")


if __name__ == "__main__":
    main()
