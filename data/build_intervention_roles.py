"""Assign the gender samples used by the O2 interventions to disjoint roles.

direction_fit             tier-A rows (seen by the classifier during fine-tuning);
                          used only to estimate the AlterRep gender subspace
independent_y_eval_probe  strict rows; trains the independent gender probe
selection_validation      strict rows; scores the AlterRep validity gates
effect_audit              strict rows; reserved for the output-effect test,
                          which is run only if an intervention is valid

The three strict roles together form the 407 rows used for LEACE. Rows in the
minimal pairs and rows whose target span lies inside the offensive span are
excluded. Allocation is exact per class (40/30/30) with a fixed seed.

Input : data/splits/gender_pool.csv
Output: data/splits/intervention_roles.json   (GUIDs only)

Run   : python data/build_intervention_roles.py
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import SPLITS, config, write_json  # noqa: E402

CFG = config("data")["intervention_roles"]
POOL = SPLITS / "gender_pool.csv"
OUT = SPLITS / "intervention_roles.json"
STRICT_ROLES = ("independent_y_eval_probe", "selection_validation", "effect_audit")


def guid_hash(guids) -> str:
    return hashlib.sha256("\n".join(sorted(guids)).encode("utf-8")).hexdigest()


def stratified_exact_split(df: pd.DataFrame, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    roles = {k: [] for k in STRICT_ROLES}
    f_probe = CFG["fractions"]["independent_y_eval_probe"]
    f_sel = CFG["fractions"]["selection_validation"]
    for label in sorted(df.Y_label.unique()):
        guids = np.asarray(sorted(df.loc[df.Y_label == label, "guid"].astype(str)))
        rng.shuffle(guids)
        n = len(guids)
        n_probe, n_sel = int(round(f_probe * n)), int(round(f_sel * n))
        roles["independent_y_eval_probe"] += guids[:n_probe].tolist()
        roles["selection_validation"] += guids[n_probe:n_probe + n_sel].tolist()
        roles["effect_audit"] += guids[n_probe + n_sel:].tolist()
    return {k: sorted(v) for k, v in roles.items()}


def summary(guids, labels, seen_in_finetuning: bool) -> dict:
    counts = {"female": int(sum(labels[g] == 0 for g in guids)),
              "male": int(sum(labels[g] == 1 for g in guids))}
    return {"n": len(guids), "counts": counts, "seen_in_finetuning": seen_in_finetuning,
            "guid_sha256": guid_hash(guids), "guids": guids}


def main() -> None:
    pool = pd.read_csv(POOL)
    keep = ~pool.in_minimal_pairs.astype(bool) & pool.overlap_degree.ne("full")
    direction = pool[keep & pool.tier.eq("A")]
    strict = pool[keep & pool.tier.eq("strict")]

    assert len(direction) == 691 and len(strict) == 407, (len(direction), len(strict))
    assert not direction.in_excluded_eval_pool.any()
    assert strict.in_excluded_eval_pool.all()

    roles = stratified_exact_split(strict, CFG["seed"])
    sets = [set(v) for v in roles.values()]
    assert all(a.isdisjoint(b) for i, a in enumerate(sets) for b in sets[i + 1:])
    assert set().union(*sets) == set(strict.guid.astype(str))

    labels = dict(zip(pool.guid.astype(str), pool.Y_label.astype(int)))
    out = {
        "seed": CFG["seed"],
        "label_map": {"0": "female", "1": "male"},
        "roles": {
            "direction_fit": summary(sorted(direction.guid.astype(str)), labels, True),
            **{k: summary(v, labels, False) for k, v in roles.items()},
        },
    }
    write_json(OUT, out)
    for name, r in out["roles"].items():
        print(f"{name:26s} n={r['n']:4d} {r['counts']}  {r['guid_sha256'][:16]}")


if __name__ == "__main__":
    main()
