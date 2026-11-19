"""O2-a: LEACE erasure validity across five repeated splits.

For each model and split seed, the 407 strict gender samples (unseen during
fine-tuning) are split 70/30 into fit and audit sets. LEACE is fitted on the
fit set and applied to the audit set. A fresh probe is then trained on the
audit vectors (repeated 5-fold CV, 5x10) before and after erasure, scored by
balanced accuracy (BAcc).

A split passes when
  original BAcc > q95 of its label-permutation null (gender is decodable),
  erased BAcc <= q95 of its own null (erasure reached chance level), and
  the paired drop is positive.
Erasure counts as consistent only if all five splits pass.

Input : activations/o2_l12_{model}.npz, data/splits/intervention_roles.json
Output: results/o2_leace_repeated_splits.json

Run   : python O2_intervention_validity/leace/leace_repeated_splits.py
"""
import importlib.metadata
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from intervention_utils import gender_labels, load_l12, role_guids  # noqa: E402
from common.paths import MODELS, RESULTS, config, write_json  # noqa: E402

CFG = config("leace")
OUT = RESULTS / "o2_leace_repeated_splits.json"
STRICT_ROLES = ("independent_y_eval_probe", "selection_validation", "effect_audit")


def make_probe(seed: int):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=CFG["probe"]["C"], max_iter=CFG["probe"]["max_iter"], random_state=seed))


def cv_bacc(x, y, splits, seed: int) -> float:
    scores = []
    for tr, te in splits:
        clf = make_probe(seed).fit(x[tr], y[tr])
        scores.append(balanced_accuracy_score(y[te], clf.predict(x[te])))
    return float(np.mean(scores))


def perm_q95(x, y, outer_seed: int, model: str, tag: int) -> float:
    p = CFG["permutation"]
    base = p["seed_offset"] + outer_seed + p["model_offset"][model] + tag
    rng = np.random.default_rng(base)
    means = []
    for pid in range(p["n"]):
        yp = rng.permutation(y)
        cv = StratifiedKFold(p["n_splits"], shuffle=True, random_state=base + 10_000 + pid)
        means.append(cv_bacc(x, yp, list(cv.split(x, yp)), base + pid))
    return float(np.quantile(means, p["quantile"]))


def main() -> None:
    import torch
    from concept_erasure import LeaceFitter

    guids = sorted(g for r in STRICT_ROLES for g in role_guids(r))
    y = gender_labels(guids)
    assert len(guids) == 407
    rows = []
    for model in MODELS:
        x = load_l12(model, guids)
        for seed in CFG["outer_seeds"]:
            fit_idx, audit_idx = train_test_split(
                np.arange(len(y)), test_size=CFG["audit_fraction"], random_state=seed, stratify=y)
            xf, yf = x[fit_idx], y[fit_idx].astype(np.float32)
            xa, ya = x[audit_idx], y[audit_idx]
            fitter = LeaceFitter(x_dim=768, z_dim=1, dtype=torch.float32, **CFG["eraser"])
            fitter.update(torch.from_numpy(xf), torch.from_numpy(yf)[:, None])
            xe = fitter.eraser(torch.from_numpy(xa)).detach().cpu().numpy().astype(np.float32)

            cv_seed = CFG["cv"]["seed_offset"] + seed
            cv = RepeatedStratifiedKFold(n_splits=CFG["cv"]["n_splits"],
                                         n_repeats=CFG["cv"]["n_repeats"], random_state=cv_seed)
            splits = list(cv.split(xa, ya))
            original = cv_bacc(xa, ya, splits, cv_seed)
            erased = cv_bacc(xe, ya, splits, cv_seed)
            q95_o = perm_q95(xa, ya, seed, model, CFG["permutation"]["tag_original"])
            q95_e = perm_q95(xe, ya, seed, model, CFG["permutation"]["tag_erased"])
            checks = {"original_above_null": original > q95_o,
                      "erased_not_above_null": erased <= q95_e,
                      "drop_positive": original - erased > 0}
            rows.append({"model": model, "seed": seed, "fit_n": len(fit_idx),
                         "audit_n": len(audit_idx), "original_bacc": original,
                         "erased_bacc": erased, "null_q95_original": q95_o,
                         "null_q95_erased": q95_e, "checks": checks,
                         "pass": bool(all(checks.values()))})
            print(f"{model} seed {seed}: {original:.4f} -> {erased:.4f} "
                  f"(q95 {q95_e:.4f}) pass={rows[-1]['pass']}")

    summary = {}
    for model in MODELS:
        mr = [r for r in rows if r["model"] == model]
        n_pass = sum(r["pass"] for r in mr)
        summary[model] = {"passes": n_pass, "of": len(mr), "consistent": n_pass == len(mr),
                          "mean_original_bacc": float(np.mean([r["original_bacc"] for r in mr])),
                          "mean_erased_bacc": float(np.mean([r["erased_bacc"] for r in mr]))}
    write_json(OUT, {
        "config": CFG,
        "packages": {n: importlib.metadata.version(n)
                     for n in ("numpy", "scikit-learn", "torch", "concept-erasure")},
        "rows": rows, "summary": summary})
    for model, s in summary.items():
        print(f"{model}: {s['passes']}/{s['of']} splits pass")


if __name__ == "__main__":
    main()
