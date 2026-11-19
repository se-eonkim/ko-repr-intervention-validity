"""O2-b: AlterRep counterfactual-flip validity on the final layer.

1. Gender subspace: INLP (iterated linear SVMs, Ravfogel et al.) on the
   direction_fit role gives up to 8 orthonormal gender directions U.
2. Independent gender probe q_Y: trained on independent_y_eval_probe; it must
   decode gender on selection_validation (BAcc >= 0.65 and above its
   permutation q95) before any flip can be judged.
3. For each setting (m in {4, 8}, alpha in {0.5, 1, 2, 4}) the selection
   vectors are flipped toward the opposite gender and scored by three gates:

   C  Completeness   in both directions, mean target probability under q_Y
                     >= max(0.70, natural target mean - 0.05), hard target
                     rate >= 0.70, paired increase > 0, and above the best of
                     10 random-subspace edits with the same m and alpha.
   S  Target-free    on 750 target-free offensive comments, editing toward
      survival       either gender lowers mean P(offensive) and the
                     offensive rate by at most 0.05 (frozen classifier head).
   R  Restricted-X   mean change of a linear offensiveness probe q_X is no
                     larger than the largest change from the 10 random edits.

A setting is valid only if q_Y is competent and C, S and R all pass. The
effect_audit role is evaluated only for valid settings; none were valid, so
no output effect is computed.

Input : activations/o2_l12_{model}.npz, data/splits/intervention_roles.json,
        data/splits/probe_splits.json, data/splits/target_free_controls.json,
        checkpoints/f_{kcbert,klue}
Output: results/o2_alterrep_validity.json

Run   : python O2_intervention_validity/alterrep/alterrep_validity.py
"""
import importlib.metadata
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from intervention_utils import (alterrep, gender_labels, load_l12,  # noqa: E402
                                random_orthonormal, role_guids)
from common.encode import head_offensive_prob, load_classifier  # noqa: E402
from common.paths import MODELS, RESULTS, SPLITS, config, read_json, write_json  # noqa: E402

CFG = config("alterrep")
G = CFG["gates"]
OUT = RESULTS / "o2_alterrep_validity.json"


def choose_c(xc, ysign, grid):
    cv = StratifiedKFold(5, shuffle=True, random_state=CFG["inlp"]["cv_seed"])
    scores = {}
    for c in grid:
        clf = LinearSVC(C=c, fit_intercept=False, class_weight="balanced", max_iter=20_000)
        scores[str(c)] = float(cross_val_score(clf, xc, ysign, cv=cv, scoring="balanced_accuracy").mean())
    best = max(scores.values())
    return min(float(k) for k, v in scores.items() if abs(v - best) <= 1e-12), scores


def fit_inlp(x, y, grid, max_m):
    """INLP with Gram-Schmidt; each direction is signed so female scores higher."""
    mu = x.mean(axis=0).astype(np.float64)
    xc = np.asarray(x, np.float64) - mu
    ysign = np.where(y == 0, 1, -1)
    c_sel, c_scores = choose_c(xc, ysign, grid)
    cols, its = [], []
    for it in range(max_m):
        res = xc if not cols else xc - (xc @ np.column_stack(cols)) @ np.column_stack(cols).T
        cv = StratifiedKFold(5, shuffle=True, random_state=CFG["inlp"]["cv_seed"] + it)
        probe = LinearSVC(C=c_sel, fit_intercept=False, class_weight="balanced", max_iter=20_000)
        bacc = float(cross_val_score(probe, res, ysign, cv=cv, scoring="balanced_accuracy").mean())
        probe.fit(res, ysign)
        w = np.asarray(probe.coef_[0], np.float64)
        for _ in range(2):
            for u in cols:
                w -= np.dot(w, u) * u
        norm = float(np.linalg.norm(w))
        if norm <= 1e-10:
            raise RuntimeError(f"INLP rank exhausted at iteration {it + 1}")
        u = w / norm
        if (res[y == 0] @ u).mean() < (res[y == 1] @ u).mean():
            u = -u
        cols.append(u)
        its.append({"iteration": it + 1, "cv_bacc": bacc})
    U = np.column_stack(cols)
    meta = {"selected_C": c_sel, "C_cv": c_scores, "iterations": its,
            "ortho_gap": float(np.max(np.abs(U.T @ U - np.eye(max_m))))}
    return mu.astype(np.float32), U.astype(np.float32), meta


def permutation_q95(x, y):
    p = CFG["gender_probe"]
    rng = np.random.default_rng(p["permutation_seed"])
    vals = []
    for i in range(p["permutation_n"]):
        yp = rng.permutation(y)
        cv = StratifiedKFold(5, shuffle=True, random_state=30_000 + i)
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            C=p["C"], class_weight="balanced", max_iter=p["max_iter"]))
        vals.append(float(cross_val_score(clf, x, yp, cv=cv, scoring="balanced_accuracy").mean()))
    return float(np.quantile(vals, 0.95))


def offensive_probe(model):
    """q_X: linear offensiveness probe on raw L12 [CLS] (no scaling)."""
    axis = read_json(SPLITS / "probe_splits.json")["axes"]["offensive"]
    guids = sorted(axis["members"])
    labels = np.asarray([axis["members"][g] for g in guids], int)
    train = set(axis["splits"][CFG["offensive_probe"]["train_split_seed"]]["train"])
    idx = [i for i, g in enumerate(guids) if g in train]
    x = load_l12(model, guids)
    return LogisticRegression(C=CFG["offensive_probe"]["C"],
                              max_iter=CFG["offensive_probe"]["max_iter"]).fit(x[idx], labels[idx])


def direction_metrics(qy, x, y, edited, source, target):
    src, tgt = y == source, y == target
    col = list(qy.classes_).index(target)
    before = qy.predict_proba(x[src])[:, col]
    after = qy.predict_proba(edited[src])[:, col]
    natural = float(qy.predict_proba(x[tgt])[:, col].mean())
    return {"n": int(src.sum()), "mean_target_prob": float(after.mean()),
            "natural_target_mean": natural,
            "threshold": max(G["completeness_min_target_prob"], natural - G["completeness_natural_margin"]),
            "hard_target_rate": float((qy.predict(edited[src]) == target).mean()),
            "paired_increase": float((after - before).mean())}


def run_model(model):
    xfit = load_l12(model, role_guids("direction_fit"))
    yfit = gender_labels(role_guids("direction_fit"))
    xp, yp = load_l12(model, role_guids("independent_y_eval_probe")), gender_labels(role_guids("independent_y_eval_probe"))
    xs, ys = load_l12(model, role_guids("selection_validation")), gender_labels(role_guids("selection_validation"))
    mu, U, inlp = fit_inlp(xfit, yfit, CFG["inlp"]["C_grid"], CFG["inlp"]["max_m"])

    p = CFG["gender_probe"]
    qy = make_pipeline(StandardScaler(), LogisticRegression(
        C=p["C"], class_weight="balanced", max_iter=p["max_iter"])).fit(xp, yp)
    comp_bacc = float(balanced_accuracy_score(ys, qy.predict(xs)))
    comp_q95 = permutation_q95(xs, ys)
    competent = comp_bacc >= p["competence_min_bacc"] and comp_bacc > comp_q95

    qx = offensive_probe(model)
    off_col = list(qx.classes_).index(1)
    qx_before = qx.predict_proba(xs)[:, off_col]
    _, clf = load_classifier(model)
    head = head_offensive_prob(clf)
    x_tf = load_l12(model, read_json(SPLITS / "target_free_controls.json")["guids"])
    tf_before = head(x_tf)
    signs = np.where(ys == 0, -1, 1)  # female -> male (-1), male -> female (+1)

    candidates, removal = [], []
    for m in CFG["candidates"]["m"]:
        u = U[:, :m]
        for alpha in CFG["candidates"]["alpha"]:
            edited, norms = alterrep(xs, u, mu, alpha, signs)
            f2m = direction_metrics(qy, xs, ys, edited, 0, 1)
            m2f = direction_metrics(qy, xs, ys, edited, 1, 0)
            if alpha == 0:
                removal.append({"m": m, "female_to_male": f2m, "male_to_female": m2f})
                continue
            dx = float(np.abs(qx.predict_proba(edited)[:, off_col] - qx_before).mean())
            surv = {}
            for sign in (-1, 1):
                poff = head(alterrep(x_tf, u, mu, alpha, sign)[0])
                surv[str(sign)] = {"mean_drop": float(tf_before.mean() - poff.mean()),
                                   "rate_drop": float((tf_before >= .5).mean() - (poff >= .5).mean())}
            rand = []
            for rs in CFG["random_direction_seeds"]:
                red, rnorm = alterrep(xs, random_orthonormal(768, m, rs), mu, alpha, signs)
                rand.append({"seed": rs,
                             "female_to_male": direction_metrics(qy, xs, ys, red, 0, 1)["mean_target_prob"],
                             "male_to_female": direction_metrics(qy, xs, ys, red, 1, 0)["mean_target_prob"],
                             "D_X": float(np.abs(qx.predict_proba(red)[:, off_col] - qx_before).mean()),
                             "mean_edit_norm": float(rnorm.mean())})
            c_pass = all([
                f2m["mean_target_prob"] >= f2m["threshold"], f2m["hard_target_rate"] >= G["completeness_min_hard_rate"],
                f2m["paired_increase"] > 0,
                m2f["mean_target_prob"] >= m2f["threshold"], m2f["hard_target_rate"] >= G["completeness_min_hard_rate"],
                m2f["paired_increase"] > 0,
                f2m["mean_target_prob"] > max(r["female_to_male"] for r in rand),
                m2f["mean_target_prob"] > max(r["male_to_female"] for r in rand)])
            s_pass = all(v["mean_drop"] <= G["survival_max_mean_drop"]
                         and v["rate_drop"] <= G["survival_max_rate_drop"] for v in surv.values())
            r_pass = dx <= max(r["D_X"] for r in rand)
            candidates.append({
                "m": m, "alpha": alpha, "female_to_male": f2m, "male_to_female": m2f,
                "survival": surv, "restricted_D_X": dx, "mean_edit_norm": float(norms.mean()),
                "random": rand,
                "gates": {"C": c_pass, "S": s_pass, "R": r_pass},
                "valid": bool(competent and c_pass and s_pass and r_pass)})
    return {"inlp": inlp,
            "gender_probe": {"bacc": comp_bacc, "null_q95": comp_q95, "competent": competent},
            "removal_baselines": removal, "candidates": candidates,
            "n_valid": sum(c["valid"] for c in candidates)}


def main():
    results = {m: run_model(m) for m in MODELS}
    write_json(OUT, {
        "config": CFG,
        "packages": {n: importlib.metadata.version(n)
                     for n in ("numpy", "scikit-learn", "torch", "transformers")},
        "results": results})
    for model, r in results.items():
        g = [c["gates"] for c in r["candidates"]]
        print(f"{model}: gender probe BAcc {r['gender_probe']['bacc']:.3f} "
              f"(q95 {r['gender_probe']['null_q95']:.3f}) | C/S/R passed "
              f"{sum(x['C'] for x in g)}/{sum(x['S'] for x in g)}/{sum(x['R'] for x in g)} "
              f"of {len(g)} | valid {r['n_valid']}")


if __name__ == "__main__":
    main()
