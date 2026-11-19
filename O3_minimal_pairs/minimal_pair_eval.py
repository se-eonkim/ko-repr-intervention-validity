"""O3: input-level output difference on 39 gender minimal pairs.

For each pair, delta = offensive logit(female version) - offensive logit(male
version), regardless of which side is the original KOLD comment. The two
models are evaluated separately.

Primary   mean delta with a 95% CI from 10,000 pair-level bootstrap resamples.
Secondary median and absolute delta, direction counts, label flips,
          a sign-flip permutation p-value (descriptive only), and the mean
          over female-origin and male-origin pairs weighted equally.

This measures how the frozen classifier responds to the swapped inputs. It is
not evidence about whether internal gender information is used.

Input : data/processed/minimal_pairs_text.json, checkpoints/f_{kcbert,klue}
Output: results/o3_minimal_pairs.csv, results/o3_minimal_pairs_summary.json

Run   : python O3_minimal_pairs/minimal_pair_eval.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.encode import ENC, load_classifier  # noqa: E402
from common.paths import MODELS, PROCESSED, RESULTS, config, read_json, write_json  # noqa: E402

CFG = config("minimal_pairs")
OFF = ENC["offensive_label_index"]
PAIRS_CSV = RESULTS / "o3_minimal_pairs.csv"
SUMMARY = RESULTS / "o3_minimal_pairs_summary.json"


def female_male(rec):
    if rec["source_group"] == "gender-female":
        return rec["original_text"], rec["counterpart_text"]
    return rec["counterpart_text"], rec["original_text"]


@torch.no_grad()
def score(tok, clf, texts):
    enc = tok(texts, padding=True, truncation=True, max_length=ENC["max_length"], return_tensors="pt")
    logits = clf(**enc).logits
    return (logits[:, OFF].numpy(), F.softmax(logits, dim=1)[:, OFF].numpy(),
            logits.argmax(dim=1).numpy())


def eval_model(model, recs):
    tok, clf = load_classifier(model)
    fl, fp, fpred = score(tok, clf, [female_male(r)[0] for r in recs])
    ml, mp, mpred = score(tok, clf, [female_male(r)[1] for r in recs])
    delta, n = fl - ml, len(recs)

    rows = [{"model": model, "pair_id": i, "guid": r["guid"], "source_group": r["source_group"],
             "logit_F": float(fl[i]), "logit_M": float(ml[i]),
             "prob_F": float(fp[i]), "prob_M": float(mp[i]),
             "delta_logit": float(delta[i]), "pred_F": int(fpred[i]), "pred_M": int(mpred[i]),
             "flip": bool(fpred[i] != mpred[i])} for i, r in enumerate(recs)]

    b = CFG["bootstrap"]
    idx = np.random.default_rng(b["seed"]).integers(0, n, size=(b["n"], n))
    boot = delta[idx].mean(axis=1)
    p = CFG["sign_flip_permutation"]
    signs = np.random.default_rng(p["seed"]).choice([-1.0, 1.0], size=(p["n"], n))
    perm = (signs * delta).mean(axis=1)
    fem = delta[[i for i, r in enumerate(recs) if r["source_group"] == "gender-female"]]
    mal = delta[[i for i, r in enumerate(recs) if r["source_group"] == "gender-male"]]

    summary = {
        "n_pairs": n,
        "mean_delta_logit": float(delta.mean()),
        "bootstrap_95ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "dz": float(delta.mean() / delta.std(ddof=1)),
        "median_delta_logit": float(np.median(delta)),
        "mean_abs_delta_logit": float(np.abs(delta).mean()),
        "direction": {"female_higher": int((delta > 0).sum()), "male_higher": int((delta < 0).sum()),
                      "zero": int((delta == 0).sum())},
        "label_flips": int(sum(r["flip"] for r in rows)),
        "sign_flip_p_two_sided": float((np.sum(np.abs(perm) >= abs(delta.mean())) + 1) / (p["n"] + 1)),
        "source_balanced_mean_delta": float(0.5 * fem.mean() + 0.5 * mal.mean()),
    }
    return rows, summary


def main():
    recs = read_json(PROCESSED / "minimal_pairs_text.json")["records"]
    all_rows, summaries = [], {}
    for model in MODELS:
        rows, summaries[model] = eval_model(model, recs)
        all_rows += rows
    PAIRS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(PAIRS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    write_json(SUMMARY, {"config": CFG, "models": summaries})
    for model, s in summaries.items():
        lo, hi = s["bootstrap_95ci"]
        d = s["direction"]
        print(f"{model}: mean delta {s['mean_delta_logit']:+.3f} [{lo:+.3f}, {hi:+.3f}] "
              f"dz {s['dz']:+.2f}  direction {d['female_higher']}/{d['male_higher']}")


if __name__ == "__main__":
    main()
