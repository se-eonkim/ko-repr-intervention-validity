"""O1: layer-wise linear decoding of gender from [CLS] vectors.

For each model, layer (L0-L12) and split seed, a linear probe is trained on
the train GUIDs and scored on the test GUIDs (macro-F1). The permutation
control trains the same probe on permuted training labels (one fixed
permutation per seed, reused across layers) and scores it on the true test
labels. High real F1 above the control means gender is linearly decodable;
it does not mean the classifier uses it.

Input : activations/o1_cls_{model}.npz, data/splits/probe_splits.json
Output: results/o1_layerwise_probe.csv

Run   : python O1_decodability/layerwise_probe.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ACTIVATIONS, MODELS, RESULTS, SPLITS, config, read_json  # noqa: E402

CFG = config("o1_probe")
N_LAYERS = config("encoding")["n_hidden_states"]
OUT = RESULTS / "o1_layerwise_probe.csv"


def macro_f1(xtr, ytr, xte, yte) -> float:
    sc = StandardScaler().fit(xtr)
    clf = LogisticRegression(C=CFG["C"], max_iter=CFG["max_iter"]).fit(sc.transform(xtr), ytr)
    return f1_score(yte, clf.predict(sc.transform(xte)), average="macro")


def main():
    axis = read_json(SPLITS / "probe_splits.json")["axes"]["gender"]
    members = axis["members"]
    rows = []
    for model in MODELS:
        with np.load(ACTIVATIONS / f"o1_cls_{model}.npz", allow_pickle=True) as d:
            gidx = {str(g): i for i, g in enumerate(d["guids"])}
            acts = {L: d[f"L{L}"] for L in range(N_LAYERS)}
        for seed, split in axis["splits"].items():
            tr, te = split["train"], split["test"]
            ytr = np.array([members[g] for g in tr])
            yte = np.array([members[g] for g in te])
            ytr_perm = np.random.RandomState(int(seed)).permutation(ytr)
            tri, tei = [gidx[g] for g in tr], [gidx[g] for g in te]
            for L in range(N_LAYERS):
                x = acts[L]
                rows.append({"model": model, "layer": L, "seed": int(seed),
                             "real_f1": macro_f1(x[tri], ytr, x[tei], yte),
                             "ctrl_f1": macro_f1(x[tri], ytr_perm, x[tei], yte)})
        print(f"{model} done")
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    g = df.groupby(["model", "layer"])[["real_f1", "ctrl_f1"]].agg(["mean", "std"])
    for model in MODELS:
        m = g.loc[model]
        peak = int(m[("real_f1", "mean")].idxmax())
        print(f"{model}: L12 {m.loc[12, ('real_f1', 'mean')]:.3f}±{m.loc[12, ('real_f1', 'std')]:.3f}"
              f" | peak L{peak} {m.loc[peak, ('real_f1', 'mean')]:.3f}"
              f" | control mean L1-12 {m.loc[1:12, ('ctrl_f1', 'mean')].mean():.3f}")


if __name__ == "__main__":
    main()
