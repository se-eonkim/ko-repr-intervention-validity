"""Extract final-layer (L12) [CLS] vectors for every sample used in O2.

Covers the four gender roles, the 450 offensive-probe samples and the 750
target-free control comments. Interventions act on these raw vectors, before
the classifier's pooler.

Input : data/splits/intervention_roles.json, data/splits/probe_splits.json,
        data/splits/target_free_controls.json, checkpoints/f_{kcbert,klue}
Output: activations/o2_l12_{model}.npz   (guids, L12 [n, 768])

Run   : python O2_intervention_validity/extract_l12_pool.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.encode import cls_states, load_classifier  # noqa: E402
from common.paths import ACTIVATIONS, MODELS, SPLITS, config, kold_text, read_json  # noqa: E402

LAYER = config("leace")["layer"]


def main():
    roles = read_json(SPLITS / "intervention_roles.json")["roles"]
    guids = {g for r in roles.values() for g in r["guids"]}
    guids |= set(read_json(SPLITS / "probe_splits.json")["axes"]["offensive"]["members"])
    guids |= set(read_json(SPLITS / "target_free_controls.json")["guids"])
    guids = sorted(guids)
    text = kold_text()
    ACTIVATIONS.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        tok, clf = load_classifier(model)
        x = cls_states(tok, clf, [text[g] for g in guids], layers=[LAYER])[LAYER]
        out = ACTIVATIONS / f"o2_l12_{model}.npz"
        np.savez_compressed(out, guids=np.array(guids, dtype=object), L12=x)
        print(f"saved {out.name}  ({len(guids)} samples)")


if __name__ == "__main__":
    main()
