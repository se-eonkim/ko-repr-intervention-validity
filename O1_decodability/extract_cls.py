"""Extract [CLS] vectors from all 13 hidden states (embedding output L0 and
Transformer layers L1-L12) for every sample in the O1 probe splits.

Input : data/splits/probe_splits.json, checkpoints/f_{kcbert,klue}
Output: activations/o1_cls_{model}.npz   (arrays L0 ... L12, each [n, 768])

Run   : python O1_decodability/extract_cls.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.encode import cls_states, load_classifier  # noqa: E402
from common.paths import ACTIVATIONS, MODELS, SPLITS, kold_text, read_json  # noqa: E402


def main():
    splits = read_json(SPLITS / "probe_splits.json")
    guids = sorted({g for ax in splits["axes"].values() for g in ax["members"]})
    text = kold_text()
    ACTIVATIONS.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        tok, clf = load_classifier(model)
        states = cls_states(tok, clf, [text[g] for g in guids])
        out = ACTIVATIONS / f"o1_cls_{model}.npz"
        np.savez_compressed(out, guids=np.array(guids, dtype=object),
                            **{f"L{L}": v for L, v in states.items()})
        print(f"saved {out.name}  ({len(guids)} samples x {len(states)} layers)")


if __name__ == "__main__":
    main()
