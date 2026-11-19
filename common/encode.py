"""Load a fine-tuned classifier and read its [CLS] hidden states.

The classifier is frozen (eval mode, float32, no gradients). Inputs are the
comment text only, truncated to 128 tokens, as in fine-tuning.
"""
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common.paths import CHECKPOINTS, config

ENC = config("encoding")


def load_classifier(model: str):
    tok = AutoTokenizer.from_pretrained(str(CHECKPOINTS[model]))
    clf = AutoModelForSequenceClassification.from_pretrained(str(CHECKPOINTS[model]))
    return tok, clf.float().eval().requires_grad_(False)


@torch.no_grad()
def cls_states(tok, clf, texts: list, layers=None) -> dict:
    """Return {layer: [n, 768] float32} raw [CLS] vectors for the given layers."""
    layers = list(range(ENC["n_hidden_states"])) if layers is None else list(layers)
    out = {L: np.zeros((len(texts), 768), dtype=np.float32) for L in layers}
    bs = ENC["batch_size"]
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True,
                  max_length=ENC["max_length"], return_tensors="pt")
        hs = clf(**enc, output_hidden_states=True).hidden_states
        for L in layers:
            out[L][i:i + bs] = hs[L][:, 0, :].cpu().numpy()
    return out


def head_offensive_prob(clf):
    """Map raw final-layer [CLS] vectors to P(offensive) through the frozen head
    (BertPooler -> dropout in eval mode -> classifier)."""
    off = ENC["offensive_label_index"]

    @torch.no_grad()
    def predict(h: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.asarray(h, np.float32))
        pooled = clf.dropout(torch.tanh(clf.bert.pooler.dense(t)))
        return torch.softmax(clf.classifier(pooled), -1)[:, off].cpu().numpy()
    return predict
