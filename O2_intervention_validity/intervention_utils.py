"""Helpers shared by the LEACE and AlterRep validity scripts."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ACTIVATIONS, SPLITS, read_json  # noqa: E402,F401


def roles() -> dict:
    return read_json(SPLITS / "intervention_roles.json")["roles"]


def role_guids(name: str) -> list:
    return list(map(str, roles()[name]["guids"]))


def gender_labels(guids: list) -> np.ndarray:
    """0 = female target, 1 = male target."""
    import pandas as pd
    df = pd.read_csv(SPLITS / "gender_pool.csv")
    mp = dict(zip(df.guid.astype(str), df.Y_label.astype(int)))
    return np.asarray([mp[g] for g in guids], dtype=np.int64)


def load_l12(model: str, guids: list) -> np.ndarray:
    with np.load(ACTIVATIONS / f"o2_l12_{model}.npz", allow_pickle=True) as d:
        order = {str(g): i for i, g in enumerate(d["guids"].tolist())}
        missing = [g for g in guids if g not in order]
        assert not missing, f"{model}: {len(missing)} activations missing"
        x = np.asarray(d["L12"], np.float32)[[order[g] for g in guids]]
    return x


def alterrep(h: np.ndarray, u: np.ndarray, mu: np.ndarray, alpha: float,
             target_sign) -> tuple:
    """AlterRep counterfactual edit (Ravfogel et al., 2021).

    Removes the projection of the centered vector on the gender subspace U and
    adds it back with magnitude alpha * |c| on the side of the target class.
    alpha = 0 removes the subspace component only.
    Returns the edited vectors and the per-sample edit norm.
    """
    hc = np.asarray(h, np.float64) - np.asarray(mu, np.float64)
    u = np.asarray(u, np.float64)
    c = hc @ u
    hn = hc - c @ u.T
    s = np.asarray(target_sign, np.float64)
    s = np.full((len(h), 1), float(s)) if s.ndim == 0 else s.reshape(-1, 1)
    hp = np.asarray(mu, np.float64) + hn + (float(alpha) * s * np.abs(c)) @ u.T
    return hp.astype(np.float32), np.linalg.norm(hp - h, axis=1)


def random_orthonormal(d: int, m: int, seed: int) -> np.ndarray:
    """Random m-dimensional subspace used as the edit-direction control."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, m)))
    return q[:, :m]
