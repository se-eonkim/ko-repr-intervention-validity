"""Repository paths and config loading shared by all scripts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = ("kcbert", "klue")
KOLD = ROOT / "data" / "raw" / "kold_v1.json"
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
CHECKPOINTS = {m: ROOT / "checkpoints" / f"f_{m}" for m in MODELS}
ACTIVATIONS = ROOT / "activations"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def config(name: str) -> dict:
    return json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def kold_text() -> dict:
    """guid -> comment text (the comment only; titles are not used)."""
    return {r["guid"]: r["comment"] for r in json.loads(KOLD.read_text(encoding="utf-8"))}
