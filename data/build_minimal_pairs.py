"""Rebuild the 39 gender minimal pairs from KOLD.

Each pair is a KOLD comment and a counterpart in which only the target-gender
expression is swapped. The released file stores the GUID and the character
edits, not the text. This script applies the edits to the KOLD comment.

The 39 pairs were selected by manual review: 70 automatically generated
candidates were checked against 16 criteria (only the target gender changes;
context and offensive expression stay the same).

Input : data/raw/kold_v1.json, data/splits/minimal_pairs.json
Output: data/processed/minimal_pairs_text.json   (contains text; not tracked)

Run   : python data/build_minimal_pairs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROCESSED, SPLITS, kold_text, read_json, write_json  # noqa: E402

OUT = PROCESSED / "minimal_pairs_text.json"


def apply_edits(text: str, edits: list) -> str:
    for start, end, repl in sorted(edits, reverse=True):
        text = text[:start] + repl + text[end:]
    return text


def main():
    text = kold_text()
    pairs = read_json(SPLITS / "minimal_pairs.json")["pairs"]
    records = []
    for p in pairs:
        original = text[p["guid"]]
        records.append({**{k: p[k] for k in ("guid", "source_group", "counterpart_group")},
                        "original_text": original,
                        "counterpart_text": apply_edits(original, p["edits"])})
    write_json(OUT, {"n_pairs": len(records), "records": records})
    print(f"saved {OUT.relative_to(OUT.parents[2])} ({len(records)} pairs)")


if __name__ == "__main__":
    main()
