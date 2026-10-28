"""Legend-driven symbol classification: show Spark a legend sheet (numbered reference symbols) next
to a crop and ask which legend entry it matches. In a real drawing the legend comes from the
drawing's own legend block; here it is built from reference crops.
"""
import json
import re
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from . import baseten_client

CELL = 128


def build_sheet(refs: Dict[int, Image.Image], cols: int = 8) -> Image.Image:
    """Grid of reference symbols, each stamped with its legend id."""
    cells = []
    for i in sorted(refs):
        im = refs[i].convert("RGB")
        s = (CELL - 16) / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
        cell = np.full((CELL, CELL, 3), 255, np.uint8)
        y, x = (CELL - im.height) // 2 + 8, (CELL - im.width) // 2
        cell[y:y + im.height, x:x + im.width] = np.array(im)[: CELL - y, : CELL - x]
        cv2.rectangle(cell, (0, 0), (CELL - 1, CELL - 1), (170, 170, 170), 1)
        cv2.putText(cell, str(i), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2)
        cells.append(cell)
    while len(cells) % cols:
        cells.append(np.full((CELL, CELL, 3), 255, np.uint8))
    rows = [np.hstack(cells[r:r + cols]) for r in range(0, len(cells), cols)]
    return Image.fromarray(np.vstack(rows))


def make_prompt(names: Dict[int, str]) -> str:
    lines = "\n".join(f"{i}: {n}" for i, n in sorted(names.items()))
    return (
        "Image 1 is a legend sheet from a piping and instrumentation diagram: each cell is a "
        "reference symbol with its legend number in red.\n" + lines + "\n"
        "Image 2 is one symbol cropped from the drawing, possibly rotated or mirrored and at lower "
        "resolution. Decide which legend number it matches. Do not deliberate over near ties; "
        "commit to the closest match quickly.\n"
        "If the symbol contains printed text, copy it as 'tag' (letters, then numbers, joined with a "
        'space); otherwise tag is "".\n'
        'Answer with only JSON: {"match": <legend number>, "tag": "...", "confidence": 0.0-1.0}'
    )


def _parse(text: str, valid) -> Dict:
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            k = int(d.get("match"))
            if k in valid:
                return {"match": k, "tag": str(d.get("tag", "")).strip(),
                        "confidence": float(d.get("confidence", 0) or 0), "ok": True}
        except (ValueError, TypeError):
            pass
    return {"match": -1, "tag": "", "confidence": 0.0, "ok": False}


def classify(sheet: Image.Image, prompt: str, crop: Image.Image, valid) -> Dict:
    r = baseten_client.spark_open_vision(prompt, [sheet, crop], max_tokens=4000)
    out = _parse(r["text"], valid)
    if not out["ok"]:
        r = baseten_client.spark_open_vision(prompt, [sheet, crop], max_tokens=9000)
        out = _parse(r["text"], valid)
    out["in_tokens"], out["out_tokens"], out["cached"] = r["in_tokens"], r["out_tokens"], r["cached"]
    return out
