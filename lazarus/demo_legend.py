"""Legend for the Digitize-PID synthetic dataset: names for its 32 visual symbol classes and a
builder that cuts reference crops from a chosen set of drawings. Names are hand-written from the
reference symbols; in a real drawing the legend block supplies them.
"""
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

from PIL import Image

from . import legend
from .classify import crop_symbol

Image.MAX_IMAGE_PIXELS = None

NAMES = {
    1: "valve, bow-tie with X through it", 2: "valve, bow-tie with circle and X",
    3: "valve, bow-tie with solid dot", 4: "valve, plain bow-tie (open)",
    5: "valve, bow-tie with oval", 6: "valve, diverter with solid dot",
    7: "valve, bow-tie with stem bar", 8: "check valve, triangle with bar",
    9: "valve, bow-tie with D-shaped arc", 10: "three-way valve, solid triangle",
    11: "valve, solid bow-tie, vertical", 12: "valve, solid bow-tie, horizontal",
    13: "valve, solid bow-tie with narrow waist", 14: "valve, bow-tie with half-circle actuator",
    15: "inline device, open circle on pipe", 16: "inline device, solid circle on pipe",
    17: "equipment, bars with oval", 18: "equipment, bars with solid and open oval",
    19: "equipment, bars with stacked ovals", 20: "equipment, trapezoid hopper",
    21: "flange, two bars across pipe", 22: "equipment, box with zigzag coil",
    23: "equipment, twin cylinders", 24: "marker, solid arrowhead",
    25: "valve, circle containing X", 26: "instrument, circle with text (GRO/GRI/GI)",
    27: "instrument, circle with text (RO-10)", 28: "instrument, circle with text (SDL)",
    29: "instrument, circle with text (DDL)", 30: "instrument, box with text (STA)",
    31: "instrument, box with circle (ZLC)", 32: "instrument, box with text (LG-10)",
}


def kind_of(name: str) -> str:
    head = name.split(",")[0]
    return {"check valve": "valve", "three-way valve": "valve", "inline device": "equipment",
            "flange": "fitting"}.get(head, head)


KIND = {k: kind_of(v) for k, v in NAMES.items()}


def build(data_dir: Path, drawings: Iterable[str], seed: int = 1) -> Tuple[Image.Image, str, Dict]:
    """Legend sheet + prompt from one random reference crop per class, drawn only from `drawings`."""
    rows = {}
    for line in (data_dir / "metadata.jsonl").open():
        r = json.loads(line)
        if r["file_name"] in set(drawings):
            rows[r["file_name"]] = r
    by: Dict[int, list] = {}
    for fn, r in rows.items():
        for b, l in zip(r["symbols"]["bbox"], r["symbols"]["labels"]):
            by.setdefault(l, []).append((fn, b))
    rng = random.Random(seed)
    imgs = {fn: Image.open(data_dir / fn).convert("RGB") for fn in rows}
    refs = {}
    for l in sorted(NAMES):
        fn, b = rng.choice(by[l])
        refs[l] = crop_symbol(imgs[fn], b, size=160)
    return legend.build_sheet(refs), legend.make_prompt(NAMES), refs
