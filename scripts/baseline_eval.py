"""Baseline for the connection benchmark: one plain Muse Spark call per labelled tile.

The model is shown the same tile the labels were drawn on, with the ground-truth symbols already boxed
and numbered (our pipeline has to find symbols itself, so this baseline gets that part for free), and is
asked to list which numbered symbols are joined by a pipe. Scored with the same truth sets as
scripts/connection_eval.py: recall over all labelled links, recall over solid pipes, precision.

Usage: python -m scripts.baseline_eval
"""
import json
import re

import cv2
import numpy as np
from PIL import Image

from lazarus import meta_client
from scripts.connection_eval import LABELS, closure, rows  # noqa: F401  (importing runs the pipeline eval; output is above ours)
from scripts.sam_prompt_explore import DATA, TILE_H, TILE_W, densest_tile

Image.MAX_IMAGE_PIXELS = None

PROMPT = (
    "This is part of a piping and instrumentation diagram (P&ID). Symbols are boxed and numbered in red.\n"
    "List every pair of numbered symbols that are joined by a pipe or line with no other numbered symbol "
    "between them. Lines may pass through junctions and corners. Ignore symbols you cannot connect.\n"
    "Answer with only JSON: a list of [a, b] pairs of symbol numbers, e.g. [[0, 3], [3, 7]]."
)


def render(name):
    row = rows[name]
    img = Image.open(DATA / name)
    (ox, oy), _ = densest_tile(img.size, row["symbols"]["bbox"])
    tile = np.array(img.crop((ox, oy, ox + TILE_W, oy + TILE_H)).convert("RGB"))
    sel = [i for i, b in enumerate(row["symbols"]["bbox"])
           if ox <= (b[0] + b[2]) / 2 < ox + TILE_W and oy <= (b[1] + b[3]) / 2 < oy + TILE_H]
    for k, i in enumerate(sel):
        x1, y1, x2, y2 = [int(v) for v in row["symbols"]["bbox"][i]]
        x1, x2, y1, y2 = x1 - ox, x2 - ox, y1 - oy, y2 - oy
        cv2.rectangle(tile, (x1, y1), (x2, y2), (255, 0, 0), 3)
        for color, w in (((255, 255, 255), 6), ((255, 0, 0), 3)):
            cv2.putText(tile, str(k), (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, w)
    return Image.fromarray(tile)


def parse(text):
    m = re.search(r"\[\s*\[.*\]\s*\]", text, re.S)
    if not m:
        return set()
    try:
        return {tuple(sorted((int(a), int(b)))) for a, b in json.loads(m.group(0)) if a != b}
    except (ValueError, TypeError):
        return set()


tot = dict(truth=0, rec=0, truth_solid=0, rec_solid=0, pred=0, cor=0)
print("\n=== BASELINE: one Muse Spark call, ground-truth symbols given ===")
for name, lab in LABELS.items():
    if name.startswith("_"):
        continue
    excl = set(lab["exclude"])
    syms = lambda x: isinstance(x, int)
    truth = {e for e in closure([(a, b) for a, b, d in lab["segments"]], syms) if not (set(e) & excl)}
    truth_solid = {e for e in closure([(a, b) for a, b, d in lab["segments"] if not d], syms) if not (set(e) & excl)}
    r = meta_client.spark_vision(PROMPT, [render(name)], max_output_tokens=40000)
    pred = {e for e in parse(r["text"]) if not (set(e) & excl)}
    rec, rec_s, cor = truth & pred, truth_solid & pred, pred & truth
    print(f"{name:9s} truth {len(truth):3d} recovered {len(rec):3d} | predicted {len(pred):3d} correct {len(cor):3d} | solid {len(rec_s)}/{len(truth_solid)}  ({'cached' if r['cached'] else 'live'})")
    for k, v in (("truth", len(truth)), ("rec", len(rec)), ("truth_solid", len(truth_solid)),
                 ("rec_solid", len(rec_s)), ("pred", len(pred)), ("cor", len(cor))):
        tot[k] += v
print(f"\nbaseline recall, all labelled links: {tot['rec']}/{tot['truth']} = {tot['rec'] / tot['truth']:.0%}")
print(f"baseline recall, solid pipes only:   {tot['rec_solid']}/{tot['truth_solid']} = {tot['rec_solid'] / tot['truth_solid']:.0%}")
print(f"baseline precision:                  {tot['cor']}/{tot['pred']} = {tot['cor'] / max(1, tot['pred']):.0%}")
