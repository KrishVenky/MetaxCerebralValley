"""A/B: does showing the legend image next to each symbol change the classifier's answers?

Re-classifies the symbols of an existing real_test run twice on the same boxes, with the legend text
context only (A) and with the legend crop as a second image (B), then writes a contact sheet of the
disagreements so they can be judged by eye (the real scans have no ground truth).

Usage: python -m scripts.legend_ref_test omre_pid.jpg
"""
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image

from lazarus import classify, pipeline
from lazarus.codes import Codes

Image.MAX_IMAGE_PIXELS = None
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
name = sys.argv[1]
d = json.load(open(ROOT / ".cache" / "out" / "real" / f"{name}.json"))
img = Image.open(ROOT / "data" / "real" / name).convert("RGB")
scale = d["scale"]
img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
codes = Codes.from_entries(d["legend"])
ref = img.crop(tuple(d["legend_box"]))
k = pipeline.LEGEND_REF_SIDE / max(ref.size)
ref = ref.resize((int(ref.width * k), int(ref.height * k)), Image.LANCZOS)
nodes = [n for n in d["graph"]["nodes"] if n["kind"] != "junction"]
crops = [classify.crop_symbol(img, n["box"]) for n in nodes]
ctx_text = codes.context()


def run(reference):
    with ThreadPoolExecutor(12) as ex:
        return list(ex.map(lambda c: classify.classify(c, ctx_text, reference), crops))


A, B = run(None), run(ref)
tok = lambda rs: sum(r["in_tokens"] + r["out_tokens"] for r in rs if not r["cached"])
same = sum(a["type"] == b["type"] for a, b in zip(A, B))
print(f"{len(nodes)} symbols; same type {same}; kind changed {sum(a['kind'] != b['kind'] for a, b in zip(A, B))}")
print("A types", dict(Counter(a["type"] for a in A)))
print("B types", dict(Counter(b["type"] for b in B)))
print(f"tag valid A {sum(codes.valid(a['tag']) for a in A if a['tag'])}/{sum(bool(a['tag']) for a in A)}, "
      f"B {sum(codes.valid(b['tag']) for b in B if b['tag'])}/{sum(bool(b['tag']) for b in B)}")
print(f"mean confidence A {np.mean([a['confidence'] for a in A]):.2f} B {np.mean([b['confidence'] for b in B]):.2f}; "
      f"tokens this run (uncached) {tok(A) + tok(B)}")
dis = [(i, a, b) for i, (a, b) in enumerate(zip(A, B)) if a["type"] != b["type"]]
print("disagreements:")
rows = []
for i, a, b in dis:
    print(f"  #{i} id {nodes[i]['id']} A={a['type']} ({a['confidence']:.2f}) B={b['type']} ({b['confidence']:.2f})")
    c = np.array(classify.crop_symbol(img, nodes[i]["box"], size=160))
    pad = np.full((160, 160, 3), 255, np.uint8)
    pad[:c.shape[0], :c.shape[1]] = c
    cv2.putText(pad, f"#{i}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 1)
    rows.append((pad, f"A {a['type']}", f"B {b['type']}"))
if rows:
    cols = 6
    h = (len(rows) + cols - 1) // cols
    sheet = np.full((h * 200, cols * 170, 3), 255, np.uint8)
    for j, (pad, ta, tb) in enumerate(rows):
        y, x = (j // cols) * 200, (j % cols) * 170
        sheet[y:y + 160, x:x + 160] = pad
        cv2.putText(sheet, ta[:24], (x, y + 176), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(sheet, tb[:24], (x, y + 192), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 160), 1)
    cv2.imwrite(str(ROOT / ".cache" / "out" / "legend_ref_disagree.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
