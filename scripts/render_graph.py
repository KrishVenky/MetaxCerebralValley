"""Overlay a saved graph on its tile: boxes by kind, junction dots, edges, and verifier flags.

Usage: python -m scripts.render_graph 118.jpg
"""
import json
import sys

import cv2
import numpy as np
from PIL import Image

from scripts.sam_prompt_explore import DATA, TILE_H, TILE_W

Image.MAX_IMAGE_PIXELS = None
COLORS = {"valve": (0, 150, 0), "instrument": (200, 0, 200), "equipment": (0, 120, 220),
          "fitting": (120, 120, 0), "marker": (0, 150, 150), "unknown": (90, 90, 90)}

name = sys.argv[1]
data = json.load(open(DATA.parent.parent / ".cache" / "out" / "graphs" / f"{name}.json"))
ox, oy = data["tile_origin"]
tile = np.array(Image.open(DATA / name).crop((ox, oy, ox + TILE_W, oy + TILE_H)).convert("RGB"))
out = (tile * 0.5 + 125).astype(np.uint8)
nodes = {n["id"]: n for n in data["graph"]["nodes"]}
ctr = {i: ((n["box"][0] + n["box"][2]) // 2, (n["box"][1] + n["box"][3]) // 2) for i, n in nodes.items()}
for a, b in data["graph"]["edges"]:
    cv2.line(out, ctr[a], ctr[b], (220, 60, 60), 2)
for i, n in nodes.items():
    if n["kind"] == "junction":
        cv2.circle(out, ctr[i], 7, (255, 140, 0), -1)
    else:
        x1, y1, x2, y2 = n["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), COLORS.get(n["kind"], (90, 90, 90)), 3)
for f in data["findings"]:
    if f["kind"] == "dangling_end":
        cv2.circle(out, tuple(f["pos"]), 18, (0, 190, 255), 4)
cv2.imwrite(str(DATA.parent.parent / ".cache" / "out" / f"graph_{name}"),
            cv2.cvtColor(cv2.resize(out, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
