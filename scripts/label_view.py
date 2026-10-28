"""Render a tile with numbered ground-truth symbol boxes, to hand-label true connections.

Usage: python -m scripts.label_view 14.jpg
"""
import json
import sys

import cv2
import numpy as np
from PIL import Image

from scripts.sam_prompt_explore import DATA, TILE_H, TILE_W, densest_tile

Image.MAX_IMAGE_PIXELS = None
name = sys.argv[1]
row = [json.loads(l) for l in (DATA / "metadata.jsonl").open() if json.loads(l)["file_name"] == name][0]
img = Image.open(DATA / name)
(ox, oy), _ = densest_tile(img.size, row["symbols"]["bbox"])
tile = np.array(img.crop((ox, oy, ox + TILE_W, oy + TILE_H)).convert("RGB"))
sel = [i for i, b in enumerate(row["symbols"]["bbox"])
       if ox <= (b[0] + b[2]) / 2 < ox + TILE_W and oy <= (b[1] + b[3]) / 2 < oy + TILE_H]
out = tile.copy()
for k, i in enumerate(sel):
    x1, y1, x2, y2 = [int(v) for v in row["symbols"]["bbox"][i]]
    x1, x2, y1, y2 = x1 - ox, x2 - ox, y1 - oy, y2 - oy
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 3)
    t = str(k)
    cv2.putText(out, t, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 6)
    cv2.putText(out, t, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3)
p = DATA.parent.parent / ".cache" / "out" / f"label_{name}"
cv2.imwrite(str(p), cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
json.dump({"origin": [ox, oy], "ids": sel, "n": len(sel)}, open(str(p) + ".json", "w"))
print(name, len(sel), "symbols ->", p)
