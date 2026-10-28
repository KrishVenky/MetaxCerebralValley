"""Render pipe-tracing overlays on dense tiles using (cached) SAM 'symbol' boxes.

Usage: python -m scripts.pipe_test 102.jpg 106.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from lazarus import config, meta_client, pipes
from scripts.sam_prompt_explore import TILE_H, TILE_W, densest_tile

import json

Image.MAX_IMAGE_PIXELS = None
DATA = Path(__file__).resolve().parent.parent / "data" / "pid_val"
OUT = config.CACHE_DIR / "out"
OUT.mkdir(parents=True, exist_ok=True)
rows = {json.loads(l)["file_name"]: json.loads(l) for l in (DATA / "metadata.jsonl").open()}

for name in sys.argv[1:]:
    img = Image.open(DATA / name)
    (ox, oy), _ = densest_tile(img.size, rows[name]["symbols"]["bbox"])
    tile = img.crop((ox, oy, ox + TILE_W, oy + TILE_H)).convert("RGB")
    sam = meta_client.sam_segment(tile, "symbol")          # cached: free
    boxes = [(b["x1"], b["y1"], b["x2"], b["y2"]) for b in sam["boxes"]]
    arr = np.array(tile)
    res = pipes.trace(arr, boxes)
    print(f"{name}: {len(boxes)} symbols, {res['n_lines']} line components, {len(res['edges'])} edges")
    vis = pipes.render(arr, boxes, res)
    path = OUT / f"pipes_{name}"
    cv2.imwrite(str(path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    # Side-by-side with the raw tile, downsized for viewing.
    both = np.concatenate([arr, vis], axis=1)
    both = cv2.resize(both, (both.shape[1] // 2, both.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / f"pipes_{name}_pair.jpg"), cv2.cvtColor(both, cv2.COLOR_RGB2BGR))
    print("  wrote", OUT / f"pipes_{name}_pair.jpg")
