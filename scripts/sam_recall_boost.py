"""Can we raise SAM symbol recall? Compare strategies on the same 12 dense tiles.

A  full tile, prompt "symbol"                       (baseline, cached)
B  A, plus extra prompts only when the tile is sparse or silent
C  2x2 overlapping sub-tiles, prompt "symbol", boxes mapped back and de-duplicated
D  C upscaled 2x before sending

Usage: python -m scripts.sam_recall_boost
"""
import json

import numpy as np
from PIL import Image

from lazarus import ledger, meta_client, pipes
from scripts.sam_prompt_explore import DATA, TILE_H, TILE_W, densest_tile, iou

Image.MAX_IMAGE_PIXELS = None
EXTRA = ["symbols", "diagram symbol", "small symbol", "component"]
rows = {json.loads(l)["file_name"]: json.loads(l) for l in (DATA / "metadata.jsonl").open()}


def boxes_of(r):
    return [(b["x1"], b["y1"], b["x2"], b["y2"], b.get("conf") or 0.0) for b in r["boxes"]]


def nms(bs, thr=0.5):
    out = []
    for b in sorted(bs, key=lambda b: -(b[4] or 0)):
        if all(iou(b[:4], o[:4]) < thr for o in out):
            out.append(b)
    return out


def subtiles(tile, scale=1):
    w, h = tile.size
    sw, sh = int(w * 0.55), int(h * 0.55)
    for x in (0, w - sw):
        for y in (0, h - sh):
            sub = tile.crop((x, y, x + sw, y + sh))
            if scale != 1:
                sub = sub.resize((sw * scale, sh * scale), Image.LANCZOS)
            yield x, y, sub


def run_subtiles(tile, scale):
    allb = []
    for x, y, sub in subtiles(tile, scale):
        for bx in boxes_of(meta_client.sam_segment(sub, "symbol")):
            allb.append((bx[0] / scale + x, bx[1] / scale + y, bx[2] / scale + x, bx[3] / scale + y, bx[4]))
    return nms(allb)


def score(preds, gt):
    p = [q[:4] for q in preds]
    hit = sum(1 for g in gt if any(iou(g, q) >= 0.5 for q in p))
    tp = sum(1 for q in p if any(iou(g, q) >= 0.5 for g in gt))
    return hit, tp, len(p)


names = sorted(p.name for p in DATA.glob("*.jpg"))
strategies = ["A", "B", "C", "D"]
tot = {s: [0, 0, 0] for s in strategies}
gt_total = 0
print(f"{'drawing':9s} {'GT':>3s} " + " ".join(f"{s}:hit/pred" for s in strategies))
for name in names:
    img = Image.open(DATA / name)
    gt_all = rows[name]["symbols"]["bbox"]
    (ox, oy), _ = densest_tile(img.size, gt_all)
    tile = img.crop((ox, oy, ox + TILE_W, oy + TILE_H)).convert("RGB")
    gt = [[b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy] for b in gt_all
          if ox <= (b[0] + b[2]) / 2 < ox + TILE_W and oy <= (b[1] + b[3]) / 2 < oy + TILE_H]
    gt_total += len(gt)

    a = nms(boxes_of(meta_client.sam_segment(tile, "symbol")))
    b = list(a)
    if len(a) < 12:                       # sparse or silent: try more phrasings
        for p in EXTRA:
            b = nms(b + boxes_of(meta_client.sam_segment(tile, p)))
    c = run_subtiles(tile, 1)
    d = run_subtiles(tile, 2)
    cells = []
    for s, preds in zip(strategies, (a, b, c, d)):
        hit, tp, n = score(preds, gt)
        tot[s][0] += hit; tot[s][1] += tp; tot[s][2] += n
        cells.append(f"{hit:3d}/{n:<3d}")
    print(f"{name:9s} {len(gt):3d} " + "  ".join(cells))

print("\n=== totals ===")
for s in strategies:
    hit, tp, n = tot[s]
    print(f"{s}: recall {hit}/{gt_total} = {hit / gt_total:.0%}   precision {tp}/{n} = {tp / max(1, n):.0%}")
print("ledger total $%.4f" % ledger.total_usd())
