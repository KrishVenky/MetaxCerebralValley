"""Which SAM 3.1 noun-phrase prompts find P&ID symbols? Class-agnostic recall vs ground truth.

Usage: python -m scripts.sam_prompt_explore [n_drawings]
"""
import json
import sys
from pathlib import Path

from PIL import Image

from lazarus import ledger, meta_client

Image.MAX_IMAGE_PIXELS = None
DATA = Path(__file__).resolve().parent.parent / "data" / "pid_val"
TILE_W, TILE_H = 1792, 1140
import os
PROMPTS = os.environ.get("SAM_PROMPTS", "valve,pump,instrument,symbol,circle,tank,pipe fitting,gate valve").split(",")


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def densest_tile(img_size, gt):
    """Tile origin (x, y) containing the most ground-truth symbol centers, on a coarse grid."""
    w, h = img_size
    best, best_n = (0, 0), -1
    for x in range(0, w - TILE_W + 1, 448):
        for y in range(0, h - TILE_H + 1, 285):
            n = sum(1 for b in gt if x <= (b[0] + b[2]) / 2 < x + TILE_W and y <= (b[1] + b[3]) / 2 < y + TILE_H)
            if n > best_n:
                best, best_n = (x, y), n
    return best, best_n


def main():
    n_draw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    rows = {json.loads(l)["file_name"]: json.loads(l) for l in (DATA / "metadata.jsonl").open()}
    names = sorted(p.name for p in DATA.glob("*.jpg"))[:n_draw]
    totals = {p: {"hit50": 0, "hit30": 0, "pred": 0, "tp50": 0, "gt": 0} for p in PROMPTS}

    for name in names:
        img = Image.open(DATA / name)
        gt_all = rows[name]["symbols"]["bbox"]
        (ox, oy), n = densest_tile(img.size, gt_all)
        tile = img.crop((ox, oy, ox + TILE_W, oy + TILE_H))
        gt = [[b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy] for b in gt_all
              if ox <= (b[0] + b[2]) / 2 < ox + TILE_W and oy <= (b[1] + b[3]) / 2 < oy + TILE_H]
        print(f"\n{name}: tile at ({ox},{oy}) with {len(gt)} GT symbols")
        for p in PROMPTS:
            r = meta_client.sam_segment(tile, p)
            preds = [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in r["boxes"]]
            h50 = sum(1 for g in gt if any(iou(g, q) >= 0.5 for q in preds))
            h30 = sum(1 for g in gt if any(iou(g, q) >= 0.3 for q in preds))
            tp50 = sum(1 for q in preds if any(iou(g, q) >= 0.5 for g in gt))
            t = totals[p]
            t["hit50"] += h50; t["hit30"] += h30; t["pred"] += len(preds); t["tp50"] += tp50; t["gt"] += len(gt)
            print(f"  {p:14s} preds={len(preds):3d}  recall@.5={h50}/{len(gt)}  recall@.3={h30}/{len(gt)}")

    print("\n=== totals (class-agnostic) ===")
    for p, t in totals.items():
        rec = t["hit50"] / t["gt"] if t["gt"] else 0
        prec = t["tp50"] / t["pred"] if t["pred"] else 0
        print(f"{p:14s} recall@.5={rec:5.1%}  precision@.5={prec:5.1%}  preds={t['pred']}")
    print("\nledger total $%.4f" % ledger.total_usd())


if __name__ == "__main__":
    main()
