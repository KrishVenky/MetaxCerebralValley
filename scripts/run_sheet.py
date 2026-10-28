"""Run the whole pipeline on full drawings (not one tile) and score symbols against ground truth.

Usage: python -m scripts.run_sheet 14.jpg [118.jpg ...]
Legend references come from the first six drawings; run held-out ones (112 114 118 122 129 14).
"""
import json
import sys
import time

import numpy as np
from PIL import Image

from lazarus import demo_legend, ledger, pipeline
from scripts.sam_prompt_explore import DATA, iou

Image.MAX_IMAGE_PIXELS = None
OUT = DATA.parent.parent / ".cache" / "out" / "sheets"
OUT.mkdir(parents=True, exist_ok=True)
names = sorted(p.name for p in DATA.glob("*.jpg"))
rows = {json.loads(l)["file_name"]: json.loads(l) for l in (DATA / "metadata.jsonl").open()}
sheet_img, prompt, _ = demo_legend.build(DATA, names[:6])
ctx = pipeline.Context(sheet_img, prompt, demo_legend.NAMES, demo_legend.KIND)


def match(gt, boxes):
    pairs = sorted(((iou(g, b), gi, bi) for gi, g in enumerate(gt) for bi, b in enumerate(boxes)), reverse=True)
    g2b, b2g = {}, {}
    for v, gi, bi in pairs:
        if v < 0.5:
            break
        if gi not in g2b and bi not in b2g:
            g2b[gi], b2g[bi] = bi, gi
    return g2b, b2g


for name in sys.argv[1:]:
    img = Image.open(DATA / name).convert("RGB")
    gt = [list(map(float, b)) for b in rows[name]["symbols"]["bbox"]]
    gt_cls = rows[name]["symbols"]["labels"]
    t0 = time.time()
    sr = pipeline.run_sheet(img, ctx)
    res, excl = sr.result, sr.excluded
    t1 = t2 = time.time()
    print(f"{name}: scale {sr.scale:g}x, vessels {sr.vessels}, non-drawing regions {excl}")
    for label, g in (("first", res.first_graph), ("final", res.graph)):
        syms = [n for n in g.nodes if n.kind != "junction"]
        g2n, _ = match(gt, [n.box for n in syms])
        cls_ok = sum(syms[bi].legend_id == gt_cls[gi] for gi, bi in g2n.items())
        print(f"{name} {label}: symbols {len(syms)} found {len(g2n)}/{len(gt)} = {len(g2n) / len(gt):.0%}, "
              f"precision {len(g2n) / max(1, len(syms)):.0%}, type exact {cls_ok}/{len(g2n)} = {cls_ok / max(1, len(g2n)):.0%}, "
              f"junctions {sum(n.kind == 'junction' for n in g.nodes)}, edges {len(g.edges)}")
    kinds = {}
    for f in res.findings:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1
    print(f"  rounds {res.rounds}\n  needs-human findings {kinds}\n  time {t2 - t0:.0f}s, ledger ${ledger.total_usd():.3f}")
    (OUT / f"{name}.json").write_text(json.dumps({
        "tile_origin": [0, 0], "size": list(sr.size), "scale": sr.scale, "excluded": [list(b) for b in excl], "graph": res.graph.to_dict(),
        "findings": [f.__dict__ for f in res.findings], "rounds": res.rounds},
        default=lambda o: int(o) if isinstance(o, np.integer) else list(o)))
