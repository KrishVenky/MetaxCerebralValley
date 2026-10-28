"""How much of the connectivity survives when SAM, not ground truth, supplies the symbols?

Reference graph = pipe tracer run on the ground-truth symbol boxes (a proxy, not hand-checked
connectivity). Candidate graph = the same tracer run on SAM boxes. SAM boxes are matched to GT
boxes by IoU, then we count reference edges that are recovered and candidate edges that are
supported. Uses cached SAM results only.

Usage: python -m scripts.recall_cap [prompt]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from lazarus import ledger, meta_client, pipes
from scripts.sam_prompt_explore import DATA, TILE_H, TILE_W, densest_tile, iou

Image.MAX_IMAGE_PIXELS = None
prompt = sys.argv[1] if len(sys.argv) > 1 else "symbol"
rows = {json.loads(l)["file_name"]: json.loads(l) for l in (DATA / "metadata.jsonl").open()}

tot = dict(ref_edges=0, cap_edges=0, rec_edges=0, cand_edges=0, sup_edges=0, gt=0, matched=0)
print(f"{'drawing':10s} {'GT':>3s} {'SAM':>3s} {'match':>5s} | ref_e cap_e rec_e | cand_e sup_e")
for name in sorted(p.name for p in DATA.glob("*.jpg")):
    img = Image.open(DATA / name)
    gt_all = rows[name]["symbols"]["bbox"]
    (ox, oy), _ = densest_tile(img.size, gt_all)
    tile = img.crop((ox, oy, ox + TILE_W, oy + TILE_H)).convert("RGB")
    arr = np.array(tile)
    gt = [[int(b[0] - ox), int(b[1] - oy), int(b[2] - ox), int(b[3] - oy)] for b in gt_all
          if ox <= (b[0] + b[2]) / 2 < ox + TILE_W and oy <= (b[1] + b[3]) / 2 < oy + TILE_H]
    if prompt == "boost":
        sb = meta_client.sam_symbols(tile)
    else:
        sb = [(b["x1"], b["y1"], b["x2"], b["y2"]) for b in meta_client.sam_segment(tile, prompt)["boxes"]]

    ref = pipes.trace(arr, [tuple(g) for g in gt])["edges"]
    cand = pipes.trace(arr, sb)["edges"] if sb else []

    # greedy one-to-one match, GT index -> SAM index
    pairs = sorted(((iou(g, s), gi, si) for gi, g in enumerate(gt) for si, s in enumerate(sb)), reverse=True)
    g2s, s2g = {}, {}
    for v, gi, si in pairs:
        if v < 0.5:
            break
        if gi not in g2s and si not in s2g:
            g2s[gi], s2g[si] = si, gi

    ref_set = {tuple(sorted(e)) for e in ref}
    cand_in_gt = set()
    for a, b in cand:
        if a in s2g and b in s2g:
            cand_in_gt.add(tuple(sorted((s2g[a], s2g[b]))))
    cap = [e for e in ref_set if e[0] in g2s and e[1] in g2s]
    rec = [e for e in cap if e in cand_in_gt]
    sup = [e for e in cand_in_gt if e in ref_set]

    print(f"{name:10s} {len(gt):3d} {len(sb):3d} {len(g2s):5d} | {len(ref_set):5d} {len(cap):5d} {len(rec):5d} | "
          f"{len(cand):6d} {len(sup):5d}")
    tot["gt"] += len(gt); tot["matched"] += len(g2s)
    tot["ref_edges"] += len(ref_set); tot["cap_edges"] += len(cap); tot["rec_edges"] += len(rec)
    tot["cand_edges"] += len(cand); tot["sup_edges"] += len(sup)

t = tot
print("\n=== totals over %d drawings (prompt=%r) ===" % (len(rows) and len(list(DATA.glob('*.jpg'))), prompt))
print("symbols found by SAM:        %d/%d = %.0f%%" % (t["matched"], t["gt"], 100 * t["matched"] / t["gt"]))
print("ref edges with both ends found (recall cap): %d/%d = %.0f%%" % (t["cap_edges"], t["ref_edges"], 100 * t["cap_edges"] / max(1, t["ref_edges"])))
print("ref edges actually recovered: %d/%d = %.0f%%" % (t["rec_edges"], t["ref_edges"], 100 * t["rec_edges"] / max(1, t["ref_edges"])))
print("  ...of those reachable:      %d/%d = %.0f%%" % (t["rec_edges"], t["cap_edges"], 100 * t["rec_edges"] / max(1, t["cap_edges"])))
print("SAM-graph edges also in ref:  %d/%d = %.0f%%" % (t["sup_edges"], t["cand_edges"], 100 * t["sup_edges"] / max(1, t["cand_edges"])))
print("ledger total $%.4f" % ledger.total_usd())
