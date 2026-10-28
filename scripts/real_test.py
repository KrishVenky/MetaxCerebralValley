"""Run the pipeline on real (non-synthetic) drawings from data/real/ and write overlays.

No ground truth exists, so this is a qualitative check: do detection, classification and pipe
tracing hold up on genuine scans? Uses the fixed-taxonomy classifier (no legend).

Usage: python -m scripts.real_test [--scale N] omre_pid.jpg ri_sample_iso10628.png
Runs pipeline.run_sheet, which upscales small scans automatically; --scale N overrides the factor.
"""
import json
import sys
import time

import cv2
import numpy as np
from PIL import Image

from lazarus import ledger, pipeline

Image.MAX_IMAGE_PIXELS = None
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / ".cache" / "out" / "real"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {"valve": (0, 150, 0), "instrument": (200, 0, 200), "equipment": (0, 120, 220),
          "fitting": (120, 120, 0), "marker": (0, 150, 150), "other": (90, 90, 90), "unknown": (90, 90, 90)}

args = sys.argv[1:]
scale = float(args.pop(args.index("--scale") + 1)) if "--scale" in args else None
if "--scale" in args:
    args.remove("--scale")

for name in args:
    img = Image.open(ROOT / "data" / "real" / name).convert("RGB")
    t0 = time.time()
    sr = pipeline.run_sheet(img, pipeline.Context(), scale=scale)
    res, excl = sr.result, sr.excluded
    img = img.resize(sr.size, Image.LANCZOS) if sr.scale != 1 else img
    print(f"{name}: {sr.scale:g}x -> {img.size}, vessels {sr.vessels}, excluded regions {excl}, "
          f"legend {sr.legend_box} with {len(sr.legend)} entries")
    kinds = {}
    for n in res.graph.nodes:
        kinds[n.kind] = kinds.get(n.kind, 0) + 1
    fk = {}
    for f in res.findings:
        fk[f.kind] = fk.get(f.kind, 0) + 1
    print(f"  nodes by kind {kinds}, edges {len(res.graph.edges)}, rounds {res.rounds}")
    print(f"  findings {fk}; {time.time() - t0:.0f}s; ledger ${ledger.total_usd():.3f}")
    (OUT / f"{name}.json").write_text(json.dumps({
        "tile_origin": [0, 0], "size": list(img.size), "scale": sr.scale, "excluded": [list(b) for b in excl],
        "legend_box": sr.legend_box, "legend": sr.legend,
        "graph": res.graph.to_dict(), "findings": [f.__dict__ for f in res.findings], "rounds": res.rounds},
        default=lambda o: int(o) if isinstance(o, np.integer) else list(o)))

    s = 1800 / max(img.size)
    out = (np.array(img.resize((int(img.width * s), int(img.height * s)))) * 0.6 + 100).astype(np.uint8)
    ctr = {n.id: (int(n.center[0] * s), int(n.center[1] * s)) for n in res.graph.nodes}
    for a, b in res.graph.edges:
        cv2.line(out, ctr[a], ctr[b], (220, 60, 60), 1)
    for n in res.graph.nodes:
        if n.kind == "junction":
            cv2.circle(out, ctr[n.id], 3, (255, 140, 0), -1)
        else:
            x1, y1, x2, y2 = [int(v * s) for v in n.box]
            cv2.rectangle(out, (x1, y1), (x2, y2), COLORS.get(n.kind, (90, 90, 90)), 2)
    cv2.imwrite(str(OUT / f"{name}.overlay.jpg"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
