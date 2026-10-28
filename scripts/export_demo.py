"""Build the self-contained demo bundle in demo/data/ from saved whole-sheet graphs.

Per drawing: a downscaled JPEG of the sheet and an entry in index.json carrying the graph (in sheet
pixel coordinates), pipe routes along the traced skeleton, findings, the excluded regions and the
default source ids. Synthetic benchmark sheets come from .cache/out/sheets (run_sheet.py); real scans
from .cache/out/real (real_test.py). Idempotent; reads only local files.

Usage: python scripts/export_demo.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lazarus import routes, simulate  # noqa: E402
from lazarus.demo_legend import KIND, NAMES  # noqa: E402
from lazarus.graph import Graph, Node  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT = ROOT / "demo" / "data"
IMG_WIDTH = 3200                      # displayed width of the sheet image; graph stays in sheet pixels

# (graph json, source image, label, scale applied to the source before analysis is stored in the json)
SHEETS = [(ROOT / ".cache/out/sheets" / f"{n}.jpg.json", ROOT / "data/pid_val" / f"{n}.jpg", f"{n}.jpg", "benchmark")
          for n in (14, 129, 118, 112, 114, 122)]
REAL = [(ROOT / ".cache/out/real/ri_sample_iso10628.png.json", ROOT / "data/real/ri_sample_iso10628.png",
         "ri_sample_iso10628.png", "real scan"),
        (ROOT / ".cache/out/real/omre_pid.jpg.json", ROOT / "data/real/omre_pid.jpg", "omre_pid.jpg", "real scan")]


def build(gp: Path, src: Path, name: str, group: str):
    d = json.loads(gp.read_text())
    w, h = d["size"]
    scale = d.get("scale", 1)
    graph = Graph([Node(**n) for n in d["graph"]["nodes"]], [tuple(e) for e in d["graph"]["edges"]])
    with Image.open(src) as im:
        im = im.convert("RGB")
        if scale != 1:
            im = im.resize((w, h), Image.LANCZOS)
        rgb = np.array(im)
        small = im.resize((IMG_WIDTH, round(IMG_WIDTH * h / w)), Image.LANCZOS) if w > IMG_WIDTH else im
        small.save(OUT / f"{name}.jpg" if not name.endswith(".jpg") else OUT / name, quality=82)
    paths = routes.route_edges(rgb, graph, d.get("excluded", []))
    real = group == "real scan"
    return {
        "name": name, "group": group, "image": f"data/{name if name.endswith('.jpg') else name + '.jpg'}",
        "size": [w, h], "graph": d["graph"], "paths": paths, "findings": d["findings"],
        "excluded": d.get("excluded", []), "legend_box": d.get("legend_box"), "legend": d.get("legend", []),
        "suggested_sources": simulate.suggest_sources(d),
        "rounds": d.get("rounds", []), "taxonomy": real,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.jpg"):
        old.unlink()
    drawings = []
    for gp, src, name, group in SHEETS + REAL:
        if not gp.exists():
            print(f"skip {name}: no {gp.name}")
            continue
        e = build(gp, src, name, group)
        drawings.append(e)
        print(f"{name}: {len(e['graph']['nodes'])} nodes, {len(e['graph']['edges'])} edges, "
              f"{sum(p is None for p in e['paths'])} unrouted, {len(e['findings'])} findings")
    legend = {str(k): {"name": v, "kind": KIND[k]} for k, v in NAMES.items()}
    (OUT / "index.json").write_text(json.dumps({"drawings": drawings, "legend": legend}, separators=(",", ":")))


if __name__ == "__main__":
    main()
