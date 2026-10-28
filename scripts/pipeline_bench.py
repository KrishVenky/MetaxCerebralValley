"""End-to-end benchmark on drawings NOT used to build the legend.

Legend references come from the first six drawings; the benchmark runs the densest tile of the
other six. Reports symbol recall/precision, legend-type accuracy, connection recovery (vs the
tracer on ground-truth boxes: a proxy), and how the verifier's flags line up with real misses,
before and after the re-inspection loop.

Usage: python -m scripts.pipeline_bench [max_rounds=2]
"""
import json
import sys

import numpy as np
from PIL import Image

from lazarus import demo_legend, ledger, pipeline, pipes
from scripts.sam_prompt_explore import DATA, TILE_H, TILE_W, densest_tile, iou

Image.MAX_IMAGE_PIXELS = None
max_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
names = sorted(p.name for p in DATA.glob("*.jpg"))
ref_drawings, test_drawings = names[:6], names[6:]
rows = {json.loads(l)["file_name"]: json.loads(l) for l in (DATA / "metadata.jsonl").open()}

sheet, prompt, _ = demo_legend.build(DATA, ref_drawings)
ctx = pipeline.Context(sheet, prompt, demo_legend.NAMES, demo_legend.KIND)
out_dir = DATA.parent.parent / ".cache" / "out" / "graphs"
out_dir.mkdir(parents=True, exist_ok=True)


def symbol_edges(graph):
    """Symbol-symbol links, treating a chain of junctions as a shared bus (so comparable to the proxy)."""
    syms = {n.id for n in graph.nodes if n.kind != "junction"}
    adj = {n.id: set() for n in graph.nodes}
    for a, b in graph.edges:
        adj[a].add(b)
        adj[b].add(a)
    out, seen = set(), set()
    for start in adj:
        if start in syms or start in seen:
            continue
        comp, stack = set(), [start]
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack += [y for y in adj[x] if y not in syms]
        seen |= comp
        members = sorted({y for x in comp for y in adj[x] if y in syms})
        out |= {(a, b) for i, a in enumerate(members) for b in members[i + 1:]}
    out |= {(min(a, b), max(a, b)) for a, b in graph.edges if a in syms and b in syms}
    return out


def score(graph, gt, gt_cls, ref_edges):
    graph = type(graph)([n for n in graph.nodes if n.kind != "junction"], sorted(symbol_edges(graph)))
    boxes = [n.box for n in graph.nodes]
    pairs = sorted(((iou(g, b), gi, bi) for gi, g in enumerate(gt) for bi, b in enumerate(boxes)), reverse=True)
    g2n, n2g = {}, {}
    for v, gi, bi in pairs:
        if v < 0.5:
            break
        if gi not in g2n and bi not in n2g:
            g2n[gi], n2g[bi] = bi, gi
    cls_ok = sum(graph.nodes[bi].legend_id == gt_cls[gi] for gi, bi in g2n.items())
    kind_ok = sum(graph.nodes[bi].kind == demo_legend.KIND[gt_cls[gi]] for gi, bi in g2n.items())
    cand = {tuple(sorted((n2g[a], n2g[b]))) for a, b in graph.edges if a in n2g and b in n2g}
    rec = sum(1 for e in ref_edges if e in cand)
    sup = sum(1 for e in cand if e in ref_edges)
    return dict(found=len(g2n), pred=len(boxes), cls_ok=cls_ok, kind_ok=kind_ok, rec=rec, cand=len(graph.edges),
                sup=sup, missed=[gi for gi in range(len(gt)) if gi not in g2n])


tot = {s: dict(gt=0, found=0, pred=0, cls_ok=0, kind_ok=0, rec=0, ref=0, cand=0, sup=0) for s in ("first", "final")}
flags = dict(n=0, near_miss=0, missed=0, missed_flagged=0)
kinds_first, kinds_final = {}, {}
print(f"{'drawing':9s} {'GT':>3s} | first: found/pred cls | final: found/pred cls | edges rec/ref first->final | rounds")
for name in test_drawings:
    img = Image.open(DATA / name)
    gt_all = rows[name]["symbols"]["bbox"]
    lab_all = rows[name]["symbols"]["labels"]
    (ox, oy), _ = densest_tile(img.size, gt_all)
    tile = img.crop((ox, oy, ox + TILE_W, oy + TILE_H)).convert("RGB")
    sel = [i for i, b in enumerate(gt_all) if ox <= (b[0] + b[2]) / 2 < ox + TILE_W and oy <= (b[1] + b[3]) / 2 < oy + TILE_H]
    gt = [[int(gt_all[i][0] - ox), int(gt_all[i][1] - oy), int(gt_all[i][2] - ox), int(gt_all[i][3] - oy)] for i in sel]
    gt_cls = [lab_all[i] for i in sel]
    ref_edges = {tuple(sorted(e)) for e in pipes.trace(np.array(tile), [tuple(g) for g in gt])["edges"]}

    res = pipeline.run_tile(tile, ctx, max_rounds=max_rounds)
    s1, s2 = score(res.first_graph, gt, gt_cls, ref_edges), score(res.graph, gt, gt_cls, ref_edges)
    for key, s in (("first", s1), ("final", s2)):
        t = tot[key]
        t["gt"] += len(gt); t["ref"] += len(ref_edges)
        for k in ("found", "pred", "cls_ok", "kind_ok", "rec", "cand", "sup"):
            t[k] += s[k]
    # dangling-flag quality (first pass): is there really a missed symbol near it?
    gc = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in gt]
    dang = [f for f in res.first_findings if f.kind == "dangling_end"]
    near = lambda p, q: (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= 110 ** 2
    flags["n"] += len(dang)
    flags["near_miss"] += sum(any(near(f.pos, gc[gi]) for gi in s1["missed"]) for f in dang)
    flags["missed"] += len(s1["missed"])
    flags["missed_flagged"] += sum(any(near(f.pos, gc[gi]) for f in dang) for gi in s1["missed"])
    for f in res.first_findings:
        kinds_first[f.kind] = kinds_first.get(f.kind, 0) + 1
    for f in res.findings:
        kinds_final[f.kind] = kinds_final.get(f.kind, 0) + 1
    print(f"{name:9s} {len(gt):3d} | {s1['found']:3d}/{s1['pred']:<3d} {s1['cls_ok']:3d} | {s2['found']:3d}/{s2['pred']:<3d} {s2['cls_ok']:3d}"
          f" | {s1['rec']:2d}->{s2['rec']:<2d}/{len(ref_edges):2d} | {res.rounds}")
    (out_dir / f"{name}.json").write_text(json.dumps({
        "tile_origin": [ox, oy], "graph": res.graph.to_dict(),
        "findings": [f.__dict__ for f in res.findings], "rounds": res.rounds},
        default=lambda o: int(o) if isinstance(o, np.integer) else list(o), indent=1))

print("\n=== totals (6 held-out drawings) ===")
for key in ("first", "final"):
    t = tot[key]
    print(f"{key:5s}: symbol recall {t['found']}/{t['gt']} = {t['found'] / t['gt']:.0%}, precision {t['found']}/{t['pred']} = "
          f"{t['found'] / max(1, t['pred']):.0%} | type exact {t['cls_ok']}/{t['found']} = {t['cls_ok'] / max(1, t['found']):.0%}, "
          f"kind {t['kind_ok']}/{t['found']} | edges recovered {t['rec']}/{t['ref']} = {t['rec'] / max(1, t['ref']):.0%}, "
          f"edge precision {t['sup']}/{t['cand']} = {t['sup'] / max(1, t['cand']):.0%}")
print(f"dangling flags (first pass): {flags['n']}, near a truly missed symbol: {flags['near_miss']} "
      f"({flags['near_miss'] / max(1, flags['n']):.0%}); missed symbols with a flag nearby: "
      f"{flags['missed_flagged']}/{flags['missed']}")
print("findings first pass:", kinds_first)
print("findings after loop (needs human):", kinds_final)
print("ledger total $%.4f" % ledger.total_usd())
