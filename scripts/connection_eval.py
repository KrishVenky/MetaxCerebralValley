"""Score the pipeline's connections against the hand-labeled set (data/labels/connections.json).

Two symbols are 'linked' when a pipe path joins them without passing through another symbol
(junctions and corners allowed), the same definition for the labels and the predictions. We report
recall over all labeled links (a link needs both symbols found, so missed symbols count against
us), precision over predicted links between matched symbols, and the same limited to solid pipes.

Usage: python -m scripts.connection_eval [graph_dir=.cache/out/graphs]   (use .cache/out/sheets for full sheets)
"""
import sys
import json
from collections import defaultdict

from scripts.sam_prompt_explore import DATA, densest_tile, iou

LABELS = json.load(open(DATA.parent / "labels" / "connections.json"))
GRAPHS = DATA.parent.parent / (sys.argv[1] if len(sys.argv) > 1 else ".cache/out/graphs")
rows = {json.loads(l)["file_name"]: json.loads(l) for l in (DATA / "metadata.jsonl").open()}


def closure(segments, is_sym):
    """Symbol pairs joined through chains of non-symbol nodes. Returns set of (a, b)."""
    adj = defaultdict(set)
    for a, b in segments:
        adj[a].add(b)
        adj[b].add(a)
    out, seen = set(), set()
    for s in adj:
        if is_sym(s) or s in seen:
            continue
        comp, stack = set(), [s]
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack += [y for y in adj[x] if not is_sym(y)]
        seen |= comp
        members = sorted({y for x in comp for y in adj[x] if is_sym(y)})
        out |= {(a, b) for i, a in enumerate(members) for b in members[i + 1:]}
    out |= {(min(a, b), max(a, b)) for a, b in segments if is_sym(a) and is_sym(b)}
    return out


tot = defaultdict(int)
print(f"{'drawing':9s} {'links':>5s} {'found both':>10s} {'recovered':>9s} | {'pred':>4s} {'correct':>7s} | solid-only rec")
for name, lab in LABELS.items():
    if name.startswith("_"):
        continue
    g = json.load(open(GRAPHS / f"{name}.json"))
    gx, gy = g["tile_origin"]                                   # graph coordinates -> sheet coordinates
    row = rows[name]
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    (ox, oy), _ = densest_tile(Image.open(DATA / name).size, row["symbols"]["bbox"])   # where the labels live
    sel = [i for i, b in enumerate(row["symbols"]["bbox"])
           if ox <= (b[0] + b[2]) / 2 < ox + 1792 and oy <= (b[1] + b[3]) / 2 < oy + 1140]
    gt = {k: [row["symbols"]["bbox"][i][0] - ox, row["symbols"]["bbox"][i][1] - oy,
              row["symbols"]["bbox"][i][2] - ox, row["symbols"]["bbox"][i][3] - oy] for k, i in enumerate(sel)}
    excl = set(lab["exclude"])
    syms = lambda x: isinstance(x, int)
    segs_all = [(a, b) for a, b, d in lab["segments"]]
    segs_solid = [(a, b) for a, b, d in lab["segments"] if not d]
    truth = {e for e in closure(segs_all, syms) if not (set(e) & excl)}
    truth_solid = {e for e in closure(segs_solid, syms) if not (set(e) & excl)}

    nodes = [dict(n, box=[n["box"][0] + gx - ox, n["box"][1] + gy - oy, n["box"][2] + gx - ox, n["box"][3] + gy - oy])
             for n in g["graph"]["nodes"] if n["kind"] != "junction"]
    sym_ids = {n["id"] for n in nodes}
    pairs = sorted(((iou(gt[k], n["box"]), k, n["id"]) for k in gt for n in nodes), reverse=True)
    g2n, n2g = {}, {}
    for v, k, nid in pairs:
        if v < 0.5:
            break
        if k not in g2n and nid not in n2g:
            g2n[k], n2g[nid] = nid, k
    pred = closure([tuple(e) for e in g["graph"]["edges"]], lambda x: x in sym_ids)
    pred_gt = {tuple(sorted((n2g[a], n2g[b]))) for a, b in pred if a in n2g and b in n2g and not ({n2g[a], n2g[b]} & excl)}
    both = {e for e in truth if e[0] in g2n and e[1] in g2n}
    rec = truth & pred_gt
    rec_solid = truth_solid & pred_gt
    cor = pred_gt & truth
    print(f"{name:9s} {len(truth):5d} {len(both):10d} {len(rec):9d} | {len(pred_gt):4d} {len(cor):7d} | {len(rec_solid)}/{len(truth_solid)}")
    for key, v in (("truth", len(truth)), ("both", len(both)), ("rec", len(rec)), ("pred", len(pred_gt)),
                   ("cor", len(cor)), ("truth_solid", len(truth_solid)), ("rec_solid", len(rec_solid)),
                   ("both_solid", len({e for e in truth_solid if e[0] in g2n and e[1] in g2n}))):
        tot[key] += v
    miss = sorted(truth - pred_gt)
    wrong = sorted(pred_gt - truth)
    print("   missed (both found):", [e for e in miss if e in both])
    print("   wrong:", wrong)

print(f"\nrecall over all labeled links: {tot['rec']}/{tot['truth']} = {tot['rec'] / tot['truth']:.0%}")
print(f"  of links whose both symbols were detected: {tot['rec']}/{tot['both']} = {tot['rec'] / tot['both']:.0%}")
print(f"precision: {tot['cor']}/{tot['pred']} = {tot['cor'] / tot['pred']:.0%}")
print(f"solid pipes only, recall {tot['rec_solid']}/{tot['truth_solid']} = {tot['rec_solid'] / tot['truth_solid']:.0%}")
