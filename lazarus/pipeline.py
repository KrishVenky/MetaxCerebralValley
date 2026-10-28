"""Tile -> verified graph: SAM finds symbols, Spark classifies them against a legend, the pipe
tracer joins them, the verifier flags what looks wrong, and a bounded re-inspection loop zooms in
on the actionable flags. Whatever is still flagged afterwards is reported as needing a human.
"""
import dataclasses
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from . import baseten_client, legend, meta_client, pipes, regions, sheet, verify
from . import classify as classify_mod
from .classify import crop_symbol
from .codes import Codes
from .graph import Graph, Node


@dataclass
class Context:
    """Legend mode: sheet/prompt/names/kinds describe the drawing's legend. With sheet=None the
    classifier falls back to a fixed taxonomy (for drawings whose legend we do not have)."""
    sheet: Optional[Image.Image] = None
    prompt: str = ""
    names: Dict[int, str] = field(default_factory=dict)
    kinds: Dict[int, str] = field(default_factory=dict)
    reference: Optional[Image.Image] = None   # legend crop shown to the classifier next to each symbol
    codes: Optional[Codes] = None       # the drawing's own instrument-code vocabulary, read from its legend


@dataclass
class Result:
    graph: Graph
    findings: List[verify.Finding]
    first_graph: Graph
    first_findings: List[verify.Finding]
    rounds: List[Dict] = field(default_factory=list)


def _classify(tile: Image.Image, boxes, ctx: Context, workers: int = 10) -> List[Dict]:
    crops = [crop_symbol(tile, b) for b in boxes]
    if ctx.sheet is None:
        types = list(classify_mod.TAXONOMY)
        def one(c):
            r = classify_mod.classify(c, ctx.codes.context() if ctx.codes else "", ctx.reference)
            return dict(r, match=types.index(r["type"]), name=r["type"], kind=r["kind"])
    else:
        valid = set(ctx.names)
        def one(c):
            r = legend.classify(ctx.sheet, ctx.prompt, c, valid)
            return dict(r, name=ctx.names.get(r["match"], "unknown"), kind=ctx.kinds.get(r["match"], "unknown"))
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, crops))


def _assemble(tile_arr: np.ndarray, items: List[Dict], ctx: Context) -> Tuple[Graph, Dict]:
    boxes = [it["box"] for it in items]
    trace = pipes.trace(tile_arr, boxes, junctions=True) if boxes else {
        "edges": [], "junctions": [], "skeleton": np.zeros(tile_arr.shape[:2], bool),
        "owner": np.full(tile_arr.shape[:2], -1, np.int32)}
    nodes = []
    for i, it in enumerate(items):
        c = it["cls"]
        lid = c["match"] if c["ok"] else -1
        nodes.append(Node(i, it["box"], lid, c["name"] if c["ok"] else "unknown", c["kind"] if c["ok"] else "unknown",
                          c["tag"], c["confidence"], it["source"]))
    for k, (x, y) in enumerate(trace["junctions"]):
        nodes.append(Node(len(items) + k, (x - 3, y - 3, x + 3, y + 3), 0, "junction", "junction"))
    return Graph(nodes, [(int(a), int(b)) for a, b in trace["edges"]]), trace


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def reinspect_dangling(tile: Image.Image, pos, boxes, window: int = 360, zoom: int = 2,
                       reach: int = 110) -> List[Tuple[int, int, int, int]]:
    """Zoom on the open end of a pipe and ask SAM again for a symbol the first pass missed."""
    w, h = tile.size
    x0 = int(min(max(0, pos[0] - window // 2), max(0, w - window)))
    y0 = int(min(max(0, pos[1] - window // 2), max(0, h - window)))
    win = tile.crop((x0, y0, min(w, x0 + window), min(h, y0 + window)))
    big = win.resize((win.width * zoom, win.height * zoom), Image.LANCZOS)
    found = []
    for prompt in ("symbol", "symbols"):
        r = meta_client.sam_segment(big, prompt)
        for b in r["boxes"]:
            found.append((int(b["x1"] / zoom + x0), int(b["y1"] / zoom + y0),
                          int(b["x2"] / zoom + x0), int(b["y2"] / zoom + y0)))
        if found:
            break
    out = []
    for f in found:
        cx, cy = (f[0] + f[2]) / 2, (f[1] + f[3]) / 2
        if (cx - pos[0]) ** 2 + (cy - pos[1]) ** 2 > reach ** 2:
            continue
        if any(_iou(f, b) > 0.3 or (b[0] <= cx <= b[2] and b[1] <= cy <= b[3]) for b in boxes):
            continue
        out.append(f)
    return out


_TAG_PROMPT = ("This is a crop of a symbol from a piping and instrumentation diagram. Read the text printed "
               "in it exactly: letters first, then numbers, joined with a space. Do not deliberate; if a "
               'character is ambiguous pick the closest letter or digit. Answer with only JSON: {"tag": "..."}')


def reread_tag(tile: Image.Image, box, codes: Optional[Codes] = None) -> str:
    prompt = _TAG_PROMPT if codes is None else _TAG_PROMPT + "\n" + codes.context()
    r = baseten_client.spark_open_vision(prompt, [crop_symbol(tile, box, size=448)], max_tokens=6000)
    m = re.search(r"\{.*?\}", r["text"], re.S)
    if not m:
        return ""
    try:
        return str(json.loads(m.group(0)).get("tag", "")).strip()
    except ValueError:
        return ""


ACCEPT_CONF = 0.7


def _accept(arr, items, cand, todo, ctx, radius: int = 40, win: int = 700) -> bool:
    """Keep a symbol found on re-inspection only if the classifier is sure of it, it ends up wired
    into the graph, and it removes the dangling pipe end that prompted the search. Checked on a
    local window so a full-sheet drawing is not re-traced per candidate."""
    c = cand["cls"]
    if not c["ok"] or c["confidence"] < ACCEPT_CONF:
        return False
    H, W = arr.shape[:2]
    cx, cy = (cand["box"][0] + cand["box"][2]) / 2, (cand["box"][1] + cand["box"][3]) / 2
    x0, y0 = int(max(0, min(cx - win // 2, W - win))), int(max(0, min(cy - win // 2, H - win)))
    x1, y1 = min(W, x0 + win), min(H, y0 + win)
    local = [b for b in [it["box"] for it in items] if x0 <= (b[0] + b[2]) / 2 < x1 and y0 <= (b[1] + b[3]) / 2 < y1]
    local.append(cand["box"])
    shifted = [(b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0) for b in local]
    trace = pipes.trace(arr[y0:y1, x0:x1], shifted, junctions=True)
    k = len(shifted) - 1
    if not any(k in e for e in trace["edges"]):
        return False
    ends = verify.dangling_ends(trace, shifted)
    targets = [(f.pos[0] - x0, f.pos[1] - y0) for f in todo if f.kind == "dangling_end"]
    lc = (cx - x0, cy - y0)
    near = [t for t in targets if (t[0] - lc[0]) ** 2 + (t[1] - lc[1]) ** 2 <= 130 ** 2]
    return any(all((e[0] - t[0]) ** 2 + (e[1] - t[1]) ** 2 > radius ** 2 for e in ends) for t in near)


def run_tile(tile: Image.Image, ctx: Context, max_rounds: int = 2, min_expected: int = 12,
             boxes=None, workers: int = 10, exclude=()) -> Result:
    """Run the full pipeline on one image. `boxes` skips detection (used for stitched sheets)."""
    arr = np.array(tile.convert("RGB"))
    if len(exclude):                                   # title block / notes: not process drawing
        arr = regions.mask(arr, list(exclude))
    if boxes is None:
        boxes = meta_client.sam_symbols(tile, min_expected=min_expected)
    boxes = [tuple(map(int, b)) for b in boxes]
    cls = _classify(tile, boxes, ctx, workers)
    items = [{"box": b, "cls": c, "source": "sam"} for b, c in zip(boxes, cls)]
    graph, trace = _assemble(arr, items, ctx)
    findings = verify.check(graph, trace, ctx.codes)
    first_graph, first_findings = graph, findings
    rounds = []

    for r in range(max_rounds):
        todo = [f for f in findings if f.kind in verify.ACTIONABLE]
        if not todo:
            break
        added, retagged = 0, 0
        known = [it["box"] for it in items]
        new_boxes: List[Tuple[int, int, int, int]] = []
        for f in todo:
            if f.kind == "dangling_end":
                for nb in reinspect_dangling(tile, f.pos, known + new_boxes):
                    new_boxes.append(nb)
        if new_boxes:
            for b, c in zip(new_boxes, _classify(tile, new_boxes, ctx, workers)):
                if _accept(arr, items, {"box": b, "cls": c, "source": "reinspect"}, todo, ctx):
                    items.append({"box": b, "cls": c, "source": "reinspect"})
                    added += 1
        for f in todo:
            if f.kind == "tag_prefix":
                it = items[f.node]
                tag = reread_tag(tile, it["box"], ctx.codes)
                if tag and verify.tag_prefix(tag) == f.data["expected"]:
                    it["cls"] = dict(it["cls"], tag=tag)
                    retagged += 1
            elif f.kind == "tag_code":
                it = items[f.node]
                tag = reread_tag(tile, it["box"], ctx.codes)
                if tag and ctx.codes.valid(tag):
                    it["cls"] = dict(it["cls"], tag=tag)
                    retagged += 1
        graph, trace = _assemble(arr, items, ctx)
        findings = verify.check(graph, trace, ctx.codes)
        rounds.append({"round": r + 1, "todo": len(todo), "nodes_added": added, "tags_fixed": retagged,
                       "findings_after": len(findings)})
        if not added and not retagged:
            break
    findings = [dataclasses.replace(f, kind="open_end", message=f"pipe ends at {f.pos} with no symbol after re-inspection "
                f"(likely an off-sheet or unfinished connection)") if f.kind == "dangling_end" else f for f in findings]
    return Result(graph, findings, first_graph, first_findings, rounds)


TARGET_LONG_SIDE = 6600           # SAM-friendly sheet size: symbols of ~100px on a drawing of this width
MAX_SCALE = 3.0
LEGEND_REF_SIDE = 1400


def auto_scale(size: Tuple[int, int]) -> float:
    """Upscale factor for small scans, rounded to 0.5. Sheets already at or above the target stay 1x.
    A size heuristic: symbol size is unknown before detection, and sheets are drawn to similar layouts."""
    f = min(MAX_SCALE, max(1.0, TARGET_LONG_SIDE / max(size)))
    return round(f * 2) / 2


@dataclass
class SheetResult:
    result: Result
    scale: float                                     # graph coordinates are in the scaled image
    size: Tuple[int, int]                            # scaled image size
    excluded: List[Tuple[int, int, int, int]]
    vessels: List[Tuple[int, int, int, int]]
    legend_box: Optional[Tuple[int, int, int, int]] = None
    legend: List[Dict] = field(default_factory=list)       # entries Spark read from the legend table


def run_sheet(img: Image.Image, ctx: Context, scale: Optional[float] = None, workers: int = 16,
              max_rounds: int = 2, legend_image: bool = False) -> SheetResult:
    """Whole-sheet pipeline: upscale small drawings, SAM over overlapping tiles, Muse Spark for the big
    outlines SAM never returns (vessels, exchangers) and for non-drawing regions, then run_tile."""
    img = img.convert("RGB")
    scale = auto_scale(img.size) if scale is None else scale
    vessels = [tuple(int(v * scale) for v in b) for b in regions.find_equipment(img)]
    legend_box = regions.find_legend(img)
    legend = regions.read_legend(img, legend_box) if legend_box else []
    if legend_box:
        legend_box = tuple(int(v * scale) for v in legend_box)
    if scale != 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    boxes = sheet.detect(img)
    if vessels:
        boxes = [b for b in boxes if all(_iou(b, v) < 0.3 for v in vessels)] + vessels
    if legend_box:                                   # the legend explains symbols; it is not part of the process
        boxes = [b for b in boxes if not (legend_box[0] <= (b[0] + b[2]) / 2 <= legend_box[2]
                                          and legend_box[1] <= (b[1] + b[3]) / 2 <= legend_box[3])]
    excluded = regions.find(img, boxes) + ([legend_box] if legend_box else [])
    if ctx.sheet is None and ctx.codes is None:      # no hand-supplied legend: use the one the drawing carries
        codes = Codes.from_entries(legend)
        if codes.usable():
            ctx = dataclasses.replace(ctx, codes=codes)
        if legend_image and legend_box:
            ref = img.crop(legend_box)
            k = LEGEND_REF_SIDE / max(ref.size)
            ctx = dataclasses.replace(ctx, reference=ref.resize((int(ref.width * k), int(ref.height * k)), Image.LANCZOS))
    res = run_tile(img, ctx, max_rounds=max_rounds, boxes=boxes, workers=workers, exclude=excluded)
    return SheetResult(res, scale, img.size, excluded, vessels, legend_box, legend)
