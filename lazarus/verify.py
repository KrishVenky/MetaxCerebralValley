"""Deterministic verifier: no model in the loop. Takes a graph plus the pipe skeleton and returns
findings. Some findings are actionable (the pipeline re-inspects them); the rest go to a human.

Kinds
  dangling_end   a traced pipe ends in empty space, usually where a symbol was missed
  isolated       a symbol with no pipe connections
  degree         an in-line valve whose connection count is not 2
  tag_missing    an instrument with no readable tag
  tag_prefix     tag letters differ from the majority of symbols of the same legend type
  tag_code       an instrument's tag letters are not a code the drawing's legend defines
  low_conf       the classifier was unsure
"""
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage as ndi

from .graph import Graph

ACTIONABLE = {"dangling_end", "tag_prefix", "tag_code"}
LOW_CONF = 0.6


@dataclass
class Finding:
    kind: str
    message: str
    node: int = -1
    pos: Tuple[int, int] = (0, 0)
    data: Dict = field(default_factory=dict)


def tag_prefix(tag: str) -> str:
    m = re.match(r"[^A-Za-z]*([A-Za-z]+)", tag or "")
    return m.group(1).upper() if m else ""


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    return ndi.convolve(skel.astype(np.int32), np.ones((3, 3), np.int32), mode="constant") - skel.astype(np.int32)


def _is_spur(skel, nb, y, x, max_steps=15) -> bool:
    """Walk from an endpoint along the skeleton; a junction within a few pixels means a spur."""
    h, w = skel.shape
    prev, cur = None, (y, x)
    for _ in range(max_steps):
        nxt = [(cur[0] + dy, cur[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dy or dx) and 0 <= cur[0] + dy < h and 0 <= cur[1] + dx < w
               and skel[cur[0] + dy, cur[1] + dx] and (cur[0] + dy, cur[1] + dx) != prev]
        if not nxt:
            return False
        prev, cur = cur, nxt[0]
        if nb[cur] >= 3:
            return True
    return False


def dangling_ends(trace: Dict, boxes, margin: int = 16, edge: int = 24, min_comp: int = 60) -> List[Tuple[int, int]]:
    skel, owner = trace["skeleton"], trace["owner"]
    h, w = skel.shape
    nb = _neighbour_count(skel)
    comp, n = ndi.label(skel, structure=np.ones((3, 3)))
    sizes = ndi.sum(skel, comp, index=np.arange(1, n + 1))
    anchored = {c for c in np.unique(comp[(owner >= 0) & (owner < len(boxes)) & skel]) if c > 0 and sizes[c - 1] >= min_comp}
    pts: List[Tuple[int, int]] = []
    for y, x in np.argwhere(skel & (nb == 1)):
        if comp[y, x] not in anchored:
            continue
        if x < edge or y < edge or x >= w - edge or y >= h - edge:      # pipe leaves the tile
            continue
        if any(b[0] - margin <= x <= b[2] + margin and b[1] - margin <= y <= b[3] + margin for b in boxes):
            continue
        if _is_spur(skel, nb, y, x):
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < 30 ** 2 for px, py in pts):
            continue
        pts.append((int(x), int(y)))
    return pts


def check(graph: Graph, trace: Dict, codes=None) -> List[Finding]:
    out: List[Finding] = []
    boxes = [n.box for n in graph.nodes]
    for x, y in dangling_ends(trace, boxes):
        out.append(Finding("dangling_end", f"pipe ends in empty space near ({x},{y})", pos=(x, y)))

    deg = graph.degree()
    for n in graph.nodes:
        label = f"{n.kind} {n.tag or n.name}"
        if deg[n.id] == 0:
            out.append(Finding("isolated", f"{label} has no pipe connections", node=n.id, pos=n.center))
        elif n.kind == "valve" and deg[n.id] != 2:
            out.append(Finding("degree", f"{label} has {deg[n.id]} connections (expected 2)", node=n.id, pos=n.center))
        if n.kind == "instrument" and not n.tag:
            out.append(Finding("tag_missing", f"{label} has no readable tag", node=n.id, pos=n.center))
        if n.confidence and n.confidence < LOW_CONF:
            out.append(Finding("low_conf", f"{label} classified with confidence {n.confidence:.2f}",
                               node=n.id, pos=n.center))

    if codes is not None:
        for n in graph.nodes:
            if n.kind == "instrument" and n.tag and not codes.valid(n.tag):
                out.append(Finding("tag_code", f"tag {n.tag!r} is not an instrument code defined in this drawing's legend",
                                   node=n.id, pos=n.center))

    # Tag letters should agree among symbols of the same legend type.
    groups: Dict[int, List] = defaultdict(list)
    for n in graph.nodes:
        if n.kind == "instrument" and tag_prefix(n.tag):
            groups[n.legend_id].append(n)
    for lid, ns in groups.items():
        if len(ns) < 3:
            continue
        top, cnt = Counter(tag_prefix(n.tag) for n in ns).most_common(1)[0]
        if cnt / len(ns) < 0.6:
            continue
        for n in ns:
            if tag_prefix(n.tag) != top:
                out.append(Finding("tag_prefix", f"tag {n.tag!r} differs from the usual {top!r} for this symbol type",
                                   node=n.id, pos=n.center, data={"expected": top}))
    return out
