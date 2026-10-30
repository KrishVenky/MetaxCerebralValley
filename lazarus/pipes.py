"""Classical pipe tracing: which symbols are joined by line work?

Input: a tile image and symbol boxes (from SAM). Symbol interiors are erased, the remaining ink
is split into connected components, and a component that touches two or more symbols becomes
connectivity between them. Text blobs are filtered out by size and shape.
"""
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def ink_mask(tile_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return bw > 0


def _collapse(edges, n_sym: int):
    """Drop junctions that join fewer than three things (corners, spurs): splice or discard them."""
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    changed = True
    while changed:
        changed = False
        for j in [k for k in adj if k >= n_sym]:
            if j not in adj or len(adj[j]) > 2:
                continue
            ns = list(adj.pop(j))
            for x in ns:
                adj[x].discard(j)
            if len(ns) == 2:
                adj[ns[0]].add(ns[1])
                adj[ns[1]].add(ns[0])
            changed = True
    keep = sorted(k for k in adj if k >= n_sym)
    remap = {k: n_sym + i for i, k in enumerate(keep)}
    out = {(min(remap.get(a, a), remap.get(b, b)), max(remap.get(a, a), remap.get(b, b)))
           for a in adj for b in adj[a] if a != b}
    return sorted(out), keep


def _contract_close(edges, jpos, n_sym: int, dist: int = 32):
    """Merge directly linked junctions that sit within `dist` px (text and ticks on a pipe make
    clusters of fake branch points). Returns the rewritten edges; jpos is updated in place."""
    parent = {}

    def find(a):
        while parent.get(a, a) != a:
            a = parent[a]
        return a

    for a, b in sorted(edges):
        if a >= n_sym and b >= n_sym:
            (ax, ay), (bx, by) = jpos[a - n_sym], jpos[b - n_sym]
            if (ax - bx) ** 2 + (ay - by) ** 2 <= dist ** 2:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups = defaultdict(list)
    for a in {x for e in edges for x in e if x >= n_sym}:
        groups[find(a)].append(a)
    for r, members in groups.items():
        jpos[r - n_sym] = (int(np.mean([jpos[m - n_sym][0] for m in members])),
                           int(np.mean([jpos[m - n_sym][1] for m in members])))
    out = {(min(find(a), find(b)), max(find(a), find(b))) for a, b in edges if find(a) != find(b)}
    return out


def trace(tile_rgb: np.ndarray, boxes: List[Tuple[int, int, int, int]],
          margin: int = 8, min_len: int = 40, bridge: int = 3, junctions: bool = False,
          axis_gap: int = 0) -> Dict:
    """`axis_gap` > 0 additionally closes gaps shorter than that many px along the horizontal and vertical
    axes only, so a dashed line becomes one component. Text merges too, so only use it as a second pass."""
    ink = ink_mask(tile_rgb)
    h, w = ink.shape
    lines = ink.copy()
    for x1, y1, x2, y2 in boxes:
        lines[max(0, y1):min(h, y2 + 1), max(0, x1):min(w, x2 + 1)] = False

    # Close tiny gaps (dashed or broken lines) before labelling.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * bridge + 1, 2 * bridge + 1))
    closed = cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    if axis_gap:
        for shape in ((axis_gap, 1), (1, axis_gap)):
            ke = cv2.getStructuringElement(cv2.MORPH_RECT, shape)
            closed |= cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_CLOSE, ke) > 0
    labels, n = ndi.label(closed, structure=np.ones((3, 3)))

    objs = ndi.find_objects(labels)
    comps = []
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if max(hh, ww) < min_len:           # too short: text, specks
            continue
        comps.append((i, sl))

    keep = np.zeros_like(closed)
    for ci, sl in comps:
        keep[sl] |= labels[sl] == ci
    skel = skeletonize(keep)

    # Seed each symbol from skeleton pixels inside its margin-expanded box, then flood along
    # the skeleton. Where two floods touch, those two symbols are direct neighbours.
    owner = np.full((h, w), -1, dtype=np.int32)
    queue: deque = deque()
    for bi, (x1, y1, x2, y2) in enumerate(boxes):
        ys, xs = np.nonzero(skel[max(0, y1 - margin):min(h, y2 + margin + 1),
                                 max(0, x1 - margin):min(w, x2 + margin + 1)])
        for y, x in zip(ys + max(0, y1 - margin), xs + max(0, x1 - margin)):
            if owner[y, x] == -1:
                owner[y, x] = bi
                queue.append((y, x))
    n_sym = len(boxes)
    jpos: List[Tuple[int, int]] = []
    if junctions:
        # Pixels with 3+ skeleton neighbours are branch points; nearby ones form one junction node.
        nb = ndi.convolve(skel.astype(np.int32), np.ones((3, 3), np.int32), mode="constant") - skel
        jm = cv2.dilate((skel & (nb >= 3)).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
        jl, jn = ndi.label(jm, structure=np.ones((3, 3)))
        for j, sl in enumerate(ndi.find_objects(jl), start=1):
            ys, xs = np.nonzero((jl[sl] == j) & skel[sl] & (owner[sl] == -1))
            if len(ys) == 0:
                continue
            ys, xs = ys + sl[0].start, xs + sl[1].start
            jid = n_sym + len(jpos)
            jpos.append((int(xs.mean()), int(ys.mean())))
            for y, x in zip(ys, xs):
                owner[y, x] = jid
                queue.append((y, x))
    edges = set()
    while queue:
        y, x = queue.popleft()
        o = owner[y, x]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if (dy or dx) and 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                    p = owner[ny, nx]
                    if p == -1:
                        owner[ny, nx] = o
                        queue.append((ny, nx))
                    elif p != o:
                        edges.add((min(o, p), max(o, p)))

    touched = np.unique(owner[(owner >= 0) & (owner < n_sym)]).tolist()
    jcenters: List[Tuple[int, int]] = []
    if junctions:
        edge_list, kept = _collapse(_contract_close(edges, jpos, n_sym), n_sym)
        edges = set(edge_list)
        jcenters = [jpos[k - n_sym] for k in kept]
    comp_ids = {ci: [] for ci, _ in comps}
    return {"labels": labels, "components": comp_ids, "edges": sorted(edges),
            "n_lines": len(comps), "skeleton": skel, "connected_symbols": len(touched),
            "owner": owner, "junctions": jcenters}


def render(tile_rgb: np.ndarray, boxes, result: Dict) -> np.ndarray:
    out = (tile_rgb * 0.45 + 140).astype(np.uint8)          # fade the drawing
    out[cv2.dilate(result["skeleton"].astype(np.uint8), np.ones((3, 3))) > 0] = (60, 60, 200)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 160, 0), 3)
    for a, b in result["edges"]:
        ca = ((boxes[a][0] + boxes[a][2]) // 2, (boxes[a][1] + boxes[a][3]) // 2)
        cb = ((boxes[b][0] + boxes[b][2]) // 2, (boxes[b][1] + boxes[b][3]) // 2)
        cv2.line(out, ca, cb, (220, 0, 0), 2)
    return out
