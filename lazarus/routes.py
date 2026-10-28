"""Pipe routes for display: for each graph edge, the path along the traced pipe between its two nodes.

The graph itself only says "A joins B". The demo wants to draw the pipe, so each edge is routed over the
pipe skeleton (the same one the tracer used) with a shortest-path search from the first node's pipe
pixels to the second's, then simplified to a few corner points. Display only; nothing here feeds
the verifier or the simulator.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from . import pipes, regions
from .graph import Graph

Point = Tuple[int, int]
_STEPS = ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5))


def _skeleton_graph(skel: np.ndarray):
    ys, xs = np.nonzero(skel)
    index = np.full(skel.shape, -1, np.int32)
    index[ys, xs] = np.arange(len(ys), dtype=np.int32)
    rows, cols, w = [], [], []
    h, wd = skel.shape
    for dy, dx, cost in _STEPS:
        ok = (ys + dy < h) & (xs + dx >= 0) & (xs + dx < wd)
        a = index[ys[ok], xs[ok]]
        b = index[ys[ok] + dy, xs[ok] + dx]
        hit = b >= 0
        rows.append(a[hit]); cols.append(b[hit]); w.append(np.full(hit.sum(), cost))
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
    n = len(ys)
    m = csr_matrix((np.concatenate([v, v]), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(n, n))
    return m, ys, xs, index


def route_edges(rgb: np.ndarray, graph: Graph, excluded: Sequence = (), margin: int = 8,
                epsilon: float = 3.0) -> List[Optional[List[Point]]]:
    """One polyline per graph edge (in graph order): node centre, pipe corners, node centre. None when no
    pipe path was found, in which case the caller draws a straight line."""
    if len(excluded):
        rgb = regions.mask(rgb, list(excluded))
    nodes = {n.id: n for n in graph.nodes}
    boxes = [n.box for n in graph.nodes if n.kind != "junction"]
    skel = pipes.trace(rgb, boxes)["skeleton"]
    m, ys, xs, index = _skeleton_graph(skel)
    if len(ys) == 0:
        return [None] * len(graph.edges)
    tree = cKDTree(np.column_stack([xs, ys]))
    h, w = skel.shape

    def pixels(n) -> np.ndarray:
        cx, cy = n.center
        if n.kind == "junction":
            d, i = tree.query([cx, cy])
            return np.array([i]) if d <= 30 else np.array([], int)
        x1, y1, x2, y2 = n.box
        sub = index[max(0, y1 - margin):min(h, y2 + margin + 1), max(0, x1 - margin):min(w, x2 + margin + 1)]
        ids = sub[sub >= 0]
        if len(ids):
            return ids
        d, i = tree.query([cx, cy])
        return np.array([i]) if d <= 60 else np.array([], int)

    cache: Dict[int, np.ndarray] = {}
    out: List[Optional[List[Point]]] = []
    for a, b in graph.edges:
        na, nb = nodes[a], nodes[b]
        src = cache.setdefault(a, pixels(na))
        dst = cache.setdefault(b, pixels(nb))
        if not len(src) or not len(dst):
            out.append(None)
            continue
        straight = np.hypot(na.center[0] - nb.center[0], na.center[1] - nb.center[1])
        dist, pred, _ = dijkstra(m, directed=False, indices=src, min_only=True, return_predecessors=True,
                                 limit=3 * straight + 300)
        best = dst[np.argmin(dist[dst])]
        if not np.isfinite(dist[best]):
            out.append(None)
            continue
        path = [int(best)]
        while pred[path[-1]] >= 0:
            path.append(int(pred[path[-1]]))
        pts = np.array([[xs[i], ys[i]] for i in reversed(path)], np.int32)
        if len(pts) > 2:
            pts = cv2.approxPolyDP(pts.reshape(-1, 1, 2), epsilon, False).reshape(-1, 2)
        out.append([tuple(map(int, na.center))] + [(int(x), int(y)) for x, y in pts] + [tuple(map(int, nb.center))])
    return out
