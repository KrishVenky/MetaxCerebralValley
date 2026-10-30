"""Dashed signal links: symbol pairs that only a dashed line joins.

The solid-pipe tracer cannot see them: each dash is shorter than its text filter, so the whole line is
discarded. A second trace closes gaps along the horizontal and vertical axes first, which makes a dashed
line one component; any symbol pair that this trace joins and the solid trace did not is a signal link.
These are kept out of `Graph.edges` (they carry signals, not process fluid, so isolation analysis must
not traverse them). Each link is then checked against the pixels by `dashed.check_dashed` (drafted by Muse
Code): links it confirms are marked verified, the rest stay "inferred" and go to a human.
"""
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import dashed, pipes

AXIS_GAP = 21               # px; closes the gap between dashes. Tuned on the benchmark's dash pitch.
NEAR = 60                   # only symbols this close to a link's two boxes can block it


def pairs_through_junctions(edges: Sequence[Tuple[int, int]], n_sym: int) -> set:
    """Symbol pairs joined by a chain of edges whose inner nodes are all junctions (ids >= n_sym)."""
    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    out = set()
    for s in range(n_sym):
        seen, stack = {s}, [s]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y in seen:
                    continue
                seen.add(y)
                if y < n_sym:
                    out.add((min(s, y), max(s, y)))
                else:
                    stack.append(y)
    return out


def find(tile_rgb: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]], solid_edges: Sequence[Tuple[int, int]],
         axis_gap: int = AXIS_GAP) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], List[Tuple[int, int]]]]:
    """Returns (signal links, {pair: route} for the subset whose pixels check_dashed confirms). Pairs are
    symbol indices; a route is the corner points of the dashed line the checker found."""
    n = len(boxes)
    if n < 2:
        return [], {}
    gapped = pipes.trace(tile_rgb, list(boxes), junctions=True, axis_gap=axis_gap)
    links = sorted(pairs_through_junctions(gapped["edges"], n) - pairs_through_junctions(solid_edges, n))
    ink = pipes.ink_mask(tile_rgb)
    verified = {}
    for a, b in links:
        lo = (min(boxes[a][0], boxes[b][0]) - NEAR, min(boxes[a][1], boxes[b][1]) - NEAR)
        hi = (max(boxes[a][2], boxes[b][2]) + NEAR, max(boxes[a][3], boxes[b][3]) + NEAR)
        others = [o for k, o in enumerate(boxes) if k not in (a, b)
                  and o[2] >= lo[0] and o[0] <= hi[0] and o[3] >= lo[1] and o[1] <= hi[1]]
        res = dashed.check_dashed(ink, boxes[a], boxes[b], others)
        if res.ok:
            verified[(a, b)] = res.path
    return links, verified
