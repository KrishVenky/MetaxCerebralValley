"""Deterministic check for dashed signal lines between two symbol boxes.

`lazarus.pipes.trace` recovers solid pipes but misses dashed instrument lines:
gaps break the flood along the skeleton. A vision model may *propose* that two
symbols are joined by a dashed line; this module accepts or rejects one such
proposal by looking at the pixels between the two boxes. No model, no network.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

Box = Tuple[int, int, int, int]
Point = Tuple[int, int]


@dataclass
class DashedLink:
    ok: bool                      # accepted as a dashed line joining the two boxes
    score: float                  # 0..1, how dash-like the evidence is (0 when no evidence)
    n_dashes: int                 # number of dash segments found along the best path
    coverage: float               # fraction of the path length that is ink (0..1)
    path: List[Tuple[int, int]]   # best path as corner points (x, y), box edge to box edge
    reason: str                   # short human-readable why, e.g. "ok", "too few dashes"


def _as_box(b) -> Box:
    x1, y1, x2, y2 = b
    return (int(x1), int(y1), int(x2), int(y2))


def _boxes_overlap(a: Box, b: Box) -> bool:
    # Inclusive edges: sharing even one pixel row/column counts as overlap/touch.
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _inside(x: int, y: int, b: Box) -> bool:
    return b[0] <= x <= b[2] and b[1] <= y <= b[3]


def _center(b: Box) -> Point:
    return ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)


def _walk_corners(corners: Sequence[Point]) -> List[Point]:
    """Expand an axis-aligned polyline into 1px steps, corners included once."""
    corners = [(int(x), int(y)) for x, y in corners]
    if len(corners) == 0:
        return []
    pts = [corners[0]]
    for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
        dx = 0 if x1 == x0 else (1 if x1 > x0 else -1)
        dy = 0 if y1 == y0 else (1 if y1 > y0 else -1)
        x, y = x0, y0
        # Axis-aligned by construction; a diagonal fallback steps both axes.
        while (x, y) != (x1, y1):
            if x != x1 and y != y1 and dx != 0 and dy != 0:
                x += dx
                y += dy
            elif x != x1 and (dx != 0 or dy == 0):
                x += dx if dx != 0 else (1 if x1 > x else -1)
            else:
                y += dy if dy != 0 else (1 if y1 > y else -1)
            pts.append((x, y))
            if len(pts) > 100000:  # pathological input guard; never hit in practice
                break
    return pts


def _straight_candidate(a: Box, b: Box):
    """Horizontal/vertical run through the middle of the x/y overlap, if any."""
    ox0, ox1 = max(a[0], b[0]), min(a[2], b[2])
    oy0, oy1 = max(a[1], b[1]), min(a[3], b[3])
    # Rates: x-overlap -> vertical run; y-overlap -> horizontal run.
    if ox0 <= ox1 and not (oy0 <= oy1):
        xm = (ox0 + ox1) // 2
        if a[3] < b[1]:
            return [(xm, a[3]), (xm, b[1])]
        if b[3] < a[1]:
            return [(xm, b[3]), (xm, a[1])]
        return None
    if oy0 <= oy1 and not (ox0 <= ox1):
        ym = (oy0 + oy1) // 2
        if a[2] < b[0]:
            return [(a[2], ym), (b[0], ym)]
        if b[2] < a[0]:
            return [(b[2], ym), (a[0], ym)]
        return None
    return None


def _trim_l(p0: Point, p1: Point, p2: Point, a: Box, b: Box):
    """Trim a centre-to-centre L to box edges; None when there is no gap."""
    walk = _walk_corners([p0, p1, p2])
    if len(walk) == 0:
        return None
    # Last pixel inside A (on its edge) scanning from the start.
    i0 = None
    for i, (x, y) in enumerate(walk):
        if _inside(x, y, a):
            i0 = i
        elif i0 is not None:
            break
    if i0 is None:
        return None
    # First pixel inside B.
    i1 = None
    for i, (x, y) in enumerate(walk):
        if _inside(x, y, b):
            i1 = i
            break
    if i1 is None or i1 <= i0:
        return None
    start = walk[i0]
    end = walk[i1]
    # Corner index of the full centre-to-centre walk.
    l1 = abs(p1[0] - p0[0]) + abs(p1[1] - p0[1])
    if i0 <= l1 <= i1 and not (start == p1 or end == p1):
        # Corner survives trimming (and is not an endpoint already).
        if p1 == start or p1 == end:
            return [start, end]
        return [start, p1, end]
    return [start, end]


def _l_candidates(a: Box, b: Box):
    ca = _center(a)
    cb = _center(b)
    out = []
    h_first = _trim_l(ca, (cb[0], ca[1]), cb, a, b)
    if h_first is not None:
        out.append(h_first)
    v_first = _trim_l(ca, (ca[0], cb[1]), cb, a, b)
    if v_first is not None:
        out.append(v_first)
    return out


def _facing_x(a: Box, b: Box):
    ca_x = (a[0] + a[2]) // 2
    cb_x = (b[0] + b[2]) // 2
    if ca_x <= cb_x:
        return a[2], b[0]
    return a[0], b[2]


def _facing_y(a: Box, b: Box):
    ca_y = (a[1] + a[3]) // 2
    cb_y = (b[1] + b[3]) // 2
    if ca_y <= cb_y:
        return a[3], b[1]
    return a[1], b[3]


def _trim_polyline(corners: Sequence[Point], a: Box, b: Box):
    """Trim an N-point polyline to box edges; None when there is no gap."""
    corners = [(int(x), int(y)) for x, y in corners]
    if len(corners) < 2:
        return None
    walk = _walk_corners(corners)
    if len(walk) == 0:
        return None
    i0 = None
    for i, (x, y) in enumerate(walk):
        if _inside(x, y, a):
            i0 = i
        elif i0 is not None:
            break
    if i0 is None:
        return None
    i1 = None
    for i, (x, y) in enumerate(walk):
        if _inside(x, y, b):
            i1 = i
            break
    if i1 is None or i1 <= i0:
        return None
    start = walk[i0]
    end = walk[i1]
    idx = [0]
    for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
        idx.append(idx[-1] + max(abs(x1 - x0), abs(y1 - y0)))
    trimmed: List[Point] = [start]
    for k in range(1, len(corners) - 1):
        ck = corners[k]
        if k >= len(idx) or idx[k] >= len(walk):
            continue
        if i0 < idx[k] < i1 and ck != start and ck != end and ck != trimmed[-1]:
            trimmed.append(ck)
    if end != trimmed[-1]:
        trimmed.append(end)
    dedup = [trimmed[0]]
    for p in trimmed[1:]:
        if p != dedup[-1]:
            dedup.append(p)
    trimmed = dedup
    changed = True
    while changed and len(trimmed) > 2:
        changed = False
        new = [trimmed[0]]
        for i in range(1, len(trimmed) - 1):
            p0, p1, p2 = new[-1], trimmed[i], trimmed[i + 1]
            if p0[0] == p1[0] == p2[0] and min(p0[1], p2[1]) <= p1[1] <= max(p0[1], p2[1]):
                changed = True
                continue
            if p0[1] == p1[1] == p2[1] and min(p0[0], p2[0]) <= p1[0] <= max(p0[0], p2[0]):
                changed = True
                continue
            new.append(p1)
        new.append(trimmed[-1])
        trimmed = new
    if len(trimmed) < 2 or trimmed[0] == trimmed[-1]:
        return None
    walk_t = _walk_corners(trimmed)
    walk_out = [(x, y) for x, y in walk_t
                if not _inside(x, y, a) and not _inside(x, y, b)]
    if len(walk_out) == 0:
        return None
    return trimmed


def _offset_straight_candidates(a: Box, b: Box, band: int):
    band_i = max(0, int(band))
    ox0, ox1 = max(a[0], b[0]), min(a[2], b[2])
    oy0, oy1 = max(a[1], b[1]), min(a[3], b[3])
    out = []
    if ox0 <= ox1 and not (oy0 <= oy1):
        width = ox1 - ox0 + 1
        if width >= 3 * band_i:
            xm = (ox0 + ox1) // 2
            off = width // 3
            if off > 0:
                for xm2 in (xm - off, xm + off):
                    if xm2 == xm or not (ox0 <= xm2 <= ox1):
                        continue
                    if a[3] < b[1]:
                        cand = [(xm2, a[3]), (xm2, b[1])]
                    elif b[3] < a[1]:
                        cand = [(xm2, b[3]), (xm2, a[1])]
                    else:
                        continue
                    walk = _walk_corners(cand)
                    outside = [p for p in walk
                               if not _inside(p[0], p[1], a) and not _inside(p[0], p[1], b)]
                    if len(outside) == 0:
                        continue
                    out.append(cand)
    if oy0 <= oy1 and not (ox0 <= ox1):
        width = oy1 - oy0 + 1
        if width >= 3 * band_i:
            ym = (oy0 + oy1) // 2
            off = width // 3
            if off > 0:
                for ym2 in (ym - off, ym + off):
                    if ym2 == ym or not (oy0 <= ym2 <= oy1):
                        continue
                    if a[2] < b[0]:
                        cand = [(a[2], ym2), (b[0], ym2)]
                    elif b[2] < a[0]:
                        cand = [(b[2], ym2), (a[0], ym2)]
                    else:
                        continue
                    walk = _walk_corners(cand)
                    outside = [p for p in walk
                               if not _inside(p[0], p[1], a) and not _inside(p[0], p[1], b)]
                    if len(outside) == 0:
                        continue
                    out.append(cand)
    return out


def _z_candidates(a: Box, b: Box):
    out = []
    ca = _center(a)
    cb = _center(b)
    if a[2] < b[0] or b[2] < a[0]:
        sx, ex = _facing_x(a, b)
        s = (sx, ca[1])
        e = (ex, cb[1])
        for frac in (0.25, 0.5, 0.75):
            xm = int(round(sx + frac * (ex - sx)))
            trimmed = _trim_polyline([s, (xm, s[1]), (xm, e[1]), e], a, b)
            if trimmed is not None:
                out.append(trimmed)
    if a[3] < b[1] or b[3] < a[1]:
        sy, ey = _facing_y(a, b)
        s = (ca[0], sy)
        e = (cb[0], ey)
        for frac in (0.25, 0.5, 0.75):
            ym = int(round(sy + frac * (ey - sy)))
            trimmed = _trim_polyline([s, (s[0], ym), (e[0], ym), e], a, b)
            if trimmed is not None:
                out.append(trimmed)
    return out


def _edge_l_candidates(a: Box, b: Box):
    out = []
    ca = _center(a)
    cb = _center(b)
    sx, ex = _facing_x(a, b)
    sy, ey = _facing_y(a, b)
    s1, e1, c1 = (sx, ca[1]), (cb[0], ey), (cb[0], ca[1])
    if not (s1 == c1 == e1):
        t1 = _trim_polyline([s1, c1, e1], a, b)
        if t1 is not None:
            out.append(t1)
    s2, e2, c2 = (ca[0], sy), (ex, cb[1]), (ca[0], cb[1])
    if not (s2 == c2 == e2):
        t2 = _trim_polyline([s2, c2, e2], a, b)
        if t2 is not None:
            out.append(t2)
    return out


def _sample_signal(ink: np.ndarray, walk: List[Point], band: int):
    """1-D on/off signal: True where ink is within `band` perpendicular to travel.

    Horizontal steps consult a vertical window, vertical steps a horizontal
    window. Samples whose centre is outside the image count as no ink.
    """
    h, w = ink.shape
    band = max(0, int(band))
    n = len(walk)
    sig = np.zeros(n, dtype=bool)
    if n == 0:
        return sig
    # Outgoing direction per sample (last sample reuses its incoming step).
    for i, (x, y) in enumerate(walk):
        if i + 1 < n:
            dx = walk[i + 1][0] - x
            dy = walk[i + 1][1] - y
        elif n > 1:
            dx = x - walk[i - 1][0]
            dy = y - walk[i - 1][1]
        else:
            dx, dy = 1, 0
        horiz = abs(dx) >= abs(dy)
        if horiz:
            if not (0 <= x < w):
                continue
            y0 = max(0, y - band)
            y1 = min(h - 1, y + band)
            if y1 < y0 or not (0 <= y0 < h):
                continue
            if np.any(ink[y0:y1 + 1, x]):
                sig[i] = True
        else:
            if not (0 <= y < h):
                continue
            x0 = max(0, x - band)
            x1 = min(w - 1, x + band)
            if x1 < x0 or not (0 <= x0 < w):
                continue
            if np.any(ink[y, x0:x1 + 1]):
                sig[i] = True
    return sig


def _runs(sig: np.ndarray):
    """Lengths of True runs (dashes) and False runs (gaps), in order."""
    if sig.size == 0:
        return [], []
    change = np.nonzero(np.diff(sig.astype(np.int8)) != 0)[0] + 1
    bounds = np.concatenate(([0], change, [sig.size]))
    dashes, gaps = [], []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if sig[s]:
            dashes.append(int(e - s))
        else:
            gaps.append(int(e - s))
    return dashes, gaps


def _cv(values) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if mean <= 0:
        return 0.0
    return float(arr.std() / mean)


def check_dashed(ink: np.ndarray, box_a, box_b, other_boxes=(), *, band: int = 4,
                 min_dashes: int = 3, min_dash: int = 4, max_dash: int = 60, min_gap: int = 2,
                 max_gap: int = 60, min_coverage: float = 0.25, max_coverage: float = 0.85) -> DashedLink:
    """Accept or reject a proposed dashed link between two symbol boxes.

    Candidates: one straight run through the middle of the x/y overlap when the
    boxes overlap in x or y, plus the two L-shapes (horizontal-then-vertical
    and vertical-then-horizontal) between box centres, trimmed to the box
    edges, plus offset straights (+/-1/3 overlap), Z-shapes (middle at
    25/50/75% of the gap) and edge-based L-shapes, all trimmed to box edges.
    Every candidate is walked pixel by pixel; each step counts as ink
    when any ink pixel lies within `band` perpendicular to travel. Samples
    inside either box are dropped, giving one 1-D on/off signal per candidate.

    Score (0..1; 0 when no dash found or the path is blocked)::

        f_dash = min(1, n_valid_dashes / min_dashes)
        f_gap  = min(1, n_valid_gaps / max(1, min_dashes - 1))
        f_cov  = 1 when coverage is inside [min_coverage, max_coverage], else 0
        f_reg  = max(0, 1 - max(cv_dash, cv_gap) / 0.6)
        score  = 0.35 * f_dash + 0.20 * f_gap + 0.20 * f_cov + 0.25 * f_reg

    Regularity drops a leading partial dash when the signal starts with ink
    and a trailing one when it ends with ink, and uses interior gaps only
    (edge gaps are distances to the boxes, not pattern gaps). A 1px interior
    gap is merged into the surrounding dashes first. New (non-original)
    candidates need coverage >= max(min_coverage, 0.30) and one extra dash.
    The best candidate is the highest score; exact ties between two ok
    candidates of different shapes keep the fewer-cornered one, else earliest.
    """
    try:
        ink_arr = np.asarray(ink)
        if ink_arr.ndim != 2:
            return DashedLink(False, 0.0, 0, 0.0, [], "bad ink")
        ink_bool = ink_arr.astype(bool, copy=False)
    except Exception:
        return DashedLink(False, 0.0, 0, 0.0, [], "bad ink")

    try:
        a = _as_box(box_a)
        b = _as_box(box_b)
        others = [_as_box(o) for o in (other_boxes or ())]
    except Exception:
        return DashedLink(False, 0.0, 0, 0.0, [], "bad box")

    if _boxes_overlap(a, b):
        return DashedLink(False, 0.0, 0, 0.0, [], "boxes overlap")

    candidates: List[Tuple[List[Point], bool]] = []
    straight = _straight_candidate(a, b)
    if straight is not None:
        candidates.append((straight, True))
    for cand in _l_candidates(a, b):
        candidates.append((cand, True))
    for cand in _offset_straight_candidates(a, b, band):
        candidates.append((cand, False))
    for cand in _z_candidates(a, b):
        candidates.append((cand, False))
    for cand in _edge_l_candidates(a, b):
        candidates.append((cand, False))
    if not candidates:
        return DashedLink(False, 0.0, 0, 0.0, [], "no path")

    need_gaps = max(0, int(min_dashes) - 1)
    best = None  # (score, index, DashedLink, corners_len)

    for idx, (corners, is_orig) in enumerate(candidates):
        walk = _walk_corners(corners)
        # Drop samples inside either endpoint box.
        keep = [i for i, (x, y) in enumerate(walk)
                if not _inside(x, y, a) and not _inside(x, y, b)]
        walk_out = [walk[i] for i in keep]
        if len(walk_out) == 0:
            sig = np.zeros(0, dtype=bool)
        else:
            # Resample along the outside part; ends keep their travel direction
            # from the full walk so the band orientation stays correct.
            full_sig = _sample_signal(ink_bool, walk, band)
            sig = full_sig[np.array(keep, dtype=int)] if len(keep) else np.zeros(0, dtype=bool)

        # Blocked when the outside centreline enters any other box.
        blocked = False
        for ob in others:
            for (x, y) in walk_out:
                if _inside(x, y, ob):
                    blocked = True
                    break
            if blocked:
                break

        # Merge a 1px interior gap (shorter than min_gap and 2px) into dashes.
        if sig.size >= 3 and min_gap > 1:
            orig = sig.copy()
            mask = (~orig[1:-1]) & orig[:-2] & orig[2:]
            if np.any(mask):
                sig = sig.copy()
                sig[np.where(mask)[0] + 1] = True

        dashes, gaps = _runs(sig)
        n_dashes = len(dashes)
        n_valid_d = sum(1 for d in dashes if min_dash <= d <= max_dash)
        n_valid_g = sum(1 for g in gaps if min_gap <= g <= max_gap)
        coverage = float(sig.mean()) if sig.size else 0.0

        # Regularity inputs: drop partial edge dashes; interior gaps only.
        dash_reg = list(dashes)
        if sig.size and sig[0] and dash_reg:
            dash_reg = dash_reg[1:]
        if sig.size and sig[-1] and dash_reg:
            # If the signal is a single dash it is both leading and trailing.
            dash_reg = dash_reg[:-1] if len(dash_reg) >= 1 else []
            if len(dashes) == 1:
                dash_reg = []
        gap_reg = list(gaps)
        if sig.size and not sig[0] and gap_reg:
            gap_reg = gap_reg[1:]
        if sig.size and not sig[-1] and gap_reg:
            gap_reg = gap_reg[:-1] if len(gap_reg) >= 1 else []
            if len(gaps) == 1:
                gap_reg = []
        cv_d = _cv(dash_reg)
        cv_g = _cv(gap_reg)

        if n_dashes == 0 or blocked:
            score = 0.0
        else:
            f_dash = min(1.0, n_valid_d / max(1, int(min_dashes)))
            f_gap = min(1.0, n_valid_g / max(1, need_gaps)) if need_gaps else 1.0
            f_cov = 1.0 if min_coverage <= coverage <= max_coverage else 0.0
            f_reg = max(0.0, 1.0 - max(cv_d, cv_g) / 0.6)
            score = 0.35 * f_dash + 0.20 * f_gap + 0.20 * f_cov + 0.25 * f_reg
            score = float(min(1.0, max(0.0, score)))

        if is_orig:
            cov_needed = float(min_coverage)
            dash_needed = int(min_dashes)
        else:
            cov_needed = max(float(min_coverage), 0.30)
            dash_needed = int(min_dashes) + 1

        if sig.size < 10:
            ok, reason = False, "path too short"
        elif blocked:
            ok, reason = False, "path blocked"
        elif coverage > max_coverage:
            ok, reason = False, "solid"
        elif n_valid_d < dash_needed:
            ok, reason = False, "too few dashes"
        elif n_valid_g < need_gaps:
            ok, reason = False, "too few gaps"
        elif coverage < cov_needed:
            ok, reason = False, "low coverage"
        elif cv_d >= 0.6:
            ok, reason = False, "irregular dashes"
        elif cv_g >= 0.6:
            ok, reason = False, "irregular gaps"
        else:
            ok, reason = True, "ok"

        link = DashedLink(ok, float(score), int(n_dashes), float(coverage),
                          [(int(x), int(y)) for x, y in corners], str(reason))
        corners_len = len(corners)
        if best is None:
            best = (float(score), idx, link, corners_len)
        else:
            b_score, _b_idx, b_link, b_len = best
            if float(score) > b_score:
                best = (float(score), idx, link, corners_len)
            elif float(score) == b_score:
                if link.ok and b_link.ok and corners_len != b_len:
                    if corners_len < b_len:
                        best = (float(score), idx, link, corners_len)
                # Else keep earliest; best already >= second-best by >= 0.0.

    assert best is not None
    return best[2]
