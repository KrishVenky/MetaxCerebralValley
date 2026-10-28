"""Whole-sheet detection: SAM only sees ~1800px tiles, so slide overlapping tiles over the drawing,
map boxes back to sheet coordinates, drop boxes clipped by an interior tile edge (the neighbouring
tile sees them whole), and merge duplicates from the overlap.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from PIL import Image

from . import meta_client

TILE_W, TILE_H = 1792, 1140
Box = Tuple[int, int, int, int]


def tile_origins(size: Tuple[int, int], overlap: float = 0.25) -> List[Tuple[int, int]]:
    w, h = size
    sx, sy = int(TILE_W * (1 - overlap)), int(TILE_H * (1 - overlap))
    xs = list(range(0, max(1, w - TILE_W + 1), sx))
    ys = list(range(0, max(1, h - TILE_H + 1), sy))
    if xs[-1] + TILE_W < w:
        xs.append(w - TILE_W)
    if ys[-1] + TILE_H < h:
        ys.append(h - TILE_H)
    return [(x, y) for y in ys for x in xs]


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


EQUIPMENT_PROMPTS = ("tank", "vessel", "pump", "heat exchanger")


def _contains(big, small, frac: float = 0.8) -> bool:
    ix1, iy1, ix2, iy2 = max(big[0], small[0]), max(big[1], small[1]), min(big[2], small[2]), min(big[3], small[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = (small[2] - small[0]) * (small[3] - small[1])
    return area > 0 and inter / area >= frac


def detect(img: Image.Image, overlap: float = 0.25, workers: int = 8, border: int = 6,
           extra: Tuple[str, ...] = ()) -> List[Box]:
    """`extra` adds SAM prompts (e.g. EQUIPMENT_PROMPTS) for large items the generic "symbol" prompt
    skips. An extra box that swallows two or more generic boxes is a false merge and is dropped."""
    img = img.convert("RGB")
    w, h = img.size
    origins = tile_origins(img.size, overlap)

    def one(o):
        ox, oy = o
        tile = img.crop((ox, oy, min(w, ox + TILE_W), min(h, oy + TILE_H)))
        out = []
        found = [(b, False) for b in meta_client.sam_symbols(tile)]
        for p in extra:
            found += [((b["x1"], b["y1"], b["x2"], b["y2"]), True) for b in meta_client.sam_segment(tile, p)["boxes"]]
        generic = [b for b, e in found if not e]
        for (x1, y1, x2, y2), is_extra in found:
            if is_extra and sum(_contains((x1, y1, x2, y2), g) for g in generic) >= 2:
                continue
            tw, th = tile.size
            clipped = ((x1 <= border and ox > 0) or (y1 <= border and oy > 0)
                       or (x2 >= tw - border and ox + tw < w) or (y2 >= th - border and oy + th < h))
            if clipped:
                continue
            margin = min(x1, y1, tw - x2, th - y2)           # how far from the tile edge: prefer the interior view
            out.append((margin, (int(x1) + ox, int(y1) + oy, int(x2) + ox, int(y2) + oy)))
        return out

    with ThreadPoolExecutor(workers) as ex:
        cands = [c for part in ex.map(one, origins) for c in part]
    kept: List[Box] = []
    for _, b in sorted(cands, key=lambda c: -c[0]):
        if all(_iou(b, k) < 0.4 for k in kept):
            kept.append(b)
    return sorted(kept, key=lambda b: (b[1], b[0]))
