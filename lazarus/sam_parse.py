"""Parse SAM 3.1 output text.

Only boxes (and confidences) are parsed for now. Mask payloads are base85 with an
undocumented layout in the pages we have, so masks are ignored until needed.
Grammar: dev.meta.ai/docs/sam/reading-segmentation
"""
import re
from typing import Dict, List

_BOX_RE = re.compile(
    r"(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([\d.]+))?\|>"
)
_FRAME_RE = re.compile(r"<(\d+)f>")


def parse_boxes(text: str) -> List[Dict]:
    """Return a list of {frame, object_id, x1, y1, x2, y2, w, h, conf} for every record."""
    out: List[Dict] = []
    for line in text.splitlines():
        fm = _FRAME_RE.match(line)
        frame = int(fm.group(1)) if fm else 0
        for m in _BOX_RE.finditer(line):
            oid, x1, y1, x2, y2, w, h, c = m.groups()
            out.append(
                {
                    "frame": frame,
                    "object_id": int(oid),
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "w": int(w), "h": int(h),
                    "conf": float(c) if c else None,
                }
            )
    return out
