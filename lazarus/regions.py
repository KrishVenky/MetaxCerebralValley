"""Find the parts of a sheet that are not process drawing (title block, notes, revision tables).

Muse Spark looks at a downscaled copy of the whole sheet and returns pixel boxes; the answer is
validated deterministically (inside the image, not most of the sheet, containing no detected
symbols) before it is used to mask those areas out of pipe tracing.
"""
import json
import re
from typing import List, Tuple

from PIL import Image

from . import meta_client

Box = Tuple[int, int, int, int]
VIEW = 1600

PROMPT = (
    "This image is a full engineering drawing sheet ({w} x {h} pixels). Find every rectangular area that "
    "is NOT process drawing: the title block, general notes panels, legend or revision tables, and "
    "border text. Do not include the area that holds the piping and instruments. Return pixel boxes in "
    "this image's own coordinates as only JSON: "
    '{{"regions": [{{"label": "...", "x1": 0, "y1": 0, "x2": 0, "y2": 0}}]}}. Return an empty list if '
    "there are none. Do not deliberate; give your best estimate."
)


def find(img: Image.Image, symbol_boxes: List[Box] = ()) -> List[Box]:
    w, h = img.size
    s = VIEW / max(w, h)
    small = img.convert("RGB").resize((int(w * s), int(h * s)), Image.LANCZOS)
    r = meta_client.spark_vision(PROMPT.format(w=small.width, h=small.height), [small], max_output_tokens=8000)
    m = re.search(r"\{.*\}", r["text"], re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0)).get("regions", [])
    except ValueError:
        return []
    out: List[Box] = []
    for it in items:
        try:
            b = tuple(int(float(it[k]) / s) for k in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError):
            continue
        b = (max(0, min(b[0], b[2])), max(0, min(b[1], b[3])), min(w, max(b[0], b[2])), min(h, max(b[1], b[3])))
        area = (b[2] - b[0]) * (b[3] - b[1])
        if area < 0.002 * w * h or area > 0.4 * w * h:
            continue
        inside = sum(b[0] <= (x1 + x2) / 2 <= b[2] and b[1] <= (y1 + y2) / 2 <= b[3] for x1, y1, x2, y2 in symbol_boxes)
        if inside > 1:                       # a region holding symbols is drawing, not title block
            continue
        out.append(b)
    return out


def mask(arr, regions: List[Box]):
    """Copy of the image with the regions painted white, for tracing."""
    out = arr.copy()
    for x1, y1, x2, y2 in regions:
        out[y1:y2, x1:x2] = 255
    return out


EQUIP_PROMPT = (
    "This image is a full piping and instrumentation drawing ({w} x {h} pixels). Find every LARGE piece of "
    "process equipment drawn as an outline: vessels, tanks, columns, drums, reactors and heat exchangers. "
    "Do NOT include pumps, valves, instruments, motors, pipes or text. Give each a tight pixel box in this "
    "image's own coordinates as only JSON: "
    '{{"equipment": [{{"label": "...", "x1": 0, "y1": 0, "x2": 0, "y2": 0}}]}}. Return an empty list if '
    "there are none. Do not deliberate; give your best estimate."
)


def find_equipment(img: Image.Image, symbol_boxes: List[Box] = ()) -> List[Box]:
    """Large vessel-like equipment that SAM does not return as a symbol. Muse Spark proposes boxes on
    a downscaled sheet; a box is kept only if it is large, in the image, and swallows at most one
    already-detected symbol (a box over many symbols is a region, not a vessel)."""
    w, h = img.size
    s = VIEW / max(w, h)
    small = img.convert("RGB").resize((int(w * s), int(h * s)), Image.LANCZOS)
    r = meta_client.spark_vision(EQUIP_PROMPT.format(w=small.width, h=small.height), [small], max_output_tokens=8000)
    m = re.search(r"\{.*\}", r["text"], re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0)).get("equipment", [])
    except ValueError:
        return []
    out: List[Box] = []
    for it in items:
        try:
            b = tuple(int(float(it[k]) / s) for k in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError):
            continue
        b = (max(0, min(b[0], b[2])), max(0, min(b[1], b[3])), min(w, max(b[0], b[2])), min(h, max(b[1], b[3])))
        area = (b[2] - b[0]) * (b[3] - b[1])
        if area < 0.01 * w * h or area > 0.4 * w * h:
            continue
        inside = sum(b[0] <= (x1 + x2) / 2 <= b[2] and b[1] <= (y1 + y2) / 2 <= b[3] for x1, y1, x2, y2 in symbol_boxes)
        if inside > 1:
            continue
        out.append(b)
    return out


LEGEND_PROMPT = (
    "This image is a full piping and instrumentation drawing sheet ({w} x {h} pixels). Find the legend: "
    "the table or panel that explains the drawing's symbols, instrument letter codes or line types "
    "(often titled LEGEND, SYMBOLS or KEY). Return one tight pixel box in this image's own coordinates "
    "covering the whole legend, including its symbol samples, as only JSON: "
    '{{"legend": {{"x1": 0, "y1": 0, "x2": 0, "y2": 0}}}}, or {{"legend": null}} if the sheet has none. '
    "Do not deliberate; give your best estimate."
)

READ_PROMPT = (
    "This is the legend of a piping and instrumentation drawing. List every entry it defines. For each, "
    "give what is defined (a symbol, an instrument letter code, a line type or an abbreviation) and its "
    "meaning exactly as printed. Answer with only JSON: "
    '{"entries": [{"item": "...", "meaning": "..."}]}. Do not deliberate.'
)


def find_legend(img: Image.Image) -> Box:
    """The legend's box, or None. Muse Spark proposes it on a downscaled sheet. Unlike find(), the box
    is expected to hold symbols, so only its size is checked."""
    w, h = img.size
    s = VIEW / max(w, h)
    small = img.convert("RGB").resize((int(w * s), int(h * s)), Image.LANCZOS)
    r = meta_client.spark_vision(LEGEND_PROMPT.format(w=small.width, h=small.height), [small], max_output_tokens=8000)
    m = re.search(r"\{.*\}", r["text"], re.S)
    if not m:
        return None
    try:
        it = json.loads(m.group(0)).get("legend")
        b = tuple(int(float(it[k]) / s) for k in ("x1", "y1", "x2", "y2"))
    except (ValueError, KeyError, TypeError):
        return None
    b = (max(0, min(b[0], b[2])), max(0, min(b[1], b[3])), min(w, max(b[0], b[2])), min(h, max(b[1], b[3])))
    area = (b[2] - b[0]) * (b[3] - b[1])
    return b if 0.005 * w * h <= area <= 0.25 * w * h else None


def read_legend(img: Image.Image, box: Box) -> List[dict]:
    """What the legend defines, read by Muse Spark from a crop at its native resolution."""
    crop = img.convert("RGB").crop(box)
    if max(crop.size) > 1800:
        k = 1800 / max(crop.size)
        crop = crop.resize((int(crop.width * k), int(crop.height * k)), Image.LANCZOS)
    r = meta_client.spark_vision(READ_PROMPT, [crop], max_output_tokens=12000)
    m = re.search(r"\{.*\}", r["text"], re.S)
    if not m:
        return []
    try:
        return [e for e in json.loads(m.group(0)).get("entries", []) if isinstance(e, dict) and e.get("item")]
    except ValueError:
        return []
