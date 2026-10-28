"""Classify one cropped P&ID symbol with Muse Spark Open (Baseten): equipment type, kind, and any
printed tag. Output is constrained to a fixed taxonomy so the verifier can reason over it.
"""
import json
import re
from typing import Dict

from PIL import Image

from . import baseten_client

# type -> kind. "kind" is what the verifier and simulator care about.
TAXONOMY = {
    "gate_valve": "valve", "globe_valve": "valve", "ball_valve": "valve", "plug_valve": "valve",
    "butterfly_valve": "valve", "check_valve": "valve", "three_way_valve": "valve",
    "control_valve": "valve", "relief_valve": "valve", "angle_valve": "valve",
    "pump": "equipment", "compressor": "equipment", "heat_exchanger": "equipment",
    "vessel": "equipment", "filter_strainer": "equipment", "motor_or_actuator": "equipment",
    "instrument_bubble": "instrument", "instrument_box": "instrument",
    "reducer": "fitting", "flange_or_connector": "fitting", "arrow_or_flow_marker": "marker",
    "reference_marker": "marker", "other": "other",
}

_PROMPT_HEAD = (
    "This image is one cropped symbol from a piping and instrumentation diagram (P&ID).\n"
    "Pick the single best type from this list: " + ", ".join(TAXONOMY) + ".\n"
    "Hints: valves are bow-tie or triangle shapes (filled or empty, with or without an extra "
    "circle, bar or actuator); instrument_bubble is a circle with text; instrument_box is a "
    "square or rectangle with text; flange_or_connector is bars or plates across a pipe.\n"
    "If the symbol contains printed text, copy it exactly as 'tag' (letters on one line, numbers "
    "on the next joined with a space); otherwise tag is \"\".\n"
    "Do not deliberate over fine distinctions; the crop is low resolution, so commit to the "
    "closest match quickly.\n"
    'Answer with only JSON: {"type": "...", "tag": "...", "confidence": 0.0-1.0}'
)
PROMPT = _PROMPT_HEAD


def prompt_for(context: str = "") -> str:
    """The classifier prompt, with the drawing's legend vocabulary (codes.Codes.context) when known."""
    return _PROMPT_HEAD if not context else _PROMPT_HEAD + "\n" + context


def _parse(text: str) -> Dict:
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            t = str(d.get("type", "other")).strip().lower()
            return {"type": t if t in TAXONOMY else "other", "tag": str(d.get("tag", "")).strip(),
                    "confidence": float(d.get("confidence", 0) or 0), "ok": True}
        except (ValueError, TypeError):
            pass
    return {"type": "other", "tag": "", "confidence": 0.0, "ok": False}


def crop_symbol(tile: Image.Image, box, pad: int = 8, size: int = 224) -> Image.Image:
    x1, y1, x2, y2 = box
    w, h = tile.size
    c = tile.crop((max(0, int(x1) - pad), max(0, int(y1) - pad), min(w, int(x2) + pad), min(h, int(y2) + pad)))
    s = size / max(c.size)
    return c.resize((max(1, int(c.width * s)), max(1, int(c.height * s))), Image.LANCZOS)


REFERENCE_NOTE = ("A second image shows this drawing's legend, with sample drawings of its symbols. When the "
                  "first image matches one of those samples, choose the matching type.")


def classify(crop: Image.Image, context: str = "", reference: Image.Image = None) -> Dict:
    prompt = prompt_for(context)
    images = [crop]
    if reference is not None:
        prompt += "\n" + REFERENCE_NOTE
        images.append(reference)
    r = baseten_client.spark_open_vision(prompt, images, max_tokens=3000)
    out = _parse(r["text"])
    if not out["ok"]:               # reasoning ate the budget on an ambiguous crop: one bigger retry
        r = baseten_client.spark_open_vision(prompt, images, max_tokens=9000)
        out = _parse(r["text"])
    out["kind"] = TAXONOMY[out["type"]]
    out["out_tokens"], out["in_tokens"], out["cached"] = r["out_tokens"], r["in_tokens"], r["cached"]
    return out
