"""Meta Model API wrapper (Responses API): SAM 3.1 segmentation and Muse Spark vision.

Every call is cached on disk and logged to the spend ledger. A cache hit is free.
"""
import base64
import io
from typing import Dict, List, Optional

from openai import OpenAI
from PIL import Image

from . import cache, config, ledger
from .sam_parse import parse_boxes

_client: Optional[OpenAI] = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not config.META_API_KEY:
            raise RuntimeError("META_API_KEY is empty. Set it in .env")
        _client = OpenAI(base_url=config.META_API_BASE_URL, api_key=config.META_API_KEY)
    return _client


def to_data_url(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=quality)
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def sam_segment(img: Image.Image, prompt: str, confidence: bool = True) -> Dict:
    """Concept-prompted SAM 3.1 call on one still image.

    Returns {"boxes": [...], "raw_len": int, "cached": bool}. `prompt` is a short noun phrase.
    """
    data_url = to_data_url(img)
    key = cache.make_key("sam", config.META_SAM_MODEL, prompt, confidence, data_url)
    hit = cache.get(key)
    if hit is not None:
        hit["cached"] = True
        return hit

    ledger.check()
    metadata = {"mask_encoding": "one_bit"}
    if confidence:
        metadata["include_confidence"] = "true"
    resp = client().responses.create(
        model=config.META_SAM_MODEL,
        input=[{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
        metadata=metadata,
    )
    text = resp.output_text or ""
    result = {"boxes": parse_boxes(text), "raw_len": len(text), "size": list(img.size)}
    cache.put(key, result)
    ledger.record("sam", config.PRICE_SAM_PER_IMAGE, prompt=prompt)
    result["cached"] = False
    return result


def spark_vision(prompt: str, images: List[Image.Image], model: Optional[str] = None,
                 max_output_tokens: int = 6000) -> Dict:
    """Muse Spark on text plus images. Returns {"text", "in_tokens", "out_tokens", "cached"}."""
    model = model or config.META_SPARK_MODEL
    urls = [to_data_url(im) for im in images]
    key = cache.make_key("spark", model, prompt, max_output_tokens, *urls)
    hit = cache.get(key)
    if hit is not None:
        hit["cached"] = True
        return hit

    ledger.check()
    content = [{"type": "input_text", "text": prompt}]
    content += [{"type": "input_image", "image_url": u} for u in urls]
    resp = client().responses.create(
        model=model,
        input=[{"type": "message", "role": "user", "content": content}],
        max_output_tokens=max_output_tokens,
    )
    usage = getattr(resp, "usage", None)
    t_in = getattr(usage, "input_tokens", 0) or 0
    t_out = getattr(usage, "output_tokens", 0) or 0
    result = {"text": resp.output_text or "", "in_tokens": t_in, "out_tokens": t_out}
    cache.put(key, result)
    usd = t_in * config.PRICE_SPARK_IN_PER_TOKEN + t_out * config.PRICE_SPARK_OUT_PER_TOKEN
    ledger.record("spark", usd, model=model, in_tokens=t_in, out_tokens=t_out)
    result["cached"] = False
    return result


FALLBACK_PROMPTS = ["symbols", "diagram symbol", "small symbol", "component"]


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def sam_symbols(img: Image.Image, min_expected: int = 12) -> List[tuple]:
    """Symbol boxes (x1, y1, x2, y2) on one tile. "symbol" first; if the tile comes back sparse or
    silent, also try alternate phrasings and merge by non-max suppression."""
    def as_boxes(r):
        return [(b["x1"], b["y1"], b["x2"], b["y2"], b.get("conf") or 0.0) for b in r["boxes"]]

    def nms(bs):
        out = []
        for b in sorted(bs, key=lambda b: -b[4]):
            if all(_iou(b[:4], o[:4]) < 0.5 for o in out):
                out.append(b)
        return out

    boxes = nms(as_boxes(sam_segment(img, "symbol")))
    if len(boxes) < min_expected:
        for p in FALLBACK_PROMPTS:
            boxes = nms(boxes + as_boxes(sam_segment(img, p)))
    return [b[:4] for b in boxes]
