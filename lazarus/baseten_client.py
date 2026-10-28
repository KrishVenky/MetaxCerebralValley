"""Baseten wrapper for Muse Spark Open (OpenAI chat-completions style).

Credits are a flat $100 grant, so cost is tracked in tokens (ledger usd = 0).
Muse Spark Open always reasons; reasoning content is returned separately when present.
"""
import random
import time
from typing import Dict, List, Optional

from openai import OpenAI, RateLimitError
from PIL import Image

from . import cache, config, ledger
from .meta_client import to_data_url

_client: Optional[OpenAI] = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not config.BASETEN_API_KEY:
            raise RuntimeError("BASETEN_API_KEY is empty. Set it in .env")
        _client = OpenAI(base_url=config.BASETEN_BASE_URL, api_key=config.BASETEN_API_KEY)
    return _client


def spark_open_vision(prompt: str, images: List[Image.Image], max_tokens: int = 2048) -> Dict:
    urls = [to_data_url(im) for im in images]
    key = cache.make_key("baseten", config.BASETEN_MODEL, prompt, max_tokens, *urls)
    hit = cache.get(key)
    if hit is not None:
        hit["cached"] = True
        return hit

    ledger.check()
    content = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    for attempt in range(8):                    # 429s: back off and retry
        try:
            resp = client().chat.completions.create(
                model=config.BASETEN_MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
            )
            break
        except RateLimitError:
            if attempt == 7:
                raise
            time.sleep(min(60, 2 ** attempt) + random.random() * 2)
    msg = resp.choices[0].message
    usage = resp.usage
    result = {
        "text": msg.content or "",
        "reasoning": getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None),
        "in_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "out_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }
    cache.put(key, result)
    ledger.record("baseten", 0.0, model=config.BASETEN_MODEL,
                  in_tokens=result["in_tokens"], out_tokens=result["out_tokens"])
    result["cached"] = False
    return result
