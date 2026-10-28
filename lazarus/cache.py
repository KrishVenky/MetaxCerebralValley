"""Disk cache keyed by request content. Cached calls cost nothing."""
import hashlib
import json
from typing import Any, Optional

from . import config

_DIR = config.CACHE_DIR / "responses"


def make_key(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def get(key: str) -> Optional[Any]:
    path = _DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def put(key: str, value: Any) -> None:
    _DIR.mkdir(parents=True, exist_ok=True)
    (_DIR / f"{key}.json").write_text(json.dumps(value), encoding="utf-8")
