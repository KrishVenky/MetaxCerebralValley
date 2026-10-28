"""Settings loaded from .env. Keys are never printed or logged."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _get(name: str, default: str = "") -> str:
    # A blank value in .env (e.g. `FOO=`) counts as unset and falls back to the default.
    return (os.environ.get(name) or default).strip()


META_API_KEY = _get("META_API_KEY") or _get("MODEL_API_KEY")
META_API_BASE_URL = _get("META_API_BASE_URL", "https://api.meta.ai/v1")
META_SPARK_MODEL = _get("META_SPARK_MODEL", "muse-spark-1.3")
META_SAM_MODEL = _get("META_SAM_MODEL", "sam-3.1")

BASETEN_API_KEY = _get("BASETEN_API_KEY")
BASETEN_BASE_URL = _get("BASETEN_BASE_URL", "https://inference.baseten.co/v1")
BASETEN_MODEL = _get("BASETEN_MODEL", "meta-models/muse-spark-open-1026")

CACHE_DIR = ROOT / _get("CACHE_DIR", ".cache")
SPEND_GUARD_USD = float(_get("SPEND_GUARD_USD", "100"))

# Published Meta Model API prices (USD), standard tier.
PRICE_SAM_PER_IMAGE = 2.50 / 1000
PRICE_SPARK_IN_PER_TOKEN = 1.25 / 1_000_000
PRICE_SPARK_OUT_PER_TOKEN = 4.25 / 1_000_000


def key_status() -> dict:
    """Which keys are set, without revealing them."""
    return {
        "META_API_KEY": bool(META_API_KEY),
        "BASETEN_API_KEY": bool(BASETEN_API_KEY),
    }
