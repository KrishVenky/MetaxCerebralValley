"""Smoke test: one SAM call, one Meta Spark call, one Baseten call. Costs about a cent."""
from pathlib import Path

from PIL import Image

from lazarus import baseten_client, config, ledger, meta_client

Image.MAX_IMAGE_PIXELS = None
DATA = Path(__file__).resolve().parent.parent / "data" / "pid_val"

print("keys set:", config.key_status())
img = Image.open(DATA / "0.jpg")
print("drawing size:", img.size)

# A 1792x1140 tile from the middle of the drawing (native resolution).
w, h = img.size
tile = img.crop((w // 2 - 896, h // 2 - 570, w // 2 + 896, h // 2 + 570))
tile.save(config.CACHE_DIR.parent / "data" / "probe_tile.jpg") if False else None

print("\n-- SAM 3.1: 'valve' on one native-resolution tile")
try:
    r = meta_client.sam_segment(tile, "valve")
    print("boxes:", len(r["boxes"]), "| raw chars:", r["raw_len"], "| cached:", r["cached"])
    for b in r["boxes"][:5]:
        print("  ", b)
except Exception as e:
    print("SAM failed:", type(e).__name__, str(e)[:400])

print("\n-- Muse Spark (Meta API): read the tags on the tile")
try:
    r = meta_client.spark_vision(
        "This is a crop of a piping and instrumentation diagram (P&ID). "
        "List every text tag you can read (e.g. FV-101, P-203), one per line.",
        [tile],
    )
    print(r["text"][:600])
    print("tokens in/out:", r["in_tokens"], r["out_tokens"])
except Exception as e:
    print("Spark failed:", type(e).__name__, str(e)[:400])

print("\n-- Muse Spark Open (Baseten): same question")
try:
    r = baseten_client.spark_open_vision(
        "This is a crop of a piping and instrumentation diagram (P&ID). "
        "List every text tag you can read (e.g. FV-101, P-203), one per line.",
        [tile],
    )
    print(r["text"][:600])
    print("tokens in/out:", r["in_tokens"], r["out_tokens"])
except Exception as e:
    print("Baseten failed:", type(e).__name__, str(e)[:400])

print("\nledger:", ledger.summary(), "| total $%.4f" % ledger.total_usd())
