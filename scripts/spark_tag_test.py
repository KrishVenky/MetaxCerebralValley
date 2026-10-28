"""Can Muse Spark read P&ID text tags? No ground truth for tags in this dataset, so we check
(a) the answer is non-empty and well-formed, (b) cost/tokens, (c) agreement between the
Meta Spark and Baseten Spark Open readings of the same tile (a rough consistency signal).
"""
import re
from pathlib import Path

from PIL import Image

from lazarus import baseten_client, ledger, meta_client

Image.MAX_IMAGE_PIXELS = None
DATA = Path(__file__).resolve().parent.parent / "data" / "pid_val"
PROMPT = ("This is a crop of a piping and instrumentation diagram (P&ID). List every text tag "
          "you can read completely (e.g. FV-101, P-203), one per line, no commentary. "
          "Skip text that is cut off at the crop edge.")


def tags(text):
    return {t.strip().upper() for t in re.findall(r"[A-Za-z]{1,4}-?\d{2,6}[A-Za-z]?", text)}


img = Image.open(DATA / "102.jpg")
tile = img.crop((448, 1710, 448 + 1792, 1710 + 1140))

a = meta_client.spark_vision(PROMPT, [tile])
b = baseten_client.spark_open_vision(PROMPT, [tile], max_tokens=6000)
ta, tb = tags(a["text"]), tags(b["text"])
print("Meta Spark   : %d tags | tokens in/out %d/%d | cached=%s" % (len(ta), a["in_tokens"], a["out_tokens"], a["cached"]))
print("Baseten Open : %d tags | tokens in/out %d/%d | cached=%s" % (len(tb), b["in_tokens"], b["out_tokens"], b["cached"]))
print("agreement (Jaccard): %.0f%%" % (100 * len(ta & tb) / max(1, len(ta | tb))))
print("Meta only:", sorted(ta - tb)[:8])
print("Baseten only:", sorted(tb - ta)[:8])
print("sample Meta output:", a["text"][:300].replace("\n", " | "))
print("ledger:", ledger.summary(), "total $%.4f" % ledger.total_usd())
