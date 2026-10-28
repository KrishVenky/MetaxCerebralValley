"""Spark Open symbol classification test on ground-truth symbol crops.

Dataset classes are numeric (32 visual types, no names), so we measure consistency, not accuracy:
for each ground-truth class, how often does Spark give its modal type (purity)? We also print the
modal type per class so the mapping can be eyeballed against the contact sheet.

Usage: python -m scripts.classify_test [per_class=4] [workers=8]
"""
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from lazarus import classify, ledger
from scripts.sam_prompt_explore import DATA

Image.MAX_IMAGE_PIXELS = None
per_class = int(sys.argv[1]) if len(sys.argv) > 1 else 4
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8

rows = [json.loads(l) for l in (DATA / "metadata.jsonl").open()]
rows = [r for r in rows if (DATA / r["file_name"]).exists()]
by = defaultdict(list)
for r in rows:
    for b, l in zip(r["symbols"]["bbox"], r["symbols"]["labels"]):
        by[l].append((r["file_name"], b))

random.seed(0)
jobs = []
for l in sorted(by):
    for fn, b in random.sample(by[l], min(per_class, len(by[l]))):
        jobs.append((l, fn, b))

imgs = {}
def crop_of(fn, b):
    if fn not in imgs:
        imgs[fn] = Image.open(DATA / fn).convert("RGB")
    return classify.crop_symbol(imgs[fn], b)

crops = [crop_of(fn, b) for _, fn, b in jobs]
with ThreadPoolExecutor(workers) as ex:
    res = list(ex.map(classify.classify, crops))

parsed_ok = sum(r["ok"] for r in res)
print(f"{len(res)} crops, parsed {parsed_ok}, tokens in/out avg "
      f"{sum(r['in_tokens'] for r in res) / len(res):.0f}/{sum(r['out_tokens'] for r in res) / len(res):.0f}")

pred_by = defaultdict(list)
for (l, fn, b), r in zip(jobs, res):
    pred_by[l].append(r)
pur_sum = 0
print(f"\n{'cls':>3s} {'modal type':20s} {'purity':>6s}  tags")
for l in sorted(pred_by):
    c = Counter(r["type"] for r in pred_by[l])
    t, n = c.most_common(1)[0]
    pur_sum += n
    tags = [r["tag"] for r in pred_by[l] if r["tag"]][:3]
    print(f"{l:3d} {t:20s} {n}/{len(pred_by[l])}   {tags}")
print(f"\noverall class-purity {pur_sum}/{len(res)} = {pur_sum / len(res):.0%}")
kinds = Counter(r["kind"] for r in res)
print("kinds:", dict(kinds))
modal = {l: Counter(r["type"] for r in v).most_common(1)[0][0] for l, v in pred_by.items()}
print("distinct modal types used:", len(set(modal.values())), "of 32 classes")
out_path = DATA.parent.parent / ".cache" / "out" / "classify_test.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(
    [{"cls": l, "file": fn, "box": b, **{k: r[k] for k in ("type", "tag", "confidence", "kind")}}
     for (l, fn, b), r in zip(jobs, res)], indent=1))
print("ledger total $%.4f" % ledger.total_usd())
