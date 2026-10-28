"""Legend experiment: does showing Spark Open a legend sheet fix symbol-type classification?

Dataset legend ids are the 32 visual classes, so accuracy is real (predicted legend id == class).
Legend names below are my own descriptions of the reference symbols. Reference crops (2 per class)
are excluded from the test sample. Baseline from classify_test (no legend): 79% class purity,
84% valve-kind recall on the valve classes.

Usage: python -m scripts.legend_test [per_class=4] [workers=10]
"""
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from lazarus import ledger, legend
from lazarus.classify import crop_symbol
from scripts.sam_prompt_explore import DATA

Image.MAX_IMAGE_PIXELS = None
per_class = int(sys.argv[1]) if len(sys.argv) > 1 else 4
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10

NAMES = {
    1: "valve, bow-tie with X through it", 2: "valve, bow-tie with circle and X",
    3: "valve, bow-tie with solid dot", 4: "valve, plain bow-tie (open)",
    5: "valve, bow-tie with oval", 6: "valve, diverter with solid dot",
    7: "valve, bow-tie with stem bar", 8: "check valve, triangle with bar",
    9: "valve, bow-tie with D-shaped arc", 10: "three-way valve, solid triangle",
    11: "valve, solid bow-tie, vertical", 12: "valve, solid bow-tie, horizontal",
    13: "valve, solid bow-tie with narrow waist", 14: "valve, bow-tie with half-circle actuator",
    15: "inline device, open circle on pipe", 16: "inline device, solid circle on pipe",
    17: "equipment, bars with oval", 18: "equipment, bars with solid and open oval",
    19: "equipment, bars with stacked ovals", 20: "equipment, trapezoid hopper",
    21: "flange, two bars across pipe", 22: "equipment, box with zigzag coil",
    23: "equipment, twin cylinders", 24: "marker, solid arrowhead",
    25: "valve, circle containing X", 26: "instrument, circle with text (GRO/GRI/GI)",
    27: "instrument, circle with text (RO-10)", 28: "instrument, circle with text (SDL)",
    29: "instrument, circle with text (DDL)", 30: "instrument, box with text (STA)",
    31: "instrument, box with circle (ZLC)", 32: "instrument, box with text (LG-10)",
}
KIND = {k: v.split(",")[0].replace("check valve", "valve").replace("three-way valve", "valve")
        for k, v in NAMES.items()}
KIND = {k: {"inline device": "equipment", "flange": "fitting", "marker": "marker"}.get(v, v)
        for k, v in KIND.items()}

rows = [json.loads(l) for l in (DATA / "metadata.jsonl").open()]
rows = [r for r in rows if (DATA / r["file_name"]).exists()]
by = defaultdict(list)
for r in rows:
    for b, l in zip(r["symbols"]["bbox"], r["symbols"]["labels"]):
        by[l].append((r["file_name"], tuple(b)))

random.seed(1)
imgs = {r["file_name"]: Image.open(DATA / r["file_name"]).convert("RGB") for r in rows}
refs, ref_keys = {}, set()
for l in sorted(by):
    fn, b = random.choice(by[l])
    refs[l] = crop_symbol(imgs[fn], b, size=160)
    ref_keys.add((fn, b))
sheet = legend.build_sheet(refs)
sheet.save(DATA.parent.parent / ".cache" / "out" / "legend_sheet.png")
prompt = legend.make_prompt(NAMES)

random.seed(0)
jobs = []
for l in sorted(by):
    pool = [x for x in by[l] if x not in ref_keys]
    for fn, b in random.sample(pool, per_class):
        jobs.append((l, fn, b))
crops = [crop_symbol(imgs[fn], b) for _, fn, b in jobs]
valid = set(NAMES)
with ThreadPoolExecutor(workers) as ex:
    res = list(ex.map(lambda c: legend.classify(sheet, prompt, c, valid), crops))

n = len(res)
exact = sum(r["match"] == l for (l, _, _), r in zip(jobs, res))
kind_ok = sum(KIND.get(r["match"]) == KIND[l] for (l, _, _), r in zip(jobs, res))
print(f"{n} crops, parsed {sum(r['ok'] for r in res)}; tokens in/out avg "
      f"{sum(r['in_tokens'] for r in res) / n:.0f}/{sum(r['out_tokens'] for r in res) / n:.0f}")
print(f"exact legend-id accuracy: {exact}/{n} = {exact / n:.0%}")
print(f"kind accuracy (valve/instrument/equipment/...): {kind_ok}/{n} = {kind_ok / n:.0%}")
valve = [(l, r) for (l, _, _), r in zip(jobs, res) if KIND[l] == "valve"]
print("valve classes: exact %d/%d, kind %d/%d" % (
    sum(r["match"] == l for l, r in valve), len(valve), sum(KIND.get(r["match"]) == "valve" for _, r in valve), len(valve)))
conf = Counter((l, r["match"]) for (l, _, _), r in zip(jobs, res) if r["match"] != l)
print("top confusions (true -> predicted):", conf.most_common(12))
per = defaultdict(list)
for (l, _, _), r in zip(jobs, res):
    per[l].append(r["match"] == l)
print("per-class correct:", {l: f"{sum(v)}/{len(v)}" for l, v in sorted(per.items())})
inst = [(l, r["tag"]) for (l, _, _), r in zip(jobs, res) if l >= 26]
print("tags:", inst)
print("ledger total $%.4f" % ledger.total_usd())
