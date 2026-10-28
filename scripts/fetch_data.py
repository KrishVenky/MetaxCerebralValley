"""Download the 12 Digitize-PID validation drawings the benchmark uses into data/pid_val/.

Dataset: hamzas/digitize-pid-symbols on Hugging Face (Paliwal et al., 2021, arXiv:2109.03794). It
declares no licence, so the images are not redistributed in this repo: run this once instead.
The first six drawings seed the legend reference set, the last six are the held-out test set.

Usage: python -m scripts.fetch_data
"""
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "hamzas/digitize-pid-symbols"
DRAWINGS = ["0", "100", "102", "106", "108", "11", "112", "114", "118", "122", "129", "14"]
OUT = Path(__file__).resolve().parent.parent / "data" / "pid_val"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for n in DRAWINGS:
        shutil.copy(hf_hub_download(REPO, f"DigitizePID_Dataset/val/{n}.jpg", repo_type="dataset"), OUT / f"{n}.jpg")
    meta = Path(hf_hub_download(REPO, "DigitizePID_Dataset/val/metadata.jsonl", repo_type="dataset"))
    keep = {f"{n}.jpg" for n in DRAWINGS}
    rows = [l for l in meta.read_text().splitlines() if json.loads(l)["file_name"] in keep]
    (OUT / "metadata.jsonl").write_text("\n".join(rows) + "\n")
    print(f"wrote {len(DRAWINGS)} drawings and {len(rows)} metadata rows to {OUT}")


if __name__ == "__main__":
    main()
