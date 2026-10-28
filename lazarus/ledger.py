"""Append-only spend ledger plus a guard that stops runaway API spend."""
import json
import time

from . import config

_LEDGER = config.CACHE_DIR / "ledger.jsonl"


class SpendGuardError(RuntimeError):
    pass


def record(kind: str, usd: float, **extra) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    row = {"t": time.time(), "kind": kind, "usd": round(usd, 6), **extra}
    with _LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def total_usd() -> float:
    if not _LEDGER.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in _LEDGER.open(encoding="utf-8") if line.strip())


def check() -> None:
    spent = total_usd()
    if spent >= config.SPEND_GUARD_USD:
        raise SpendGuardError(
            f"Estimated spend ${spent:.2f} reached the guard of ${config.SPEND_GUARD_USD:.2f}"
        )


def summary() -> dict:
    out: dict = {}
    if _LEDGER.exists():
        for line in _LEDGER.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            k = out.setdefault(r["kind"], {"calls": 0, "usd": 0.0, "in_tokens": 0, "out_tokens": 0})
            k["calls"] += 1
            k["usd"] += r["usd"]
            k["in_tokens"] += r.get("in_tokens", 0)
            k["out_tokens"] += r.get("out_tokens", 0)
    return out
