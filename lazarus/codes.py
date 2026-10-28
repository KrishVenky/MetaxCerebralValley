"""What a drawing's own legend says about its vocabulary, as data the classifier and verifier can use.

Muse Spark reads the legend into {item, meaning} entries (regions.read_legend). Here they become an
instrument-code grammar (modifier letters around a base code, e.g. B + TI + M for "BTIM") and a list
of the valve and fitting symbols the legend names.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set

_SYMBOL_WORDS = ("valve symbol",)


@dataclass
class Codes:
    bases: Set[str] = field(default_factory=set)         # TRC, TI, PIC ...
    mods: Dict[str, str] = field(default_factory=dict)   # B -> MAIN CONTROL BOARD ...
    symbols: List[str] = field(default_factory=list)     # GLOBE, GATE, STRAINER ...
    _rx: "re.Pattern" = None

    @classmethod
    def from_entries(cls, entries: List[Dict]) -> "Codes":
        c = cls()
        for e in entries:
            item = re.sub(r"[^A-Za-z. ]", "", str(e.get("item", ""))).strip()
            meaning = str(e.get("meaning", "")).strip()
            if re.fullmatch(r"[A-Z]{2,4}", item):
                c.bases.add(item)
            elif re.fullmatch(r"[A-Z]", item):
                c.mods[item] = meaning
            elif item.lower() in _SYMBOL_WORDS and meaning:
                c.symbols.append(meaning.title())
        if c.bases:
            alt = "|".join(sorted(c.bases, key=len, reverse=True))
            m = "".join(sorted(c.mods)) or "$^"
            c._rx = re.compile(rf"(?:[{m}])*(?:{alt})(?:[{m}])*" if c.mods else rf"(?:{alt})")
        return c

    def usable(self) -> bool:
        return len(self.bases) >= 3

    @staticmethod
    def letters(tag: str) -> str:
        """The letter part of a tag: everything before the first digit, spaces and punctuation dropped."""
        return re.sub(r"[^A-Za-z]", "", re.split(r"\d", tag or "", maxsplit=1)[0]).upper()

    def valid(self, tag: str) -> bool:
        return bool(self._rx and self._rx.fullmatch(self.letters(tag)))

    def context(self) -> str:
        """Text for the classifier prompt."""
        lines = ["This drawing's own legend defines its vocabulary:",
                 "- Instrument tags are a base code from: " + ", ".join(sorted(self.bases)) + "."]
        if self.mods:
            lines.append("- Single letters may be added before or after a base code: "
                         + "; ".join(f"{k} = {v.lower()}" for k, v in sorted(self.mods.items())) + ".")
        if self.symbols:
            lines.append("- Symbols the legend names: " + ", ".join(self.symbols) + ".")
        lines.append("- A circle holding only a letter and a number that is not such a code (for example "
                     "'E 5') is a reference marker, not an instrument: answer reference_marker.")
        return "\n".join(lines)
