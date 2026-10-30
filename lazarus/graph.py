"""The structure Lazarus produces: typed nodes joined by edges."""
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Node:
    id: int
    box: Tuple[int, int, int, int]
    legend_id: int
    name: str
    kind: str
    tag: str = ""
    confidence: float = 0.0
    source: str = "sam"           # "sam", or "reinspect" when added by the verifier loop

    @property
    def center(self) -> Tuple[int, int]:
        return (self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2


@dataclass
class Graph:
    nodes: List[Node] = field(default_factory=list)
    edges: List[Tuple[int, int]] = field(default_factory=list)
    # Dashed signal links (lazarus/signals.py). Kept apart from `edges` on purpose: they carry signals,
    # not process fluid, so isolation analysis must not traverse them. `signal_verified` is the subset
    # whose pixels a deterministic check confirmed; the rest are inferred from the gap-closed trace.
    signal_edges: List[Tuple[int, int]] = field(default_factory=list)
    signal_verified: List[Tuple[int, int]] = field(default_factory=list)
    signal_paths: List[List[Tuple[int, int]]] = field(default_factory=list)      # one route per verified link

    def degree(self) -> Dict[int, int]:
        c: Counter = Counter()
        for a, b in self.edges:
            c[a] += 1
            c[b] += 1
        return {n.id: c.get(n.id, 0) for n in self.nodes}

    def to_dict(self) -> Dict:
        return {"nodes": [asdict(n) for n in self.nodes], "edges": [list(e) for e in self.edges],
                "signal_edges": [list(e) for e in self.signal_edges],
                "signal_verified": [list(e) for e in self.signal_verified],
                "signal_paths": [[list(p) for p in path] for path in self.signal_paths]}
