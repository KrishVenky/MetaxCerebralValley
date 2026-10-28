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

    def degree(self) -> Dict[int, int]:
        c: Counter = Counter()
        for a, b in self.edges:
            c[a] += 1
            c[b] += 1
        return {n.id: c.get(n.id, 0) for n in self.nodes}

    def to_dict(self) -> Dict:
        return {"nodes": [asdict(n) for n in self.nodes], "edges": [list(e) for e in self.edges]}
