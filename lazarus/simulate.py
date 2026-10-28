"""Connectivity / isolation analysis on a verified P&ID graph.

This is NOT hydraulics: flow direction, pressure and flow rates are unknown in this dataset. The
graph is treated as an undirected network and the only question answered is "if these valves are
closed, which symbols can still be reached from this source, and which lose their connection?"

Model
- Every node passes connections through, except closed valves.
- A closed valve is still reachable (it sits on the wet side that touches it) but nothing is
  reachable *through* it. If the source itself is a closed valve it still expands (it is the origin).
- "junction" nodes (kind == "junction") are bare pipe joints: they pass connections through and are
  never reported as instruments/equipment. Graphs without junction nodes work unchanged.

The JS port in demo/app.js must stay consistent with this module.
"""
import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

GRAPHS = Path(__file__).resolve().parents[1] / ".cache" / "out" / "graphs"


def _graph(d: dict) -> dict:
    return d["graph"] if "graph" in d else d


def _nodes(d: dict) -> Dict[int, dict]:
    return {n["id"]: n for n in _graph(d)["nodes"]}


def _adjacency(d: dict) -> Dict[int, Set[int]]:
    g = _graph(d)
    adj = {n["id"]: set() for n in g["nodes"]}
    for a, b in g["edges"]:
        if a in adj and b in adj and a != b:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def _reach(adj, source, blocked: Set[int]) -> Set[int]:
    """BFS from source; blocked nodes are reached but not expanded (the source always expands)."""
    seen, queue = {source}, deque([source])
    while queue:
        u = queue.popleft()
        if u in blocked and u != source:
            continue
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return seen


def connected_components(graph_dict: dict, closed_valves: Iterable[int] = ()) -> List[List[int]]:
    """Components of the network with `closed_valves` cutting passage, largest first.

    A closed valve belongs to every component that touches it (as a dead end) but never bridges two
    of them. Node ids include junctions.
    """
    adj, nodes = _adjacency(graph_dict), _nodes(graph_dict)
    closed = {c for c in closed_valves if c in nodes}
    comps, seen = [], set()
    for start in adj:
        if start in closed or start in seen:
            continue
        comp, queue = {start}, deque([start])
        while queue:
            for v in adj[queue.popleft()]:
                if v in comp:
                    continue
                comp.add(v)
                if v not in closed:
                    queue.append(v)
        seen |= comp - closed
        comps.append(sorted(comp))
    in_any = {i for c in comps for i in c}
    comps += [[c] for c in sorted(closed) if c not in in_any]
    return sorted(comps, key=lambda c: (-len(c), c[0]))


def effective_degree(graph_dict: dict, node: int) -> int:
    """Number of distinct non-junction nodes adjacent to `node`, seeing through junction chains."""
    adj, nodes = _adjacency(graph_dict), _nodes(graph_dict)
    seen, out, stack = {node}, set(), [node]
    while stack:
        for v in adj[stack.pop()]:
            if v in seen:
                continue
            seen.add(v)
            if nodes[v]["kind"] == "junction":
                stack.append(v)
            else:
                out.add(v)
    return len(out)


_MAJOR = re.compile(r"vessel|tank|pump|exchanger|compressor|column|reactor|drum", re.I)


def suggest_sources(graph_dict: dict, limit: int = 3) -> List[int]:
    """Defensible default sources, best first.

    A process source is usually a big piece of equipment (tank, vessel, pump) feeding the system.
    Rank equipment by (size of its connected component, major process equipment by name, effective degree, lowest id): the hub of
    the biggest network gives the richest isolation story. With no equipment, the same ranking is
    applied to every non-junction node. Junctions are never suggested.
    """
    nodes = _nodes(graph_dict)
    comp_size = {i: len(c) for c in connected_components(graph_dict) for i in c}
    real = [i for i, n in nodes.items() if n["kind"] != "junction"]
    pool = [i for i in real if nodes[i]["kind"] == "equipment"] or real
    ranked = sorted(pool, key=lambda i: (-comp_size[i], not _MAJOR.search(nodes[i].get("name", "")),
                                         -effective_degree(graph_dict, i), i))
    return ranked[:limit]


def _group(ids: Iterable[int], nodes: Dict[int, dict]) -> Dict[str, List[dict]]:
    out = defaultdict(list)
    for i in sorted(ids):
        n = nodes[i]
        if n["kind"] != "junction":
            out[n["kind"]].append({"id": i, "tag": n.get("tag", ""), "name": n.get("name", "")})
    return dict(out)


def isolate(graph_dict: dict, closed_valves: Iterable[int], source: int) -> dict:
    """Which nodes stay connected to `source` with `closed_valves` shut.

    Returns node-id lists (junctions included) and per-kind groupings (junctions excluded):
      reachable / isolated   now (isolated = everything not reachable from source)
      lost                   reachable with all valves open but not now (the change vs all-open)
      unreachable_baseline   not reachable from source even with every valve open
      reachable_by_kind, isolated_by_kind, lost_by_kind   {kind: [{id, tag, name}]}
    """
    adj, nodes = _adjacency(graph_dict), _nodes(graph_dict)
    if source not in nodes:
        raise KeyError(f"unknown source node {source}")
    closed = {c for c in closed_valves if c in nodes and nodes[c]["kind"] == "valve"}
    base, now = _reach(adj, source, set()), _reach(adj, source, closed)
    lost, isolated = base - now, set(nodes) - now
    return {
        "source": source, "closed": sorted(closed),
        "reachable": sorted(now), "isolated": sorted(isolated), "lost": sorted(lost),
        "unreachable_baseline": sorted(set(nodes) - base),
        "reachable_by_kind": _group(now, nodes), "isolated_by_kind": _group(isolated, nodes),
        "lost_by_kind": _group(lost, nodes),
    }


def rank_valves(graph_dict: dict, source: int) -> List[dict]:
    """Close each valve alone; rank by how many symbols (junctions excluded) it cuts off."""
    rows = []
    for i, n in _nodes(graph_dict).items():
        if n["kind"] == "valve":
            rep = isolate(graph_dict, [i], source)
            rows.append({"valve": i, "tag": n.get("tag", ""), "report": rep,
                         "lost": sum(len(v) for v in rep["lost_by_kind"].values())})
    return sorted(rows, key=lambda r: (-r["lost"], r["valve"]))


def _fmt(items: List[dict]) -> str:
    return ", ".join(f"#{x['id']}" + (f" {x['tag']}" if x["tag"] else "") for x in items)


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description="Connectivity/isolation report (not hydraulics).")
    ap.add_argument("drawing", help="drawing name (112 or 112.jpg) or a path to a graph json")
    ap.add_argument("--source", type=int, help="source node id (default: suggest_sources)")
    ap.add_argument("--top", type=int, default=10, help="valves to list")
    a = ap.parse_args(argv)
    p = Path(a.drawing)
    if not p.exists():
        p = GRAPHS / f"{a.drawing if a.drawing.endswith('.jpg') else a.drawing + '.jpg'}.json"
    data = json.loads(p.read_text())
    nodes = _nodes(data)
    src = a.source if a.source is not None else suggest_sources(data)[0]
    s = nodes[src]
    reach = sum(len(v) for v in isolate(data, [], src)["reachable_by_kind"].values())
    print(f"{p.name}: {len(nodes)} nodes, {len(_graph(data)['edges'])} edges, "
          f"{len(connected_components(data))} components")
    print(f"source #{src} ({s['name']}{', ' + s['tag'] if s.get('tag') else ''}): "
          f"{reach} symbols reachable with all valves open (connectivity only, not flow)")
    print("\nclosing each valve alone, ranked by symbols cut off from the source:")
    for r in rank_valves(data, src)[:a.top]:
        detail = "; ".join(f"{k}: {_fmt(v)}" for k, v in r["report"]["lost_by_kind"].items())
        print(f"  valve #{r['valve']:<3}{r['tag']:<10} cuts off {r['lost']:>2}  {detail}")


if __name__ == "__main__":
    main()
