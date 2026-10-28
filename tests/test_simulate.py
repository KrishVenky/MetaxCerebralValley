"""Toy graph:  T-1 (equip 0) - v1 (valve 1) - J (junction 2) - PI-1 (instr 3)
                                                 |
                                           v4 (valve 4) - P-2 (equip 5) - TI-2 (instr 6)
plus a disconnected pair: equip 7 - instr 8. The no-junction variant wires 1 directly to 3 and 4.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lazarus import simulate as sim  # noqa: E402


def node(i, kind, tag=""):
    return {"id": i, "kind": kind, "tag": tag, "name": kind, "box": [0, 0, 1, 1]}


def toy(junction=True):
    nodes = [node(0, "equipment", "T-1"), node(1, "valve"), node(3, "instrument", "PI-1"), node(4, "valve"),
             node(5, "equipment", "P-2"), node(6, "instrument", "TI-2"), node(7, "equipment"), node(8, "instrument")]
    if junction:
        nodes.append(node(2, "junction"))
        edges = [[0, 1], [1, 2], [2, 3], [2, 4], [4, 5], [5, 6], [7, 8]]
    else:
        edges = [[0, 1], [1, 3], [1, 4], [4, 5], [5, 6], [7, 8]]
    return {"graph": {"nodes": nodes, "edges": edges}}


def test_all_open_reaches_component():
    r = sim.isolate(toy(), [], 0)
    assert set(r["reachable"]) == {0, 1, 2, 3, 4, 5, 6}
    assert r["lost"] == [] and set(r["unreachable_baseline"]) == {7, 8}


def test_closed_valve_splits_line():
    r = sim.isolate(toy(), [4], 0)
    assert set(r["lost"]) == {5, 6}
    assert 4 in r["reachable"]  # the closed valve itself stays on the wet side
    assert [x["tag"] for x in r["lost_by_kind"]["instrument"]] == ["TI-2"]
    assert [x["tag"] for x in r["lost_by_kind"]["equipment"]] == ["P-2"]


def test_junction_passthrough_and_hidden_from_reports():
    r = sim.isolate(toy(), [1], 0)
    assert set(r["lost"]) == {2, 3, 4, 5, 6}
    assert all(x["id"] != 2 for g in r["lost_by_kind"].values() for x in g)  # junction never reported
    assert 3 in sim.isolate(toy(), [4], 0)["reachable"]  # junction passes 1 -> 3 while 4 is shut


def test_works_without_junctions():
    r = sim.isolate(toy(False), [4], 0)
    assert set(r["lost"]) == {5, 6} and 3 in r["reachable"]


def test_closed_source_valve_still_expands():
    assert set(sim.isolate(toy(), [1], 1)["reachable"]) == {0, 1, 2, 3, 4, 5, 6}


def test_non_valve_close_is_ignored():
    assert sim.isolate(toy(), [5], 0)["lost"] == []


def test_components():
    assert [len(c) for c in sim.connected_components(toy())] == [7, 2]
    comps = sim.connected_components(toy(), [4])
    assert sorted(map(tuple, comps)) == [(0, 1, 2, 3, 4), (4, 5, 6), (7, 8)]  # valve 4 touches both sides


def test_suggest_sources():
    assert sim.suggest_sources(toy())[0] == 5  # P-2 (degree 2) beats T-1 (degree 1) in the big component
    assert sim.suggest_sources(toy(False))[0] == 5
    assert 2 not in sim.suggest_sources(toy(), limit=99)  # junctions are never suggested


def test_rank_valves():
    rows = sim.rank_valves(toy(), 0)
    assert [r["valve"] for r in rows] == [1, 4] and [r["lost"] for r in rows] == [4, 2]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
