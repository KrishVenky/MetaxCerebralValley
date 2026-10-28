"""Tests for lazarus/routes.py: pipe routes between two nodes on synthetic images."""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lazarus.graph import Graph, Node  # noqa: E402
from lazarus.routes import route_edges  # noqa: E402


def make_graph(box_a, box_b):
    nodes = [
        Node(0, box_a, 0, "A", "equipment"),
        Node(1, box_b, 0, "B", "equipment"),
    ]
    return Graph(nodes=nodes, edges=[(0, 1)])


def test_route_straight_starts_and_ends_at_centres():
    img = np.full((100, 200, 3), 255, np.uint8)
    cv2.line(img, (20, 50), (180, 50), (0, 0, 0), thickness=5, lineType=cv2.LINE_8)
    box_a = (10, 40, 30, 60)
    box_b = (170, 40, 190, 60)
    graph = make_graph(box_a, box_b)

    routes = route_edges(img, graph)

    assert len(routes) == 1
    route = routes[0]
    assert route is not None
    assert route[0] == (20, 50)
    assert route[-1] == (180, 50)


def test_route_l_has_corner_near_bend():
    img = np.full((200, 200, 3), 255, np.uint8)
    cv2.line(img, (30, 30), (30, 150), (0, 0, 0), thickness=5, lineType=cv2.LINE_8)
    cv2.line(img, (30, 150), (150, 150), (0, 0, 0), thickness=5, lineType=cv2.LINE_8)
    box_a = (20, 20, 40, 40)
    box_b = (140, 140, 160, 160)
    bend = (30, 150)
    graph = make_graph(box_a, box_b)

    routes = route_edges(img, graph)

    assert len(routes) == 1
    route = routes[0]
    assert route is not None
    assert route[0] == (30, 30)
    assert route[-1] == (150, 150)
    mids = route[1:-1]
    assert len(mids) >= 1
    assert min(math.hypot(x - bend[0], y - bend[1]) for x, y in mids) < 15
    # A straight diagonal from (30, 30) to (150, 150) has x == y; the L stays far from it.
    assert any(abs(x - y) > 30 for x, y in mids)


def test_route_no_pipe_returns_none():
    img = np.full((100, 200, 3), 255, np.uint8)
    box_a = (10, 40, 30, 60)
    box_b = (170, 40, 190, 60)
    graph = make_graph(box_a, box_b)

    routes = route_edges(img, graph)

    assert len(routes) == 1
    assert routes[0] is None


def test_route_none_when_pipe_does_not_join_the_nodes():
    # Added by hand (not Muse Code). A pipe exists in the image, but it runs elsewhere: neither node touches it.
    img = np.full((200, 200, 3), 255, np.uint8)
    cv2.line(img, (20, 170), (180, 170), (0, 0, 0), thickness=5, lineType=cv2.LINE_8)
    graph = make_graph((10, 20, 30, 40), (170, 20, 190, 40))

    assert route_edges(img, graph) == [None]
