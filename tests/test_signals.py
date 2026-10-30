import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lazarus import pipes, signals  # noqa: E402

BOXES = [(10, 80, 50, 120), (350, 80, 390, 120)]


def sheet(dashed: bool) -> np.ndarray:
    """White page, two symbol boxes, and a 5 px line between them: dashes of 30 px with 20 px gaps, or solid."""
    img = np.full((200, 400, 3), 255, np.uint8)
    for x1, y1, x2, y2 in BOXES:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 3)
    if dashed:
        for x in range(50, 350, 50):
            cv2.line(img, (x, 100), (min(x + 30, 350), 100), (0, 0, 0), 5, cv2.LINE_8)
    else:
        cv2.line(img, (50, 100), (350, 100), (0, 0, 0), 5, cv2.LINE_8)
    return img


def test_plain_trace_misses_a_dashed_line_but_the_gap_closed_trace_finds_it():
    img = sheet(dashed=True)
    assert pipes.trace(img, BOXES, junctions=True)["edges"] == []
    links, _ = signals.find(img, BOXES, [])
    assert links == [(0, 1)]


def test_a_solid_pipe_is_not_a_signal_link():
    img = sheet(dashed=False)
    solid = pipes.trace(img, BOXES, junctions=True)["edges"]
    assert solid == [(0, 1)]
    assert signals.find(img, BOXES, solid)[0] == []


def test_clean_dashed_line_is_pixel_verified():
    links, verified = signals.find(sheet(dashed=True), BOXES, [])
    assert links == [(0, 1)] and list(verified) == [(0, 1)]
    route = verified[(0, 1)]
    assert route[0][0] <= 51 and route[-1][0] >= 349          # runs from one box edge to the other


def test_pairs_through_junctions_stop_at_symbols():
    # symbols 0,1,2; junction 3 joins 0 and 1; 1 is joined to 2 directly. 0-2 is not a link: 1 is in the way.
    assert signals.pairs_through_junctions([(0, 3), (1, 3), (1, 2)], 3) == {(0, 1), (1, 2)}
