"""Tests for lazarus/dashed.py: dashed signal-line check on synthetic masks."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lazarus.dashed import check_dashed  # noqa: E402

DASH = 8
GAP = 6


def _h_dashes(ink, y, x0, count, dash=DASH, gap=GAP, thick=3):
    h, w = ink.shape
    for i in range(count):
        x = x0 + i * (dash + gap)
        ink[max(0, y - thick // 2):min(h, y + thick // 2 + 1), x:x + dash] = True


def _v_dashes(ink, x, y0, count, dash=DASH, gap=GAP, thick=3):
    h, w = ink.shape
    for i in range(count):
        y = y0 + i * (dash + gap)
        ink[y:y + dash, max(0, x - thick // 2):min(w, x + thick // 2 + 1)] = True


def test_horizontal_dashed_ok_exact_count():
    # n_dashes equals what was drawn (no end effects: leading/trailing gaps).
    ink = np.zeros((200, 400), bool)
    box_a = (20, 90, 60, 110)
    box_b = (340, 90, 380, 110)
    _h_dashes(ink, 100, 70, 19)
    link = check_dashed(ink, box_a, box_b)
    assert link.ok, link.reason
    assert link.n_dashes == 19
    assert link.reason == "ok"
    assert 0.25 <= link.coverage <= 0.85
    # Roughly straight: a 2-point horizontal run at y=100, box edge to box edge.
    assert len(link.path) == 2
    assert link.path[0] == (60, 100) and link.path[-1] == (340, 100)


def test_vertical_dashed_ok():
    ink = np.zeros((400, 200), bool)
    box_a = (90, 20, 110, 60)
    box_b = (90, 340, 110, 380)
    _v_dashes(ink, 100, 70, 19)
    link = check_dashed(ink, box_a, box_b)
    assert link.ok, link.reason
    assert link.n_dashes == 19
    assert len(link.path) == 2
    assert link.path[0] == (100, 60) and link.path[-1] == (100, 340)


def test_l_shaped_dashed_ok_corner_near_bend():
    ink = np.zeros((300, 400), bool)
    box_a = (20, 20, 60, 60)
    box_b = (340, 240, 380, 280)
    bend = (360, 40)
    # Horizontal leg y=40 from x=70 (21 dashes ending at 358), vertical leg
    # x=360 from y=41 (14 dashes ending at 231); corner gap is 3 px, valid.
    _h_dashes(ink, 40, 70, 21)
    _v_dashes(ink, 360, 41, 14)
    link = check_dashed(ink, box_a, box_b)
    assert link.ok, link.reason
    assert len(link.path) == 3
    mids = link.path[1:-1]
    assert min(math.hypot(x - bend[0], y - bend[1]) for x, y in mids) < 5
    assert link.path[0] == (60, 40) and link.path[-1] == (360, 240)


def test_solid_line_rejected_as_solid():
    ink = np.zeros((200, 400), bool)
    box_a = (20, 90, 60, 110)
    box_b = (340, 90, 380, 110)
    ink[99:102, 60:340] = True
    link = check_dashed(ink, box_a, box_b)
    assert not link.ok
    assert link.reason == "solid"


def test_no_ink_rejected():
    ink = np.zeros((200, 400), bool)
    box_a = (20, 90, 60, 110)
    box_b = (340, 90, 380, 110)
    link = check_dashed(ink, box_a, box_b)
    assert not link.ok
    assert link.n_dashes == 0
    assert link.score == 0


def test_dashed_through_other_box_rejected():
    ink = np.zeros((200, 400), bool)
    box_a = (20, 90, 60, 110)
    box_b = (340, 90, 380, 110)
    other = (190, 90, 230, 110)
    _h_dashes(ink, 100, 70, 19)
    link = check_dashed(ink, box_a, box_b, [other])
    assert not link.ok
    assert link.reason == "path blocked"


def test_stray_dots_rejected():
    ink = np.zeros((200, 400), bool)
    box_a = (20, 90, 60, 110)
    box_b = (340, 90, 380, 110)
    ink[99:102, 150:153] = True
    ink[99:102, 200:203] = True
    link = check_dashed(ink, box_a, box_b)
    assert not link.ok
    assert link.n_dashes < 3


def test_irregular_blobs_rejected():
    rng = np.random.default_rng(0)
    # Bimodal lengths guarantee CV > 0.6 for any draw in these ranges.
    dash_lens = np.concatenate([rng.integers(4, 8, size=2), rng.integers(50, 60, size=3)])
    gap_lens = np.concatenate([rng.integers(2, 6, size=2), rng.integers(40, 60, size=2)])
    rng.shuffle(dash_lens)
    rng.shuffle(gap_lens)
    assert float(dash_lens.std() / dash_lens.mean()) > 0.6
    assert float(gap_lens.std() / gap_lens.mean()) > 0.6
    ink = np.zeros((200, 600), bool)
    box_a = (20, 90, 60, 110)
    box_b = (540, 90, 580, 110)
    x = 70
    for i, dl in enumerate(dash_lens):
        ink[99:102, x:x + int(dl)] = True
        x += int(dl)
        if i < len(gap_lens):
            x += int(gap_lens[i])
    link = check_dashed(ink, box_a, box_b)
    assert not link.ok
    assert "irregular" in link.reason


def test_overlap_and_blank_masks_no_exception():
    ink = np.zeros((200, 400), bool)
    # Overlapping boxes.
    link = check_dashed(ink, (20, 90, 60, 110), (50, 90, 90, 110))
    assert not link.ok
    assert "overlap" in link.reason
    # Touching boxes share an edge pixel row.
    link = check_dashed(ink, (20, 90, 60, 110), (60, 90, 100, 110))
    assert not link.ok
    # All-False and all-True masks with separated boxes.
    box_a = (20, 90, 60, 110)
    box_b = (340, 90, 380, 110)
    assert not check_dashed(np.zeros((200, 400), bool), box_a, box_b).ok
    assert not check_dashed(np.ones((200, 400), bool), box_a, box_b).ok


def test_short_path_and_clipped_boxes_no_exception():
    ink = np.zeros((200, 400), bool)
    # 5 px gap -> outside signal shorter than 10 px.
    link = check_dashed(ink, (20, 90, 60, 110), (65, 90, 105, 110))
    assert not link.ok
    # Boxes partly outside the image are clipped, not an error.
    link = check_dashed(ink, (-10, 90, 60, 110), (340, 90, 410, 110))
    assert not link.ok


def test_z_shaped_dashed_ok():
    ink = np.zeros((300, 400), bool)
    box_a = (20, 20, 60, 60)
    box_b = (340, 240, 380, 280)
    # H-V-H at xm=200 (50% of x-gap 60..340).
    _h_dashes(ink, 40, 62, 10)
    _v_dashes(ink, 200, 48, 15)
    _h_dashes(ink, 260, 206, 9)
    link = check_dashed(ink, box_a, box_b)
    assert link.ok, link.reason
    assert link.reason == "ok"
    assert len(link.path) == 4
    assert link.path[0] == (60, 40) and link.path[-1] == (340, 260)
    assert link.path[1][0] == 200 and link.path[2][0] == 200
    assert link.n_dashes == 34


def test_vertical_offset_third_ok():
    ink = np.zeros((400, 200), bool)
    box_a = (50, 20, 150, 60)
    box_b = (50, 340, 150, 380)
    # x-overlap 50..150 (width 101, middle 100, off 33): draw at 133.
    _v_dashes(ink, 133, 70, 19)
    link = check_dashed(ink, box_a, box_b)
    assert link.ok, link.reason
    assert len(link.path) == 2
    assert link.path[0] == (133, 60) and link.path[-1] == (133, 340)
    assert link.n_dashes == 19


def test_z_through_other_box_rejected():
    ink = np.zeros((300, 400), bool)
    box_a = (20, 20, 60, 60)
    box_b = (340, 240, 380, 280)
    _h_dashes(ink, 40, 62, 10)
    _v_dashes(ink, 200, 48, 15)
    _h_dashes(ink, 260, 206, 9)
    other = (190, 100, 210, 120)
    link = check_dashed(ink, box_a, box_b, [other])
    assert not link.ok


def test_solid_z_rejected_as_solid():
    ink = np.zeros((300, 400), bool)
    box_a = (20, 20, 60, 60)
    box_b = (340, 240, 380, 280)
    ink[39:42, 60:201] = True
    ink[40:261, 199:202] = True
    ink[259:262, 200:341] = True
    # Block the competing L and side-Z paths so the solid Z is the best.
    others = [(290, 30, 310, 50), (30, 140, 50, 160),
              (120, 140, 140, 160), (260, 140, 280, 160)]
    link = check_dashed(ink, box_a, box_b, others)
    assert not link.ok
    assert link.reason == "solid"


def test_no_ink_new_candidates_zero():
    ink = np.zeros((300, 400), bool)
    box_a = (20, 20, 60, 60)
    box_b = (340, 240, 380, 280)
    link = check_dashed(ink, box_a, box_b)
    assert not link.ok
    assert link.score == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
