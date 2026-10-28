"""Tests for lazarus/codes.py: legend vocabulary (bases, single-letter modifiers)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lazarus.codes import Codes  # noqa: E402


def make_codes():
    entries = [
        {"item": "TI", "meaning": "TEMPERATURE MEASURING INDICATE"},
        {"item": "PIC", "meaning": "PRESSURE INDICATE CONTROL"},
        {"item": "TRC", "meaning": "TEMPERATURE RECORD CONTROL"},
        {"item": "B", "meaning": "MAIN CONTROL BOARD"},
        {"item": "M", "meaning": "MIDDLE"},
    ]
    return Codes.from_entries(entries)


def test_from_entries_reads_bases_and_mods():
    codes = make_codes()
    assert codes.bases == {"TI", "PIC", "TRC"}
    assert set(codes.mods) == {"B", "M"}
    assert codes.mods["B"] == "MAIN CONTROL BOARD"
    assert codes.mods["M"] == "MIDDLE"


def test_valid_plain_base_with_spaces_and_digits():
    codes = make_codes()
    assert codes.valid("TI 12")
    assert codes.valid("PIC 101")
    assert codes.valid("PIC-101")
    assert codes.valid("TRC 5")


def test_valid_with_prefix_and_suffix_modifiers():
    codes = make_codes()
    assert codes.valid("BTIM 255")
    assert codes.valid("BTI 10")
    assert codes.valid("TIM 7")


def test_valid_rejects_non_codes():
    codes = make_codes()
    assert not codes.valid("E 5")
    assert not codes.valid("305-1-0")
    assert not codes.valid("")


def test_letters_strips_spaces_digits_punctuation():
    assert Codes.letters("BTIM 255") == "BTIM"
    assert Codes.letters("TI 12") == "TI"
    assert Codes.letters("PIC-101") == "PIC"
    assert Codes.letters("E 5") == "E"
    assert Codes.letters("305-1-0") == ""
    assert Codes.letters("") == ""


def test_usable_needs_three_bases():
    two = Codes.from_entries([
        {"item": "TI", "meaning": "TEMPERATURE MEASURING INDICATE"},
        {"item": "PIC", "meaning": "PRESSURE INDICATE CONTROL"},
    ])
    assert not two.usable()
    assert make_codes().usable()


def test_context_mentions_base_codes():
    ctx = make_codes().context()
    assert "TI" in ctx
    assert "PIC" in ctx
    assert "TRC" in ctx
