"""Tests for the Processes panel's ``gui_hint`` display.

Created by Claude Opus 5 (Anthropic), 2026-08-17.

The wording of a hint is a matter of judgement and is not tested. Two mechanical
things are:

1. Every class the panel offers has a hint at all — in particular
   ``StaticGeneration``, which the runner wires no ramp for (A50/A51): it appears
   in the selector like any other process, so it needs one to send the user
   elsewhere.
2. The hint survives the trip to the browser. Hints are plain text written
   alongside the physics, not markup, and the panel interpolates them into a
   ``<div>`` — so ``'<name>/generation'`` is parsed as a tag and disappears
   unless it is escaped (A53).

Needs ``panel`` but not FEniCS; see ``README.md`` for why these live outside
``simudo/``.
"""

from __future__ import annotations

import os
from html.parser import HTMLParser

import pytest

# These tests exercise the GUI, which is a separate concern from the solver:
# they need panel (the ``gui`` extra), which the Docker image deliberately
# does not carry, and they need ``simudo`` itself importable because the GUI
# modules import each other by package path.  Skip cleanly when either is
# absent rather than erroring, so `pytest` over the whole tree stays green in
# a solver-only environment.
pytest.importorskip(
    "panel",
    reason="GUI tests need the gui extra: pip install 'simudo[gui]'")
pytest.importorskip(
    "simudo.gui.model",
    reason="GUI tests need simudo importable (run from the repo root, "
           "or pip install simudo)")

_HERE   = os.path.dirname(os.path.abspath(__file__))
GUI_DIR = os.path.normpath(os.path.join(_HERE, "..", "simudo", "gui"))


def _visible_text(markup: str) -> str:
    """The text a browser would actually show for ``markup``."""
    class _P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.chunks = []

        def handle_data(self, data):
            self.chunks.append(data)

    p = _P()
    p.feed(markup)
    return "".join(p.chunks)


def test_every_offered_class_has_a_hint():
    """A class in the selector with no hint shows an empty pane and says nothing."""
    from simudo.gui.panels.processes import _PROCESS_CLASSES, _PROCESS_HINTS

    missing = [c for c in _PROCESS_CLASSES if not _PROCESS_HINTS.get(c)]
    assert not missing, f"EOP classes offered without a gui_hint: {missing}"
    print("[ok] every_offered_class_has_a_hint")


def test_hint_markup_is_escaped():
    """Angle-bracketed placeholders must reach the pane, not be eaten as tags."""
    from simudo.gui.panels.processes import _PROCESS_HINTS, _hint_html

    shown = _visible_text(_hint_html("key '<name>/generation' & <b>bold</b>"))
    assert "<name>/generation" in shown, shown
    assert "<b>bold</b>" in shown, "markup in a hint must display literally"

    # Every real hint must survive too — several use <name> placeholders.
    for cls, hint in _PROCESS_HINTS.items():
        assert _visible_text(_hint_html(hint)) == hint, \
            f"{cls}'s hint is altered on the way to the browser"

    # An empty hint yields an empty pane, which is what `visible` keys off.
    assert _hint_html("") == ""
    print("[ok] hint_markup_is_escaped")


def main():
    test_every_offered_class_has_a_hint()
    test_hint_markup_is_escaped()
    print("\nAll process-hint tests passed.")


if __name__ == "__main__":
    main()
