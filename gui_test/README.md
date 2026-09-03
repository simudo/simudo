# GUI tests — run these on the host

Tests for the GUI front end. **Deliberately outside the `simudo/` package.**

The GUI runs on your machine and submits work to a container, so the host has
`panel` but not Simudo's numerical stack. Anything pytest collects from under
`simudo/` makes it resolve `simudo` and import `simudo/__init__.py`, which
pulls in `future`, `petsc4py` and the rest — so these cannot live in there,
whichever subdirectory is used.

Run from the repo root, in the environment that has `panel`:

    python -m pytest gui_test/

The library's own tests are the other half of the split: they need FEniCS, ship
inside the wheel, and run in the container:

    python -m pytest simudo/test/
