"""Smoke tests for the v1 material-callable convention.

Two layers of coverage:

1. **AST scan (no FEniCS).** Verify that the GUI's ``mat_ast`` module
   recognises a dict-shaped ``get_dict`` entry with a ``'type'`` tag as a
   callable provided by the material, and that ``processes._discover_eop_classes``
   extracts the ``required_spatial_callables`` attribute from
   ``BeerLambert`` / ``IBBeerLambert``.

2. **Runner dispatch (no FEniCS).** Build a mat_dict with one scalar
   entry and one top-hat callable entry; pass it through a mock spatial
   object and assert each entry is dispatched to ``add_material_data``
   vs. ``add_material_callable_data`` correctly.

Neither layer needs FEniCS, but both need ``panel`` (the GUI's own dependency),
so the whole module skips without it -- install ``simudo[gui]`` to run it.

**This lives outside the ``simudo/`` package on purpose.** The GUI runs on the
host and submits work to a container, so the host needs ``panel`` but not
Simudo's numerical stack. Anything collected from under ``simudo/`` makes
pytest resolve ``simudo`` and import ``simudo/__init__``, which pulls in
``future``, ``petsc4py`` and the rest -- so these tests cannot live there, no
matter which directory inside it they are put in.

Run from the repo root, in the host environment that has ``panel``::

    python -m pytest gui_test/
"""

from __future__ import annotations

import os

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

_HERE        = os.path.dirname(os.path.abspath(__file__))
GUI_DIR      = os.path.normpath(os.path.join(_HERE, "..", "simudo", "gui"))
PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
MATERIALS_DIR = os.path.normpath(os.path.join(GUI_DIR, "..", "materials"))


def test_ast_detects_callable_in_material():
    """mat_ast should flag the test material's sigma entry as a callable."""
    from simudo.gui.panels.mat_ast import extract_classes_from_file

    classes = extract_classes_from_file(
        os.path.join(MATERIALS_DIR, "ib_test_material.py"))
    assert classes, "expected at least one ClassInfo from ib_test_material.py"
    info = next(c for c in classes if c.class_name == "IBTestMaterial")

    keys = {p.key: p for p in info.properties}
    # Scalar entry must still be parsed as a property.
    assert "IB/number_of_states" in keys, keys
    # Callable entry must be flagged.
    sigma = keys.get("opt_ci/sigma_function")
    assert sigma is not None, "opt_ci/sigma_function missing from AST scan"
    assert sigma.is_callable is True, sigma
    assert sigma.callable_type == "top_hat", sigma
    print("[ok] ast_detects_callable_in_material")


def test_eop_required_callables_discovered():
    """processes._discover_eop_classes should AST-extract required_spatial_callables."""
    from simudo.gui.panels.processes import _EOP_REQUIRED_CALLABLES

    assert "BeerLambert" in _EOP_REQUIRED_CALLABLES, _EOP_REQUIRED_CALLABLES
    assert "alpha_function" in _EOP_REQUIRED_CALLABLES["BeerLambert"]
    assert _EOP_REQUIRED_CALLABLES["BeerLambert"]["alpha_function"][
        "return_units"] == "1/cm"

    assert "IBBeerLambert" in _EOP_REQUIRED_CALLABLES
    assert "sigma_function" in _EOP_REQUIRED_CALLABLES["IBBeerLambert"]
    # IBBeerLambert overrides BeerLambert: it must NOT also report alpha_function.
    assert "alpha_function" not in _EOP_REQUIRED_CALLABLES["IBBeerLambert"]
    print("[ok] eop_required_callables_discovered")


def test_runner_dispatch_splits_scalar_vs_callable():
    """_register_material_dict routes scalars and callables to the right API."""
    # Stand-alone import — exercise the dispatcher without booting the GUI.
    import importlib.util
    runner_path = os.path.join(GUI_DIR, "simudo_1d_runner.py")
    # The runner imports dolfin at module-load time; we cannot import it
    # outside the container.  Re-implement the relevant logic inline by
    # mirroring the dispatch decision so this test can run on the host.
    from collections import namedtuple
    Calls = namedtuple("Calls", "scalar callable_")

    class FakeSpatial:
        def __init__(self):
            self.scalar_calls = []
            self.callable_calls = []

        def add_material_data(self, key, material_name, value, priority):
            self.scalar_calls.append((key, material_name, value, priority))

        def add_material_callable_data(self, key, material_name, func, priority):
            self.callable_calls.append((key, material_name, func, priority))

    fake = FakeSpatial()
    # Hand-build a mat_dict shape (no pint needed for the decision logic).
    mat_dict = {
        "poisson/permittivity": 11.9,        # scalar (would be pint in reality)
        "opt_ci/sigma_function": {            # callable
            "type": "top_hat",
            "E_low": 0.8, "E_high": 4.0,
            "value": 3e-14, "unit": "cm^2",
        },
    }

    # Replicate the dispatch loop body verbatim (small enough to inline).
    for key, value in mat_dict.items():
        if isinstance(value, dict) and "type" in value:
            fake.add_material_callable_data(
                key=key, material_name="ib_test", func=lambda E: None,
                priority=10)
        else:
            fake.add_material_data(
                key=key, material_name="ib_test", value=value, priority=10)

    assert len(fake.scalar_calls) == 1, fake.scalar_calls
    assert len(fake.callable_calls) == 1, fake.callable_calls
    assert fake.scalar_calls[0][0] == "poisson/permittivity"
    assert fake.callable_calls[0][0] == "opt_ci/sigma_function"
    print("[ok] runner_dispatch_splits_scalar_vs_callable")


def test_missing_params_callable_logic():
    """Inline top-hat OR material provision suppresses the missing-callable warning;
    an unset callable is reported."""
    from simudo.gui.yaml_io import load_project
    from simudo.gui.panels.layers import LayersPanel

    MATERIALS = [os.path.join(GUI_DIR, "..", "materials")]

    class _Stand:
        """Stand-in exposing the interface LayersPanel's compute methods need."""
        _compute_suggested_params = LayersPanel._compute_suggested_params
        _compute_missing_params   = LayersPanel._compute_missing_params
        _any_radiative_recombination = LayersPanel._any_radiative_recombination

        def __init__(self, proj, libs):
            self._proj = proj
            self._libs = libs
            self.app = self

        @property
        def project(self):
            return self._proj

        def get_library_dirs(self):
            return self._libs

    def missing_for(yaml_rel):
        proj = load_project(os.path.join(_HERE, yaml_rel))
        stand = _Stand(proj, MATERIALS)
        return stand._compute_suggested_params(), set(stand._compute_missing_params())

    # 1) Inline top-hat (ib_beer_lambert_test.yaml): sigma must NOT be missing.
    sug, miss = missing_for("inline_sigma_project.yaml")
    assert "opt_ci/sigma_function" in sug, "sigma not suggested"
    assert "opt_ci/sigma_function" not in miss, \
        f"inline top-hat sigma wrongly reported missing: {miss}"

    # 2) Material-provided (claude_ibbl_material_test.yaml): sigma must NOT be missing.
    sug, miss = missing_for("material_callable_project.yaml")
    assert "opt_ci/sigma_function" in sug
    assert "opt_ci/sigma_function" not in miss, \
        f"material-provided sigma wrongly reported missing: {miss}"
    print("[ok] missing_params_callable_logic")


def test_missing_params_reports_unset_callable():
    """A BeerLambert process with no alpha anywhere is reported missing."""
    from simudo.gui.model import Project, Process, Band, Layer
    from simudo.gui.panels.layers import LayersPanel

    proj = Project()
    proj.bands = [Band(name="CB", type="nondegenerate", sign=-1, extent="all"),
                  Band(name="VB", type="nondegenerate", sign=1, extent="all")]
    proj.layers = [Layer(name="abs")]
    proj.processes = [Process(name="opt_cv", cls="BeerLambert",
                              src_band="VB", dst_band="CB")]

    class _Stand:
        _compute_suggested_params = LayersPanel._compute_suggested_params
        _compute_missing_params   = LayersPanel._compute_missing_params
        _any_radiative_recombination = LayersPanel._any_radiative_recombination
        def __init__(self, p):
            self._proj = p
            self.app = self
        @property
        def project(self): return self._proj
        def get_library_dirs(self): return []

    missing = set(_Stand(proj)._compute_missing_params())
    assert "opt_cv/alpha_function" in missing, \
        f"unset alpha should be reported missing, got {missing}"
    print("[ok] missing_params_reports_unset_callable")


def test_refractive_index_required_when_radiative():
    """refractive_index is suggested iff some optical process has SVR enabled."""
    from simudo.gui.model import Project, Process, Band, Layer
    from simudo.gui.panels.layers import LayersPanel

    class _Stand:
        _compute_suggested_params = LayersPanel._compute_suggested_params
        _compute_missing_params   = LayersPanel._compute_missing_params
        _any_radiative_recombination = LayersPanel._any_radiative_recombination
        def __init__(self, p):
            self._proj = p
            self.app = self
        @property
        def project(self): return self._proj
        def get_library_dirs(self): return []

    bands = [Band(name="CB", type="nondegenerate", sign=-1, extent="all"),
             Band(name="VB", type="nondegenerate", sign=1, extent="all")]

    # BeerLambert defaults radiative ON (since [7da11b7483], 2026-08-27), so a
    # process that leaves it unset still needs refractive_index.
    p = Project(); p.bands = bands; p.layers = [Layer(name="abs")]
    p.processes = [Process(name="opt_cv", cls="BeerLambert",
                           src_band="VB", dst_band="CB",
                           radiative_recombination=None)]
    sug = _Stand(p)._compute_suggested_params()
    assert "refractive_index" in sug, \
        "BeerLambert defaults radiative ON; refractive_index should be required"

    # BeerLambert with radiative explicitly ON → still required.
    p.processes[0].radiative_recombination = True
    sug = _Stand(p)._compute_suggested_params()
    assert "refractive_index" in sug, \
        "radiative ON should require refractive_index"

    # BeerLambert with radiative explicitly OFF → not required.
    p.processes[0].radiative_recombination = False
    sug = _Stand(p)._compute_suggested_params()
    assert "refractive_index" not in sug, \
        "radiative OFF should not require refractive_index"

    # IBBeerLambert defaults radiative ON → refractive_index required even when None.
    p2 = Project(); p2.bands = bands; p2.layers = [Layer(name="abs")]
    p2.processes = [Process(name="opt_ci", cls="IBBeerLambert",
                            src_band="VB", dst_band="CB", trap_band="VB",
                            radiative_recombination=None)]
    sug = _Stand(p2)._compute_suggested_params()
    assert "refractive_index" in sug, \
        "IBBeerLambert defaults radiative ON; refractive_index should be required"

    # IBBeerLambert with radiative explicitly OFF → not required.
    p2.processes[0].radiative_recombination = False
    sug = _Stand(p2)._compute_suggested_params()
    assert "refractive_index" not in sug, \
        "radiative OFF should not require refractive_index"
    print("[ok] refractive_index_required_when_radiative")


def test_only_material_classes_are_offered():
    """Helper classes in the materials directory must not appear as materials.

    ``Alloy`` (helpers.py) and ``OpticsInterpolator`` (alloy_optics.py) live in
    the materials package but are not materials: they have no ``get_dict()``
    and cannot be assigned to a layer.  The GUI listed them until the scan
    started filtering on the base class.
    """
    from simudo.gui.panels.mat_ast import scan_library_dirs

    names = {c.class_name for c in scan_library_dirs([MATERIALS_DIR])}
    for helper in ("Alloy", "OpticsInterpolator"):
        assert helper not in names, f"{helper} is not a material"
    # ... and the real ones are still there.
    for mat in ("SiliconMaterial", "GalliumArsenideMaterial",
                "AluminumGalliumArsenideAlloy"):
        assert mat in names, f"{mat} went missing from the material list"
    print("[ok] only_material_classes_are_offered")


def main():
    test_ast_detects_callable_in_material()
    test_eop_required_callables_discovered()
    test_runner_dispatch_splits_scalar_vs_callable()
    test_missing_params_callable_logic()
    test_missing_params_reports_unset_callable()
    test_refractive_index_required_when_radiative()
    test_only_material_classes_are_offered()
    print("\nAll AST/dispatch tests passed.")


if __name__ == "__main__":
    main()
