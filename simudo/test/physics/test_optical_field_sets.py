#!/usr/bin/env python3
"""
Tests for simudo_1d_runner.py optical field machinery.

_expand_field_set tests (mock FEniCS objects):
  - correct number of fields registered
  - field names follow the {prefix}_{i:03d} pattern
  - flux values sum to the expected AM1.5G integral over the energy range
  - name collision detection raises ValueError

_parse_alpha_spec tests (require dolfin, no mesh needed):
  - constant alpha form
  - energies_eV inline table (interpolation + zero extrapolation)
  - wavelengths_nm inline table (wavelength→energy conversion)
  - table_file CSV form
  - bad spec raises ValueError
"""
import os

import pytest

# dolfin must be initialised before the runner is imported.
dolfin = pytest.importorskip(
    'dolfin', reason='solver tests require a FEniCS/dolfin installation')

from simudo.gui import simudo_1d_runner as runner
from simudo.util import make_unit_registry, AM15Spectrum

_expand_field_set  = runner._expand_field_set
_parse_alpha_spec  = runner._parse_alpha_spec
parse_quantity     = runner.parse_quantity

U = make_unit_registry()

# ---- Minimal mock objects -------------------------------------------------

class MockOptical:
    def __init__(self):
        self.fields = []   # list of (name, photon_energy, direction)
    def easy_add_field(self, name, photon_energy, direction):
        self.fields.append((name, photon_energy, direction))

class MockOSpatial:
    def __init__(self):
        self.bcs = []   # list of (key, contact, value)
    def add_BC(self, key, contact, value):
        self.bcs.append((key, contact, value))

class MockF:
    left_contact  = 'LEFT'
    right_contact = 'RIGHT'

# ---- Test 1: AM1.5G, N=5 bins --------------------------------------------

def test_am15g_bins():
    optical  = MockOptical()
    ospatial = MockOSpatial()
    F        = MockF()
    names    = set()

    fset = {
        'name_prefix': 'sol',
        'direction':   '+x',
        'N_bins':      5,
        'E_min':       {'value': 1.1, 'unit': 'eV'},
        'E_max':       {'value': 4.0, 'unit': 'eV'},
        'spectrum': {
            'type':          'am15g',
            'concentration': 1.0,
        },
    }

    _expand_field_set(fset, U, optical, ospatial, F, names)

    # 5 fields registered
    assert len(optical.fields) == 5, f"Expected 5 fields, got {len(optical.fields)}"
    assert len(ospatial.bcs)   == 5

    # Names are sol_000 … sol_004
    field_names = [f[0] for f in optical.fields]
    assert field_names == [f'sol_{i:03d}' for i in range(5)], field_names

    # All BCs go to left_contact (direction +x)
    assert all(bc[1] == 'LEFT' for bc in ospatial.bcs)

    # All BCs key matches {name}/Phi
    for i, bc in enumerate(ospatial.bcs):
        assert bc[0] == f'sol_{i:03d}/Phi', bc[0]

    # Flux sum matches direct integral from AM15Spectrum
    spec = AM15Spectrum(U, variant='global', concentration=1.0)
    expected_total = spec.photon_flux_integral(
        U('1.1 eV'), U('4.0 eV')).m_as('1/cm^2/s')
    actual_total = sum(bc[2].m_as('1/cm^2/s') for bc in ospatial.bcs)
    rel_err = abs(actual_total - expected_total) / expected_total
    print(f"  AM1.5G total flux:    {expected_total:.4e} /cm^2/s")
    print(f"  Sum of bin fluxes:    {actual_total:.4e} /cm^2/s")
    print(f"  Relative error:       {rel_err:.2e}")
    assert rel_err < 1e-4, f"Flux conservation error too large: {rel_err:.2e}"

    # names set is updated
    assert names == {f'sol_{i:03d}' for i in range(5)}

    print("  PASS: test_am15g_bins")

# ---- Test 2: AM1.5D, direction -x ----------------------------------------

def test_am15d_direction():
    optical  = MockOptical()
    ospatial = MockOSpatial()
    F        = MockF()
    names    = set()

    fset = {
        'name_prefix': 'direct',
        'direction':   '-x',
        'N_bins':      3,
        'E_min':       {'value': 1.0, 'unit': 'eV'},
        'E_max':       {'value': 3.0, 'unit': 'eV'},
        'spectrum': {'type': 'am15d'},
    }

    _expand_field_set(fset, U, optical, ospatial, F, names)

    assert len(optical.fields) == 3
    assert all(bc[1] == 'RIGHT' for bc in ospatial.bcs), "Expected right_contact for -x"

    spec = AM15Spectrum(U, variant='direct', concentration=1.0)
    expected = spec.photon_flux_integral(U('1.0 eV'), U('3.0 eV')).m_as('1/cm^2/s')
    actual   = sum(bc[2].m_as('1/cm^2/s') for bc in ospatial.bcs)
    rel_err  = abs(actual - expected) / expected
    assert rel_err < 1e-4, f"AM1.5D flux error: {rel_err:.2e}"

    print("  PASS: test_am15d_direction")

# ---- Test 3: concentration multiplier ------------------------------------

def test_concentration():
    optical  = MockOptical()
    ospatial = MockOSpatial()
    F        = MockF()
    names    = set()

    fset = {
        'name_prefix': 'conc',
        'N_bins':      4,
        'E_min':       {'value': 1.1, 'unit': 'eV'},
        'E_max':       {'value': 2.0, 'unit': 'eV'},
        'spectrum': {'type': 'am15g', 'concentration': 10.0},
    }

    _expand_field_set(fset, U, optical, ospatial, F, names)

    spec = AM15Spectrum(U, variant='global', concentration=1.0)
    expected = 10.0 * spec.photon_flux_integral(
        U('1.1 eV'), U('2.0 eV')).m_as('1/cm^2/s')
    actual = sum(bc[2].m_as('1/cm^2/s') for bc in ospatial.bcs)
    rel_err = abs(actual - expected) / expected
    assert rel_err < 1e-4, f"Concentration flux error: {rel_err:.2e}"

    print("  PASS: test_concentration")

# ---- Test 4: name collision detection ------------------------------------

def test_collision():
    optical  = MockOptical()
    ospatial = MockOSpatial()
    F        = MockF()
    names    = {'sol_001'}   # pre-occupy one name

    fset = {
        'name_prefix': 'sol',
        'N_bins':      3,
        'E_min':       {'value': 1.1, 'unit': 'eV'},
        'E_max':       {'value': 2.0, 'unit': 'eV'},
        'spectrum': {'type': 'am15g'},
    }

    try:
        _expand_field_set(fset, U, optical, ospatial, F, names)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'sol_001' in str(e), str(e)

    print("  PASS: test_collision")

# ---- Test 5: bad spectrum type -------------------------------------------

def test_bad_spectrum_type():
    optical  = MockOptical()
    ospatial = MockOSpatial()
    F        = MockF()
    names    = set()

    fset = {
        'name_prefix': 'x',
        'N_bins':      2,
        'E_min':       {'value': 1.0, 'unit': 'eV'},
        'E_max':       {'value': 2.0, 'unit': 'eV'},
        'spectrum': {'type': 'solar_wind'},
    }

    try:
        _expand_field_set(fset, U, optical, ospatial, F, names)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'solar_wind' in str(e), str(e)

    print("  PASS: test_bad_spectrum_type")

# ===========================================================================
# _parse_alpha_spec tests
# ===========================================================================

def _eval(alpha_func, E_eV):
    """Call alpha_func at E_eV and return the float value in cm^-1."""
    E    = E_eV * U.eV
    result = alpha_func(E)
    return float(result.m_as('1/cm'))


# ---- Test 6: constant alpha -----------------------------------------------

def test_alpha_constant():
    spec = {'constant': {'value': 1234.5, 'unit': '1/cm'}}
    f    = _parse_alpha_spec(spec, U)

    assert abs(_eval(f, 1.0)  - 1234.5) < 1e-9
    assert abs(_eval(f, 2.5)  - 1234.5) < 1e-9
    assert abs(_eval(f, 0.01) - 1234.5) < 1e-9, "constant should not depend on E"

    print("  PASS: test_alpha_constant")


# ---- Test 7: energies_eV inline table ------------------------------------

def test_alpha_energy_table():
    # Simple two-point table: alpha rises linearly from 0 at 1.0 eV to 1000 at 2.0 eV.
    spec = {
        'energies_eV': [1.0, 2.0],
        'alphas':       [0.0, 1000.0],
        'unit':         '1/cm',
    }
    f = _parse_alpha_spec(spec, U)

    # Interpolated midpoint
    mid = _eval(f, 1.5)
    assert abs(mid - 500.0) < 1e-6, f"Midpoint should be 500, got {mid}"

    # At endpoints
    assert abs(_eval(f, 1.0) - 0.0)    < 1e-9
    assert abs(_eval(f, 2.0) - 1000.0) < 1e-9

    # Outside range: extrapolate_zero=True → zero
    assert abs(_eval(f, 0.5)) < 1e-9,  "Below table range should be zero"
    assert abs(_eval(f, 3.0)) < 1e-9,  "Above table range should be zero"

    print("  PASS: test_alpha_energy_table")


# ---- Test 8: wavelengths_nm inline table ---------------------------------

def test_alpha_wavelength_table():
    # Table by wavelength: 1000 nm = 1.240 eV, 500 nm = 2.480 eV.
    # Alpha: 200 at 1000 nm, 800 at 500 nm.
    # That means alpha should interpolate to 500 at the midpoint energy
    # ~1.860 eV (620 nm).
    wl_nm  = [1000.0, 500.0]
    alphas = [200.0,  800.0]
    spec   = {'wavelengths_nm': wl_nm, 'alphas': alphas, 'unit': '1/cm'}
    f      = _parse_alpha_spec(spec, U)

    # Convert wavelengths to energies: hc ≈ 1239.84 eV·nm
    HC = 1239.8419843320028
    E_lo = HC / 1000.0   # 1.2398 eV → alpha = 200
    E_hi = HC /  500.0   # 2.4797 eV → alpha = 800

    assert abs(_eval(f, E_lo) - 200.0) < 1e-4, f"Got {_eval(f, E_lo)}"
    assert abs(_eval(f, E_hi) - 800.0) < 1e-4, f"Got {_eval(f, E_hi)}"

    # Midpoint energy
    E_mid = 0.5 * (E_lo + E_hi)
    mid   = _eval(f, E_mid)
    assert 200.0 < mid < 800.0, f"Midpoint alpha {mid} not between endpoints"

    # Outside range: zero
    assert abs(_eval(f, 0.5)) < 1e-9,  "Below table range should be zero"
    assert abs(_eval(f, 5.0)) < 1e-9,  "Above table range should be zero"

    print("  PASS: test_alpha_wavelength_table")


# ---- Test 9: table_file CSV ----------------------------------------------

def test_alpha_table_file():
    import tempfile, numpy as np_

    # Write a temporary CSV: two columns — energy_eV, alpha_cm-1
    rows = np_.array([[1.0, 0.0], [2.0, 2000.0], [3.0, 4000.0]])
    tmp  = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
    for row in rows:
        tmp.write(f"{row[0]},{row[1]}\n")
    tmp.close()

    spec = {'table_file': tmp.name, 'unit': '1/cm'}
    f    = _parse_alpha_spec(spec, U)

    assert abs(_eval(f, 1.0) - 0.0)    < 1e-9
    assert abs(_eval(f, 2.0) - 2000.0) < 1e-9
    assert abs(_eval(f, 3.0) - 4000.0) < 1e-9
    assert abs(_eval(f, 1.5) - 1000.0) < 1e-6,  "Midpoint interpolation"
    assert abs(_eval(f, 0.5)) < 1e-9,            "Below range → zero"
    assert abs(_eval(f, 4.0)) < 1e-9,            "Above range → zero"

    os.unlink(tmp.name)
    print("  PASS: test_alpha_table_file")


# ---- Test 10: bad spec raises ValueError ---------------------------------

def test_alpha_bad_spec():
    spec = {'nonsense_key': 42}
    try:
        _parse_alpha_spec(spec, U)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'nonsense_key' in str(e), str(e)

    print("  PASS: test_alpha_bad_spec")


# ---- Run all tests --------------------------------------------------------

if __name__ == '__main__':
    print("test_am15g_bins")
    test_am15g_bins()
    print("test_am15d_direction")
    test_am15d_direction()
    print("test_concentration")
    test_concentration()
    print("test_collision")
    test_collision()
    print("test_bad_spectrum_type")
    test_bad_spectrum_type()
    print("test_alpha_constant")
    test_alpha_constant()
    print("test_alpha_energy_table")
    test_alpha_energy_table()
    print("test_alpha_wavelength_table")
    test_alpha_wavelength_table()
    print("test_alpha_table_file")
    test_alpha_table_file()
    print("test_alpha_bad_spec")
    test_alpha_bad_spec()
    print("\nAll tests passed.")
