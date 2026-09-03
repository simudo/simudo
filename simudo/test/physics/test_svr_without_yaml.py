"""Radiative recombination must work without going through the YAML runner.

Created by Claude Opus 5 (Anthropic), 2026-08-21.

The Shockley-van Roosbroeck integral needs a quadrature grid over photon
energy. That grid used to be a pair of private attributes that only
``gui/simudo_1d_runner.py`` ever set, from a ``radiative_recombination:`` block
in a project YAML. Anyone assembling a problem in Python therefore hit either

    RuntimeError: ... Add an 'integration_bins' ... entry under
                  'radiative_recombination:' in the process YAML config

which is advice they cannot follow, or -- on the accumulator path, guarded by
``and self._svr_E_grid`` -- silence, with no radiative recombination at all.

The grid is now derived from public attributes with working defaults, so it
exists for every caller. These tests build the processes directly.
"""

import pytest

from .conftest import make_ib_config, make_pn_config
from .test_config_smoke import ALL_GOALS, build


def _process(problems, name):
    pd = problems['full']
    return next(p for p in pd.pdd.electro_optical_processes if p.name == name)


@pytest.mark.solver
def test_beer_lambert_has_a_working_grid_by_default(runner):
    """No YAML, no private attributes: the defaults must be usable."""
    config = make_pn_config()
    config['materials'] = {'Si': {'source': 'library://SiliconMaterial'}}
    for layer in config['layers']:
        layer['material'] = 'Si'
    config['processes'].append(
        {'class': 'BeerLambert', 'name': 'opt_cv',
         'dst_band': 'CB', 'src_band': 'VB'})
    proc = _process(build(runner, config, goals=ALL_GOALS), 'opt_cv')

    E_grid, dE = proc._svr_grid
    assert len(E_grid) == proc.svr_integration_bins == 100
    # Midpoints of uniform bins spanning the default 0-4 eV.
    U = proc.unit_registry
    assert dE.m_as('eV') == pytest.approx(0.04)
    assert E_grid[0].m_as('eV') == pytest.approx(0.02)
    assert E_grid[-1].m_as('eV') == pytest.approx(3.98)


@pytest.mark.solver
def test_grid_follows_the_public_knobs(runner):
    """Set as constructor kwargs, the way a Python user would."""
    config = make_pn_config()
    config['materials'] = {'Si': {'source': 'library://SiliconMaterial'}}
    for layer in config['layers']:
        layer['material'] = 'Si'
    config['processes'].append(
        {'class': 'BeerLambert', 'name': 'opt_cv',
         'dst_band': 'CB', 'src_band': 'VB'})
    proc = _process(build(runner, config, goals=ALL_GOALS), 'opt_cv')

    U = proc.unit_registry
    proc.svr_integration_bins = 4
    proc.svr_E_min = 1.0 * U('eV')
    proc.svr_E_max = 3.0 * U('eV')
    E_grid, dE = proc._svr_grid
    assert len(E_grid) == 4
    assert dE.m_as('eV') == pytest.approx(0.5)
    assert [E.m_as('eV') for E in E_grid] == pytest.approx([1.25, 1.75, 2.25, 2.75])


@pytest.mark.solver
def test_radiative_recombination_assembles_without_yaml(runner):
    """The whole point: enable SVR in Python and build the forms."""
    config = make_pn_config()
    config['materials'] = {'Si': {'source': 'library://SiliconMaterial'}}
    for layer in config['layers']:
        layer['material'] = 'Si'
    config['regions'][0]['properties']['refractive_index'] = {
        'value': 3.5, 'unit': ''}
    config['processes'].append(
        {'class': 'BeerLambert', 'name': 'opt_cv',
         'dst_band': 'CB', 'src_band': 'VB'})
    problems = build(runner, config, goals=ALL_GOALS)
    proc = _process(problems, 'opt_cv')

    proc.enable_radiative_recombination = True
    proc.svr_integration_bins = 3          # keep the form small
    proc.svr_E_min = 1.1 * proc.unit_registry('eV')
    # Would previously raise RuntimeError telling the user to edit a YAML file.
    assert proc.get_radiative_recombination() is not None


@pytest.mark.solver
def test_ib_beer_lambert_radiative_recombination_without_yaml(runner):
    """Same for the IB variant, which recomputes the integral every solve."""
    problems = build(runner, make_ib_config(), goals=ALL_GOALS)
    proc = _process(problems, 'opt_ci')

    E_grid, dE = proc._svr_grid
    assert len(E_grid) == 100
    proc.enable_radiative_recombination = True
    proc.svr_integration_bins = 3
    assert len(proc.__dict__.pop('_svr_grid', ())) in (0, 2)   # clear the cache
    assert len(proc._svr_grid[0]) == 3


@pytest.mark.solver
def test_bad_grid_settings_are_rejected(runner):
    config = make_pn_config()
    config['processes'].append(
        {'class': 'BeerLambert', 'name': 'opt_cv',
         'dst_band': 'CB', 'src_band': 'VB'})
    proc = _process(build(runner, config, goals=ALL_GOALS), 'opt_cv')
    U = proc.unit_registry

    proc.svr_integration_bins = 0
    with pytest.raises(ValueError, match='svr_integration_bins'):
        proc._svr_grid

    proc.__dict__.pop('_svr_grid', None)
    proc.svr_integration_bins = 10
    proc.svr_E_min = 3.0 * U('eV')
    proc.svr_E_max = 1.0 * U('eV')
    with pytest.raises(ValueError, match='svr_E_max'):
        proc._svr_grid


# ── Quadrature accuracy (A64) ────────────────────────────────────────────────
# Added by Claude Opus 5 (Anthropic), 2026-08-31.
#
# The SVR integrand is alpha(E) * E^2 * exp(-E/kT).  With alpha a top hat, the
# exact integral is available in closed form, so the midpoint rule's error can
# be measured rather than guessed.  These tests need no solver: they exercise
# the quadrature that `_svr_grid` lays down, which is what sets the accuracy.
#
# The finding these pin: the blackbody weight peaks at the absorption
# threshold, so the bin containing the threshold is the largest in the sum and
# gets counted or dropped whole.  Leaving svr_E_min at 0 therefore costs far
# more than the midpoint bias does -- and costs it in either direction.
# Everything below is evaluated at 300 K; the 0.6 eV window is ~23 kT there and
# would need to be wider at higher temperature.

import math

_KT_300K_EV = 1.380649e-23 * 300.0 / 1.602176634e-19   # 0.025852 eV


def _exact_tail(E, kT=_KT_300K_EV):
    """int_E^inf E'^2 exp(-E'/kT) dE', in closed form."""
    return kT * math.exp(-E / kT) * (E * E + 2 * kT * E + 2 * kT * kT)


def _midpoint(E_lo, E_hi, n_bins, threshold, kT=_KT_300K_EV):
    """Midpoint quadrature of the same integrand with a top-hat alpha."""
    dE = (E_hi - E_lo) / n_bins
    return sum(E * E * math.exp(-E / kT) * dE
               for E in (E_lo + (i + 0.5) * dE for i in range(n_bins))
               if E >= threshold)


@pytest.mark.parametrize('threshold', [0.611, 1.215, 1.826])
def test_aligned_but_wide_window_is_a_few_percent_low(threshold):
    """With E_min on the threshold, what is left is the midpoint bias.

    One-signed and second order in dE/kT.  This is the *smaller* of the two
    errors; see the misalignment test below for the one that dominates.
    """
    exact = _exact_tail(threshold)
    error = _midpoint(threshold, 4.0, 100, threshold) / exact - 1.0
    assert -0.07 < error < -0.02, f"expected a few percent low, got {error:.1%}"


@pytest.mark.parametrize('threshold', [0.611, 1.215, 1.826])
def test_leaving_E_min_at_zero_is_wrong_by_tens_of_percent(threshold):
    """The shipped 0-4 eV default does not align to any threshold.

    Not a regression test -- the defaults are deliberately unchanged, since only
    the caller knows where alpha turns on -- but a record of why the docs and
    the GUI both push the user to set E_min, and a tripwire if anyone decides
    the default window is good enough.
    """
    exact = _exact_tail(threshold)
    error = _midpoint(0.0, 4.0, 100, threshold) / exact - 1.0
    assert abs(error) > 0.30, (
        f"expected the unaligned default to be badly wrong, got {error:.1%}")


def test_misalignment_dominates_the_midpoint_bias():
    """Slide a threshold across one bin: the swing dwarfs the aligned error."""
    errors = [_midpoint(0.0, 4.0, 100, 1.10 + i * 0.004) / _exact_tail(1.10 + i * 0.004) - 1.0
              for i in range(10)]
    assert max(errors) - min(errors) > 0.5, (
        f"expected a large swing across one bin, got {max(errors) - min(errors):.1%}")


@pytest.mark.parametrize('threshold', [0.611, 1.215, 1.826])
def test_narrow_window_is_accurate_at_the_same_cost(threshold):
    """E_min at threshold, E_max 0.6 eV higher (23 kT at 300 K): sub-percent."""
    exact = _exact_tail(threshold)
    error = _midpoint(threshold, threshold + 0.6, 100, threshold) / exact - 1.0
    assert abs(error) < 0.01, f"expected under 1%, got {error:.2%}"


@pytest.mark.parametrize('threshold', [0.611, 1.215, 1.826])
def test_truncating_at_the_narrow_window_discards_nothing(threshold):
    """The justification for narrowing: the tail above E_min + 0.6 eV is gone.

    0.6 eV is ~23 kT at 300 K, so the blackbody weight has fallen by e^-23.
    The truncation introduced is many orders below the discretisation error it
    removes, which is what makes the narrow window a free improvement rather
    than a trade.
    """
    discarded = _exact_tail(threshold + 0.6) / _exact_tail(threshold)
    assert discarded < 1e-6, f"tail is {discarded:.2e} of the integral"
