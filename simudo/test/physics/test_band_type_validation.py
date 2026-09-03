"""EOP classes must object when applied to the wrong kind of band (A38).

Created by Claude Opus 5 (Anthropic), 2026-08-17.

The motivating case: ``ShockleyReadBand2BandTrap`` was applied between a sharp
IB and a standard VB. It ran, and produced meaningless physics silently. Its
derivation assumes the receiving band is essentially empty, which a band whose
finite filling is the whole point violates outright.

Bands now declare ``is_trap_band``, and processes declare which kind they are
derived for. Mismatches **warn** -- a wrong pairing is still a runnable
configuration, and the user may know something the class does not. The one
exception raises: a ``trap_band`` that is neither ``dst_band`` nor ``src_band``
leaves ``reg_band`` undefined, so the process cannot be assembled at all.

What is deliberately *not* checked: B2B also assumes the receiving band is
nearly empty, which is a statement about occupancy rather than band type and
cannot be settled at construction.
"""

import logging

import pytest

from .conftest import make_ib_config, make_pn_config
from .test_config_smoke import ALL_GOALS, build


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.levelno >= logging.WARNING]


def _trap_band_warnings(caplog):
    return [m for m in _warnings(caplog) if 'trap band' in m]


# ---------------------------------------------------------------------------
# The configurations that should complain
# ---------------------------------------------------------------------------

@pytest.mark.solver
def test_band_to_band_trap_between_ib_and_vb_warns(runner, caplog):
    """The A38 case itself: B2B is derived for dispersing bands."""
    config = make_ib_config()
    config['processes'] = [
        {'class': 'ShockleyReadBand2BandTrap', 'name': 'SR_B2B',
         'dst_band': 'IB', 'src_band': 'VB'},
    ]
    with caplog.at_level(logging.WARNING):
        build(runner, config, goals=ALL_GOALS)

    msgs = _trap_band_warnings(caplog)
    assert msgs, f"expected a band-type warning, got {_warnings(caplog)}"
    assert any('ShockleyReadBand2BandTrap' in m and 'IB' in m for m in msgs), msgs


@pytest.mark.solver
def test_trap_to_trap_between_dispersing_bands_warns(runner, caplog):
    """The mirror image: T2T requires sharp bands on both sides."""
    config = make_pn_config()
    config['processes'] = [
        {'class': 'ShockleyReadTrap2Trap', 'name': 'SR_T2T',
         'dst_band': 'CB', 'src_band': 'VB'},
    ]
    for layer in config['layers']:
        layer['properties']['SR_T2T/CB/capture_rate'] = {
            'value': 1e-9, 'unit': 'cm^3/s'}
    with caplog.at_level(logging.WARNING):
        build(runner, config, goals=ALL_GOALS)

    msgs = _trap_band_warnings(caplog)
    assert msgs, f"expected a band-type warning, got {_warnings(caplog)}"
    # Both bands are wrong here, so both should be named.
    assert any('dst_band' in m for m in msgs), msgs
    assert any('src_band' in m for m in msgs), msgs


@pytest.mark.solver
def test_beer_lambert_on_the_ib_warns(runner, caplog):
    """Plain BeerLambert takes alpha as a material property; an IB transition
    needs IBBeerLambert, whose alpha tracks the occupancy."""
    config = make_ib_config()
    config['processes'] = [
        {'class': 'BeerLambert', 'name': 'opt_ci',
         'dst_band': 'CB', 'src_band': 'IB'},
    ]
    with caplog.at_level(logging.WARNING):
        build(runner, config, goals=ALL_GOALS)

    assert _trap_band_warnings(caplog), _warnings(caplog)


# ---------------------------------------------------------------------------
# The configuration that cannot work at all
# ---------------------------------------------------------------------------

@pytest.mark.solver
def test_missing_trap_band_raises(runner):
    """``TrapEOPMixin`` says it requires an explicit trap_band, and without one
    ``reg_band`` is None as surely as in the case below. ``SRHRecombination``
    is the class for traps with no Band object."""
    config = make_ib_config()
    config['processes'] = [
        {'class': 'NonRadiativeTrap', 'name': 'nr_top',
         'dst_band': 'CB', 'src_band': 'IB'},
    ]
    with pytest.raises(ValueError, match='trap_band'):
        build(runner, config, goals=ALL_GOALS)


@pytest.mark.solver
def test_trap_band_outside_the_transition_raises(runner):
    """``reg_band`` is whichever of dst/src is not the trap band. Name a third
    band and it is None, which fails later as an unrelated AttributeError."""
    config = make_ib_config()
    config['processes'] = [
        {'class': 'NonRadiativeTrap', 'name': 'nr_top',
         'dst_band': 'CB', 'src_band': 'VB', 'trap_band': 'IB'},
    ]
    with pytest.raises(ValueError, match='trap_band'):
        build(runner, config, goals=ALL_GOALS)


# ---------------------------------------------------------------------------
# ...and the configurations that should stay silent
# ---------------------------------------------------------------------------

@pytest.mark.solver
def test_correct_ib_configuration_is_silent(runner, caplog):
    """The stock IB device -- SRH, two NonRadiativeTraps and IBBeerLambert, all
    correctly paired -- must not warn. Guards against a check that fires on
    everything, which would be worse than no check."""
    with caplog.at_level(logging.WARNING):
        build(runner, make_ib_config(), goals=ALL_GOALS)

    assert not _trap_band_warnings(caplog), _trap_band_warnings(caplog)


@pytest.mark.solver
def test_correct_pn_configuration_is_silent(runner, caplog):
    with caplog.at_level(logging.WARNING):
        build(runner, make_pn_config(), goals=ALL_GOALS)

    assert not _trap_band_warnings(caplog), _trap_band_warnings(caplog)
