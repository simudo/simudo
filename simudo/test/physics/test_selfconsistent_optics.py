"""Self-consistent optics: the feedback loop between absorption and occupancy.

``selfconsistent_optics`` only means anything when the absorption coefficient
depends on the carrier densities. For plain Beer-Lambert with a constant alpha
there is no loop and the flag is inert; for ``IBBeerLambert`` alpha is
proportional to the IB occupancy, so illumination changes what the light sees.

The physics under test is **optical bleaching**. Photons pump electrons out of
the IB into the CB, so f_IB falls, so alpha falls, so the device becomes more
transparent -- and being more transparent it absorbs less, which limits the
bleaching. Every test here is a signed statement about that loop, not a stored
number.

Measured 2026-08-15 across the intensity range, on the fixture below:

=============  ==========  =============  ==============
concentration  self-cons.  min f_IB       transmission
=============  ==========  =============  ==============
1e-3           either      0.50000        0.023508
1e5            off         0.49209        0.024407
1e5            on          0.49209        0.024436
1e7            off         0.02038        0.243716
1e7            on          0.05133        0.264680
=============  ==========  =============  ==============

At 1e-3 suns the two agree to seven digits; at 1e7 the band is nearly emptied
and the loop is worth 8.6 % of the transmitted flux.
"""

import pytest

from .conftest import make_ib_config, read_spatial, run_project

PHI = 'opt_Phi_mono_000'
ALPHA = 'opt_alpha_mono_000'

#: Well below the intensity at which the IB starts to empty.
WEAK = 1e-3

#: Hard enough to strip most of the band, so the feedback is unambiguous.
STRONG = 1e7


def solve_at(runner, directory, concentration, selfconsistent):
    config = make_ib_config(filling=0.5, sigma=1e-13)
    config['optical_field_sets'][0]['concentration'] = concentration
    config['simulation']['intensity_ramp']['selfconsistent_optics'] = selfconsistent
    config['simulation']['voltage_sweep']['selfconsistent_optics'] = selfconsistent
    return read_spatial(run_project(runner, config, directory))


def min_filling(profile):
    return min(u / n for u, n in
               zip(profile['u_IB'], profile['number_of_states_IB']))


def transmission(profile):
    return profile[PHI][-1] / profile[PHI][0]


@pytest.fixture(scope='module')
def weak(runner, tmp_path_factory):
    return {sc: solve_at(runner, tmp_path_factory.mktemp(f'weak_{sc}'), WEAK, sc)
            for sc in (False, True)}


@pytest.fixture(scope='module')
def strong(runner, tmp_path_factory):
    return {sc: solve_at(runner, tmp_path_factory.mktemp(f'strong_{sc}'), STRONG, sc)
            for sc in (False, True)}


# ---------------------------------------------------------------------------
# The weak-intensity limit
# ---------------------------------------------------------------------------

@pytest.mark.solver
@pytest.mark.slow
def test_selfconsistency_is_irrelevant_at_low_intensity(weak):
    """With nothing to bleach, the flag must make no difference.

    The limit that makes the flag's meaning precise: self-consistency is a
    correction for the occupancy *changing*, so where the occupancy does not
    change, turning it on must be a no-op. If these two ever diverge, the
    self-consistent path is doing something beyond closing the loop.
    """
    off, on = weak[False], weak[True]
    assert transmission(on) == pytest.approx(transmission(off), rel=1e-6), (
        f'at {WEAK:g} suns, self-consistent transmission {transmission(on):.8e} '
        f'!= frozen {transmission(off):.8e}')
    assert min_filling(on) == pytest.approx(min_filling(off), rel=1e-6)


@pytest.mark.solver
@pytest.mark.slow
def test_weak_illumination_leaves_the_band_half_full(weak):
    """The dark occupancy must survive weak illumination.

    Establishes that the fixture really is in the unperturbed limit, so the
    previous test is not passing merely because both runs were broken in the
    same way.
    """
    for selfconsistent, profile in weak.items():
        assert min_filling(profile) == pytest.approx(0.5, abs=1e-4), (
            f'selfconsistent={selfconsistent}: filling moved to '
            f'{min_filling(profile):.5f} under negligible illumination')


# ---------------------------------------------------------------------------
# Bleaching
# ---------------------------------------------------------------------------

@pytest.mark.solver
@pytest.mark.slow
@pytest.mark.parametrize('selfconsistent', [False, True])
def test_strong_illumination_empties_the_band(weak, strong, selfconsistent):
    """Pumping IB -> CB must reduce the IB occupancy.

    The sign is what matters: photons remove electrons from the IB, so the
    filling can only fall. A model that pumped the wrong way, or that ignored
    the optical term in the IB's balance, would raise it or leave it flat.
    """
    dark_filling = min_filling(weak[selfconsistent])
    pumped_filling = min_filling(strong[selfconsistent])
    assert pumped_filling < dark_filling - 0.05, (
        f'selfconsistent={selfconsistent}: filling went {dark_filling:.5f} -> '
        f'{pumped_filling:.5f} under strong pumping; expected a large drop')


@pytest.mark.solver
@pytest.mark.slow
@pytest.mark.parametrize('selfconsistent', [False, True])
def test_emptying_the_band_bleaches_the_absorption(weak, strong, selfconsistent):
    """Less filling must mean more transmission.

    Couples the two halves of the loop: alpha is proportional to the occupancy,
    so a band that has been emptied has to pass more light. This is the step
    that fails if the optical solve uses a stale occupancy.
    """
    dark = transmission(weak[selfconsistent])
    pumped = transmission(strong[selfconsistent])
    assert pumped > dark * 2, (
        f'selfconsistent={selfconsistent}: transmission only went '
        f'{dark:.6e} -> {pumped:.6e} while the band was being emptied')


@pytest.mark.solver
@pytest.mark.slow
def test_selfconsistency_limits_the_bleaching(strong):
    """Closing the loop must leave *more* of the band filled, not less.

    The feedback is self-limiting: as the front of the device bleaches, alpha
    there drops, so it absorbs less, so it bleaches less. Freezing alpha at its
    dark value removes that brake and over-pumps.

    Measured 0.05133 with the loop closed against 0.02038 with it frozen -- a
    factor of 2.5 in the residual filling, so the sign is unambiguous rather
    than marginal.
    """
    frozen, closed = strong[False], strong[True]
    assert min_filling(closed) > min_filling(frozen), (
        f'self-consistent filling {min_filling(closed):.5f} is not above '
        f'frozen {min_filling(frozen):.5f}; the feedback should be a brake')
    assert transmission(closed) > transmission(frozen), (
        f'self-consistent transmission {transmission(closed):.6e} is not above '
        f'frozen {transmission(frozen):.6e}')


@pytest.mark.solver
@pytest.mark.slow
def test_picard_freeze_leaves_only_a_small_residual_inconsistency(strong):
    """How stale the frozen alpha is by the time the solve converges.

    alpha is projected once, before the first Newton iteration, and held fixed
    through the solve (``pre_first_iteration_hook``); the occupancy reported
    afterwards is the converged one. So alpha = sigma * u_IB is **not** expected
    to hold exactly here -- the two are from different iterates, and the gap is
    the residual of the fixed-point iteration.

    In the dark they do match exactly, because the occupancy does not move
    during the solve; ``test_ib_optics.py`` asserts that at rel=1e-6.

    What matters is that the gap stays small: it is the amount by which the
    converged solution fails to be self-consistent. Measured 0.18 % on a
    device bleached from f = 0.5 to f = 0.05, so the bound below is ~5x the
    observed value. If it grows, the once-per-solve refresh (A5) is no longer
    sufficient and would need revisiting -- which is exactly the re-test that
    A5 says is worthwhile once a regression case exists. This is that case.
    """
    profile = strong[True]
    worst = max(
        abs(alpha - 1e-13 * u) / (1e-13 * u)
        for alpha, u in zip(profile[ALPHA], profile['u_IB'])
        if u > 0)
    assert worst < 1e-2, (
        f'frozen alpha is {worst:.2%} away from sigma*u_IB at the converged '
        f'occupancy; the Picard fixed point has not settled')
