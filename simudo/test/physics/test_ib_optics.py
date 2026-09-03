"""IBBeerLambert: absorption through an intermediate band.

``IBBeerLambert`` is the feature that most distinguishes Simudo, and until now
it had no coverage beyond "it constructs". It is harder to test than plain
Beer-Lambert because its absorption coefficient depends on the IB occupancy,
which depends back on the generation -- there is no closed form in general.

**Turning self-consistent optics off removes that loop.** The optical problem
is then solved against a frozen absorption coefficient, so the profile is an
ordinary exponential again and can be checked against arithmetic rather than
against a previous run.

Setting the filling
-------------------
Charge neutrality fixes the IB occupancy: donors at ``f * N_IB`` must be
balanced by that many electrons in the IB, giving a filling fraction of exactly
``f``. Measured f is 0.5000 across the device for ``filling=0.5``.

**f = 0.5, not 0 or 1.** At the endpoints the filling dependence of alpha drops
out -- alpha is either zero or saturated -- and a test there would pass whether
or not the code used the occupancy at all. A half-filled band exercises it, and
:py:func:`test_alpha_tracks_the_ib_filling_fraction` pins the dependence
directly by comparing two fillings.

Note on naming
--------------
``IBBeerLambert.get_alpha`` computes ``sigma * f`` where its local ``f`` is
``IB.u``, a **density** in cm^-3, not the dimensionless filling fraction the
docstring's "filling factor" suggests -- sigma is in cm^2, so only a density
gives alpha in cm^-1. The two descriptions agree, since IB.u = N_IB * f_IB, and
the tests below are written in terms of the fraction.
"""

import math

import pytest

from .conftest import IB_NUMBER_OF_STATES, make_ib_config, read_spatial, run_project

#: cm^2, the cross-section the baseline device uses.
SIGMA = 1.0e-13

PHI = 'opt_Phi_mono_000'
ALPHA = 'opt_alpha_mono_000'


def solve_ib(runner, directory, **kwargs):
    return read_spatial(run_project(runner, make_ib_config(**kwargs), directory))


@pytest.fixture(scope='module')
def half_filled(runner, tmp_path_factory):
    """The baseline: f_IB = 0.5, sigma = 1e-13 cm^2. Shared -- the solve is slow."""
    return solve_ib(runner, tmp_path_factory.mktemp('ib_half'),
                    filling=0.5, sigma=SIGMA)


@pytest.fixture(scope='module')
def quarter_filled(runner, tmp_path_factory):
    """Same device at f_IB = 0.25, for the filling-dependence comparison."""
    return solve_ib(runner, tmp_path_factory.mktemp('ib_quarter'),
                    filling=0.25, sigma=SIGMA)


@pytest.fixture(scope='module')
def transparent(runner, tmp_path_factory):
    """sigma = 0: the IB is present but optically inert."""
    return solve_ib(runner, tmp_path_factory.mktemp('ib_zero'),
                    filling=0.5, sigma=0.0)


def filling_fraction(profile):
    return [u / n for u, n in zip(profile['u_IB'], profile['number_of_states_IB'])]


# ---------------------------------------------------------------------------
# The filling itself
# ---------------------------------------------------------------------------

@pytest.mark.solver
@pytest.mark.slow
def test_doping_sets_the_ib_filling_fraction(half_filled):
    """Charge neutrality must put the band exactly half full.

    If this drifts, every other test in the file is measuring something other
    than what it claims to, so it is asserted separately and tightly.
    """
    f = filling_fraction(half_filled)
    assert min(f) == pytest.approx(0.5, abs=1e-3), f'min filling {min(f)}'
    assert max(f) == pytest.approx(0.5, abs=1e-3), f'max filling {max(f)}'


@pytest.mark.solver
@pytest.mark.slow
def test_ib_occupancy_stays_within_its_physical_bounds(half_filled, quarter_filled):
    """0 <= u_IB <= number_of_states, everywhere.

    A sharp bound with no tolerance to argue about: a band cannot hold fewer
    than no electrons or more than it has states. This is the invariant that
    the A38 family of misconfigurations violates -- a process derived for a
    band with unlimited receiving states, applied to a sharp IB, will drive the
    occupancy outside these bounds rather than merely give a wrong number.
    """
    for name, profile in (('half', half_filled), ('quarter', quarter_filled)):
        for u, n in zip(profile['u_IB'], profile['number_of_states_IB']):
            assert 0.0 <= u <= n, (
                f'{name}: IB occupancy {u:.6e} outside [0, {n:.6e}]')


# ---------------------------------------------------------------------------
# alpha = sigma * N_IB * f_IB
# ---------------------------------------------------------------------------

@pytest.mark.solver
@pytest.mark.slow
def test_alpha_equals_sigma_times_filled_state_density(half_filled):
    """alpha = sigma * N_IB * f_IB, pointwise.

    Checks the absorption coefficient against arithmetic, using the filling the
    solver actually reached rather than the one requested -- so it tests the
    construction of alpha, not the doping.
    """
    for alpha, u in zip(half_filled[ALPHA], half_filled['u_IB']):
        assert alpha == pytest.approx(SIGMA * u, rel=1e-6), (
            f'alpha {alpha:.6e} != sigma*u_IB {SIGMA * u:.6e}')


@pytest.mark.solver
@pytest.mark.slow
def test_alpha_tracks_the_ib_filling_fraction(half_filled, quarter_filled):
    """Halving the filling must halve alpha.

    **The point of the whole file.** Comparing two fillings tests the
    dependence itself, which a single-filling test cannot: with only f = 0.5 an
    implementation that ignored occupancy entirely and hard-coded
    ``N_IB / 2`` would pass.
    """
    f_half = sum(filling_fraction(half_filled)) / len(filling_fraction(half_filled))
    f_quarter = (sum(filling_fraction(quarter_filled))
                 / len(filling_fraction(quarter_filled)))
    alpha_half = max(half_filled[ALPHA])
    alpha_quarter = max(quarter_filled[ALPHA])

    assert f_quarter < f_half, 'the two fillings are not distinguishable'
    expected = alpha_half * (f_quarter / f_half)
    assert alpha_quarter == pytest.approx(expected, rel=1e-3), (
        f'alpha did not follow the filling: f {f_half:.4f} -> {f_quarter:.4f} '
        f'but alpha {alpha_half:.6e} -> {alpha_quarter:.6e}, '
        f'expected {expected:.6e}')


# ---------------------------------------------------------------------------
# The resulting profile
# ---------------------------------------------------------------------------

@pytest.mark.solver
@pytest.mark.slow
@pytest.mark.parametrize('fixture_name', ['half_filled', 'quarter_filled'])
def test_flux_follows_the_analytic_exponential(request, fixture_name):
    """Phi(x) = Phi_0 exp(-alpha x), with alpha from the filling.

    Exact here only because self-consistent optics is off and the IB filling is
    uniform, so alpha is a constant. Run at both fillings so the check covers
    two different decay rates rather than one.
    """
    profile = request.getfixturevalue(fixture_name)
    x, phi = profile['coord_x'], profile[PHI]
    alpha_per_um = profile[ALPHA][0] * 1e-4      # cm^-1 -> um^-1

    worst = max(
        abs((phi[i] / phi[0]) / math.exp(-alpha_per_um * x[i]) - 1.0)
        for i in range(len(x)))
    assert worst < 1e-2, (
        f'{fixture_name}: flux departs from exp(-alpha x) by up to {worst:.2e} '
        f'(alpha = {profile[ALPHA][0]:.4e} cm^-1)')


@pytest.mark.solver
@pytest.mark.slow
def test_zero_cross_section_makes_the_ib_optically_inert(transparent, half_filled):
    """sigma -> 0: alpha vanishes and the flux passes through undiminished.

    The neutral element of the IB absorption. The IB is still present and still
    half full, so this isolates the optical coupling from everything else the
    band does -- if any absorption survived sigma = 0 it would have to come
    from somewhere other than the cross-section.
    """
    assert max(abs(a) for a in transparent[ALPHA]) == pytest.approx(0.0, abs=1e-12), (
        'alpha is non-zero with sigma = 0')

    phi = transparent[PHI]
    variation = (max(phi) - min(phi)) / max(phi)
    assert variation < 1e-6, (
        f'flux varies by {variation:.2e} across a device that cannot absorb')

    # And the baseline must actually absorb, or the comparison proves nothing.
    absorbed = 1.0 - half_filled[PHI][-1] / half_filled[PHI][0]
    assert absorbed > 0.5, (
        f'baseline device only absorbed {absorbed:.1%}; test is not meaningful')


@pytest.mark.solver
@pytest.mark.slow
def test_generation_follows_the_absorbed_flux(half_filled):
    """Generation into the CB must track alpha * Phi.

    Distinguishes an error in the optical solve from one in converting absorbed
    photons into carriers -- the two are separate code paths and only this
    comparison sees the second one.
    """
    alpha_cm, phi, g = (half_filled[ALPHA], half_filled[PHI],
                        half_filled['g_opt_ci_CB'])
    reference = None
    for i in range(0, len(g), max(1, len(g) // 50)):
        if g[i] <= 0:
            continue
        ratio = g[i] / (alpha_cm[i] * phi[i])
        if reference is None:
            reference = ratio
        assert ratio == pytest.approx(reference, rel=1e-6), (
            f'generation / (alpha*Phi) is not constant: {ratio:.6e} vs '
            f'{reference:.6e} -- absorbed photons are not becoming carriers '
            f'at a fixed rate')
    assert reference is not None, 'no positive generation anywhere'
