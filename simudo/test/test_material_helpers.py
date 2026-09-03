"""Unit tests for :py:mod:`simudo.materials.helpers`.

These need no dolfin: the formulas are exercised with pint quantities and plain
numbers.  That matters because three of the four alloys are internal-only and
so are barely covered end-to-end, which makes this the place where the
interpolation and the thermal-velocity convention are actually pinned.

Expected values are written out longhand rather than by re-applying the
implementation's own formula, so these would catch a bad rewrite. In particular
the Richardson factor of 4 used by
:py:meth:`~simudo.physics.heterojunction.ThermionicHeterojunction.emission_velocity`
is derived here from sqrt(kT/2*pi*m), not by dividing the mean speed.
"""

import pytest

from simudo.materials.helpers import Alloy, thermal_velocity
from simudo.util.pint import make_unit_registry


@pytest.fixture(scope='module')
def U():
    return make_unit_registry()


# ---------------------------------------------------------------------------
# thermal_velocity
# ---------------------------------------------------------------------------

def test_thermal_velocity_is_the_maxwell_boltzmann_mean_speed(U):
    """<v> = sqrt(8 kT / pi m*), the average an SRH capture coefficient needs."""
    from math import pi, sqrt
    T = U('300 K')
    got = thermal_velocity(U, T, 0.5).m_as('m/s')
    kT = (U.boltzmann_constant * T).m_as('J')
    expected = sqrt(8 * kT / (pi * 0.5 * (1 * U.electron_mass).m_as('kg')))
    assert got == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize('T_kelvin', [100.0, 300.0, 600.0])
def test_thermal_velocity_scales_as_sqrt_T(U, T_kelvin):
    ref = thermal_velocity(U, U('300 K'), 0.5).m_as('m/s')
    got = thermal_velocity(U, U(f'{T_kelvin} K'), 0.5).m_as('m/s')
    assert got == pytest.approx(ref * (T_kelvin / 300.0) ** 0.5, rel=1e-12)


@pytest.mark.parametrize('m', [0.1, 0.5, 2.0])
def test_thermal_velocity_scales_as_inverse_sqrt_mass(U, m):
    ref = thermal_velocity(U, U('300 K'), 1.0).m_as('m/s')
    got = thermal_velocity(U, U('300 K'), m).m_as('m/s')
    assert got == pytest.approx(ref * m ** -0.5, rel=1e-12)


def test_richardson_velocity_is_a_quarter_of_the_mean_speed(U):
    """The conversion ThermionicHeterojunction.emission_velocity applies.

    v_R = sqrt(kT / 2 pi m*) and <v> = sqrt(8 kT / pi m*), so <v>/v_R = 4
    exactly. Computed here from first principles rather than by dividing, so
    the factor in heterojunction.py is genuinely checked.
    """
    from math import pi, sqrt
    T = U('300 K')
    kT = (U.boltzmann_constant * T).m_as('J')
    m = 0.5 * (1 * U.electron_mass).m_as('kg')
    v_R = sqrt(kT / (2 * pi * m))
    assert thermal_velocity(U, T, 0.5).m_as('m/s') / 4 == pytest.approx(v_R, rel=1e-12)


# ---------------------------------------------------------------------------
# Alloy
# ---------------------------------------------------------------------------

@pytest.fixture
def parents(U):
    """Two toy parents. ``at_x0`` is recovered at x=0, ``at_x1`` at x=1."""
    gaas = {'Eg': U('1.42 eV'), 'mu': U('8500 cm^2/V/s'), 'only_in_0': U('7 eV')}
    alas = {'Eg': U('2.16 eV'), 'mu': U('250 cm^2/V/s'), 'only_in_1': U('9 eV')}
    return gaas, alas


# ---------------------------------------------------------------------------
# vegard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('x', [0.0, 0.25, 0.5, 1.0])
def test_vegard_linear_without_bowing(U, parents, x):
    at_x0, at_x1 = parents
    alloy = Alloy(U, x, at_x0=at_x0, at_x1=at_x1)
    expected = U('2.16 eV') * x + U('1.42 eV') * (1 - x)
    assert alloy.vegard('Eg', U('0 eV')).m_as('eV') == pytest.approx(
        expected.m_as('eV'))


def test_vegard_recovers_each_parent_at_the_endpoints(U, parents):
    at_x0, at_x1 = parents
    assert Alloy(U, 0.0, at_x0=at_x0, at_x1=at_x1).vegard(
        'Eg', U('-0.37 eV')).m_as('eV') == pytest.approx(1.42)
    assert Alloy(U, 1.0, at_x0=at_x0, at_x1=at_x1).vegard(
        'Eg', U('-0.37 eV')).m_as('eV') == pytest.approx(2.16)


def test_vegard_applies_the_bowing_term(U, parents):
    at_x0, at_x1 = parents
    alloy = Alloy(U, 0.5, at_x0=at_x0, at_x1=at_x1)
    # linear midpoint 1.79 eV, plus C*x*(1-x) = -0.37 * 0.25
    assert alloy.vegard('Eg', U('-0.37 eV')).m_as('eV') == pytest.approx(
        1.79 - 0.0925)


@pytest.mark.parametrize('zero', ['bare', 'dimensioned', 'plain_int'])
def test_vegard_accepts_a_zero_bowing_parameter_of_any_dimension(U, parents, zero):
    """A46: a bare ``U("0")`` is dimensionless.

    Adding it to a quantity in eV is a pint ``DimensionalityError``, and most
    call sites in the alloy files spell no-bowing exactly that way. Zero has to
    drop the term, not add it.
    """
    at_x0, at_x1 = parents
    C = {'bare': U('0'), 'dimensioned': U('0 eV'), 'plain_int': 0}[zero]
    alloy = Alloy(U, 0.5, at_x0=at_x0, at_x1=at_x1)
    assert alloy.vegard('Eg', C).m_as('eV') == pytest.approx(1.79)


def test_vegard_default_bowing_is_zero(U, parents):
    at_x0, at_x1 = parents
    alloy = Alloy(U, 0.5, at_x0=at_x0, at_x1=at_x1)
    assert alloy.vegard('Eg').m_as('eV') == pytest.approx(1.79)


# ---------------------------------------------------------------------------
# missing parameters
# ---------------------------------------------------------------------------

def test_missing_parameter_raises_by_default(U, parents):
    at_x0, at_x1 = parents
    alloy = Alloy(U, 0.5, at_x0=at_x0, at_x1=at_x1)
    with pytest.raises(KeyError):
        alloy.vegard('only_in_0', U('0'))


@pytest.mark.parametrize('param,expected', [('only_in_0', 7.0), ('only_in_1', 9.0)])
def test_fallback_returns_the_parent_that_defines_it(U, parents, param, expected):
    """GaAsN's behaviour: uninterpolated, from whichever parent has the key."""
    at_x0, at_x1 = parents
    alloy = Alloy(U, 0.5, at_x0=at_x0, at_x1=at_x1, missing='fallback')
    assert alloy.vegard(param, U('0')).m_as('eV') == pytest.approx(expected)


def test_fallback_still_raises_when_neither_parent_has_it(U, parents):
    at_x0, at_x1 = parents
    alloy = Alloy(U, 0.5, at_x0=at_x0, at_x1=at_x1, missing='fallback')
    with pytest.raises(KeyError):
        alloy.vegard('in_neither', U('0'))


def test_invalid_missing_mode_is_rejected(U, parents):
    at_x0, at_x1 = parents
    with pytest.raises(ValueError, match='raise.*fallback'):
        Alloy(U, 0.5, at_x0=at_x0, at_x1=at_x1, missing='ignore')


# ---------------------------------------------------------------------------
# mobility_bowing
# ---------------------------------------------------------------------------

def test_mobility_bowing_is_harmonic(U, parents):
    at_x0, at_x1 = parents
    alloy = Alloy(U, 0.5, at_x0=at_x0, at_x1=at_x1)
    C = U('125 cm^2/V/s')
    expected = 1.0 / (0.5 / 8500 + 0.5 / 250 + 0.25 / 125)
    assert alloy.mobility_bowing('mu', C).m_as('cm^2/V/s') == pytest.approx(expected)


@pytest.mark.parametrize('x,expected', [(0.0, 8500.0), (1.0, 250.0)])
def test_mobility_bowing_recovers_each_parent_at_the_endpoints(U, parents, x, expected):
    """At the endpoints the bowing term vanishes, whatever C is."""
    at_x0, at_x1 = parents
    alloy = Alloy(U, x, at_x0=at_x0, at_x1=at_x1)
    got = alloy.mobility_bowing('mu', U('125 cm^2/V/s')).m_as('cm^2/V/s')
    assert got == pytest.approx(expected)


def test_large_bowing_parameter_means_negligible_bowing(U, parents):
    """The convention the alloy files rely on for satellite valleys."""
    at_x0, at_x1 = parents
    alloy = Alloy(U, 0.5, at_x0=at_x0, at_x1=at_x1)
    harmonic = 1.0 / (0.5 / 8500 + 0.5 / 250)
    got = alloy.mobility_bowing('mu', U('1e6 cm^2/V/s')).m_as('cm^2/V/s')
    assert got == pytest.approx(harmonic, rel=1e-3)


@pytest.mark.parametrize('zero', ['bare', 'dimensioned', 'plain_int'])
def test_mobility_bowing_rejects_zero(U, parents, zero):
    """M11: C=0 is 'no bowing' for vegard but division by zero here.

    The two conventions are opposite and the mistake is silent, so it must
    raise rather than produce ``inf``.
    """
    at_x0, at_x1 = parents
    C = {'bare': U('0'), 'dimensioned': U('0 cm^2/V/s'), 'plain_int': 0}[zero]
    alloy = Alloy(U, 0.5, at_x0=at_x0, at_x1=at_x1)
    with pytest.raises(ValueError, match='division by zero'):
        alloy.mobility_bowing('mu', C)
