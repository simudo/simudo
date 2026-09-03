"""Tier 2: physical invariants and limits.

Every check here is something the solver must satisfy *by construction* -- an
exact identity, or one configuration agreeing with a second, independently
implemented one. None of them stores a golden number.

That is the point. Simudo's physics is actively being improved: A47 moved every
heterojunction current by a factor of ~4 on the same day the HJBC default flip
moved it by another 4 %. A golden-value suite would have failed on both and told
you nothing about whether either was *right*. These tests pass before and after
both changes, and would fail if either had broken the underlying physics.

Tolerances are relative to the device's own current scale rather than absolute,
because "zero current" in a solved problem means "small compared to the currents
this device carries", not 1e-30.  The isotype device sits at ~5e-9 of its own
0.1 V current at equilibrium, not at machine zero.
"""

import pytest

from .conftest import (
    make_heterojunction_config,
    make_isotype_config,
    make_pn_config,
    read_jv,
    read_spatial,
    run_project,
)

#: Boltzmann voltage kT/q at the 300 K the fixtures run at.
KT_OVER_Q = 0.025852

CB_RIGHT = 'avg:current_CB:right_contact'
CB_LEFT = 'avg:current_CB:left_contact'


def solve(runner, config, tmp_path, name='project.yaml'):
    """Run *config* and return ``{V: {column: value}}``."""
    return read_jv(run_project(runner, config, tmp_path, name=name))


def relative_difference(a, b):
    """|a - b| / max(|a|, |b|), and 0 when both are zero."""
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0.0 else abs(a - b) / scale


def assert_curves_agree(a, b, tol, what, column=None):
    """Assert two J(V) curves agree to *tol* of their own current scale.

    Compared against the scale of the whole curve, not against each point's own
    magnitude.  At V = 0 the current is a numerical residual -- I1 measures it
    at ~1e-9 of what the device carries at 0.1 V -- so a pointwise *relative*
    comparison there compares noise with noise and fails on curves that are
    physically identical.  Scaling by the curve instead asks the question that
    matters: do these two configurations trace the same J(V)?
    """
    column = column or CB_RIGHT
    scale = max(max(abs(r[column]) for r in a.values()),
                max(abs(r[column]) for r in b.values()))
    assert scale > 0, f'{what}: both curves are identically zero; test is vacuous'
    for v in sorted(a):
        difference = abs(a[v][column] - b[v][column])
        assert difference < tol * scale, (
            f'{what}\n  V={v}: {a[v][column]:.6e} vs {b[v][column]:.6e}\n'
            f'  differ by {difference:.3e}, which is '
            f'{difference / scale:.2e} of the curve scale {scale:.3e} '
            f'(tolerance {tol:.0e})')


# ---------------------------------------------------------------------------
# I1 -- equilibrium carries no current
# ---------------------------------------------------------------------------

@pytest.mark.solver
@pytest.mark.slow
@pytest.mark.parametrize('maker', [make_isotype_config, make_pn_config,
                                   make_heterojunction_config],
                         ids=['isotype', 'pn', 'heterojunction'])
def test_no_current_at_equilibrium_in_the_dark(runner, tmp_path, maker):
    """I1: dark and unbiased, J must vanish at both contacts.

    Exact, and the cheapest broad-spectrum check there is: a sign error, a
    misregistered BC or a stray generation term all show up here.

    Scaled against the same device's current at 0.1 V rather than an absolute
    floor, so the bound means "nine orders below what this device actually
    carries" for any device.
    """
    config = maker()
    config['simulation']['voltage_sweep']['values'] = [0.0, 0.1]
    jv = solve(runner, config, tmp_path)

    scale = abs(jv[0.1][CB_RIGHT])
    assert scale > 0, 'device carries no current at 0.1 V; test is vacuous'
    for column in (CB_RIGHT, CB_LEFT):
        assert abs(jv[0.0][column]) < 1e-6 * scale, (
            f'{column} at equilibrium is {jv[0.0][column]:.3e}, '
            f'not negligible against {scale:.3e} at 0.1 V')


@pytest.mark.solver
@pytest.mark.slow
def test_current_is_continuous_through_the_device(runner, tmp_path):
    """What flows in one contact must leave the other.

    Not a BC test -- a conservation test on the assembled weak form. The two
    contacts are computed by separate surface integrals, so agreement is a real
    statement about the solution.
    """
    config = make_heterojunction_config()
    config['simulation']['voltage_sweep']['values'] = [0.0, 0.1]
    jv = solve(runner, config, tmp_path)
    for v, row in jv.items():
        assert relative_difference(row[CB_RIGHT], row[CB_LEFT]) < 1e-6, (
            f'V={v}: current in {row[CB_LEFT]:.6e} != out {row[CB_RIGHT]:.6e}')


# ---------------------------------------------------------------------------
# I5 / I6 -- the surface-recombination BC brackets the two limiting BCs
# ---------------------------------------------------------------------------

@pytest.mark.solver
@pytest.mark.slow
def test_vanishing_srv_approaches_the_zero_current_bc(runner, tmp_path):
    """I5: as s -> 0 the Robin BC must become the blocking contact.

    ``zero_current`` is implemented independently of ``surface_recombination``
    -- an essential BC on j, not a weak-form surface term -- so this compares
    two separate code paths rather than one against itself. Also pins A22,
    where s = 0 exactly used to divide by s.
    """
    def config_with(cb_bc):
        config = make_isotype_config()
        config['contacts']['left']['bands']['CB'] = cb_bc
        config['simulation']['voltage_sweep']['values'] = [0.0, 0.1]
        return config

    tiny = solve(runner, config_with(
        {'type': 'surface_recombination',
         'srv': {'value': 1e-6, 'unit': 'cm/s'}}), tmp_path, 'tiny.yaml')
    blocking = solve(runner, config_with('zero_current'), tmp_path, 'zero.yaml')
    # Both of the above are blocking, so their own currents are a residual and
    # cannot set the scale -- comparing them relatively compares noise. The
    # scale has to come from the same device with a conducting contact.
    conducting = solve(runner, config_with('ohmic'), tmp_path, 'ohmic_ref.yaml')

    scale = max(abs(r[CB_RIGHT]) for r in conducting.values())
    for v in sorted(tiny):
        difference = abs(tiny[v][CB_RIGHT] - blocking[v][CB_RIGHT])
        assert difference < 1e-6 * scale, (
            f'V={v}: srv=1e-6 gives {tiny[v][CB_RIGHT]:.6e}, zero_current gives '
            f'{blocking[v][CB_RIGHT]:.6e}; they differ by {difference:.3e}, '
            f'which is not negligible against the {scale:.3e} this device '
            f'carries with an ohmic contact')


@pytest.mark.solver
@pytest.mark.slow
def test_large_srv_approaches_the_ohmic_bc(runner, tmp_path):
    """I6: as s -> infinity the Robin BC must pin u to equilibrium, i.e. ohmic.

    The other end of the bracket from I5, against a third independent BC
    implementation (an essential BC on u).
    """
    def config_with(cb_bc):
        config = make_isotype_config()
        config['contacts']['left']['bands']['CB'] = cb_bc
        config['simulation']['voltage_sweep']['values'] = [0.0, 0.1]
        return config

    huge = solve(runner, config_with(
        {'type': 'surface_recombination',
         'srv': {'value': 1e12, 'unit': 'cm/s'}}), tmp_path, 'huge.yaml')
    ohmic = solve(runner, config_with('ohmic'), tmp_path, 'ohmic.yaml')

    assert_curves_agree(huge, ohmic, 1e-2,
                        'srv=1e12 should behave as an ohmic contact')


# ---------------------------------------------------------------------------
# I11 was dropped -- see doc/release0_7/08_test_plan.md
#
# The plan proposed "a heterojunction with zero band offset must give the same
# J as the homojunction". That invariant is false. Thermionic emission has a
# finite velocity, so the interface always carries a resistance ~ 1/(q v_R n)
# in series with the bulk, whatever the barrier height.
#
# Measured 2026-08-15 on the isotype device: adding a zero-offset HJBC lowers J
# by 29.7%, and by the *same* 29.7% at V = 1e-4 and at V = 0.1 -- bias
# independent, i.e. a linear series element, not a current-dependent clamp.
# Interface and bulk resistance scale as mu/(L*v_R), so this device would have
# to be ~100 um long rather than 0.5 um for the interface to be negligible,
# which is far too expensive for a unit test.
#
# The 29.7% itself has not been checked against an analytic value; see
# 06_known_issues.md A48.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# I7 -- the HJBC enhancement factor converges
# ---------------------------------------------------------------------------

@pytest.mark.solver
@pytest.mark.slow
def test_heterojunction_converges_in_the_enhancement_factor(runner, tmp_path):
    """I7: raising the penalty weight must stop changing the answer.

    ``HJBC_enhancement`` is a numerical penalty on deviation from the BC, not a
    physical parameter, so J must converge as it grows. Measured 2026-08-15:
    the two BC formulations differ by 4 % at enhancement 1 and agree to 8
    significant figures at 1e6 -- the difference scales as 1/enhancement.

    Self-convergence, so it survives any change to the BC formulation while
    still failing if enforcement silently stops working.
    """
    def config_at(enhancement):
        config = make_isotype_config()
        config['interfaces'] = [{
            'left': 'wide', 'right': 'narrow',
            'bcs': [{'type': 'ThermionicHeterojunction',
                     'bands': ['CB', 'VB'],
                     'HJBC_enhancement': {'CB': enhancement, 'VB': enhancement}}],
        }]
        config['simulation']['voltage_sweep']['values'] = [0.0, 0.1]
        return config

    mid = solve(runner, config_at(1e3), tmp_path, 'e3.yaml')
    high = solve(runner, config_at(1e6), tmp_path, 'e6.yaml')

    assert_curves_agree(mid, high, 1e-3,
                        'HJBC must converge in the enhancement factor')


# ---------------------------------------------------------------------------
# I8 -- degenerate statistics reduce to Boltzmann when they should
# ---------------------------------------------------------------------------

@pytest.mark.solver
@pytest.mark.slow
def test_degenerate_band_reduces_to_nondegenerate_at_low_doping(runner, tmp_path):
    """I8: Joyce-Dixon must agree with Boltzmann far from degeneracy.

    At 1e15 cm^-3 the Fermi level sits well below the band edge, so Fermi-Dirac
    and Boltzmann statistics coincide and the two band classes must give the
    same current. Tests the whole degenerate path -- u<->qfl conversions
    included -- against the simpler one, and pins the validity domain A30 says
    belongs in the release notes.
    """
    def config_with(band_type):
        config = make_isotype_config()
        for band in config['bands']:
            band['type'] = band_type
        for layer in config['layers']:
            layer['properties']['poisson/static_rho'] = {
                'value': 1.0e15, 'unit': 'cm^-3'}
        config['simulation']['voltage_sweep']['values'] = [0.0, 0.1]
        return config

    boltzmann = solve(runner, config_with('nondegenerate'), tmp_path, 'nondeg.yaml')
    fermi = solve(runner, config_with('degenerate'), tmp_path, 'deg.yaml')

    assert_curves_agree(boltzmann, fermi, 1e-2,
                        'Joyce-Dixon must reduce to Boltzmann at 1e15 cm^-3')


# ---------------------------------------------------------------------------
# I9 -- a symmetric device has an antisymmetric J(V)
# ---------------------------------------------------------------------------

@pytest.mark.solver
@pytest.mark.slow
def test_symmetric_device_has_antisymmetric_jv(runner, tmp_path):
    """I9: uniform doping and identical contacts, so J(-V) = -J(V).

    Tests sign conventions and contact symmetry end to end. A device with no
    junction anywhere is the only case where this identity is exact, which is
    why the doping is made uniform rather than reusing a junction fixture.
    """
    config = make_isotype_config()
    for layer in config['layers']:
        layer['properties']['poisson/static_rho'] = {'value': 1.0e17, 'unit': 'cm^-3'}
        layer['properties']['CB/energy_level'] = {'value': 1.42, 'unit': 'eV'}
        layer['properties']['VB/energy_level'] = {'value': 0.0, 'unit': 'eV'}
    config['simulation']['voltage_sweep']['values'] = [-0.05, 0.0, 0.05]
    jv = solve(runner, config, tmp_path)

    forward, reverse = jv[0.05][CB_RIGHT], jv[-0.05][CB_RIGHT]
    assert relative_difference(forward, -reverse) < 1e-2, (
        f'J(+0.05) = {forward:.6e} but J(-0.05) = {reverse:.6e}; '
        f'a symmetric device must give J(-V) = -J(V)')


# ---------------------------------------------------------------------------
# I2 / I3 -- equilibrium identities in the spatial profile
# ---------------------------------------------------------------------------

def _uniform_equilibrium_profile(runner, tmp_path):
    """Solve a uniformly doped single-gap device and return its V=0 profile.

    Uniform on purpose: with no junction the equilibrium identities hold
    everywhere except within a screening length of the contacts, so the
    assertions need no special-casing of a depletion region.
    """
    config = make_isotype_config()
    for layer in config['layers']:
        layer['properties']['poisson/static_rho'] = {'value': 1.0e17, 'unit': 'cm^-3'}
        layer['properties']['CB/energy_level'] = {'value': 1.42, 'unit': 'eV'}
        layer['properties']['VB/energy_level'] = {'value': 0.0, 'unit': 'eV'}
    config['simulation']['voltage_sweep']['values'] = [0.0]
    config['simulation']['output']['spatial_profiles'] = True
    return read_spatial(run_project(runner, config, tmp_path))


@pytest.mark.solver
@pytest.mark.slow
def test_np_product_equals_ni_squared_at_equilibrium(runner, tmp_path):
    """I2: n*p = N_C N_V exp(-Eg/kT) everywhere, at equilibrium.

    Exact for Boltzmann statistics, and independent of doping, position and
    band bending -- which is what makes it sharp. It exercises the u <-> quasi
    Fermi level conversions directly: an error there moves n and p in ways that
    do not cancel in the product.
    """
    profile = _uniform_equilibrium_profile(runner, tmp_path)
    from math import exp

    n = profile['u_CB']
    p = profile['u_VB']
    NC = profile['effective_density_of_states_CB']
    NV = profile['effective_density_of_states_VB']
    band_gap = 1.42  # set on both layers above

    worst = 0.0
    for i in range(len(n)):
        ni_squared = NC[i] * NV[i] * exp(-band_gap / KT_OVER_Q)
        worst = max(worst, relative_difference(n[i] * p[i], ni_squared))
    assert worst < 1e-3, (
        f'n*p departs from ni^2 by up to {worst:.2e} across the device')


@pytest.mark.solver
@pytest.mark.slow
def test_bulk_is_charge_neutral_at_equilibrium(runner, tmp_path):
    """I3: net charge vanishes away from the contacts, at equilibrium.

    Tests that the doping reaches Poisson and that the carrier densities
    respond to it. Only the middle half of the device is checked: charge
    neutrality genuinely fails within a screening length of a contact, so
    asserting it there would be asserting the wrong physics.
    """
    profile = _uniform_equilibrium_profile(runner, tmp_path)

    x = profile['coord_x']
    rho = profile['rho']
    doping = profile['static_rho']
    span = max(x) - min(x)
    interior = [i for i, xi in enumerate(x)
                if min(x) + 0.25 * span <= xi <= min(x) + 0.75 * span]
    assert interior, 'no interior points selected'

    scale = max(abs(doping[i]) for i in interior)
    worst = max(abs(rho[i]) for i in interior) / scale
    assert worst < 1e-6, (
        f'net charge reaches {worst:.2e} of the doping in the device interior; '
        f'the bulk should be neutral at equilibrium')
