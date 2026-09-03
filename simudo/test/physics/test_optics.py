"""I4: Beer-Lambert absorption against its analytic solution.

The only *exactly* analytic check available anywhere in the suite. With a
constant absorption coefficient and self-consistent optics off, the optical
problem has a closed-form solution

.. math::  \\Phi(x) = \\Phi_0 e^{-\\alpha x}

so the profile can be compared with the answer rather than with a previous run.
Everything else in Tier 2 compares one configuration against another; this one
compares against arithmetic.

Fossil ticket b833c483b0
------------------------
*"check why optical generation rate is not correct near back surface"* --
mwilkins, 2018-10-25, Code Defect / Open / Important:

    VB-CB optical generation decays at the incorrect rate near back surface.
    Can be verified by plotting optg_cv vs. x on a log scale, and comparing
    with exp(-alpha_cv x). Not immediately critical, as the rate has decayed to
    a negligible level before the problem becomes apparent.

Re-measured 2026-08-15 at check-in 11febfab3a, over 10 e-foldings
(alpha = 1e5 cm^-1 across a 1 um device). The reported deviation is real and
reproduces -- and it is **second-order discretization error, not a defect**:

===========  =======================
mesh h (um)  worst deviation
===========  =======================
0.01         2.31e-2
0.005        5.68e-3   (ratio 4.07)
0.0025       1.41e-3   (ratio 4.03)
===========  =======================

Halving the mesh quarters the error, i.e. clean O(h^2) convergence to the
analytic profile. The deviation is largest at the back surface only because
that is where the accumulated error is largest relative to an exponentially
small local value -- which is also why mwilkins observed it there and judged it
non-critical.

:py:func:`test_beer_lambert_error_is_second_order_in_mesh_size` is the test
that pins this. If the ticket's symptom were a genuine defect -- a wrong
coefficient, a boundary term, an off-by-one in the upwinding -- the error would
stall at some floor instead of falling as h^2, and that test would fail.
**Recommend closing b833c483b0 as not-a-defect.**
"""

import math

import pytest

from .conftest import make_pn_config, read_spatial, run_project

#: cm^-1. Over the 1 um device below this is 10 e-foldings, so the profile
#: spans four and a half decades -- enough for a rate error to be unmistakable.
ALPHA_CM = 1e5

#: The same in um^-1, which is what coord_x is measured in.
ALPHA_UM = ALPHA_CM * 1e-4

PHI = 'opt_Phi_mono_000'


def make_beer_lambert_config(mesh_h):
    """A p-n device under monochromatic illumination with constant alpha.

    Self-consistent optics is off, so the optical problem is decoupled from the
    carrier densities and its solution is exactly ``Phi_0 exp(-alpha x)``.
    ``alpha`` is given as a top hat spanning all photon energies so that every
    bin sees the same coefficient, making the analytic form exact regardless of
    the spectrum used to set the incident flux.
    """
    config = make_pn_config()
    for layer in config['layers']:
        layer['thickness'] = {'value': 0.5, 'unit': 'um'}
        layer['properties']['CB/energy_level'] = {'value': 1.42, 'unit': 'eV'}
        layer['properties']['VB/energy_level'] = {'value': 0.0, 'unit': 'eV'}
        layer['mesh'] = {'start': mesh_h, 'factor': 1.0}

    alpha_spec = {
        layer['name']: {'top_hat': {
            'E_low': {'value': 0.0, 'unit': 'eV'},
            'E_high': {'value': 1e9, 'unit': 'eV'},
            'value': ALPHA_CM, 'unit': 'cm^-1'}}
        for layer in config['layers']}

    # BeerLambert enables Shockley-van Roosbroeck radiative recombination by
    # default, which needs a refractive index; supply a realistic one.
    config['regions'][0]['properties']['refractive_index'] = {
        'value': 3.5, 'unit': ''}

    config['processes'] = [
        {'class': 'SRHRecombination', 'name': 'SRH',
         'dst_band': 'CB', 'src_band': 'VB'},
        # Emission off.  These tests compare against the analytic decoupled
        # solution, and the output column 'g_opt_cv_<band>' is the process's
        # *net* contribution -- absorption minus spontaneous emission -- so
        # with SVR on it is no longer alpha*Phi and the comparison is not the
        # one the tests are written to make.  Absorption itself is unaffected,
        # so only the generation test actually breaks; the whole fixture is
        # kept on the pure optical problem rather than splitting it in two.
        {'class': 'BeerLambert', 'name': 'opt_cv',
         'dst_band': 'CB', 'src_band': 'VB', 'alpha': alpha_spec,
         'radiative_recombination': False},
    ]
    config['optical_field_sets'] = [{
        'name_prefix': 'mono', 'direction': '+x', 'N_bins': 1,
        'E_min': {'value': 1.5, 'unit': 'eV'},
        'E_max': {'value': 2.0, 'unit': 'eV'},
        'concentration': 1.0, 'spectrum': 'am15g'}]
    config['simulation']['intensity_ramp'] = {
        'enabled': True, 'selfconsistent_optics': False}
    config['simulation']['voltage_sweep'] = {
        'enabled': True, 'values': [0.0], 'selfconsistent_optics': False}
    config['simulation']['output']['spatial_profiles'] = True
    config['mesh']['max_edge_length'] = mesh_h
    return config


def worst_deviation_from_exponential(profile, column=PHI):
    """Largest |computed / analytic - 1| over the device."""
    x = profile['coord_x']
    values = profile[column]
    front = values[0]
    assert front > 0, 'no illumination at the front surface'
    return max(
        abs((values[i] / front) / math.exp(-ALPHA_UM * x[i]) - 1.0)
        for i in range(len(x)))


@pytest.fixture(scope='module')
def beer_lambert_profiles(runner, tmp_path_factory):
    """Solve the same optical problem on two meshes. Shared: each run is slow."""
    profiles = {}
    for mesh_h in (0.01, 0.005):
        directory = tmp_path_factory.mktemp(f'bl_{mesh_h}')
        profiles[mesh_h] = read_spatial(
            run_project(runner, make_beer_lambert_config(mesh_h), directory))
    return profiles


@pytest.mark.solver
@pytest.mark.slow
def test_beer_lambert_flux_follows_the_analytic_exponential(beer_lambert_profiles):
    """I4: Phi(x) must equal Phi_0 exp(-alpha x).

    Tolerance is set at the *measured* discretization error for this mesh
    rather than at an aspirational value, so this test says "the solver is as
    accurate as its discretization allows", not "the solver is perfect".
    The mesh-independent statement is the convergence test below.
    """
    worst = worst_deviation_from_exponential(beer_lambert_profiles[0.005])
    assert worst < 1e-2, (
        f'optical flux departs from exp(-alpha x) by up to {worst:.2e} '
        f'at mesh h=0.005 um')


@pytest.mark.solver
@pytest.mark.slow
def test_generation_rate_follows_the_analytic_exponential(beer_lambert_profiles):
    """The generation rate inherits the flux profile.

    This is the quantity fossil ticket b833c483b0 was written about -- the
    ticket says to plot ``optg_cv`` against ``exp(-alpha_cv x)``. Checking it
    as well as the flux distinguishes an error in the optical solve from one in
    how absorption is turned into carrier generation.
    """
    profile = beer_lambert_profiles[0.005]
    worst = worst_deviation_from_exponential(profile, column='g_opt_cv_CB')
    assert worst < 1e-2, (
        f'optical generation departs from exp(-alpha x) by up to {worst:.2e}')


@pytest.mark.solver
@pytest.mark.slow
def test_beer_lambert_error_is_second_order_in_mesh_size(beer_lambert_profiles):
    """**The ticket b833c483b0 test.** The error must fall as h^2.

    This is what separates "discretization error" from "defect". A wrong
    coefficient, a mishandled boundary term or an off-by-one in the upwinding
    would leave an error floor that refinement cannot remove; convergence at
    the expected order says the discretization is right and only finite.

    Measured 4.07x and 4.03x for two successive halvings, against the 4x an
    O(h^2) scheme gives. The bound below is deliberately loose (>3) so that
    normal solver noise does not make this flaky, while still failing outright
    if the error stalls.
    """
    coarse = worst_deviation_from_exponential(beer_lambert_profiles[0.01])
    fine = worst_deviation_from_exponential(beer_lambert_profiles[0.005])

    assert fine < coarse, (
        f'refining the mesh did not reduce the error: {coarse:.3e} -> '
        f'{fine:.3e}. That is the signature of a defect rather than '
        f'discretization -- see ticket b833c483b0.')
    ratio = coarse / fine
    assert ratio > 3.0, (
        f'error fell only {ratio:.2f}x for a 2x refinement ({coarse:.3e} -> '
        f'{fine:.3e}); second order would give 4x. An error that stalls under '
        f'refinement means ticket b833c483b0 is a real defect after all.')


@pytest.mark.solver
@pytest.mark.slow
def test_absorption_is_monotonic_through_the_device(beer_lambert_profiles):
    """Flux must decrease monotonically along the propagation direction.

    Cheap, and independent of the analytic form: a negative or oscillating
    flux would be unphysical whatever the decay rate. Guards the case where a
    profile happens to match the exponential on average while oscillating
    about it.
    """
    profile = beer_lambert_profiles[0.005]
    x, phi = profile['coord_x'], profile[PHI]
    for i in range(1, len(x)):
        assert phi[i] <= phi[i - 1] * (1 + 1e-9), (
            f'optical flux increased between x={x[i-1]:.4f} and x={x[i]:.4f}: '
            f'{phi[i-1]:.6e} -> {phi[i]:.6e}')
        assert phi[i] > 0, f'optical flux went non-positive at x={x[i]:.4f}'
