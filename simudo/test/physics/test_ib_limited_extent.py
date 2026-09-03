"""An intermediate band confined to part of the device.

Created by Claude Opus 5 (Anthropic), 2026-08-17.

Every other IB test puts the IB across the whole device (``extent: all``),
which is the one geometry where ``IBBeerLambert`` used to work. A real IB
solar cell has the IB in one layer between ordinary p and n regions, and that
configuration failed outright:

    Unable to solve linear system using PETSc Krylov solver.
    Reason: DIVERGED_PC_FAILED, residual norm ||r|| = 0.000000e+00

``||r|| = 0`` is the signature of NaN reaching the matrix. ``get_alpha``
returned ``sigma * N_IB * f_IB`` with ``f_IB = u_active / N_IB``; the two
``N_IB`` cancel, but outside the IB's extent ``N_IB`` is zero, so the
expression evaluated ``0 * (0/0) = NaN`` on every cell where the IB does not
exist. Introduced by cdc1d9bab5, whose message called the rewrite
"numerically identical to before" -- true wherever N_IB > 0, which is
everywhere in the tests that existed at the time.

``NonOverlappingTopHatBeerLambertIB`` never had the bug: it has always
returned ``sigma * trap_filling_factor`` with no cancelling division. It is
tested here alongside, since it is what proved the geometry itself was sound.
"""

import pytest

from .conftest import make_ib_config, run_project
from .test_config_smoke import ALL_GOALS, build


def limit_ib_to_front(config):
    """Confine the IB, and its IB-only parameters, to the 'front' layer."""
    for band in config['bands']:
        if band['name'] == 'IB':
            band['extent'] = ['front']
    for layer in config['layers']:
        if layer['name'] != 'front':
            for key in list(layer['properties']):
                if key.startswith('IB/') or key.startswith('nr_'):
                    del layer['properties'][key]
    for proc in config['processes']:
        if 'sigma' in proc:
            proc['sigma'] = {'front': proc['sigma']['front']}
    return config


@pytest.mark.solver
def test_alpha_is_finite_outside_a_limited_extent_ib(runner):
    """The sharp form of the bug, and cheap: no solve, just evaluate alpha.

    Fails with NaN on exactly the cells outside the IB if the cancelling
    division comes back.
    """
    import dolfin
    import numpy as np

    pd = build(runner, limit_ib_to_front(make_ib_config()),
               goals=ALL_GOALS)['full']
    proc = next(p for p in pd.pdd.electro_optical_processes
                if p.name == 'opt_ci')
    alpha = proc.get_alpha(0.9 * pd.unit_registry('eV'))
    values = dolfin.project(
        alpha.m_as('1/cm'),
        dolfin.FunctionSpace(pd.mesh_data.mesh, 'DG', 0)).vector().get_local()

    assert np.isfinite(values).all(), (
        f"alpha is NaN on {np.isnan(values).sum()} of {values.size} cells; "
        "check for a division by N_IB outside the IB extent")
    # Absorption must be zero where there is no IB, and positive where there is.
    assert values.min() == 0.0
    assert values.max() > 0.0


@pytest.mark.solver
@pytest.mark.slow
def test_ib_beer_lambert_solves_with_a_limited_extent_ib(runner, tmp_path):
    """The whole point: the device solves rather than dying in the optics."""
    run_project(runner, limit_ib_to_front(make_ib_config()), tmp_path)


@pytest.mark.solver
@pytest.mark.slow
def test_top_hat_ib_solves_with_a_limited_extent_ib(runner, tmp_path):
    """The same geometry through NonOverlappingTopHatBeerLambertIB, which
    never had the defect. Pins that the geometry itself is supported."""
    config = limit_ib_to_front(make_ib_config())
    for proc in config['processes']:
        if proc['name'] == 'opt_ci':
            proc['class'] = 'NonOverlappingTopHatBeerLambertIB'
            del proc['sigma']
    for layer in config['layers']:
        if layer['name'] == 'front':
            layer['properties']['opt_ci/sigma_opt'] = {
                'value': 1e-13, 'unit': 'cm^2'}
    run_project(runner, config, tmp_path)
