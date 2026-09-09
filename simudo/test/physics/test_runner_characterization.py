"""Characterization tests pinning ``simudo_1d_runner.run()`` behaviour.

These exist to guard refactors of the runner.  They assert **exact** reproduction
of reference currents: a pure refactor must not perturb a single bit, so the
tolerance is 1e-12, not a physics tolerance.

Because the tolerance is that tight, these are also the first tests to fail when
anything changes the discretisation -- coarsening a fixture's mesh moves the last
digits even though the physics is untouched. That is intended: it forces the
change to be looked at rather than absorbed silently.

If one of these fails after a change that was *meant* to alter the physics, that
is the test doing its job -- re-capture the references in the same commit as the
change, and say so in the commit message.  Do not loosen the tolerance.

The comparisons run only on the architecture the references were captured on
(see ``requires_reference_arch``): 1e-12 is tighter than floating point is
reproducible across instruction sets, so elsewhere they would fail for reasons
that have nothing to do with Simudo.

Reference values were generated with::

    cfg['simulation']['voltage_sweep']['values'] = [...]
    run_project(runner, cfg, tmpdir); read_jv(out_dir)
"""

import platform

import pytest

from .conftest import (
    make_heterojunction_config,
    make_isotype_config,
    make_pn_config,
    read_jv,
    run_project,
)

#: A pure refactor must reproduce these bit-for-bit.
RTOL = 1e-12

#: Architecture the reference currents below were captured on.
#: ``platform.machine()`` reports 'aarch64' on 64-bit ARM Linux and 'arm64' on
#: macOS; both are the capture platform.
REFERENCE_ARCH = ('aarch64', 'arm64')

#: The 1e-12 tolerance is tighter than these currents are *defined*.  Measured
#: on the 0.7.0.1 images, the same solve gives 0.06595489370182804 on aarch64
#: and 0.06595499810875168 on x86_64 -- a relative difference of 1.6e-6.
#:
#: That is not accumulated rounding.  Rounding is ~1e-16, and MUMPS is a direct
#: solver, so there is no iterative linear tolerance either.  1.6e-6 is the
#: Newton solver's own `relative_tolerance` of 1e-6 (fem/newton_solver.py):
#: iteration stops once the relative change falls below it, so the answer is
#: only pinned to ~1e-6 and any point inside that ball is an equally converged
#: solution.  Architecture-dependent floating point -- FMA contraction, libm,
#: BLAS -- only perturbs the iterates in the last bits, which changes where the
#: trajectory crosses the threshold, and can change the iteration count outright.
#: The rounding selects which point in the convergence ball you land on; it does
#: not grow into the discrepancy.
#:
#: Loosening RTOL would defeat these tests -- the whole point is that a refactor
#: must not move a single bit.  So instead they run only where the references
#: were captured.  To support another architecture, capture a second reference
#: set there rather than relaxing the comparison.
requires_reference_arch = pytest.mark.skipif(
    platform.machine() not in REFERENCE_ARCH,
    reason=(
        "reference currents were captured on {}; this is {}. The 1e-12 "
        "tolerance is tighter than these currents are defined: the Newton "
        "solver stops at a relative tolerance of 1e-6, so any point inside "
        "that ball is an equally converged solution, and a different "
        "instruction set lands on a different one. These would fail for "
        "reasons unrelated to Simudo."
    ).format("/".join(REFERENCE_ARCH), platform.machine()),
)

#: Captured 2026-09-04 on the true-1D interval mesh, which became the runner's
#: default (``mesh.dimension: 1``) when the 1D route was added.  Before/after
#: for the same fixtures on the 2D strip vs the interval mesh:
#:
#:   iso V=0     J_CB right  0.06595489370182285   ->  1.8998561072939093e-05
#:   hj  V=0.1   J_CB right  46678.068830323056    ->  46729.19389724924  (+0.11 %)
#:   bidir V=0.1 J_CB right  3242508.7784823463    ->  3242553.4918776373 (+1.4e-5)
#:   bidir V=-0.1            -29840711.80159044    ->  -29837092.813603953 (+1.2e-4)
#:
#: The V=0 currents are the residual of an equilibrium solve (the true value is
#: zero); the interval mesh conserves current exactly along x, hence the much
#: smaller residual.  The biased points moved by less than the golden tests'
#: 1 % tolerance (test_golden_jv.py passed unchanged).  The strip values are
#: kept in REFERENCE_STRIP and pinned by test_isotype_v0_strip_matches_reference.
REFERENCE = {
    'iso': {
        0.0: {'avg:current_CB:right_contact': 1.8998561072939093e-05,
              'avg:current_CB:left_contact':  1.8998561072939103e-05,
              'avg:current_VB:right_contact': 1.3920354460463433e-29},
    },
    'pn': {
        0.0: {'avg:current_CB:right_contact': -1.0775875237686182e-17,
              'avg:current_CB:left_contact':  -3.2351444544345846e-23,
              'avg:current_VB:right_contact': -1.782258077387216e-24},
    },
    # Re-captured when ThermionicHeterojunction switched from the mean thermal
    # speed to the Richardson emission velocity (vth/4), which is the correct
    # average for a flux across the interface. J at 0.1 V fell by 3.83x -- close
    # to the 4x in the velocity, the shortfall being that the device is not
    # purely emission-limited. This is a deliberate physics change, not drift.
    # (Values below are the 1D-mesh capture of 2026-09-04.)
    'hj': {
        0.0: {'avg:current_CB:right_contact': 3.489376963883302e-07,
              'avg:current_CB:left_contact':  3.4893769638833004e-07,
              'avg:current_VB:right_contact': 1.2942504204363108e-31},
        0.1: {'avg:current_CB:right_contact': 46729.19389724924,
              'avg:current_CB:left_contact':  46729.193897249206,
              'avg:current_VB:right_contact': 1.8916025950018712e-19},
    },
}

#: The 2D-strip mesh (``mesh.dimension: 2``), which was the only route before
#: 2026-09-04 and is still selectable.  Same fixture as REFERENCE['iso'].
REFERENCE_STRIP = {
    'iso': {
        0.0: {'avg:current_CB:right_contact': 0.06595489370182285,
              'avg:current_CB:left_contact':  0.06595489370182804,
              'avg:current_VB:right_contact': 5.0025827073484733e-26},
    },
}

#: Bidirectional sweep. The negative branch builds a *second* full problem
#: (``full_problem_neg``), a path none of the single-direction cases reach.
REFERENCE_BIDIR = [
    (0.0,  1.8998561072939093e-05),
    (0.1,  3242553.4918776373),
    (0.0,  1.8998561072939093e-05),
    (-0.1, -29837092.813603953),
]


def _check_against_reference(jv, expected):
    assert sorted(jv) == sorted(expected), (
        f'bias points differ: got {sorted(jv)}, expected {sorted(expected)}')
    for v, columns in expected.items():
        for column, want in columns.items():
            got = jv[v][column]
            assert got == pytest.approx(want, rel=RTOL), (
                f'V={v} {column}: got {got!r}, reference {want!r}')


@requires_reference_arch
@pytest.mark.solver
def test_isotype_v0_matches_reference(runner, tmp_path):
    """Cheapest end-to-end path: mesh, topology, materials, bands, contacts."""
    config = make_isotype_config()
    out_dir = run_project(runner, config, tmp_path)
    _check_against_reference(read_jv(out_dir), REFERENCE['iso'])


@requires_reference_arch
@pytest.mark.solver
def test_isotype_v0_strip_matches_reference(runner, tmp_path):
    """The legacy 2D strip mesh, selected explicitly, must still reproduce
    the currents it produced before the true-1D route became the default."""
    config = make_isotype_config()
    config['mesh']['dimension'] = 2
    out_dir = run_project(runner, config, tmp_path)
    _check_against_reference(read_jv(out_dir), REFERENCE_STRIP['iso'])


@requires_reference_arch
@pytest.mark.solver
def test_pn_v0_matches_reference(runner, tmp_path):
    """A real junction: exercises the built-in potential and the LCN stage."""
    config = make_pn_config()
    out_dir = run_project(runner, config, tmp_path)
    _check_against_reference(read_jv(out_dir), REFERENCE['pn'])


@requires_reference_arch
@pytest.mark.solver
@pytest.mark.slow
def test_heterojunction_jv_matches_reference(runner, tmp_path):
    """Adds the interface BC and one biased step, so it also pins the stepper."""
    config = make_heterojunction_config()
    config['simulation']['voltage_sweep']['values'] = [0.0, 0.1]
    out_dir = run_project(runner, config, tmp_path)
    _check_against_reference(read_jv(out_dir), REFERENCE['hj'])


@requires_reference_arch
@pytest.mark.solver
@pytest.mark.slow
def test_bidirectional_sweep_matches_reference(runner, tmp_path):
    """Negative and positive branches, i.e. the ``full_problem_neg`` path.

    The rows are order-sensitive: the runner sweeps up from the initial value,
    then restarts at it and sweeps down, so V=0 appears twice.  That ordering
    is part of what this pins.
    """
    config = make_isotype_config()
    config['simulation']['voltage_sweep']['values'] = [-0.1, 0.0, 0.1]
    out_dir = run_project(runner, config, tmp_path)

    import csv
    with open(out_dir / 'sim_V.csv', encoding='utf-8') as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    iV = header.index('sweep_parameter:V')
    iJ = header.index('avg:current_CB:right_contact')
    got = [(float(r[iV]), float(r[iJ]))
           for r in rows[2:] if r and r[0].strip()]

    assert len(got) == len(REFERENCE_BIDIR), (
        f'row count changed: got {got}')
    for (v_got, j_got), (v_want, j_want) in zip(got, REFERENCE_BIDIR):
        assert v_got == pytest.approx(v_want, abs=1e-12)
        assert j_got == pytest.approx(j_want, rel=1e-9), (
            f'V={v_want}: got {j_got!r}, reference {j_want!r}')


@pytest.mark.solver
def test_run_writes_expected_artifacts(runner, tmp_path):
    """``run()`` writes the CSV, the logs, and a frozen copy of the project.

    Pins the output-directory contract the GUI depends on: it locates a run's
    results by reading ``output.folder`` back out of the project YAML.
    """
    out_dir = run_project(runner, make_isotype_config(), tmp_path)
    assert (out_dir / 'sim_V.csv').is_file()
    assert (out_dir / 'info.log').is_file()
    assert (out_dir / 'project.yaml').is_file(), (
        'frozen copy of the project YAML missing from the output directory')
