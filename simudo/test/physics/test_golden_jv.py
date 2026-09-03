"""Tier 3: a small set of golden J(V) curves.

Deliberately small. Golden values fail whenever the physics legitimately
improves -- and Simudo's physics is being improved right now -- so they earn
their keep only where the *shape* of a curve over a real bias range is the
thing worth pinning, which the Tier 2 invariants cannot express. Everything
that can be stated as an invariant belongs in ``test_invariants.py`` instead.

Tolerance is 1 %, not 1e-12: these guard against a curve changing shape, not
against the last bit moving. Bit-level pinning of ``run()`` is
``test_runner_characterization.py``'s job.

Re-blessing procedure
---------------------
When a change legitimately moves these curves:

1. Confirm it is intended -- check it against a Tier 2 invariant or a hand
   calculation first. A golden test failing is not by itself evidence that the
   new numbers are right.
2. Regenerate with ``print_reference()`` at the bottom of this file.
3. Update ``REFERENCE`` **in the same commit as the change that moved it**,
   never in a separate "fix tests" commit.
4. Put the before/after in the commit message, with the reason.

Following that, the history explains every movement of these numbers. Not
following it turns them into noise that gets re-blessed reflexively, which is
worse than having no golden tests at all.
"""

import pytest

from .conftest import (
    make_heterojunction_config,
    make_pn_config,
    read_jv,
    run_project,
)

CB_RIGHT = 'avg:current_CB:right_contact'

#: Bias points every golden curve is sampled at.
SWEEP = [0.0, 0.2, 0.4, 0.5, 0.6]

#: Fractional tolerance on J at each bias.
RTOL = 1e-2

#: Captured 2026-08-15. Re-captured when the test fixtures' mesh was coarsened
#: (graded factor 1.2, cap 0.1 um) to bring the suite runtime down: J moved by
#: <0.04% at every forward bias, well inside the 1% tolerance below, but the
#: stored values are exact so they had to be refreshed.
REFERENCE = {
    # p-n homojunction in the dark. Spans four decades of J, so it pins the
    # shape of the exponential turn-on, not just a magnitude.
    'pn_dark': {
        0.0: -1.825770919563591e-14,
        0.2: 6.4784414238677875e-06,
        0.4: 0.0003661548441096911,
        0.5: 0.0027510062888056836,
        0.6: 0.02189839675568121,
    },
    # Isotype heterojunction at enhancement 1e6. The well-enforced regime, so
    # the values are independent of which BC formulation is in use -- see
    # 06_known_issues.md section A.3, where the two formulations agree to eight
    # significant figures at this enhancement but differ by 4 % at enhancement 1.
    # That makes this curve a much more stable reference than one taken at the
    # default enhancement.
    'hj_enhancement_1e6': {
        0.0: 0.0012103981763687925,
        0.2: 60302.516673158636,
        0.4: 85629.85320711366,
        0.5: 98276.2797441282,
        0.6: 110919.94370673913,
    },
}


def _pn_dark_config():
    config = make_pn_config()
    config['simulation']['voltage_sweep']['values'] = list(SWEEP)
    return config


def _hj_high_enhancement_config():
    config = make_heterojunction_config()
    config['interfaces'][0]['bcs'][0]['HJBC_enhancement'] = {'CB': 1e6, 'VB': 1e6}
    config['simulation']['voltage_sweep']['values'] = list(SWEEP)
    return config


CONFIGS = {
    'pn_dark': _pn_dark_config,
    'hj_enhancement_1e6': _hj_high_enhancement_config,
}


@pytest.mark.solver
@pytest.mark.slow
@pytest.mark.parametrize('curve', sorted(REFERENCE))
def test_golden_jv_curve(runner, tmp_path, curve):
    jv = read_jv(run_project(runner, CONFIGS[curve](), tmp_path))
    expected = REFERENCE[curve]

    assert sorted(jv) == sorted(expected), (
        f'{curve}: bias points changed; got {sorted(jv)}')

    # Scale absolute agreement at near-zero bias against the curve, since J(0)
    # is a numerical residual whose relative value is meaningless.
    scale = max(abs(j) for j in expected.values())
    for v in sorted(expected):
        got, want = jv[v][CB_RIGHT], expected[v]
        if abs(want) < 1e-9 * scale:
            assert abs(got) < 1e-9 * scale, (
                f'{curve} V={v}: {got:.6e} should still be a residual')
        else:
            assert got == pytest.approx(want, rel=RTOL), (
                f'{curve} V={v}: {got:.6e} vs reference {want:.6e} '
                f'({abs(got - want) / abs(want):.2%} off, tolerance {RTOL:.0%})')


@pytest.mark.solver
@pytest.mark.slow
def test_dark_diode_ideality_factor_is_physical(runner, tmp_path):
    """The dark p-n curve must have an ideality factor between 1 and 2.

    No golden value: n = (q/kT) * dV / d(ln J) is bounded below by 1 (ideal
    diffusion) and above by 2 (SRH recombination in the depletion region)
    for a diode with no series or shunt pathology. The fixture has SRH on with
    1 ns lifetimes, so n near 2 is expected.

    This is the check that would catch the exponential having the wrong slope
    while its magnitude happened to stay plausible -- something the golden
    curve above passes over, since a wrong kT would move every point together.
    """
    from math import log

    jv = read_jv(run_project(runner, _pn_dark_config(), tmp_path))
    kT_over_q = 0.025852  # volts at 300 K

    points = [(v, jv[v][CB_RIGHT]) for v in sorted(jv) if jv[v][CB_RIGHT] > 0]
    assert len(points) >= 3, f'need three forward points, got {points}'

    for (v1, j1), (v2, j2) in zip(points, points[1:]):
        n = (v2 - v1) / (kT_over_q * log(j2 / j1))
        assert 1.0 <= n <= 2.0, (
            f'ideality factor {n:.3f} between V={v1} and V={v2} is outside '
            f'[1, 2]; J went {j1:.6e} -> {j2:.6e}')


def print_reference():  # pragma: no cover - developer utility
    """Regenerate ``REFERENCE``. See the re-blessing procedure above."""
    import tempfile
    from simudo.gui import simudo_1d_runner

    for name, make in sorted(CONFIGS.items()):
        with tempfile.TemporaryDirectory() as tmp:
            jv = read_jv(run_project(simudo_1d_runner, make(), tmp))
        print(f"    {name!r}: {{")
        for v in sorted(jv):
            print(f"        {v}: {jv[v][CB_RIGHT]!r},")
        print("    },")
