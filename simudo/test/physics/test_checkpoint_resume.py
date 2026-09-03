"""Checkpoint and resume: an interrupted run must equal an uninterrupted one.

This is the most-churned part of the runner -- ``f2aa86afab``, ``293fc82224``
and ``8e95684551`` all reworked it during 2026 -- and it is the part with the
worst failure mode. A resume that silently produces slightly different numbers
looks exactly like a resume that worked, and the GUI drives it on every
"continue this sweep".

The invariant is exact and needs no stored values: **a sweep computed in two
goes must equal the same sweep computed in one.** The two runs share no code
path for the resumed portion -- one seeds its state from a solved problem in
memory, the other reloads it from HDF5 -- so agreement is a real statement
about the checkpoint round trip, not a tautology.

How a resume is wired
---------------------
Run 1 sweeps part of the range with ``checkpoint_at_end``, writing
``checkpoints/sim_V=<last>.yaml`` inside its output folder. Run 2 names that
file in ``simulation.resume_from`` (a path *relative to the output folder*),
asks for the full range, and must be given ``new_experiment=False`` so it lands
in the same folder -- otherwise ``ensure_new_dir`` moves it to a fresh one where
its own checkpoint does not exist.

The runner prunes bias points already present in ``sim_V.csv``, so run 2
recomputes nothing and appends only the new points to the same file.
"""

import pytest

from .conftest import make_pn_config, read_jv, run_project

CB_RIGHT = 'avg:current_CB:right_contact'

#: Split here: run 1 stops at 0.2, run 2 carries on to 0.4.
FIRST_LEG = [0.0, 0.1, 0.2]
FULL_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4]


def checkpointing_config(values, resume_from=None):
    """A p-n sweep that checkpoints at its last bias point."""
    config = make_pn_config()
    config['simulation']['voltage_sweep']['values'] = list(values)
    config['simulation']['checkpoints']['voltage_sweep'] = {
        'checkpoint_at_end': True, 'at_values': []}
    # A fixed folder name: the resume has to land in the same place as the run
    # that wrote the checkpoint.
    config['output']['folder'] = 'run'
    if resume_from is not None:
        config['simulation']['resume_from'] = resume_from
    return config


@pytest.fixture(scope='module')
def resumed_and_reference(runner, tmp_path_factory):
    """(resumed sweep, uninterrupted sweep) as J(V) dicts.

    Module-scoped: three solves, and every test here compares the same pair.
    """
    split_dir = tmp_path_factory.mktemp('resumed')
    out = run_project(runner, checkpointing_config(FIRST_LEG), split_dir)
    first_leg = read_jv(out)

    resumed_out = run_project(
        runner,
        checkpointing_config(FULL_SWEEP, resume_from='checkpoints/sim_V=0.2.yaml'),
        split_dir, name='resume.yaml', new_experiment=False)
    resumed = read_jv(resumed_out)

    whole_dir = tmp_path_factory.mktemp('uninterrupted')
    reference = read_jv(run_project(
        runner, checkpointing_config(FULL_SWEEP), whole_dir))

    return first_leg, resumed, reference


# ---------------------------------------------------------------------------

@pytest.mark.solver
@pytest.mark.slow
def test_checkpoint_file_is_written(runner, tmp_path):
    """``checkpoint_at_end`` must actually leave a checkpoint behind.

    Cheap, and it fails first and unambiguously if the write path breaks --
    without it the resume tests below would fail for a reason that looks like a
    resume bug.
    """
    out = run_project(runner, checkpointing_config(FIRST_LEG), tmp_path)
    checkpoint = out / 'checkpoints' / 'sim_V=0.2.yaml'
    assert checkpoint.is_file(), (
        f'no checkpoint at {checkpoint}; directory holds '
        f'{sorted(p.name for p in (out / "checkpoints").iterdir())}')


@pytest.mark.solver
@pytest.mark.slow
def test_resumed_sweep_matches_an_uninterrupted_one(resumed_and_reference):
    """**The point of this file.** Same bias points, same currents.

    The resumed run reloads its state from HDF5; the reference run carries it
    in memory. If the round trip loses anything -- a band, the optical field,
    the quasi-Fermi levels -- the recomputed points diverge from the reference
    even though both runs "succeeded".
    """
    _first_leg, resumed, reference = resumed_and_reference

    assert sorted(resumed) == sorted(reference), (
        f'resumed sweep covered {sorted(resumed)}, '
        f'uninterrupted covered {sorted(reference)}')

    scale = max(abs(row[CB_RIGHT]) for row in reference.values())
    for v in sorted(reference):
        difference = abs(resumed[v][CB_RIGHT] - reference[v][CB_RIGHT])
        assert difference < 1e-6 * scale, (
            f'V={v}: resumed {resumed[v][CB_RIGHT]:.6e} vs uninterrupted '
            f'{reference[v][CB_RIGHT]:.6e}, differing by {difference:.3e} '
            f'({difference / scale:.2e} of the curve scale)')


@pytest.mark.solver
@pytest.mark.slow
def test_resume_preserves_the_points_computed_before_it(resumed_and_reference):
    """The pre-checkpoint points must survive the resume unaltered.

    The runner prunes already-computed bias points and appends only new ones,
    so the early rows in the final CSV are the *original* run's output. If a
    resume recomputed or rewrote them they could drift; this pins that they are
    carried through byte-identically.
    """
    first_leg, resumed, _reference = resumed_and_reference

    for v in sorted(first_leg):
        assert v in resumed, f'V={v} vanished from the sweep after resuming'
        assert resumed[v][CB_RIGHT] == first_leg[v][CB_RIGHT], (
            f'V={v} changed when the run was resumed: '
            f'{first_leg[v][CB_RIGHT]!r} -> {resumed[v][CB_RIGHT]!r}')


@pytest.mark.solver
@pytest.mark.slow
def test_resume_extends_rather_than_restarting(resumed_and_reference):
    """The resumed file must hold the whole sweep, not just the new points.

    Guards the failure where a resume writes a fresh ``sim_V.csv`` into the
    same folder and the earlier points are lost -- which looks like success
    until someone plots the curve.
    """
    first_leg, resumed, _reference = resumed_and_reference
    assert set(first_leg) < set(resumed), (
        f'resume did not extend the sweep: had {sorted(first_leg)}, '
        f'now {sorted(resumed)}')
    assert sorted(resumed) == FULL_SWEEP, (
        f'expected the full sweep {FULL_SWEEP}, got {sorted(resumed)}')
