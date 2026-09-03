"""Shared fixtures and config builders for the solver tests.

The tests here build project configurations **as Python dicts** rather than
storing dozens of near-identical YAML files.  ``run_project`` writes the dict
to a temporary directory before invoking the runner, which matters: the runner
rewrites its input YAML in place (it rewrites ``output.folder`` after
``ensure_new_dir`` increments it), so a fixture file handed to it directly
would be modified.

Every solver test is marked ``solver``; tests that sweep bias are additionally
marked ``slow``.  Run only the fast ones with::

    python3 -m pytest simudo/test/ -q -m 'not slow'
"""

import csv
import os
from pathlib import Path

import pytest
import yaml

#: Directory holding the released material modules, for ``library://`` sources.
MATERIALS_DIR = str(Path(__file__).resolve().parents[2] / 'materials')


# ---------------------------------------------------------------------------
# Importing the runner
# ---------------------------------------------------------------------------

def _import_runner():
    """Import the 1D runner, skipping the whole module if dolfin is absent."""
    dolfin = pytest.importorskip(
        'dolfin', reason='solver tests require a FEniCS/dolfin installation')
    from simudo.gui import simudo_1d_runner
    return simudo_1d_runner


@pytest.fixture(scope='session')
def runner():
    return _import_runner()


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _q(value, unit):
    return {'value': value, 'unit': unit}


def make_isotype_config(**overrides):
    """An isotype n-N heterojunction: the cheapest device that has an interface.

    Both sides n-type, so there is no p-n junction and the equilibrium solve is
    fast (~3 s to V=0, against ~12 s for a p-n junction).  Behaviour under bias
    is dominated by transport across the conduction-band spike, which makes it
    the natural vehicle for heterojunction-BC tests.

    Band edges are set per layer rather than taken from a library alloy so the
    offset is exact and known: CB 1.42 -> 1.70 eV, VB 0.00 -> -0.10 eV.
    """
    config = {
        'layers': [
            _layer('wide', 'AlGaAsLike', rho=1.0e17, cb=1.70, vb=-0.10),
            _layer('narrow', 'GaAsLike', rho=1.0e17, cb=1.42, vb=0.0),
        ],
        'regions': [{
            'name': 'domain',
            'properties': {
                'temperature': _q(300.0, 'K'),
                'SRH/energy_level': _q(0.7, 'eV'),
                'SRH/CB/tau': _q(1.0e-9, 's'),
                'SRH/VB/tau': _q(1.0e-9, 's'),
            },
        }],
        'bands': [
            {'name': 'CB', 'type': 'nondegenerate', 'sign': -1, 'extent': 'all'},
            {'name': 'VB', 'type': 'nondegenerate', 'sign': 1, 'extent': 'all'},
        ],
        'processes': [
            {'class': 'SRHRecombination', 'name': 'SRH',
             'dst_band': 'CB', 'src_band': 'VB'},
        ],
        'materials': {
            'GaAsLike': {'source': 'library://GalliumArsenideMaterial'},
            'AlGaAsLike': {'source': 'library://GalliumArsenideMaterial'},
        },
        'contacts': {
            'left':  {'role': 'swept',     'bands': {'CB': 'ohmic', 'VB': 'ohmic'}},
            'right': {'role': 'reference', 'bands': {'CB': 'ohmic', 'VB': 'ohmic'}},
        },
        'simulation': {
            'intensity_ramp': {'enabled': False, 'selfconsistent_optics': False},
            'voltage_sweep': {'enabled': True, 'values': [0.0],
                              'selfconsistent_optics': False},
            'output': {'spatial_profiles': False, 'xdmf_mesh': False},
            'checkpoints': {
                'intensity_ramp': {'checkpoint_at_end': False, 'at_values': []},
                'voltage_sweep':  {'checkpoint_at_end': False, 'at_values': []},
                'directory': 'checkpoints',
            },
        },
        'output': {'folder': 'out'},
        'mesh': {'max_edge_length': 0.1},
    }
    return _apply_overrides(config, overrides)


def make_heterojunction_config(**overrides):
    """:func:`make_isotype_config` plus a ThermionicHeterojunction on CB+VB."""
    config = make_isotype_config()
    config['interfaces'] = [{
        'left': 'wide', 'right': 'narrow',
        'bcs': [{'type': 'ThermionicHeterojunction',
                 'bands': ['CB', 'VB'],
                 'HJBC_enhancement': {'CB': 1.0, 'VB': 1.0}}],
    }]
    return _apply_overrides(config, overrides)


def make_pn_config(**overrides):
    """A p-n homojunction in a single material, with no interface.

    Slower than the isotype device (~12 s to equilibrium) because of the
    built-in potential, so prefer the isotype config where the test does not
    specifically need a junction.
    """
    config = make_isotype_config()
    config['layers'] = [
        _layer('p', 'GaAsLike', rho=-1.0e17, cb=1.42, vb=0.0),
        _layer('n', 'GaAsLike', rho=1.0e17, cb=1.42, vb=0.0),
    ]
    config['materials'] = {'GaAsLike': {'source': 'library://GalliumArsenideMaterial'}}
    return _apply_overrides(config, overrides)


#: Intermediate-band density used by :func:`make_ib_config`, cm^-3.
IB_NUMBER_OF_STATES = 2.5e17


def make_ib_config(filling=0.5, sigma=1e-13, **overrides):
    """A p-i-n-free slab with an intermediate band, for IBBeerLambert tests.

    The IB spans the whole device so its filling -- and hence the absorption
    coefficient -- is uniform, which is what makes the Beer-Lambert form exact.

    *filling* is the IB filling fraction f_IB = u_IB / IB.number_of_states, set
    by charge neutrality: donors at ``filling * number_of_states`` must be
    balanced by that many electrons in the IB. Passing 0.5 gives a half-filled
    band, which is the interesting case -- at f = 0 or 1 the absorption
    coefficient's dependence on filling drops out and the test would not
    exercise it.

    All material properties are set inline rather than through a library
    material, so the device is self-contained and its band structure is exactly
    what the test asserts against.

    Radiative recombination is off: it would re-emit into the field being
    measured and destroy the closed-form profile.
    """
    def region_properties():
        return {
            'temperature': _q(300.0, 'K'),
            'CB/energy_level': _q(2.0, 'eV'),
            'IB/energy_level': _q(1.2, 'eV'),
            'VB/energy_level': _q(0.0, 'eV'),
            'IB/degeneracy': _q(1, ''),
            'CB/effective_density_of_states': _q(2.0e19, 'cm^-3'),
            'VB/effective_density_of_states': _q(2.0e19, 'cm^-3'),
            'CB/mobility': _q(500.0, 'cm^2/V/s'),
            'VB/mobility': _q(500.0, 'cm^2/V/s'),
            'CB/vth': _q(2.0e7, 'cm/s'),
            'VB/vth': _q(2.0e7, 'cm/s'),
            'poisson/permittivity': _q(13.0, 'vacuum_permittivity'),
            'SRH/energy_level': _q(1.0, 'eV'),
            'SRH/CB/tau': _q(1.0e-9, 's'),
            'SRH/VB/tau': _q(1.0e-9, 's'),
        }

    def ib_layer(name):
        return {
            'name': name,
            'thickness': _q(1.5, 'um'),
            # Graded: 5 nm at the layer edges, growing 1.2x per cell up to the
            # global cap. Uniform spacing here is what made this fixture take
            # ~17 minutes; alpha is uniform because the IB filling is uniform,
            # not because the mesh is.
            'mesh': {'start': 0.005, 'factor': 1.2},
            'properties': {
                'poisson/static_rho': _q(filling * IB_NUMBER_OF_STATES,
                                         'elementary_charge/cm^3'),
                'IB/number_of_states': _q(IB_NUMBER_OF_STATES, 'cm^-3'),
                'IB/mobility0': _q(500.0, 'cm^2/V/s'),
                'IB/simple_mobility_model': _q(True, ''),
                'nr_top/sigma_th': _q(5e-18, 'cm^2'),
                'nr_bottom/sigma_th': _q(5e-18, 'cm^2'),
            },
        }

    layer_names = ('front', 'back')
    config = {
        'layers': [ib_layer(name) for name in layer_names],
        'regions': [{'name': 'domain', 'properties': region_properties()}],
        'bands': [
            {'name': 'CB', 'type': 'nondegenerate', 'sign': -1, 'extent': 'all'},
            {'name': 'IB', 'type': 'intermediate', 'sign': -1, 'extent': 'all'},
            {'name': 'VB', 'type': 'nondegenerate', 'sign': 1, 'extent': 'all'},
        ],
        'processes': [
            {'class': 'SRHRecombination', 'name': 'SRH',
             'dst_band': 'CB', 'src_band': 'VB'},
            {'class': 'NonRadiativeTrap', 'name': 'nr_top',
             'dst_band': 'CB', 'src_band': 'IB', 'trap_band': 'IB'},
            {'class': 'NonRadiativeTrap', 'name': 'nr_bottom',
             'dst_band': 'IB', 'src_band': 'VB', 'trap_band': 'IB'},
            {'class': 'IBBeerLambert', 'name': 'opt_ci',
             'dst_band': 'CB', 'src_band': 'IB', 'trap_band': 'IB',
             'radiative_recombination': False,
             'sigma': {name: {'top_hat': {
                 'E_low': _q(0.0, 'eV'), 'E_high': _q(1e9, 'eV'),
                 'value': sigma, 'unit': 'cm^2'}} for name in layer_names}},
        ],
        'materials': {},
        # 0.9 eV excites IB -> CB (a 0.8 eV gap) but not VB -> CB (2.0 eV), so
        # the IB transition is the only absorber in the device.
        'optical_field_sets': [{
            'name_prefix': 'mono', 'direction': '+x', 'N_bins': 1,
            'E_min': _q(0.85, 'eV'), 'E_max': _q(0.95, 'eV'),
            'concentration': 1.0, 'spectrum': 'am15g'}],
        'contacts': {
            'left':  {'role': 'swept',     'bands': {'CB': 'ohmic', 'VB': 'ohmic'}},
            'right': {'role': 'reference', 'bands': {'CB': 'ohmic', 'VB': 'ohmic'}},
        },
        'simulation': {
            'intensity_ramp': {'enabled': True, 'selfconsistent_optics': False},
            'voltage_sweep': {'enabled': True, 'values': [0.0],
                              'selfconsistent_optics': False},
            'output': {'spatial_profiles': True, 'xdmf_mesh': False},
            'checkpoints': {
                'intensity_ramp': {'checkpoint_at_end': False, 'at_values': []},
                'voltage_sweep':  {'checkpoint_at_end': False, 'at_values': []},
                'directory': 'checkpoints'},
        },
        'output': {'folder': 'out'},
        'mesh': {'max_edge_length': 0.1},
    }
    return _apply_overrides(config, overrides)


def _layer(name, material, rho, cb, vb, thickness=0.25):
    return {
        'name': name,
        'thickness': _q(thickness, 'um'),
        'material': material,
        'mesh': {'start': 0.001, 'factor': 1.2},
        'properties': {
            'poisson/static_rho': _q(rho, 'cm^-3'),
            'CB/energy_level': _q(cb, 'eV'),
            'VB/energy_level': _q(vb, 'eV'),
        },
    }


def _apply_overrides(config, overrides):
    """Apply ``dotted.path=value`` overrides, e.g. ``mesh__max_edge_length``.

    Keys use ``__`` as the path separator so they are valid Python kwargs.
    A value of ``None`` deletes the key.
    """
    for dotted, value in overrides.items():
        parts = dotted.split('__')
        node = config
        for part in parts[:-1]:
            node = node[part]
        if value is None:
            node.pop(parts[-1], None)
        else:
            node[parts[-1]] = value
    return config


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_project(runner, config, tmp_path, name='project.yaml', new_experiment=True):
    """Write *config* to *tmp_path* and run it.  Returns the output directory.

    The runner resolves a relative ``output.folder`` against the YAML's own
    directory, so the run lands inside *tmp_path* and leaves the source tree
    untouched.

    Pass ``new_experiment=False`` to resume: the runner then writes into the
    folder named by ``output.folder`` instead of incrementing it, which is what
    a ``simulation.resume_from`` run needs -- the checkpoint path is resolved
    relative to that same folder, so a resume that landed in a fresh directory
    would not find its own checkpoint.  This mirrors the runner's ``--resume``
    flag.
    """
    yaml_path = Path(tmp_path) / name
    with open(yaml_path, 'w', encoding='utf-8') as fh:
        yaml.safe_dump(config, fh, sort_keys=False)

    runner.run(str(yaml_path), library_dir=MATERIALS_DIR,
               new_experiment=new_experiment)

    # run() may increment the folder; read back where it actually landed.
    with open(yaml_path, encoding='utf-8') as fh:
        actual = yaml.safe_load(fh)['output']['folder']
    return Path(tmp_path) / actual


def read_jv(out_dir, prefix='sim'):
    """Parse ``<prefix>_V.csv`` into ``{V: {column: value}}``.

    The second row of the runner's CSV is a units row, not data.
    """
    path = Path(out_dir) / f'{prefix}_V.csv'
    with open(path, encoding='utf-8') as fh:
        rows = list(csv.reader(fh))
    header, data = rows[0], [r for r in rows[2:] if r and r[0].strip()]
    iV = header.index('sweep_parameter:V')
    out = {}
    for row in data:
        out[round(float(row[iV]), 6)] = {
            col: float(val) for col, val in zip(header, row)
            if _is_float(val)
        }
    return out


def read_spatial(out_dir, value='0', prefix='sim', parameter='V'):
    """Read the spatial profile written at one bias point.

    Returns ``{column: [values...]}``, ordered by row. Requires
    ``simulation.output.spatial_profiles: true``.

    Unlike the J(V) file this one has no units row -- data starts at row 1.
    The runner appends a ``.0`` rank suffix, so the name is globbed.
    """
    matches = sorted(Path(out_dir).glob(f'{prefix}_{parameter}={value}.csv.*'))
    if not matches:
        raise FileNotFoundError(
            f'no spatial profile for {parameter}={value} in {out_dir}; '
            f'is simulation.output.spatial_profiles enabled? '
            f'found: {sorted(p.name for p in Path(out_dir).iterdir())}')
    with open(matches[0], encoding='utf-8') as fh:
        rows = list(csv.reader(fh))
    header, data = rows[0], rows[1:]
    out = {name: [] for name in header}
    for row in data:
        if not row or not row[0].strip():
            continue
        for name, cell in zip(header, row):
            out[name].append(float(cell))
    return out


def _is_float(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False
