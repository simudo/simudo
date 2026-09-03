"""Every concrete EOP class, and every alpha/sigma spec form, must build.

Construction only, and deliberately so. This is the A43 net widened: a class can
resolve by name, satisfy ``test_every_exported_eop_class_resolves``, and still
die the moment something tries to build a problem with it. A43 was exactly that
-- ``ThermionicHeterojunction`` resolved fine and then raised on ``register()``.

Three classes had no coverage of this kind until now:
``ShockleyReadTrap2Trap``, ``NonOverlappingTopHatBeerLambert`` and
``NonOverlappingTopHatBeerLambertIB``. The last is what ``gui/test/pn_opt.yaml``
uses, so it is not a hypothetical corner.

``_parse_alpha_spec`` is the other gap: five mutually exclusive spec forms --
``top_hat``, ``constant``, ``energies_eV``, ``wavelengths_nm``, ``table_file``
-- of which only ``top_hat`` was ever exercised. A table fed in the wrong order,
or a CSV column index off by one, would have gone unnoticed.

Note this is *not* the same vocabulary as ``_build_material_callable``, which
takes ``energy_table`` / ``wavelength_table`` / ``formula`` and is how a
*material* supplies alpha(E) or sigma(E). Two parsers, two schemas; the material
one is covered by ``simudo/gui/test/test_material_callables.py``.
"""

import pytest

from .conftest import IB_NUMBER_OF_STATES, make_ib_config, make_pn_config
from .test_config_smoke import ALL_GOALS, build


def _q(value, unit):
    return {'value': value, 'unit': unit}


# ---------------------------------------------------------------------------
# EOP classes that had no configuration coverage
# ---------------------------------------------------------------------------

@pytest.mark.solver
def test_non_overlapping_top_hat_beer_lambert_builds(runner):
    """Constant-alpha Beer-Lambert, clipped to an absorption window.

    ``gui/test/pn_opt.yaml`` uses this class, so it is a shipped configuration
    rather than a library curiosity. It needs an ``<name>/alpha`` spatial
    parameter per region.
    """
    config = make_pn_config()
    for layer in config['layers']:
        layer['properties']['opt_cv/alpha'] = _q(1.0e4, '1/cm')
    config['processes'] = [
        {'class': 'SRHRecombination', 'name': 'SRH',
         'dst_band': 'CB', 'src_band': 'VB'},
        {'class': 'NonOverlappingTopHatBeerLambert', 'name': 'opt_cv',
         'dst_band': 'CB', 'src_band': 'VB'},
    ]
    config['optical_field_sets'] = [{
        'name_prefix': 'sun', 'direction': '+x', 'N_bins': 2,
        'E_min': _q(1.0, 'eV'), 'E_max': _q(3.0, 'eV'),
        'concentration': 1.0, 'spectrum': 'am15g'}]
    config['simulation']['intensity_ramp'] = {
        'enabled': True, 'selfconsistent_optics': False}
    assert set(build(runner, config)) == set(ALL_GOALS)


@pytest.mark.solver
def test_non_overlapping_top_hat_beer_lambert_ib_builds(runner):
    """The IB flavour of the same class, taking a cross-section instead.

    Needs ``<name>/sigma_opt`` rather than ``<name>/alpha``, because for an IB
    transition the absorption scales with the occupancy.
    """
    config = make_ib_config()
    for layer in config['layers']:
        layer['properties']['opt_ib/sigma_opt'] = _q(1.0e-14, 'cm^2')
    config['processes'].append(
        {'class': 'NonOverlappingTopHatBeerLambertIB', 'name': 'opt_ib',
         'dst_band': 'CB', 'src_band': 'IB', 'trap_band': 'IB',
         'radiative_recombination': False})
    assert set(build(runner, config)) == set(ALL_GOALS)


@pytest.mark.solver
def test_shockley_read_trap_to_trap_builds(runner):
    """Trapping between two *sharp* bands.

    Its docstring requires both the sending and receiving bands to be trap
    (sharp) bands, so the fixture carries a second intermediate band -- the
    only configuration in the suite that does. Requires a
    ``<dst_band>/capture_rate``.
    """
    config = make_ib_config()
    config['bands'].append(
        {'name': 'IB2', 'type': 'intermediate', 'sign': -1, 'extent': 'all'})
    for layer in config['layers']:
        layer['properties'].update({
            'IB2/number_of_states': _q(IB_NUMBER_OF_STATES, 'cm^-3'),
            'IB2/mobility0': _q(500.0, 'cm^2/V/s'),
            'IB2/simple_mobility_model': _q(True, ''),
            't2t/CB/capture_rate': _q(1.0e-9, 'cm^3/s'),
            't2t/IB/capture_rate': _q(1.0e-9, 'cm^3/s'),
            't2t/IB2/capture_rate': _q(1.0e-9, 'cm^3/s'),
        })
    config['regions'][0]['properties'].update({
        'IB2/energy_level': _q(1.5, 'eV'),
        'IB2/degeneracy': _q(1, ''),
    })
    config['processes'].append(
        {'class': 'ShockleyReadTrap2Trap', 'name': 't2t',
         'dst_band': 'IB2', 'src_band': 'IB', 'trap_band': 'IB'})
    assert set(build(runner, config)) == set(ALL_GOALS)


# ---------------------------------------------------------------------------
# alpha / sigma spec forms
# ---------------------------------------------------------------------------

def _beer_lambert_config(alpha_spec):
    config = make_pn_config()
    config['processes'] = [
        {'class': 'SRHRecombination', 'name': 'SRH',
         'dst_band': 'CB', 'src_band': 'VB'},
        {'class': 'BeerLambert', 'name': 'opt_cv',
         'dst_band': 'CB', 'src_band': 'VB',
         'alpha': {layer['name']: alpha_spec for layer in config['layers']}},
    ]
    config['optical_field_sets'] = [{
        'name_prefix': 'sun', 'direction': '+x', 'N_bins': 2,
        'E_min': _q(1.0, 'eV'), 'E_max': _q(3.0, 'eV'),
        'concentration': 1.0, 'spectrum': 'am15g'}]
    config['simulation']['intensity_ramp'] = {
        'enabled': True, 'selfconsistent_optics': False}
    return config


@pytest.mark.solver
def test_top_hat_alpha_spec_builds(runner):
    """The form every other test uses, asserted here for completeness."""
    build(runner, _beer_lambert_config({'top_hat': {
        'E_low': _q(1.42, 'eV'), 'E_high': _q(1e9, 'eV'),
        'value': 1.0e4, 'unit': 'cm^-1'}}))


@pytest.mark.solver
def test_constant_alpha_spec_builds(runner):
    """Energy-independent alpha, its own branch separate from ``top_hat``."""
    build(runner, _beer_lambert_config(
        {'constant': {'value': 1.0e4, 'unit': '1/cm'}}))


@pytest.mark.solver
def test_energies_ev_table_alpha_spec_builds(runner):
    """alpha tabulated against photon energy.

    Rising with energy, as a real absorption edge does, so a table fed in the
    wrong order would give a physically backwards alpha rather than merely a
    different one.
    """
    build(runner, _beer_lambert_config({
        'energies_eV': [1.0, 1.4, 2.0, 3.0],
        'alphas': [0.0, 1.0e3, 1.0e4, 3.0e4],
        'unit': '1/cm'}))


@pytest.mark.solver
def test_wavelengths_nm_table_alpha_spec_builds(runner):
    """alpha tabulated against wavelength.

    Wavelength runs opposite to energy, so this branch has to handle the
    reversed ordering. The same physical edge as the energy table above,
    expressed in nm.
    """
    build(runner, _beer_lambert_config({
        'wavelengths_nm': [400.0, 620.0, 886.0, 1240.0],
        'alphas': [3.0e4, 1.0e4, 1.0e3, 0.0],
        'unit': '1/cm'}))


@pytest.mark.solver
def test_table_file_alpha_spec_builds(runner, tmp_path):
    """alpha read from a two-column CSV.

    The only spec form that touches the filesystem, and the only one with
    configurable column indices.
    """
    table = tmp_path / 'alpha.csv'
    table.write_text('# energy_eV, alpha_per_cm\n'
                     '1.0,0.0\n1.4,1.0e3\n2.0,1.0e4\n3.0,3.0e4\n')
    build(runner, _beer_lambert_config(
        {'table_file': str(table), 'unit': '1/cm'}))


@pytest.mark.solver
def test_unknown_alpha_spec_type_is_rejected(runner):
    """A misspelled spec type must raise rather than silently give no alpha.

    A21's family: a missing alpha used to mean zero optical generation with no
    diagnostic. The parser should refuse the input rather than let the run
    proceed with nothing absorbing.
    """
    with pytest.raises(ValueError, match='Cannot parse alpha spec'):
        build(runner, _beer_lambert_config({'enrgies_eV': [1.0, 2.0]}))
