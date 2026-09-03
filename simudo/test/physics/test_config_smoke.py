"""Tier 1: every configuration option can at least be *built*.

These tests construct the problems and stop. They do not solve, which makes
them fast (well under a second each once dolfin is warm) and makes them cover
the failure mode that motivated the suite: A43, where
``ThermionicHeterojunction`` was registered on the ``'local charge
neutrality'`` goal, whose bands have no mixedqfl machinery, and raised
``AttributeError`` before any solve. Every project using an ``interfaces:``
block was fatally broken and nothing noticed.

Hence :data:`ALL_GOALS` -- building only ``'full'`` would have missed A43
entirely. Any test that builds a configuration should build all three.
"""

import pytest

from .conftest import (
    MATERIALS_DIR,
    make_heterojunction_config,
    make_isotype_config,
    make_pn_config,
)

#: A43 lived on 'local charge neutrality'. Always build all three.
ALL_GOALS = ('local charge neutrality', 'thermal equilibrium', 'full')


def build(runner, config, goals=ALL_GOALS):
    """Build *config* for *goals*, returning the problems. Raises on failure."""
    _builder, problems = runner.build_problems(
        config, library_dir=MATERIALS_DIR, goals=goals)
    return problems


# ---------------------------------------------------------------------------
# The builder itself
# ---------------------------------------------------------------------------

@pytest.mark.solver
def test_build_problems_returns_all_goals(runner):
    problems = build(runner, make_isotype_config())
    assert set(problems) == set(ALL_GOALS)
    for goal, problem in problems.items():
        assert problem.pdd is not None, f'{goal}: no pdd'


@pytest.mark.solver
def test_build_problems_does_not_solve(runner, tmp_path):
    """Building must have no side effects on disk.

    ``run()`` creates directories, writes logs and rewrites the project YAML;
    ``build_problems`` must do none of that, or tests cannot use it freely.
    """
    before = set(tmp_path.iterdir())
    build(runner, make_isotype_config())
    assert set(tmp_path.iterdir()) == before


# ---------------------------------------------------------------------------
# Contact boundary conditions
# ---------------------------------------------------------------------------

@pytest.mark.solver
@pytest.mark.parametrize('bc', [
    'ohmic',
    'zero_current',
    pytest.param({'type': 'surface_recombination',
                  'srv': {'value': 1e5, 'unit': 'cm/s'}}, id='srv-default'),
    pytest.param({'type': 'surface_recombination',
                  'srv': {'value': 1e5, 'unit': 'cm/s'},
                  'method': 'drift_diffusion'}, id='srv-drift_diffusion'),
    pytest.param({'type': 'surface_recombination',
                  'srv': {'value': 1e5, 'unit': 'cm/s'},
                  'method': 'continuity'}, id='srv-continuity'),
    pytest.param({'type': 'surface_recombination',
                  'srv': {'value': 1e5, 'unit': 'cm/s'},
                  'method': 'continuity_corrected'}, id='srv-continuity_corrected'),
    pytest.param({'type': 'surface_recombination',
                  'srv': {'value': 0.0, 'unit': 'cm/s'}}, id='srv-zero'),
])
def test_contact_bc_builds(runner, bc):
    """Each selectable contact BC assembles on the CB at the swept contact.

    ``srv-zero`` is A22: srv=0 must register the zero-current BC rather than
    dividing by s.
    """
    config = make_isotype_config()
    config['contacts']['left']['bands']['CB'] = bc
    build(runner, config)


def _cb_band(pdd):
    for band in pdd.bands:
        if band.name == 'CB':
            return band
    raise AssertionError('no CB band')


@pytest.mark.solver
def test_zero_valued_srv_constant_warns(runner, caplog):
    """A ``dolfin.Constant`` holding 0 must warn in the log.

    A plain 0 is the no-surface-recombination limit and registers the
    zero-current BC (A22). A Constant stays on the Robin path because it is
    mutable and may be updated during the solve. The solve it produces dies
    inside DOLFIN with a division by zero that never mentions srv, so the
    warning goes to the log, where the user will look once it fails.

    Only reachable from the library API: the YAML path parses srv to a plain
    float, so a user writing ``srv: 0`` gets the zero-current BC instead.
    """
    import logging

    import dolfin

    from simudo.physics import SurfaceRecombination

    builder, problems = runner.build_problems(
        make_isotype_config(), library_dir=MATERIALS_DIR, goals=('full',))
    pdd = problems['full'].pdd
    U = pdd.mesh_util.unit_registry

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        SurfaceRecombination(
            band=_cb_band(pdd), boundary=builder.F.left_contact,
            srv=dolfin.Constant(0.0) * U('cm/s')).register()

    warnings = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    assert any('srv' in m for m in warnings), (
        f'no warning mentioning srv; got {warnings}')


@pytest.mark.solver
def test_nonzero_srv_constant_does_not_warn(runner, caplog):
    """The warning must be specific to a Constant that is actually zero.

    Otherwise it would fire on every ramped or spatially varying srv and be
    trained away as noise.
    """
    import logging

    import dolfin

    from simudo.physics import SurfaceRecombination

    builder, problems = runner.build_problems(
        make_isotype_config(), library_dir=MATERIALS_DIR, goals=('full',))
    pdd = problems['full'].pdd
    U = pdd.mesh_util.unit_registry

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        SurfaceRecombination(
            band=_cb_band(pdd), boundary=builder.F.left_contact,
            srv=dolfin.Constant(1e5) * U('cm/s')).register()

    assert not [r for r in caplog.records
                if r.levelno >= logging.WARNING and 'srv' in r.getMessage()]


@pytest.mark.solver
def test_srv_zero_detection_distinguishes_plain_from_constant(runner):
    """The two zero cases are handled differently and must not be confused.

    A plain scalar zero takes the zero-current path; a zero-valued Constant
    does not, and warns instead.
    """
    import dolfin

    from simudo.physics import SurfaceRecombination

    _builder, problems = runner.build_problems(
        make_isotype_config(), library_dir=MATERIALS_DIR, goals=('full',))
    pdd = problems['full'].pdd
    U = pdd.mesh_util.unit_registry
    band = _cb_band(pdd)

    constant_zero = SurfaceRecombination(
        band=band, boundary=None, srv=dolfin.Constant(0.0) * U('cm/s'))
    assert constant_zero._srv_is_zero_valued_constant() is True
    assert constant_zero._srv_is_zero() is False, (
        'a mutable Constant must not be taken for the immutable-zero case')

    plain_zero = SurfaceRecombination(
        band=band, boundary=None, srv=0.0 * U('cm/s'))
    assert plain_zero._srv_is_zero() is True
    assert plain_zero._srv_is_zero_valued_constant() is False


@pytest.mark.solver
def test_unknown_contact_bc_is_rejected(runner):
    """A typo in a BC name must raise, not be silently ignored.

    The A21/A23 family -- configurations that are wrong but produce a
    plausible answer -- is the thing this suite most needs to prevent.
    """
    config = make_isotype_config()
    config['contacts']['left']['bands']['CB'] = 'ohmick'
    with pytest.raises(ValueError, match='Unknown band BC type'):
        build(runner, config)


# ---------------------------------------------------------------------------
# Interface boundary conditions  (the A43 net)
# ---------------------------------------------------------------------------

@pytest.mark.solver
def test_heterojunction_builds_on_all_goals(runner):
    """**This is the A43 regression test.**

    Before the fix, this raised ``AttributeError: 'NondegenerateBand' object
    has no attribute 'mixedqfl_xi'`` on the 'local charge neutrality' goal.
    """
    problems = build(runner, make_heterojunction_config())
    assert set(problems) == set(ALL_GOALS)


@pytest.mark.solver
@pytest.mark.parametrize('bands,enhancement', [
    (['CB'], {'CB': 1.0}),
    (['CB', 'VB'], {'CB': 1.0, 'VB': 1.0}),
    (['CB', 'VB'], {'CB': 1e6, 'VB': 1e6}),
    (['CB', 'VB'], {}),          # missing keys default to 1.0
    (['CB', 'VB'], 10.0),        # scalar back-compat form
], ids=['cb-only', 'cb-vb', 'high-enhancement', 'empty-dict', 'scalar-compat'])
def test_heterojunction_enhancement_forms_build(runner, bands, enhancement):
    config = make_isotype_config()
    config['interfaces'] = [{
        'left': 'wide', 'right': 'narrow',
        'bcs': [{'type': 'ThermionicHeterojunction',
                 'bands': bands, 'HJBC_enhancement': enhancement}],
    }]
    build(runner, config)


@pytest.mark.solver
def test_unknown_interface_region_raises(runner):
    """A typo'd layer name in an interfaces: block raises rather than passing.

    The runner used to carry a "warn and skip" guard here, but CellRegions
    creates a region on attribute access so the guard could never fire; it was
    removed as dead code (A44). Raising is the acceptable behaviour -- what
    matters is that a typo is not silently ignored. The KeyError comes from
    mesh/topology.py rather than from a message the runner writes.
    """
    config = make_isotype_config()
    config['interfaces'] = [{
        'left': 'wide', 'right': 'nonexistent',
        'bcs': [{'type': 'ThermionicHeterojunction', 'bands': ['CB']}],
    }]
    with pytest.raises(KeyError, match='nonexistent'):
        build(runner, config)


# ---------------------------------------------------------------------------
# Band types
# ---------------------------------------------------------------------------

@pytest.mark.solver
@pytest.mark.parametrize('band_type', [
    'nondegenerate', 'boltzmann', 'degenerate', 'parabolic', 'fermi',
])
def test_band_type_aliases_build(runner, band_type):
    """Every documented band-type alias resolves and assembles.

    ``easy_add_band``'s docstring still says the degenerate class is "not
    implemented yet"; it is, and these aliases all map onto it.
    """
    config = make_isotype_config()
    for band in config['bands']:
        band['type'] = band_type
    build(runner, config)


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------

@pytest.mark.solver
def test_every_exported_eop_class_resolves(runner):
    """Every EOP class exported from ``simudo.physics`` is reachable by name.

    ``_get_eop_class`` resolves out of the ``simudo.physics`` namespace, so
    this covers EOP classes added in future without anyone editing this test.
    That is the point: the runner's own ``EOP_CLASSES`` table is documented as
    something you should *not* need to touch, and this pins that promise.
    """
    import simudo.physics as P
    from simudo.physics.electro_optical_process import ElectroOpticalProcess

    exported = [
        name for name in dir(P)
        if isinstance(getattr(P, name), type)
        and issubclass(getattr(P, name), ElectroOpticalProcess)
        and not name.startswith('_')
    ]
    assert exported, 'no EOP classes found in simudo.physics'

    unresolved = []
    for name in exported:
        try:
            assert runner._get_eop_class(name) is getattr(P, name)
        except Exception as exc:                      # noqa: BLE001
            unresolved.append(f'{name}: {exc!r}')
    assert not unresolved, 'EOP classes not resolvable by name:\n' + '\n'.join(unresolved)


@pytest.mark.solver
def test_unknown_process_class_is_rejected(runner):
    config = make_isotype_config()
    config['processes'] = [{'class': 'NoSuchProcess', 'name': 'x',
                            'dst_band': 'CB', 'src_band': 'VB'}]
    with pytest.raises(Exception):
        build(runner, config)


@pytest.mark.solver
@pytest.mark.parametrize('cls_name', [
    'SRHRecombination',
    'ShockleyReadBand2BandTrap',
])
def test_two_band_process_builds(runner, cls_name):
    """Processes that need only a dst/src band pair, on the CB-VB pair."""
    config = make_isotype_config()
    config['processes'] = [{'class': cls_name, 'name': 'proc',
                            'dst_band': 'CB', 'src_band': 'VB'}]
    build(runner, config)


@pytest.mark.solver
def test_static_generation_builds(runner):
    """``StaticGeneration`` assembles from a ``<name>/generation`` spatial rule.

    Construction only, deliberately. A50: the runner wires no ramp for it --
    ``UniformGenerationAdaptiveStepper`` exists for exactly that job and is
    never used -- so generation is switched on abruptly at the first solve and
    fails to converge above ~1e9 cm^-3/s on this device. Below that the
    photocurrent is close enough to the dark residual that an analytic
    q*G*L check would be measuring noise. See doc/release0_7/06_known_issues.md.
    """
    config = make_pn_config()
    for layer in config['layers']:
        layer['properties']['static/generation'] = {
            'value': 1.0e17, 'unit': 'cm^-3/s'}
    config['processes'] = [
        {'class': 'SRHRecombination', 'name': 'SRH',
         'dst_band': 'CB', 'src_band': 'VB'},
        {'class': 'StaticGeneration', 'name': 'static',
         'dst_band': 'CB', 'src_band': 'VB'},
    ]
    problems = build(runner, config)
    assert set(problems) == set(ALL_GOALS)


@pytest.mark.solver
def test_no_processes_builds(runner):
    """A device with no recombination at all is a legitimate configuration."""
    config = make_isotype_config()
    config['processes'] = []
    build(runner, config)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

#: Standalone materials cleared for release in 06 §B.4, with any spatial
#: parameters they declare via ``required_spatial_params``.
RELEASED_MATERIALS = [
    pytest.param('GalliumArsenideMaterial', {}, id='GalliumArsenideMaterial'),
    pytest.param('IndiumArsenideMaterial', {}, id='IndiumArsenideMaterial'),
    pytest.param('SiliconMaterial', {}, id='SiliconMaterial'),
    pytest.param('AluminumArsenideMaterial', {}, id='AluminumArsenideMaterial'),
    pytest.param('AluminumGalliumArsenideAlloy',
                 {'MoleFractionX': {'value': 0.3, 'unit': ''}},
                 id='AluminumGalliumArsenideAlloy'),
]


@pytest.mark.solver
@pytest.mark.parametrize('material_class,extra_properties', RELEASED_MATERIALS)
def test_released_material_builds(runner, material_class, extra_properties):
    """Each standalone material cleared for release loads and assembles.

    ``AluminumGalliumArsenideAlloy`` is the one that matters most: it imports
    its parent binaries, so this covers the relative-import fix from
    5142b8ad97, and it is the file M1/M11 live in.
    """
    config = make_pn_config()
    config['materials'] = {'GaAsLike': {'source': f'library://{material_class}'}}
    if extra_properties:
        config['regions'][0]['properties'].update(extra_properties)
    build(runner, config)


@pytest.mark.solver
def test_ib_test_material_provides_its_documented_keys(runner):
    """``IBTestMaterial`` is a *supplementary* material, not a standalone one.

    Its docstring says it supplies layer properties for the I-region of a
    p-IB-n device, and it defines only the handful of keys below -- notably no
    ``CB/effective_density_of_states`` -- so it cannot be the sole material of
    a device. This test pins what it does provide rather than pretending it is
    a full material.
    """
    from simudo.materials.ib_test_material import IBTestMaterial
    assert issubclass(IBTestMaterial, object)
    source = __import__('inspect').getsource(IBTestMaterial)
    for key in ('CB/vth', 'VB/vth', 'IB/number_of_states', 'IB/mobility0'):
        assert key in source, f'{key} no longer provided by IBTestMaterial'


@pytest.mark.solver
def test_derived_material_overrides_build(runner):
    """A material with inline property overrides, as the GUI emits."""
    config = make_pn_config()
    config['materials'] = {
        'GaAsLike': {
            'source': 'library://GalliumArsenideMaterial',
            'properties': {'CB/MC': {'value': 1.5, 'unit': ''}},
        },
    }
    build(runner, config)


@pytest.mark.solver
def test_unknown_material_is_rejected(runner):
    config = make_pn_config()
    config['materials'] = {'GaAsLike': {'source': 'library://NoSuchMaterial'}}
    with pytest.raises(ImportError):
        build(runner, config)
