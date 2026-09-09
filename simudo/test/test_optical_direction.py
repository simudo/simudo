"""An optical field's propagation direction must fit the mesh it lives on.

On the 2D strip a layered device could only be lit along x, so scripts say
``direction=(1, 0)`` or ``(-1, 0)``. Those must keep working on a true-1D
interval mesh, which wants ``(1,)`` and ``(-1,)``. A direction with a real
transverse component has no meaning on an interval mesh and must say so.

Created by Claude Opus 5 (Anthropic), 2026-09-08.
"""

import numpy as np
import pytest

from ..physics.optical import adapt_direction_to_gdim


@pytest.mark.parametrize('given, expected', [
    ((1, 0), (1.0,)),
    ((-1, 0), (-1.0,)),
    ((1.0, 0.0), (1.0,)),
    ((1,), (1.0,)),
    ((-1,), (-1.0,)),
    ((1, 0, 0), (1.0,)),
])
def test_directions_along_x_are_accepted_on_an_interval_mesh(given, expected):
    assert tuple(adapt_direction_to_gdim(given, 1)) == expected


@pytest.mark.parametrize('given', [(0, 1), (0, -1), (0.6, 0.8), (1, 0.5)])
def test_transverse_directions_are_rejected_on_an_interval_mesh(given):
    with pytest.raises(ValueError) as exc:
        adapt_direction_to_gdim(given, 1, name='sun_000')
    message = str(exc.value)
    assert 'sun_000' in message
    assert '(1,)' in message and '(-1,)' in message


def test_two_dimensional_meshes_are_untouched():
    for given in ((1, 0), (-1, 0), (0, 1), (0.6, 0.8)):
        np.testing.assert_allclose(adapt_direction_to_gdim(given, 2), given)


def test_too_few_components_is_an_error():
    with pytest.raises(ValueError) as exc:
        adapt_direction_to_gdim((1,), 2)
    assert 'fewer components' in str(exc.value)


def test_field_uses_the_adapted_direction():
    """The property on the field object, not just the helper."""
    dolfin = pytest.importorskip('dolfin')

    from ..fem import FunctionSpaceCache, FunctionSubspaceRegistry, MeshUtil
    from ..mesh import ConstructionHelperLayeredStructure
    from ..physics.optical import OpticalField
    from ..util import make_unit_registry

    U = make_unit_registry(("mesh_unit = 1 micrometer",))
    ls = ConstructionHelperLayeredStructure()
    ls.params = dict(
        edge_length=0.1, extra_regions=[], mesh_unit=U.um,
        layers=[dict(name='p', material='A', thickness=0.5,
                     mesh=dict(type='geometric', start=0.05, factor=1.3))])
    ls.run()
    mu = MeshUtil(mesh_data=ls.mesh_data, unit_registry=U,
                  function_space_cache=FunctionSpaceCache(),
                  function_subspace_registry=FunctionSubspaceRegistry())
    assert mu.gdim == 1

    class _Optical:
        mesh_util = mu

    field = OpticalField(optical=_Optical(), key='sun_000',
                         photon_energy=U('1.5 eV'), direction=(1, 0),
                         solid_angle=4 * np.pi)
    # the user's own value is left alone; the mesh-fitted one is what the
    # weak form reads
    np.testing.assert_allclose(field.direction, (1, 0))
    assert tuple(field.mesh_direction) == (1.0,)

    field.direction = np.array((0, 1))
    with pytest.raises(ValueError):
        field.mesh_direction
