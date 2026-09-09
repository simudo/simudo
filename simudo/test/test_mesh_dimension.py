"""The true-1D mesh route of ``ConstructionHelperLayeredStructure``
(``params['mesh_dimension'] = 1``) against the historical 2D strip.

Both routes are built from the same tagged x-intervals, so the x coordinates,
layer lengths, region names and contact facets must agree exactly; the 1D
mesh has half the vertices and one interval cell per x-interval instead of
two triangles.  ``MeshUtil.element_hdiv`` must hand back ``BDM`` on the strip
and a 1-component continuous Lagrange vector element on the interval mesh.

Created by Claude Fable 5.1 (Anthropic), 2026-09-04.
"""

import numpy as np
import pytest

dolfin = pytest.importorskip('dolfin')

from ..fem import MeshUtil, FunctionSpaceCache, FunctionSubspaceRegistry
from ..mesh import CellRegions, ConstructionHelperLayeredStructure, FacetRegions
from ..util import make_unit_registry


def _build(dim):
    U = make_unit_registry(("mesh_unit = 1 micrometer",))
    s = dict(type='geometric', start=0.02, factor=1.3)
    layers = [dict(name='p', material='A', thickness=0.5, mesh=dict(s)),
              dict(name='i', material='B', thickness=0.3, mesh=dict(s)),
              dict(name='n', material='A', thickness=0.7, mesh=dict(s))]
    ls = ConstructionHelperLayeredStructure()
    ls.params = dict(edge_length=0.1, layers=layers, extra_regions=[],
                     mesh_unit=U.um, mesh_dimension=dim)
    ls.run()
    return ls, U


def _topology():
    R, F = CellRegions(), FacetRegions()
    F.left_contact = R.exterior_left.boundary(R.domain)
    F.right_contact = R.domain.boundary(R.exterior_right)
    F.exterior = R.domain.boundary(R.exterior)
    F.pi = R.p.boundary(R.i)
    return R, F


def _measures(md, R, F):
    ev = md.evaluate_topology
    mesh = md.mesh
    dx = dolfin.dx(domain=mesh, subdomain_data=md.cell_function)
    ds = dolfin.ds(domain=mesh, subdomain_data=md.facet_function)
    dS = dolfin.dS(domain=mesh, subdomain_data=md.facet_function)
    one = dolfin.Constant(1.0)
    lengths = {k: sum(dolfin.assemble(one*dx(cv)) for cv in ev(getattr(R, k)))
               for k in ('p', 'i', 'n', 'domain')}
    facets = {}
    for k in ('left_contact', 'right_contact', 'pi'):
        fvs = ev(getattr(F, k))
        facets[k] = (len(fvs), sum(dolfin.assemble(one*ds(fv)) +
                                   dolfin.assemble(one*dS(fv))
                                   for fv, sign in fvs))
    n = dolfin.FacetNormal(mesh)
    (fv, sign), = ev(F.left_contact)
    facets['left_nx'] = dolfin.assemble(n[0]*ds(fv))
    return lengths, facets


def test_1d_route_matches_strip_tagging():
    ls1, _ = _build(1)
    ls2, _ = _build(2)
    m1, m2 = ls1.mesh_data.mesh, ls2.mesh_data.mesh

    assert m1.geometry().dim() == 1 and m1.topology().dim() == 1
    assert m2.geometry().dim() == 2

    xs1 = sorted(set(m1.coordinates()[:, 0]))
    xs2 = sorted(set(m2.coordinates()[:, 0]))
    np.testing.assert_allclose(xs1, xs2)
    assert m1.num_vertices() == len(xs1)
    assert m2.num_vertices() == 2 * len(xs1)
    assert m1.num_cells() == len(xs1) - 1
    assert m2.num_cells() == 2 * (len(xs1) - 1)

    assert set(ls1.mesh_data.region_name_to_cvs) == set(ls2.mesh_data.region_name_to_cvs)
    # top/bottom exist for compatibility but are empty on the interval mesh
    assert ls1.mesh_data.region_name_to_cvs['exterior_top'] == set()
    assert ls1.mesh_data.region_name_to_cvs['exterior_bottom'] == set()

    R, F = _topology()
    lengths1, facets1 = _measures(ls1.mesh_data, R, F)
    lengths2, facets2 = _measures(ls2.mesh_data, R, F)
    for k in lengths1:
        assert abs(lengths1[k] - lengths2[k]) < 1e-12
    for k in ('left_contact', 'right_contact', 'pi'):
        assert facets1[k][0] == 1
        assert abs(facets1[k][1] - 1.0) < 1e-12   # a point "has area" 1
        assert abs(facets2[k][1] - 1.0) < 1e-12   # strip is 1 mesh unit tall
    assert facets1['left_nx'] == -1.0 and facets2['left_nx'] == -1.0

    # exterior minus contacts: nothing left in 1D, the top/bottom in 2D
    nonc = F.exterior - (F.left_contact | F.right_contact).both()
    assert ls1.mesh_data.evaluate_topology(nonc) == set()
    assert len(ls2.mesh_data.evaluate_topology(nonc)) > 0


def test_interval_mesh_is_the_default():
    """No ``mesh_dimension`` in params must give the interval mesh."""
    U = make_unit_registry(("mesh_unit = 1 micrometer",))
    ls = ConstructionHelperLayeredStructure()
    ls.params = dict(edge_length=0.1, extra_regions=[], mesh_unit=U.um,
                     layers=[dict(name='p', material='A', thickness=0.5,
                                  mesh=dict(type='geometric', start=0.05,
                                            factor=1.3))])
    ls.run()
    assert ls.mesh_dimension == 1
    assert ls.mesh_data.mesh.geometry().dim() == 1


def test_customized_strip_heights_still_extrude():
    """A subclass that sets ``product2d_Ys`` gets the extruded mesh.

    Those Y coordinates are meaningless on an interval mesh, so honouring the
    1D default there would silently flatten a genuinely 2D problem.
    """
    U = make_unit_registry(("mesh_unit = 1 micrometer",))

    class Custom(ConstructionHelperLayeredStructure):
        product2d_Ys = (0.0, 0.5, 1.0)

    params = dict(edge_length=0.1, extra_regions=[], mesh_unit=U.um,
                  layers=[dict(name='p', material='A', thickness=0.5,
                               mesh=dict(type='geometric', start=0.05,
                                         factor=1.3))])
    ls = Custom()
    ls.params = dict(params)
    ls.run()
    assert ls.mesh_dimension == 2
    assert ls.mesh_data.mesh.geometry().dim() == 2

    # ... unless the caller asks for 1D explicitly.
    ls1 = Custom()
    ls1.params = dict(params, mesh_dimension=1)
    ls1.run()
    assert ls1.mesh_data.mesh.geometry().dim() == 1


@pytest.mark.parametrize('dim', [1, 2])
def test_mesh_util_dimension_helpers(dim):
    ls, U = _build(dim)
    mu = MeshUtil(mesh_data=ls.mesh_data, unit_registry=U,
                  function_space_cache=FunctionSpaceCache(),
                  function_subspace_registry=FunctionSubspaceRegistry())
    assert mu.gdim == dim
    assert mu.zerovec.ufl_shape == (dim,)
    assert mu.unitvec_x.ufl_shape == (dim,)
    el = mu.element_hdiv(2)
    if dim == 1:
        assert el.family() == 'Lagrange' and el.degree() == 2
        assert el.value_shape() == (1,)
        with pytest.raises(ValueError):
            mu.unitvec_y
    else:
        assert el.family() == 'Brezzi-Douglas-Marini' and el.degree() == 2
        assert mu.unitvec_y.ufl_shape == (2,)
    # the element must be usable in a mixed space on this mesh
    W = dolfin.FunctionSpace(mu.mesh, dolfin.MixedElement(
        [dolfin.FiniteElement('DG', mu.ufl_cell, 1), el]))
    assert W.dim() > 0
