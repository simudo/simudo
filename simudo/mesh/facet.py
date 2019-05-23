from __future__ import absolute_import, division, print_function

import itertools
from builtins import bytes, dict, int, range, str, super
from functools import partial

import numpy as np

import dolfin

from ..fem import point_to_array

__all__ = [
    'facet2d_angle',
    'mark_boundary_facets',
    'mesh_function_map',
    'FacetsManager']

def low_level_make_facets(mesh, cell_function, boundary_facet_function):
    '''Create a facet function corresponding to cell function value
boundaries.

(Don't use this, use the high level `FacetsManager` class instead.)

Parameters
----------
mesh : dolfin.Mesh
    Mesh on which to operate.
cell_function : dolfin.CellFunction
    Cell function on `mesh`.
boundary_facet_function : dolfin.FacetFunction
    Facet function on `mesh`. If a facet is a boundary facet,
    then this function will behave as if on the exterior there
    was a cell with the value given by the facet value.

Returns
-------
facet_function : dolfin.FacetFunction
    Facet function taking value `facet_value` on each facet as
    described below.
facet_value_dict : dict
    Dictionary with keys `(cell_function_value_1,
    cell_function_value_2, sign)` and value `facet_value`.
    '''
    D = mesh.topology().dim()
    mesh.init(D-1, D)

    cf = cell_function
    ff = dolfin.MeshFunction("size_t", mesh, D-1) # facet function
    bff = boundary_facet_function

    cfa = cf.array()
    ffa = ff.array()
    bffa = bff.array()

    d = {} # result dict

    # allocate same-material facet values as they may become needed when
    # subdividing the mesh at a later time
    fv = 1 # starting index
    cell_values = tuple(sorted(set(cfa)))
    d.update(((int(v), int(v), +1), i)
             for i, v in enumerate(cell_values, fv))
    fv += len(cell_values)

    for f in dolfin.facets(mesh):
        f_index = f.index()
        t = tuple(f.entities(D)) # cells on either side of facet
        n = len(t)
        if n == 2:
            x, y = t
            x = cfa[x]
            y = cfa[y]
        elif n == 1:
            x = t[0]
            x = cfa[x]
            y = bffa[f_index]
        else:
            raise AssertionError()
        if x <= y:
            sign = +1
        elif x > y:
            x, y = y, x
            sign = -1
        k = (int(x), int(y), sign)
        v = d.get(k)
        if v is None:
            d[k] = v = fv
            fv += 1
        ffa[f_index] = v
    return (ff, d)

### BROKEN!!! broken in FEniCS 2018. works in 2017.
# _boundary_mesh_entities_dim_dict = {'facet': (-1, dolfin.Facet),
#                                     'cell' : ( 0, dolfin.Cell)}
# def boundary_mesh_entities_broken(mesh, dim='facet', type='exterior', order=True):
#     '''iterator of facets or cells of BoundaryMesh'''

#     boundary_mesh = dolfin.BoundaryMesh(mesh, type=type, order=order)

#     dim, cls = _boundary_mesh_entities_dim_dict[dim]
#     dim += mesh.topology().dim()

#     for i in boundary_mesh.entity_map(dim):
#         yield cls(mesh, i)
###

def boundary_mesh_entities(mesh, dim='facet', type='exterior', order=True):
    '''iterator of facets or cells of BoundaryMesh

    slow stupid limited workaround version >_< '''
    if (dim,type,order) != ('facet', 'exterior', True): raise AssertionError()

    D = mesh.topology().dim()
    Facet = dolfin.Facet
    for f in dolfin.facets(mesh):
        if len(f.entities(D)) == 1:
            # only one adjacent cell, other side is nonexistence
            yield f

def facet2d_angle(facet):
    ''' right_value=0, up_value=1, left_value=2, down_value=3 '''
    n = point_to_array(facet.normal())
    n_ = np.abs(n)
    if n_[0] > n_[1]: # x component significant
        return 0 if n[0] > 0 else 2
    else: # y component significant
        return 1 if n[1] > 0 else 3

def mark_boundary_facets(mesh, boundary_facet_function, facet_closure,
                         offset=0):
    D = mesh.topology().dim()

    ffa = boundary_facet_function.array()

    for f in boundary_mesh_entities(mesh, 'facet'):
        v = facet_closure(f)
        if v is not None:
            ffa[f.index()] = v + offset

def _compute_refinement_undefined_value():
    '''super convoluted way to get (size_t)-1'''

    mesh = dolfin.UnitSquareMesh(2, 2)
    dim = mesh.geometry().dim()
    ff = dolfin.MeshFunction("size_t", mesh, dim-1, 42)

    old_refinement_algorithm = dolfin.parameters["refinement_algorithm"]
    try:
        dolfin.parameters["refinement_algorithm"] = 'plaza_with_parent_facets'
        mesh2 = dolfin.refine(mesh)
    finally:
        dolfin.parameters["refinement_algorithm"] = old_refinement_algorithm

    # mesh2 = mesh.child() ## <-- DOLFIN2018
    ff2 = dolfin.adapt(ff, mesh2)
    before = frozenset(ff.array())
    after = frozenset(ff2.array())
    new_facet_values = after - before
    assert len(new_facet_values) == 1
    return next(iter(new_facet_values))

refinement_undefined_value = _compute_refinement_undefined_value()

_NONEXISTENT = []
def get_or_create(dictionary, key, create_func):
    v = dictionary.get(key, _NONEXISTENT)
    if v is _NONEXISTENT:
        dictionary[key] = v = create_func()
    return v

def invert_mapping(mapping):
    '''input {key: set(values...)}
output {value: set(keys)}
'''
    r = {}
    for k, vs in mapping.items():
        for v in vs:
            get_or_create(r, v, set).add(k)
    return r

def mesh_function_map(mesh_function, func, out_type='size_t'):
    mf = mesh_function
    out = dolfin.MeshFunction(out_type, mf.mesh(), mf.dim(), 0)
    out.array()[:] = map(func, mf.array()[:])
    return out

class FacetsManager(object):
    '''Keep track of boundaries between subdomains (cell function values).'''
    def __init__(self, mesh, cell_function, boundary_facet_function):
        # TODO: doc
        self.facet_mesh_function, d = low_level_make_facets(
            mesh, cell_function, boundary_facet_function)

        self.internal_facets_dict = ifvs = {}
        self.fvs = fvs = {}
        for (cv1, cv2, sign), fv in d.items():
            if cv1 == cv2:
                ifvs[cv1] = fv
            else:
                get_or_create(fvs, (cv1, cv2), list).append((fv,  sign))
                get_or_create(fvs, (cv2, cv1), list).append((fv, -sign))

        for k, v in fvs.items():
            fvs[k] = set(v)

    def boundary(self, X, Y, intersection=None):
        '''Get signed boundary between `X` and `Y`.

If the intersection between `X` and `Y` is nonempty, you must
pass the `intersection` argument. This function will pretend
that the intersection is actually part of `X` or `Y`
(depending on the value of `intersection`). In other words,

- :code:`boundary(X, Y, 0) == boundary(X, Y-X)`, and
- :code:`boundary(X, Y, 1) == boundary(X-Y, Y)`.

Parameters
----------
X : set
    `X` and `Y` must be sets of cell marker values (representing
    subdomains). `X` and `Y` must be disjoint unless the
    `intersection` argument is used.
Y: set
    See `X`.
intersection : int, optional
    Must be 0 or 1, if specified.

Returns
-------
fvs : set
    Set of tuples :code:`(facet_value, sign)` representing facets
    between `X` and `Y`.
'''

        if intersection is not None:
            if intersection == 0:
                Y = Y - X
            elif intersection == 1:
                X = X - Y
            else:
                raise ValueError(repr(intersection))
        else:
            assert not (X & Y)

        d = self.fvs
        r = set()
        for x, y in itertools.product(X, Y):
            v = d.get((x, y), None)
            if v is not None:
                r.update(v)
        return r

    def cell_value_internal(self, X):
        '''Get internal facets excluding boundaries across cell values
(subdomains).

Parameters
----------
X : set
    Set of cell values (representing subdomains).

Returns
-------
fvs : set
    Set of `(facet_value, sign)` representing facets contained
    within `X` and not on the boundary between two cell
    values.'''

        ifvs = self.internal_facets_dict
        return set((fv, 1) for fv in (ifvs.get(x, None) for x in X)
                   if fv is not None)

    def internal(self, X):
        '''Get all internal facets.

Parameters
----------
X : set
    Set of cell values (representing subdomains).

Returns
-------
fvs : set
    Set of `(facet_value, sign)` representing facets contained
    within `X` and not on its boundary.'''

        d = self.fvs
        r = self.cell_value_internal(X)
        for x, y in itertools.combinations(X, 2):
            v = d.get((x, y), None)
            if v is not None:
                r.update(v)
        return r

    def fix_undefined_facets_after_subdivision(
            self, mesh, cell_function, facet_function):
        '''Assign markers to facets created by mesh subdivision.

Upon subdivision, cells get split and new facets are created
to separate them. The subdivision algorithm cannot know what
facet value to assign to these new facets, so it leaves them
with a large undefined value.

These facets must be fully internal to a subdomain, so the
cells on either side must have the same cell value. That cell
value corresponds to a facet value for internal facets, which
is precisely the information stored in the
`internal_facets_dict` attribute.

Parameters
----------
mesh : dolfin.Mesh
    Mesh on which to operate.
cell_function : dolfin.CellFunction
    Cell function on `mesh` representing subdomains.
facet_function : dolfin.FacetFunction
    Facet function to fix up after subdivision. This will be
    modified in place.'''
        cf = cell_function
        ff = facet_function

        D = mesh.topology().dim()
        mesh.init(D-1, D)

        cfa = cf.array()
        ffa = ff.array()

        assert len(ffa.shape) == 1
        idx = np.indices(ffa.shape)[0]
        idx = idx[ffa == refinement_undefined_value]

        ifvs = self.internal_facets_dict

        Facet = partial(dolfin.Facet, mesh)
        for f_index in idx:
            f = Facet(f_index)
            t = tuple(f.entities(D))
            x, y = t
            x = cfa[x]
            y = cfa[y]
            assert x == y
            ffa[f_index] = ifvs[x]

    def __getstate__(self):
        return {k: getattr(self, k) for k in
                ('fvs', 'internal_facets_dict')}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
