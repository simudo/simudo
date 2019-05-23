from __future__ import print_function

import unittest
import warnings
from collections import ChainMap
from itertools import chain as ichain

import numpy as np

import dolfin
import ufl

__all__ = [
    'FunctionSubspaceRegistry',
    'FSRError',
    'AssignmentError',
    'FunctionTypeError',
    'IncompatibleFunctionSpacesError',
]

def collapse_indices(obj):
    if isinstance(obj, (tuple, list)):
        return ichain.from_iterable(collapse_indices(x) for x in obj)
    elif isinstance(obj, ufl.indexed.MultiIndex):
        return collapse_indices(obj[0])
    elif isinstance(obj, ufl.indexed.FixedIndex):
        return (int(obj),)
    elif isinstance(obj, int):
        return (obj,)
    else:
        raise TypeError("unexpected type {!r} for {!r}".format(type(obj), obj))

class FSRError(Exception):
    pass

class AssignmentError(FSRError):
    pass

class FunctionTypeError(AssignmentError, TypeError):
    pass

class IncompatibleFunctionSpacesError(AssignmentError):
    pass

class FunctionSpaceWrapper(object):
    '''wew'''
    def __init__(self, function_space):
        self.function_space = function_space

    def __hash__(self):
        s = self.function_space
        return hash((s.ufl_element(), s.mesh().id()))

    def __eq__(self, other):
        return self.function_space == other.function_space

    def __repr__(self):
        return "<{} {!r}>".format(type(self).__name__, self.function_space)

# TODO: make _subspaces values be objects themselves instead of dicts

class FunctionSubspaceRegistry(object):
    '''\
The FEniCS API does not provide any obvious way of retrieving the function space
of the result of dolfin.split(mixed_function). This class keeps track of those
function spaces.

Also, FunctionAssigner is broken, so we're also taking over that functionality.
'''
    def __init__(self):
        self._subspaces = {}

    def register(self, function_space, _root_space=None, _subfunction=None):
        # TODO: do this without creating a function
        subspaces_dict = self._subspaces
        root_space = (function_space if _root_space is None else _root_space)
        if function_space is root_space:
            # avoid registering same space twice
            root_key = self._subspace_key_for_root_space(root_space)
            if root_key in subspaces_dict:
                return
            else:
                subspaces_dict[root_key] = dict(
                    root_space=root_space, subspace=root_space,
                    subspace_copy=root_space)
                self._compute_dofmap_data(root_key)

        func = (dolfin.Function(function_space)
                if _subfunction is None else _subfunction)
        xs = tuple(dolfin.split(func))

        for i, x in enumerate(xs):
            try:
                subspace = function_space.sub(i)
            except ValueError:
                if len(xs) != 1: raise AssertionError()
                break
            # subspace_copy = dolfin.FunctionSpace(
            #     subspace.mesh(), subspace.ufl_element())
            subspace_copy = subspace.collapse()
            self.register(subspace_copy)
            subspace_key = self._subspace_key(x)
            subspaces_dict[subspace_key] = dict(
                root_space=root_space, subspace=subspace,
                subspace_copy=subspace_copy)
            self._compute_dofmap_data(subspace_key)
            if subspace.num_sub_spaces():
                self.register(subspace, root_space, _subfunction=x)

    def Function(self, space, *args, **kwargs):
        func = dolfin.Function(space, *args, **kwargs)
        self.register(space, _subfunction=func)
        return func

    def _subspace_key(self, subfunction):
        _, space, indices = (
            self.get_full_function_functionspace_indices(subfunction))
        return (FunctionSpaceWrapper(space), indices)

    def _subspace_key_for_root_space(self, space):
        return (FunctionSpaceWrapper(space), None)

    def get_full_function_functionspace_indices(self, u, no_indices=False):
        if   isinstance(u, ufl.indexed.Indexed):
            coefficient, indices = u.ufl_operands
            space = coefficient.function_space()
        elif isinstance(u, ufl.tensors.ListTensor):
            entries = tuple(self.get_full_function_functionspace_indices(
                e, no_indices=no_indices) for e in list(u))
            coefficient, space = entries[0][0:2]
            if not no_indices:
                indices = tuple(e[2] for e in entries)
            else:
                indices = None
        elif (isinstance(u, ufl.coefficient.Coefficient)
              and hasattr(u, 'function_space')):
            space = u.function_space()
            coefficient = u
            indices = None
            self.register(space)
        else:
            raise FunctionTypeError()
        if indices is not None and not no_indices:
            indices = tuple(collapse_indices(indices))
        return (coefficient, space, indices)

    def get_full_function_functionspace(self, u):
        return self.get_full_function_functionspace_indices(
            u, no_indices=True)[:2]

    def get_function_space(self, subfunction, collapsed=False):
        '''Get function space for a (sub)function. Use collapsed=True to return
a fresh independent/unattached copy.'''
        # if hasattr(subfunction, 'function_space'):
        #     return subfunction.function_space()
        return self._subspaces[self._subspace_key(subfunction)][
            'subspace_copy' if collapsed else 'subspace']

    def new_independent_function(self, subfunction):
        return dolfin.Function(
            self.get_function_space(subfunction, collapsed=True))

    def _compute_dofmap_data(self, subspace_key):
        # cmapv = collapse map values
        d = self._subspaces[subspace_key]
        ss = d['subspace']
        ssc = ss if ss is d['root_space'] else ss.collapse()

        ssdofmap = ss.dofmap()
        sscdofmap = ssc.dofmap()

        d['subdofmap'] = ssdofmap
        d['dofmap'] = sscdofmap

        # # proof that sscdofmap is boring
        # a = list(sscdofmap.dofs())
        # assert a == list(range(len(a)))

        # why
        mesh = ss.mesh()
        cmapv = np.zeros(ss.dim(), dtype='int32')
        for i in range(3):
            cmapv[sscdofmap.dofs(mesh, i)] = ssdofmap.dofs(mesh, i)
        d['cmapv'] = cmapv

    # FIXME: select call based on version
    def _compute_dofmap_data_FENICS2017(self, subspace_key):
        # cmapv = collapse map values
        d = self._subspaces[subspace_key]
        ss = d['subspace']
        d['subdofmap'] = subdofmap = ss.dofmap()
        dofmap, cmap = subdofmap.collapse(ss.mesh())
        d['dofmap'] = dofmap
        d['cmap'] = cmap
        cmapv = cmap.values()
        cmapv = d['cmapv'] = np.fromiter(
            iter(cmapv), dtype=np.uint32, count=len(cmapv))

    def get_vector_and_subspace_info(self, subfunction):
        k = self._subspace_key(subfunction)
        info = self._subspaces[k]
        # if hasattr(subfunction, 'function_space'):
        #     space = subfunction.function_space()
        #     func = subfunction
        # else:
        space = info['subspace']
        func = self.get_full_function_functionspace(subfunction)[0]
        return (func.vector(), info)

    def _same_space(self, s1, s2):
        return ((s1.mesh().ufl_id() == s2.mesh().ufl_id()) and
                (s1.ufl_element()   == s2.ufl_element()))

    def get_vector_and_indices(self, function):
        v, inf = self.get_vector_and_subspace_info(function)
        return (v, inf['cmapv'])

    def entity_dofs(self, function, dim, entities):
        ''' entities must be uintp array '''
        vec, inf = self.get_vector_and_subspace_info(function)
        return (vec, inf['dofmap'].entity_dofs(
            inf['subspace'].mesh(), dim, entities))

    def assign(self, target, source, coefficient=1.0, offset=0.0, ignore_incompatible=False):
        dv, d_inf = self.get_vector_and_subspace_info(target)
        sv, s_inf = self.get_vector_and_subspace_info(source)

        dS = d_inf['subspace']
        sS = s_inf['subspace']

        ds = d_inf['cmapv']
        ss = s_inf['cmapv']

        if not self._same_space(dS, sS):
            if ignore_incompatible:
                warnings.warn(RuntimeWarning(
                    "possibly incompatible function spaces dst={!r} and src={!r}"
                    .format(dS, sS)))
            else:
                raise IncompatibleFunctionSpacesError()

        dv[ds] = sv[ss] if coefficient==1.0 else sv[ss]*coefficient
        if offset != 0.0:
            dv[ds] += offset

    def assign_scalar(self, target, scalar):
        dv, d_inf = self.get_vector_and_subspace_info(target)
        ds = d_inf['cmapv']
        dv[ds] = scalar

    def copy(self, subfunction):
        f = self.new_independent_function(subfunction)
        self.assign(f, subfunction)
        return f

    def clear(self):
        self._subspaces.clear()

    @classmethod
    def from_union(cls, function_subspace_registries):
        '''This creates a registry that is the union of several other
registries, and can therefore handle any spaces found within them.'''
        ss = ChainMap(*([{}] + [fsr._subspaces for fsr in
                                function_subspace_registries]))
        fsr = cls()
        fsr._subspaces = ss
        return fsr

# FIXME: make this unit test more complete
class FSubRegTest(unittest.TestCase):
    def setUp(self):
        from dolfin import (MixedElement as MixE, Function, FunctionSpace,
                            FiniteElement, VectorElement, UnitSquareMesh)
        self.mesh = mesh = UnitSquareMesh(10, 10)
        muc = mesh.ufl_cell

        self.fe1 = FiniteElement("CG", muc(), 1)
        self.fe2 = FiniteElement("CG", muc(), 2)
        self.ve1 = VectorElement("CG", muc(), 2, 2)

        self.me = (
            MixE([self.fe1, self.fe2]),
            MixE([MixE([self.ve1, self.fe2]), self.fe1, self.ve1, self.fe2]),
            # MixE([self.fe1]), # broken
        )

        self.W = [FunctionSpace(mesh, me) for me in self.me]

        self.w = [Function(W) for W in self.W]

        self.reg = FunctionSubspaceRegistry()

        self.x = dolfin.SpatialCoordinate(mesh)

        for W in self.W:
            self.reg.register(W)

    def FunctionSpace(self, element):
        space = dolfin.FunctionSpace(self.mesh, element)
        self.reg.register(space)
        return space

    def get_expr(self, space, constant):
        x = self.x
        return dolfin.project(x[0] - 0.5*x[1] + dolfin.Constant(constant), space)

    def get_vect_expr(self, space, constant, components):
        x = self.x
        return dolfin.project(dolfin.as_vector(
            tuple(x[0] - 0.5*x[1] + dolfin.Constant(constant/(i+1.0))
            for i in range(components))), space)

    def test_vect(self):
        a = dolfin.split(dolfin.split(self.w[1])[0])[0]
        self.assigncheck(a, self.get_vect_expr(self.FunctionSpace(self.ve1), 1, 2))
        a = dolfin.split(self.w[1])[2]
        self.assigncheck(a, self.get_vect_expr(self.FunctionSpace(self.ve1), 1, 2))

    def test_scalar_W0(self):
        w = self.w[0]
        f1, f2 = dolfin.split(w)

        gah = self.reg.get_full_function_functionspace_indices
        a = gah(f1)
        b = gah(f2)
        self.assertEqual(a[:2], b[:2])

        self.do_test_scalar(f1, self.FunctionSpace(self.fe1))
        self.do_test_scalar(f2, self.FunctionSpace(self.fe2))

    def test_scalar_W2(self):
        '''
FIXME: This DOESN'T work. Specifically, FunctionSubspaceRegistry is
broken for spaces that are MixedElement with a single element. I'm not
yet sure how to fix it.
'''
        # w = self.w[2]
        # f1, = dolfin.split(w)
        # self.do_test_scalar(f1, self.FunctionSpace(self.fe1))

    def do_test_scalar(self, subvar, space):
        x = self.x
        v = dolfin.Function(space)
        v.assign(dolfin.project(x[0]**2 - x[1]**3 + 0.5, space))

        self.assigncheck(subvar, v)

    def test_scalar(self):
        w = self.w[0]
        x = self.x
        space = self.FunctionSpace(self.fe1)
        v = dolfin.Function(space)
        v.assign(dolfin.project(x[0]**2 - x[1]**3 + 0.5, space))
        f1, f2 = dolfin.split(self.w[0])

        gah = self.reg.get_full_function_functionspace_indices
        a = gah(f1)
        b = gah(f2)
        self.assertEqual(a[:2], b[:2])

        self.assigncheck(f1, v)

    def test_entity_dofs(self):
        space = self.FunctionSpace(self.fe1)
        x = self.x

        expr = x[0] + 2*x[1]

        u = dolfin.project(expr, space)

        vertices_positions = self.mesh.coordinates()
        vertices = np.arange(vertices_positions.shape[0], dtype='uintp')
        vec, dofs = self.reg.entity_dofs(u, 0, vertices)

        # arr = vec.array() ## <-- DOLFIN2018
        arr = vec
        for vertex, position in enumerate(vertices_positions):
            dof = dofs[vertex]
            y = arr[dof]
            self.assertLessEqual(abs(y - u(position)), 10*dolfin.DOLFIN_EPS)

    def assigncheck(self, target, source):
        import numpy as np
        reg = self.reg

        reg.assign(target, source)

        # check integrated error
        error = (target - source)**2*dolfin.dx
        err = np.sqrt(np.abs(dolfin.assemble(error)))
        self.assertLessEqual(err, 1e-8)

        v0, d0 = reg.get_vector_and_subspace_info(target)
        v1, d1 = reg.get_vector_and_subspace_info(source)

        # broken on dolfin2018
        # # sanity check for dofmap/subdofmap/cmap assumptions
        # for d in (d0, d1):
        #     self.assertEqual(frozenset(d['subdofmap'].dofs()), frozenset(d['cmap'].values()))
        #     self.assertEqual(tuple(d['dofmap'].dofs()), tuple(d['cmap'].keys()))

        def tab(space):
            cs = space.tabulate_dof_coordinates()
            n = space.dim()
            d = space.mesh().geometry().dim()
            return cs.reshape((n, d))

        dc0 = tab(d0['root_space'])
        dc1 = tab(d1['root_space'])

        cm0, cm1 = (d['cmapv'] for d in (d0, d1))
        for cell in dolfin.cells(self.mesh):
            cell_index = cell.index()
            cd0 = d0['dofmap'].cell_dofs(cell_index)
            cd1 = d1['dofmap'].cell_dofs(cell_index)

            # check that values match up
            self.assertLessEqual(np.max(np.abs(v1[cm1[cd1]] - v0[cm0[cd0]])), 1e-8)

            # check that coordinates match up
            for i0, i1 in zip(cd0, cd1):
                x0 = dc0[cm0[i0]]
                x1 = dc1[cm1[i1]]
                np.testing.assert_allclose(x0, x1, rtol=0, atol=1e-8)

                # check that point is actually inside cell
                x0c = x0.copy()
                x0c.resize((3,))
                p = dolfin.Point(x0c)
                self.assertLessEqual(cell.distance(p), 1e-8)
                # self.assertTrue(cell.contains(p))  # unreliable af >_<
