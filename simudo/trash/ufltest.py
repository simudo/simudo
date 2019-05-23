from __future__ import print_function
import dolfin, ufl
import weakref

if False:

    from dolfin import *
    import numpy as np

    mesh = UnitSquareMesh(40, 40)

    n = 5
    E = FiniteElement('DG', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, E)
    MixE = dolfin.MixedElement
    M = FunctionSpace(mesh, MixE([E]*n))

    u = Function(M)
    U = u.vector()

    # Manual assign to vector
    # Extract subfunction dofs
    dofs_is = [M.sub(i).dofmap().dofs() for i in range(n)]

    for dofs_i in dofs_is:
        # Extract subfunction values
        U_i = U[dofs_i]
        # Modify values to i+1
        U_i += i+1
        # Assign
        U[dofs_i] = U_i

    # Use assigner
    assigner = FunctionAssigner(M, [V]*n)

    # Create functions in V with values of subfunctions
    functions = [Function(V) for dofs_i in dofs_is]
    for f, dofs in zip(functions, dofs_is):
        f.vector()[:] = U[dofs]
        # Parellel
        # f.vector().set_local(np.array(U[dofs], dtype='double'))
        # f.vector().apply('')

    # Modify values
    for f in functions:
        f.vector()[:] *= 10
    # Assign
    assigner.assign(u, functions)

    # Check, that subfunctions are 10*(i+1)
    print(all(near(U[dofs_i].norm('linf'), 10*(i+1)) for dofs_i in dofs_is))


    exit(0)










if False:
    from dolfin import *
    mesh = UnitSquareMesh(20,20)
    x = SpatialCoordinate(mesh)
    V = VectorFunctionSpace(mesh, "CG", 1)
    t = Constant(0.0)
    f = x[0]*t
    df_dt = diff(f, t)
    print(f)
    print(df_dt)
    #plot(df_dt, mesh=mesh)
    t.assign(0.1)
    #plot(f, mesh=mesh)
    #plot(df_dt, mesh=mesh)

import dolfin
from dolfin import *
from ufl.algorithms.apply_derivatives import apply_derivatives

mesh = UnitSquareMesh(12,12)
muc = mesh.ufl_cell

e1 = FiniteElement("CG", muc(), 1)
e2 = VectorElement("CG", muc(), 2, 3)

W0 = FunctionSpace(mesh, e1)

MixE = dolfin.MixedElement

#W = FunctionSpace(mesh, MixE([
#    MixE([e1, e1, MixE([e1, e2])]), e1, e1, e1]))

W = FunctionSpace(mesh, MixE([e1, e2, e1]))

w, w2 = Function(W), Function(W)

g = Function(W0)

def ufl_operands(value, types):
    if not isinstance(value, types):
        raise TypeError("expected types {!r} for {!r}".format(
            types, value))
    return value.ufl_operands

# def get_full_function_space_and_index(u):
#     coefficient, indices = ufl_operands(u, ufl.indexed.Indexed)
#     space = coefficient.function_space()
#     return (space, indices)

# print(dir(space))
# print(space.extract_sub_space([0, 0]))
# print((space.num_sub_spaces()))

# TODO: avoid storing all the subfunction instances; instead use the function above get_function_space2 to get the full space and then memoize only what the subspaces are for that.


class FunctionSubspaceRegistry(object):
    '''\
The FEniCS API does not provide any obvious way of retrieving the function space
of the result of dolfin.split(mixed_function). This class keeps track of those
function spaces.

Also, FunctionAssigner is broken, so we're also taking over that functionality.
'''
    def __init__(self):
        # {(root_space, index): (root_space, function_subspace)}
        self._subspaces = {}

    def register(self, function_space, _root_space=None, _subfunction=None):
        # TODO: do this without creating a function
        root_space = (function_space
                      if _root_space is None else _root_space)
        func = (dolfin.Function(function_space)
                if _subfunction is None else _subfunction)
        xs = dolfin.split(func)
        for i, x in enumerate(xs):
            subspace = function_space.sub(i)
            self._subspaces[self._subspace_key(x)] = dict(
                root_space=root_space, subspace=subspace)
            if subspace.num_sub_spaces():
                self.register(subspace, root_space, x)

    def _subspace_key(self, subfunction):
        return self.get_full_function_functionspace_indices(subfunction)[1:3]

    def get_full_function_functionspace_indices(self, u):
        if   isinstance(u, ufl.indexed.Indexed):
            coefficient, indices = u.ufl_operands
            space = coefficient.function_space()
        elif isinstance(u, ufl.tensors.ListTensor):
            entries = tuple(self.get_full_function_functionspace_indices(e)
                            for e in list(u))
            coefficient, space = entries[0][0:2]
            indices = tuple(e[2] for e in entries)
        else:
            raise TypeError("unexpected type for {!r}".format(u))
        return (coefficient, space, indices)

    def get_function_space(self, subfunction, new=False):
        '''Get function space for a (sub)function. Use new=True to return
a fresh independent/unattached copy.'''
        print(repr(subfunction))
        if hasattr(subfunction, 'function_space'):
            space = subfunction.function_space()
        else:
            space = self._subspaces[self._subspace_key(subfunction)]['subspace']
        if new:
            element = space.ufl_element()
            space = dolfin.FunctionSpace(space.mesh(), element)
        return space

    def new_independent_function(self, subfunction):
        return dolfin.Function(self.get_function_space(subfunction, new=True))

    def get_vector_and_dofslice(self, subfunction):
        if hasattr(subfunction, 'function_space'):
            return (subfunction.vector(), slice(None))
        else:
            subspace = self._subspaces[self._subspace_key(subfunction)]['subspace']
            parent = self.get_full_function_functionspace_indices(subfunction)[0]
            return (parent.vector(), subspace.dofmap().dofs())

    def assign(self, destination, source):
        dst = self.get_vector_and_dofslice(destination)
        src = self.get_vector_and_dofslice(source)
        dst[0][dst[1]] = src[0][src[1]]

    def clear(self):
        self._subspaces.clear()

reg = FunctionSubspaceRegistry()
reg.register(W)

a0, a1, a2 = dolfin.split(w)

b0, b1, a2 = [reg.new_independent_function(x) for x in (a0, a1, a2)]

reg.assign(a0, b0)
reg.assign(a1, b1)

print(repr(reg._subspaces))


# space.sub

exit(0)
print(repr(y))
print(y.ufl_operands)

l = dolfin.split(w)

print(len(l))

# R = FunctionSpace(mesh, "R", 0)
# tt = Function(R)
# t = variable(tt)
# x = SpatialCoordinate(mesh)

# f = sin(t*x[0])

# dfdt = diff(f, t)
# dfdx = diff(f, x)

# print(f)
# print(apply_derivatives(dfdt))
# print(dfdx)


