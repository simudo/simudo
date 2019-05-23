from __future__ import absolute_import, division, print_function

from builtins import bytes, dict, int, range, str, super
from functools import partial

import numpy as np

import dolfin


def refine_forever(refinement_objs, make_predicate):
    while True:
        mesh = refinement_objs['mesh']

        predicate = make_predicate()
        mf = predicate_to_meshfunction(mesh, predicate, predicate.dim)

        if not any(mf.array()):
            break

        mesh2 = adapt(mesh, mf)

        for k, v in refinement_objs.items():
            if v is mesh:
                v = mesh2
            else:
                v = adapt(v, mesh2)
            refinement_objs[k] = v

    return mesh

def predicate_to_meshfunction(mesh, predicate, dim):
    mf = dolfin.MeshFunction(
        "bool", mesh, dim, False)
    mark_using_predicate(mf, mesh, predicate)
    return mf

def mark_using_predicate(mf, mesh, predicate):
    for entity in dolfin.entities(mesh, mf.dim()):
        if predicate(entity):
            mf[entity] = True

def cell_predicate_expr_threshold(mesh, function, threshold):
    '''function must be a dolfin.FunctionSpace or pint wrapper around one'''

    if hasattr(function, 'units'):
        threshold = threshold.m_as(function.units)
        function = function.magnitude

    fvec = function.vector()
    dofmap = function.function_space().dofmap()

    def predicate(cell):
        cell_index = cell.index()
        cell_dofs = dofmap.cell_dofs(cell_index)
        val = np.amax(np.abs(fvec[cell_dofs]))
        return (val >= threshold) or None

    return predicate

def adapt(obj, *args, **kwargs):
    if isinstance(obj, (dolfin.cpp.mesh.MeshFunctionBool,
                        dolfin.cpp.mesh.MeshFunctionDouble,
                        dolfin.cpp.mesh.MeshFunctionInt,
                        dolfin.cpp.mesh.MeshFunctionSizet)):
        return adapt_mf(obj, *args, **kwargs)
    # why it's stupid:
    # https://bitbucket.org/fenics-project/dolfin/issues/319/faulty-memory-management-in-adapt
    dolfin.adapt(obj, *args, **kwargs)
    return obj.child()

def adapt_mf(mf, new_mesh):
    # why it's stupid: see adapt_workaround
    # why it's even stupider:
    # MeshFunction::child() has two C++ methods associated (differing
    # only in a const on the return type) and SWIG couldn't
    # distinguish between them, so it errors out.
    bad = dolfin.adapt(mf, new_mesh)
    good = dolfin.MeshFunction("size_t", new_mesh, mf.dim(), 0)
    good.array()[:] = bad.array()[:]
    del bad
    return good
