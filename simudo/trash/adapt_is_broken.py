import dolfin

from dolfin import (Function, FunctionSpace,
                    FiniteElement, VectorElement, UnitSquareMesh,
                     adapt, refine)

from .mesh.io import load as mesh_load

parameters = dolfin.parameters
parameters["refinement_algorithm"] = "plaza_with_parent_facets"
# parameters["form_compiler"]["cpp_optimize"] = True
# parameters["form_compiler"]["cpp_optimize_flags"] = '-O3 -funroll-loops'
# parameters["form_compiler"]["quadrature_degree"] = 2

dct = mesh_load("out/mesh")

mesh = dct['mesh']
cf = dct['cell_function']
ff = dct['facet_function']
# ff.array()[:] = range(len(ff.array()))

# mesh = UnitSquareMesh(10, 10)
muc = mesh.ufl_cell()

dim = mesh.geometry().dim()
mesh.init(dim-1, dim)
# mesh.init(dim-2, dim-1)
refine_cf = dolfin.MeshFunction("bool", mesh, dim, False)

# ff = dolfin.MeshFunction("size_t", mesh, dim-1, 1)
# ff.get_local()[:] = range(len(ff.get_local()))

# element = FiniteElement("CG", muc, 1)
# W = dolfin.FunctionSpace(mesh, element)

refine_cf.array()[0:10] = True

print(frozenset(ff.array()))

dolfin.File("finalfantasy.pvd") << ff

adapt(mesh, refine_cf)
mesh2 = mesh.child()

mesh2.init(dim-1, dim)
# mesh2.init(dim-2, dim-1)

cf2 = adapt(cf, mesh2)
ff2 = adapt(ff, mesh2)

dolfin.File("finalfantasy2.pvd") << ff2

ff = dct['facet_function']
from .mesh.facet import fix_undefined_facets_after_subdivision
fix_undefined_facets_after_subdivision(
    mesh2, cf2, ff2, dct['meta']['facets'])

dolfin.File("finalfantasy3.pvd") << ff2

print(frozenset(ff2.array()))
