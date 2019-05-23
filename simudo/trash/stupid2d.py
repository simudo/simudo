import dolfin

mesh = dolfin.UnitSquareMesh(50, 20)
el = dolfin.FiniteElement("BDM", mesh.ufl_cell(), 1)
x = dolfin.SpatialCoordinate(mesh)
expr = dolfin.as_vector((dolfin.sin(1/x[0] + x[1]), 2*x[1]))
W = dolfin.FunctionSpace(mesh, el)
w = dolfin.project(expr, W)

example_testfunc = dolfin.TestFunction(W)
example_form = dolfin.dot(w, example_testfunc) * dolfin.dx() + w[0] * dolfin.ds()

