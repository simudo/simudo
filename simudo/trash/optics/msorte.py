
import dolfin
from numpy import array as npa
import numpy as np
from ...util.newton_solver import NewtonSolver
from ...util.expr import tuple_to_point

really_squished_mesh = False


no_side_bc = really_squished_mesh
theta = (90 + 0) * (np.pi/180)
omega_tuple = (np.cos(theta), np.sin(theta))

omega = dolfin.Constant(omega_tuple)

if really_squished_mesh:
    extent, NM = (0., 0., 0.02, 1.0), (1, 20)
else:
    extent, NM = (0., 0., 1.0, 1.0), (30, 30)
mesh = dolfin.RectangleMesh(tuple_to_point(extent[:2]),
                            tuple_to_point(extent[2:]), *NM)

muc = mesh.ufl_cell()
I_el = dolfin.FiniteElement('CG', muc, 2)
space = dolfin.FunctionSpace(mesh, I_el)

u = dolfin.Function(space)
test_function = dolfin.TestFunction(space)

facet_normal = dolfin.FacetNormal(mesh)
x = dolfin.SpatialCoordinate(mesh)

def lower_boundary(x, on_boundary):
    return on_boundary and (abs(x[1] - 0) < 1e-10)

def left_boundary(x, on_boundary):
    return on_boundary and (abs(x[0] - 0) < 1e-10)

def right_boundary(x, on_boundary):
    return on_boundary and (abs(x[0] - 1) < 1e-10)

def make_form(I, v, omega, beta, S):
    ''' returns (ds terms, dx terms) '''
    dot, grad, div = dolfin.dot, dolfin.grad, dolfin.div

    a = dot(omega, grad(I))
    w = v*omega
    n = facet_normal

    a_bc = S - beta*I

    return (dot(w, n)*a_bc,
            -a*div(w) + dot(w, grad(beta*I)) - dot(w, grad(S)))

if really_squished_mesh:
    lower_boundary_value = dolfin.Constant(1.0)
else:
    lower_boundary_value = dolfin.conditional(
        dolfin.lt((x[0] - 0.5)**2, 0.35**2), 1.0, 0.0)

bcs = [dolfin.DirichletBC(space, lower_boundary_value, lower_boundary)]

if not no_side_bc:
    def _side_bc(subdomain):
        bcs.append(dolfin.DirichletBC(space, dolfin.Constant(0.0), subdomain))
    if theta < np.pi / 2:
        _side_bc(left_boundary)
    else:
        _side_bc(right_boundary)

# alpha = dolfin.Constant(1.0)
alpha = dolfin.conditional(
    dolfin.lt((x[0] - 0.5)**2 + (x[1] - 0.5)**2, 0.2**2), 6.0, 1.0)
# alpha = dolfin.conditional(dolfin.And(dolfin.gt(x[0], 0.5), dolfin.gt(x[1], 0.5)), 4.0, 1.0)

alpha = dolfin.project(alpha, space) # needs to be continuous
S = dolfin.Constant(0.0)*x[0]

dsf, dxf = make_form(I=u, v=test_function, omega=omega, beta=alpha, S=S)
form = dsf*dolfin.ds() + dxf*dolfin.dx()

solver = NewtonSolver(form, u, bcs)
solver.solve()

# dolfin.plot(u)
# dolfin.plot(alpha)

