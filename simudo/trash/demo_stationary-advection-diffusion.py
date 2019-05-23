# source: https://fenicsproject.org/qa/6951/help-on-supg-method-in-the-advection-diffusion-demo
'''
Problem:
    b*grad(u) - div(c*grad(u)) = f  in domain
                             u = g  on boundary
    where b = [1, 1], c = 1, f = cos(x[0]) + cos(x[1]) + sin(x[0]) + sin(x[1]), and g = 'sin(x[0]) + sin(x[1])'
Exact solution;
    u = sin(x[0]) + sin(x[1])
'''
from dolfin import *

# Create mesh and function space
mesh = UnitSquareMesh(100, 100)
Q = FunctionSpace(mesh, "CG", 1)

# Define velocity as vector
velocity = as_vector([1.0, 1.0])

c = 1.0

# Test and trial functions
u, v = TrialFunction(Q), TestFunction(Q)

# Manufactured right hand side
f = Expression('cos(x[0]) + cos(x[1]) + sin(x[0]) + sin(x[1])')

# Residual
r = dot(velocity, grad(u)) - c*div(grad(u)) - f

# Galerkin variational problem
F = v*dot(velocity, grad(u))*dx + c*dot(grad(v), grad(u))*dx - f*v*dx

# Add SUPG stabilisation terms
vnorm = sqrt(dot(velocity, velocity))
h = CellSize(mesh)
delta = h/(2.0*vnorm)
F += delta*dot(velocity, grad(v))*r*dx

# Create bilinear and linear forms
a = lhs(F)
L = rhs(F)

# Define boundary condition
def u0_boundary(x, on_boundary):
    return on_boundary
U0_expression = Expression('sin(x[0]) + sin(x[1])')
bc = DirichletBC(Q, U0_expression, u0_boundary)

# Solve
u = Function(Q)
solve(a == L, u, bc)

# Print out calcuated and exact solution
u_exact = interpolate(U0_expression, Q)
print 'x--', 'u------------ ', 'u_exact-------'
for i in xrange(0, 11):
    x = [0.5, i/10.0]
    print i/10.0, u(x), u_exact(x)

plot(u, interactive = True)
