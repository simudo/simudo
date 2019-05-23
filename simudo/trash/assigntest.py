from dolfin import *
import numpy as np

mesh = UnitSquareMesh(40, 40)

n = 5
V = FunctionSpace(mesh, 'DG', 1)
M = MixedFunctionSpace([V]*n)

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

# Check, that subfunctions are i+1
print all(near(U[dofs_i].norm('linf'), i+1) for dofs_i in dofs_is)

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
print all(near(U[dofs_i].norm('linf'), 10*(i+1)) for dofs_i in dofs_is)
