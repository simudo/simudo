from dolfin import *
import numpy as np

mesh = UnitSquareMesh(4,4)

# Mark vertices
subdomain = AutoSubDomain(lambda x, on_boundary: x[1]<=0.5)
disjoint_subdomain = AutoSubDomain(lambda x, on_boundary: x[1]>0.5)
vertex_domain = VertexFunction("size_t", mesh , 0)
subdomain.mark(vertex_domain, 1)

V = FunctionSpace(mesh, "P", 1)
VV = VectorFunctionSpace(mesh, "P", 1)

bc = DirichletBC(VV, Constant((0,0)), disjoint_subdomain)

e = Expression("x[0]+2*x[1]", degree=1)
ee = Expression(["x[0]+x[1]", "x[0]*x[1]"], degree=2)

# Create some coefficients
u = interpolate(e, V)
uu = interpolate(ee, VV)

# Test and trial spaces
v = TestFunction(V)
vv = TestFunction(VV)
U = TrialFunction(V)

dP = dP[vertex_domain]
scalar_value = assemble(u*dP)

result = abs(scalar_value-u.vector().sum())<10*DOLFIN_EPS
print "Scalar:", MPI.rank(mpi_comm_world()), result
assert result

point_vec_1 = assemble(u*v*dP)
point_vec_2 = assemble(inner(uu,vv)*dP)
point_vec_3 = assemble(inner(uu,vv)*dP(1))

# Assemble matrix
point_mat_1 = assemble(u*v*U*dP)

# FIXME: No test for matrix, but it seems to work :)

result = sum(np.absolute(point_vec_1.array() - u.vector().array())) < 10*DOLFIN_EPS
print "Vector #0", MPI.rank(mpi_comm_world()), result
assert result

result = sum(np.absolute(point_vec_2.array() - uu.vector().array())) < 10*DOLFIN_EPS
print "Vector #1", MPI.rank(mpi_comm_world()), result
assert result

# Apply BC to original coefficient to mimic subdomain assemble
bc.apply(uu.vector())
result = sum(np.absolute(point_vec_3.array()-uu.vector().array())) < 10*DOLFIN_EPS
print "Vector with subdomain:", MPI.rank(mpi_comm_world()), result
assert result
