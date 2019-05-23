
import dolfin

mesh = dolfin.UnitSquareMesh(5, 5)

uc = mesh.ufl_cell()

E1 = dolfin.MixedElement(dolfin.FiniteElement("CG", uc, 1),
                         dolfin.FiniteElement("DG", uc, 2))
W1 = dolfin.FunctionSpace(mesh, E1)

E2 = dolfin.FiniteElement("CG", uc, 2)
W2 = dolfin.FunctionSpace(mesh, E2)

u1 = dolfin.Function(W1)
u2 = dolfin.project(dolfin.SpatialCoordinate(mesh)[0], W2)

h5w = dolfin.HDF5File(mesh.mpi_comm(), "test.h5", "w")

h5w.write(mesh, '/mesh')
h5w.write(mesh, '/mesh_2')

h5w.write(u1, '/u1')
h5w.write(u1, '/u1_1')
h5w.write(u2, '/u2')

h5w.close()

m2 = dolfin.Mesh()

h5r = dolfin.HDF5File(m2.mpi_comm(), "test.h5", "r")

h5r.read(m2, '/mesh')
h5r.read(m2, '/mesh')


# import h5py

# h = h5py.File("test.h5")



