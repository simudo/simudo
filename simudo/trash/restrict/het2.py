from dolfin import *
import matplotlib.pyplot as plt
import pygmsh
import meshio
import numpy as np
from ...mesh.construction_helper import ConstructionHelperPygmsh
from ...util.pint import make_unit_registry
from ...api import CellRegions, FacetRegions, MeshUtil

class CH(ConstructionHelperPygmsh):
    def user_define_pygmsh_geo(self):
        geom = pygmsh.opencascade.Geometry(
           characteristic_length_min = 0.02,
           characteristic_length_max = 0.02,
           )

        r0 = geom.add_rectangle([0, 0, 0], 1., 1.)
        r1 = geom.add_rectangle([0, 1, 0], 1, 0.5)
        r2 = geom.add_rectangle([0, -0.5, 0], 1, 0.5)

        union = geom.boolean_union([r1, r0, r2])

        geom.add_physical_surface([r0], 'core')
        geom.add_physical_surface([r1], 'top')
        geom.add_physical_surface([r2], 'bottom')
        return geom

U = make_unit_registry(("mesh_unit = 1 dimensionless",))

ch = CH()
ch.params = dict(mesh_unit=U.mesh_unit)
ch.run()

mesh_data = ch.mesh_data
mesh = mesh_data.mesh
celldata = mesh_data.cell_function
mu = MeshUtil(mesh_data=mesh_data, _sloppy=True)

R = CellRegions()
F = FacetRegions()

R.live = R.core
R.dead = R.top | R.bottom

# Define finite elements spaces and build mixed space

BDM = FiniteElement("BDM", mesh.ufl_cell(), 1)
DG  = FiniteElement("DG", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, BDM * DG)

# Define trial and test functions
(sigma, u) = TrialFunctions(W)
(tau, v) = TestFunctions(W)

# Define source function
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)

dx_live = mu.region_dx(R.live)
dx_dead = mu.region_dx(R.dead)

# Define variational form
a = (dx_live*(dot(sigma, tau) + div(tau)*u + div(sigma)*v) +
     dx_dead*(u*v + dot(sigma, tau)))
L = -dx_live*f*v

a = a.to_ufl().m
L = L.to_ufl().m

# Define function G such that G \cdot n = g
class BoundarySource(UserExpression):

    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        super().__init__(**kwargs)

    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        # g = sin(5*x[0])
        g = 0
        values[0] = g*n[0]
        values[1] = g*n[1]

    def value_shape(self):
        return (2,)

G = BoundarySource(mesh, degree=2)

bcs = []

dead_cvs = mesh_data.evaluate_topology(R.dead)
deadfunc = project(mu.cell_function_function.make_cases(
    {cv: 1.0 for cv in dead_cvs}), mu.space.DG0)
# deadfunc_CG1 = project(deadfunc, mu.space.CG1)

output_u = np.array([0.0], dtype='double')
class MyRegionDomain(SubDomain):
    def inside(self, x, on_boundary):
        deadfunc.eval(output_u, x)
        if output_u[0] >= 0.9:
            return True
        return False

# BC ORDER MATTERS
# this region BC will be overwritten by the next one
bcs.append(DirichletBC(
    W.sub(0), [0.0, 0.0], MyRegionDomain(), method='pointwise'))
bcs.append(DirichletBC(
    W.sub(1), 0.0, MyRegionDomain(), method='pointwise'))

for fv, _ in mesh_data.evaluate_topology(R.live.boundary(R.dead)):
    bcs.append(DirichletBC(
        W.sub(0), G, mesh_data.facet_function, fv))

# Compute solution
w = Function(W)

from ...util.xtimeit import xtimeit

def thunk():
    w.vector()[:] = 0.0
    solve(a == L, w, bcs)

print("solve time", xtimeit(thunk, overall_time=3.0))
(sigma, u) = w.split()

with XDMFFile('het2_u.xdmf') as uf:
    uf.write(u,0)
    uf.write(sigma,1)

# Plot sigma and u
plt.figure()
plot(sigma)

plt.figure()
plot(u)

plt.figure()
plot(mesh, lw=0.5, color='k')
plot(celldata)


from ...util.plot import DolfinFunctionRenderer
from mpl_render import RenderingImShow, RenderingPlot

def justplotscalar(func):
    render = DolfinFunctionRenderer(
        func, extract_params=dict(component='magnitude'))
    fig, ax = plt.subplots()
    p = RenderingImShow(ax, kw=dict(cmap='inferno', aspect='auto'),
                        extent=(0, 1, -1, 2),
                        size=(300, 400), render_callback=render)
    p.force_single_thread = True
    ax.grid(True)

# plt.figure()
# justplotscalar(u)

# plt.figure()
# justplotscalar(deadfunc)


plt.show()



