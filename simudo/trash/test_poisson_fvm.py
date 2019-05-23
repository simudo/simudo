from __future__ import division, absolute_import, print_function
from builtins import bytes, dict, int, range, str, super

import unittest

from .util import CellMidpointExpression
from .plot import XDMFPlotter
import dolfin
from ..function_subspace_registry import FunctionSubspaceRegistry

class BaseFVMDemo:
    def __init__(self, *args, **kwargs):
        self.fsr = fsr
        super().__init__(*args, **kwargs)

    def init_space(self):
        self.mesh = mesh = dolfin.UnitSquareMesh(20, 2, "left")
        self.DG0_element = DG0e = dolfin.FiniteElement("DG", mesh.ufl_cell(), 0)
        self.DG0v_element = DG0ve = dolfin.VectorElement("DG", mesh.ufl_cell(), 0)
        self.DG0 = DG0 = dolfin.FunctionSpace(mesh, DG0e)
        self.DG0v = DG0v = dolfin.FunctionSpace(mesh, DG0ve)

        self.fsr.register(DG0)
        self.fsr.register(DG0v)

    def init_variables(self):
        mesh = self.mesh
        DG0 = self.DG0
        self.n = n = dolfin.FacetNormal(mesh)

        self.u = u = dolfin.Function(DG0)
        self.v = v = dolfin.TestFunction(DG0)

    def init_bc(self):
        eps = 1e-10
        class BL(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return abs(x[0]) < eps
        class BR(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return abs(x[0]-1) < eps

        ff = dolfin.FacetFunction('size_t', mesh, 0)
        BL().mark(ff, 1)
        BR().mark(ff, 1)

        ds = dolfin.Measure('ds', domain=mesh, subdomain_data=ff)


class AdvectiveFVMTest(unittest.TestCase):
    pass

class PoissonFVMTest(unittest.TestCase):
    def setUp(self):
        self.mesh = mesh = dolfin.UnitSquareMesh(20, 2, "left")
        self.DG0_element = DG0e = dolfin.FiniteElement("DG", mesh.ufl_cell(), 0)
        self.DG0v_element = DG0ve = dolfin.VectorElement("DG", mesh.ufl_cell(), 0)
        self.DG0 = DG0 = dolfin.FunctionSpace(mesh, DG0e)
        self.DG0v = DG0v = dolfin.FunctionSpace(mesh, DG0ve)

        self.fsr = fsr = FunctionSubspaceRegistry()
        fsr.register(DG0)
        fsr.register(DG0v)

        self.cellmid = cm = CellMidpointExpression(mesh, element=DG0ve)
        self.n = n = dolfin.FacetNormal(mesh)

        self.u = u = dolfin.Function(DG0)
        self.v = v = dolfin.TestFunction(DG0)

        u_bc = dolfin.Expression('x[0]', degree=2)

        x = dolfin.SpatialCoordinate(mesh)
        self.rho = rho = dolfin.conditional(
            dolfin.lt((x[0] - 0.5)**2 + (x[1] - 0.5)**2, 0.2**2),
            0.0, 0.0)

        dot = dolfin.dot
        cellsize = dolfin.CellSize(mesh)
        self.h = h = cm('+') - cm('-')
        self.h_boundary = h_boundary = 2*n*dot(x - cm, n)
        self.E = E = h/dot(h, h)*(u('-') - u('+'))
        self.E_boundary = E_boundary = h_boundary/dot(h_boundary, h_boundary)*(u - u_bc)
        dS = dolfin.dS

        eps = 1e-8
        class BL(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return abs(x[0]) < eps
        class BR(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return abs(x[0]-1) < eps

        ff = dolfin.FacetFunction('size_t', mesh, 0)
        BL().mark(ff, 1)
        BR().mark(ff, 1)

        ds = dolfin.Measure('ds', domain=mesh, subdomain_data=ff)

        self.F = (dot(E, n('+'))*v('+')*dS + dot(E, n('-'))*v('-')*dS
                  - v*rho*dolfin.dx
                  + dot(E_boundary, n)*v*ds(1))      

    def test_solve(self):
        from ..newtonsolver import NewtonSolver
        solver = NewtonSolver(self.F, self.u, [])
        solver.parameters['relative_tolerance'] = 1e-18
        solver.parameters['maximum_iterations'] = 6
        solver.parameters['relaxation_parameter'] = 1.0
        solver.solve()
        xp = XDMFPlotter("out/fvmplot.xdmf", self.fsr)
        xp.add(0.0, "u", 1, self.u)


# alpha/h_avg*dot(jump(v, n), jump(u, n))*dS
