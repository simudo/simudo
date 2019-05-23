from ...util.function_subspace_registry import FunctionSubspaceRegistry
from ...util.expr import DelayedForm
from ...util import sympy_dolfin_printer
from ...util.newton_solver import NewtonSolver
from ...io.xdmf import XdmfPlot

from contextlib import closing
import dolfin
import sympy
from sympy.physics.vector import ReferenceFrame, gradient, divergence

def derive_exprs():

    R = ReferenceFrame('N')
    x, y = R[0], R[1]

    w = x**2*(1-x)**2/10
    c = gradient(w, R) * sympy.exp(-w)
    f = divergence(c, R)

    return dict(quantities=dict(w=w, c=c, f=f), R=R)

def main():
    fsr = FunctionSubspaceRegistry()

    deg = 2
    mesh = dolfin.UnitSquareMesh(100, 3)
    muc = mesh.ufl_cell()
    el_w = dolfin.FiniteElement('DG', muc, deg-1)
    el_j = dolfin.FiniteElement('BDM', muc, deg)
    el_DG0 = dolfin.FiniteElement('DG', muc, 0)
    el = dolfin.MixedElement([el_w, el_j])
    space = dolfin.FunctionSpace(mesh, el)
    DG0 = dolfin.FunctionSpace(mesh, el_DG0)
    fsr.register(space)
    facet_normal = dolfin.FacetNormal(mesh)
    xyz = dolfin.SpatialCoordinate(mesh)

    trial = dolfin.Function(space)
    test = dolfin.TestFunction(space)

    w, c = dolfin.split(trial)
    v, phi = dolfin.split(test)

    sympy_exprs = derive_exprs()
    exprs = {k: sympy_dolfin_printer.to_ufl(sympy_exprs['R'], mesh, v)
             for k, v in sympy_exprs['quantities'].items()}

    f = exprs['f']
    w0 = dolfin.project(dolfin.conditional(dolfin.gt(xyz[0], 0.5), 1.0, 0.3), DG0)
    w_BC = exprs['w']

    dx = dolfin.dx()
    form = (
        + v * dolfin.div(c) * dx
        - v * f * dx

        + dolfin.exp(w + w0) * dolfin.dot(phi, c) * dx
        + dolfin.div(phi) * w * dx
        - (w_BC - w0) * dolfin.dot(phi, facet_normal) * dolfin.ds()
        - (w0('-') - w0('+')) * dolfin.dot(phi('+'), facet_normal('+')) * dolfin.dS()
    )

    solver = NewtonSolver(form, trial, [], parameters=dict(
        relaxation_parameter=1.0,
        maximum_iterations=15,
        extra_iterations=10,
        relative_tolerance=1e-6,
        absolute_tolerance=1e-7))

    solver.solve()

    with closing(XdmfPlot("out/qflop_test.xdmf", fsr)) as X:
        CG1 = dolfin.FunctionSpace(mesh, dolfin.FiniteElement('CG', muc, 1))
        X.add('w0', 1, w0, CG1)
        X.add('w_c', 1, w + w0, CG1)
        X.add('w_e', 1, exprs['w'], CG1)
        X.add('f', 1, f, CG1)
        X.add('cx_c', 1, c[0], CG1)
        X.add('cx_e', 1, exprs['c'][0], CG1)

if __name__=='__main__':
    main()

