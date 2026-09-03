import math
import unittest
import scipy.special

import ufl
from ufl.constantvalue import (
    FloatValue, IntValue, ScalarValue, Zero, as_ufl, is_true_ufl_scalar)
from ufl.core.ufl_type import ufl_type
from ufl.mathfunctions import MathFunction
from ufl.log import warning, error
from .compat import update_ufl_expr_classes
from simudo.fem import setup_dolfin_parameters

import dolfin 
import numpy as np

__all__ = ['ExpM1', 'expm1', 'Ln1P', 'ln1p', 'LambertW0', 'lambert_w0','lambert_w0_conditional']

'''
note: MathFunction.derivative() is an undocumented attribute which can
be used to specify a function's derivative; see
`ufl.algorithms.apply_derivatives.GenericDerivativeRuleset.math_function`
for usage.
'''


def ufl_mathfunction(f, cls):
    # taken from ufl.operators
    f = as_ufl(f)
    r = cls(f)
    if isinstance(r, (ScalarValue, Zero, int, float)):
        return float(r)
    return r


@ufl_type()
class ExpM1(MathFunction):
    __slots__ = ()

    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.expm1(float(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        MathFunction.__init__(self, "expm1", argument)

    def derivative(self):
        f, = self.ufl_operands
        return ufl.exp(f)

@ufl_type()
class Ln1P(MathFunction):
    __slots__ = ()

    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(math.log1p(float(argument)))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        MathFunction.__init__(self, "log1p", argument)

    def derivative(self):
        x, = self.ufl_operands
        return 1 / (x + 1)

@ufl_type()
class LambertW0(MathFunction):
    __slots__ = ()

    @staticmethod
    def _evaluate(x):
        return scipy.special.lambertw(float(x)).real

    def __new__(cls, argument):
        if isinstance(argument, (ScalarValue, Zero)):
            return FloatValue(cls._evaluate(argument))
        return MathFunction.__new__(cls)

    def __init__(self, argument):
        MathFunction.__init__(self, "simudo_safe_lambert_w0", argument)  # Claude Sonnet 4.6

    def evaluate(self, x, mapping, component, index_values):
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        return self._evaluate(a)

    def derivative(self):
        # Derivative is W/(z*(1+W)) but also 1/(z + exp(W))
        x, = self.ufl_operands
        cls = type(self)
        return 1/(x + ufl.exp(cls(x)))


def expm1(x):
    return ufl_mathfunction(x, ExpM1)

def ln1p(x):
    # TODO: unit test
    return ufl_mathfunction(x, Ln1P)

def lambert_w0(x):
    return ufl_mathfunction(x, LambertW0)

def lambert_w0_conditional(x):
    '''boost's lambert_w0 function gives an error when given infinite input.
    Use asymptotic form for large argument'''
    return ufl.conditional(
            ufl.eq(x, x), #test for nans, x!=x
            ufl.conditional(
                ufl.lt(x,1e300), #test for infinite values
                lambert_w0(x),
                ufl.ln(x) - ufl.ln(ufl.ln(x))),
            ufl.ln(x) - ufl.ln(ufl.ln(x)) #first two terms of asymptotic expansion
            )
    

# This is necessary because we've defined new ufl types.
update_ufl_expr_classes()
setup_dolfin_parameters()


class Test(unittest.TestCase):
    def setUp(self):
        self.mesh = mesh = dolfin.UnitIntervalMesh(30)
        self.element = element = dolfin.FiniteElement("CG", mesh.ufl_cell(), 3)
        self.W = W = dolfin.FunctionSpace(mesh, element)
        self.u = dolfin.Function(W, name='u')
        self.v = dolfin.TestFunction(W)
        self.x = dolfin.SpatialCoordinate(mesh)[0]

    def test_expm1(self):
        x = self.x
        u1 = expm1(x)
        u2 = dolfin.exp(x) - 1.0
        err = dolfin.assemble((u1-u2)**2*dolfin.dx)
        self.assertLessEqual(err, 1e-8)

    def test_lambert_w0(self):
        x = self.x
        W = self.W
        w = lambert_w0(x)
        u1 = w*dolfin.exp(w)
        u2 = x
        err = dolfin.assemble((u1-u2)**2*dolfin.dx)
        self.assertLessEqual(err, 1e-8)

        # u3 = lambert_w0(x*2e100)
        # xx = np.linspace(0,1,5)
        # f = dolfin.project(u3, W)
        # for i in range(len(xx)):
        #     print([xx[i], f(xx[i])])

    def test_lambert_w0_inf(self):  # Claude Sonnet 4.6
        # lambert_w0(inf) must return inf without throwing.
        # Check via assembled linear form: ∫ lambert_w0(inf)*v dx should be all-inf.
        # (dolfin.project cannot be used here because the FEM linear solve breaks
        # with infinite RHS entries.)
        v = dolfin.TestFunction(self.W)
        u = lambert_w0(dolfin.Constant(float('inf')))
        b = dolfin.assemble(u * v * dolfin.dx)
        self.assertTrue(np.all(np.isinf(b.get_local())))

    def test_lambert_w0_conditional(self):
        x = self.x
        W = self.W
        u1 = lambert_w0_conditional(x*2e100)
        xx = np.linspace(0,1,5)
        f = dolfin.project(u1, W)
        # for i in range(len(xx)):
        #     print([xx[i], f(xx[i])])

    def test_expm1_derivative(self):
        ''' check that we can take functional derivatives '''
        x = self.x
        u = self.u
        v = self.v
        dolfin.project(x, self.W, function=u)
        F = expm1(u*u)*v*dolfin.dx - (dolfin.exp(x) - 1.0)*v*dolfin.dx

        from .newton_solver import NewtonSolver
        solver = NewtonSolver(F, u, [])
        solver.parameters['relative_tolerance'] = 1e-14
        solver.solve()

        u_e = dolfin.sqrt(x)

        err = dolfin.assemble((u - u_e)**2*dolfin.dx)
        self.assertLessEqual(err, 1e-8)
