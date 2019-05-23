from __future__ import absolute_import, division, print_function

import math
import unittest
from builtins import bytes, dict, int, range, str, super

import dolfin
import ufl
from ufl.constantvalue import (
    FloatValue, IntValue, ScalarValue, Zero, as_ufl, is_true_ufl_scalar)
from ufl.core.ufl_type import ufl_type
from ufl.mathfunctions import MathFunction

__all__ = ['ExpM1', 'expm1']

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

def expm1(x):
    return ufl_mathfunction(x, ExpM1)

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
        u1 = dolfin.project(expm1(x), self.W)
        u2 = dolfin.project(dolfin.exp(x)-1.0)
        err = dolfin.assemble((u1-u2)**2*dolfin.dx)
        self.assertLessEqual(err, 1e-8)

    def test_expm1_derivative(self):
        ''' check that we can take functional derivatives '''
        x = self.x
        u = self.u
        v = self.v
        dolfin.project(x, self.W, function=u)
        F = expm1(u*u)*v*dolfin.dx - (dolfin.exp(x) - 1.0)*v*dolfin.dx

        from ..newtonsolver import NewtonSolver
        solver = NewtonSolver(F, u, [])
        solver.parameters['relative_tolerance'] = 1e-14
        solver.solve()

        u_e = dolfin.sqrt(x)

        err = dolfin.assemble((u - u_e)**2*dolfin.dx)
        self.assertLessEqual(err, 1e-8)
