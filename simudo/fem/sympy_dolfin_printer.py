import math
import operator
import unittest
from functools import partial, reduce

import numpy as np
import sympy
from sympy import Integer, Rational
from sympy.core.expr import Expr
from sympy.core.function import ArgumentIndexError, Function
from sympy.physics.vector import ReferenceFrame, Vector
from sympy.printing.str import Printer

import dolfin

__all__ = [
    'DolfinConstant',
    'DolfinPrinter',
    'sympy_to_ufl_base',
    'sympy_to_ufl']

# TODO: acosh, conditional

def my_abs(x):
    return dolfin.sqrt(x**2)

foldargs = lambda func: lambda *args: reduce(func, args)

_mapping = {
    sympy.Add: foldargs(operator.add),
    sympy.Mul: foldargs(operator.mul),
    sympy.Abs: my_abs,
    sympy.Pow: operator.pow,
    sympy.log: dolfin.ln}
for k in ('sin',  'cos',  'tan',
          'asin', 'acos', 'atan',
     #     'sinh', 'cosh', 'tanh',
          'erf',
          'exp', 'sqrt'):
    _mapping[getattr(sympy, k)] = getattr(dolfin, k)

class DolfinConstant(sympy.Function):
    nargs = 1

    @property
    def value(self):
        return self.args[0]

    @classmethod
    def release(cls, expr):
        return expr.replace(cls, sympy.Id)

class DolfinPrinter(Printer):
    """ Code printer for Theano computations """
    printmethod = "_dolfin"

    mapping = _mapping
    def __init__(self, *args, **kwargs):
        self.syms = kwargs.pop('syms')
        super(DolfinPrinter, self).__init__(*args, **kwargs)

    def _print_Basic(self, expr, **kwargs):
        op = self.mapping[type(expr)]
        children = [self._print(arg, **kwargs) for arg in expr.args]
        return op(*children)

    def _print_Symbol(self, s, **kwargs):
        return self.syms[s]

    def _print_DolfinConstant(self, expr, **kwargs):
        return dolfin.Constant(float(expr.value))

    def _print_Number(self, expr, **kwargs):
        return float(expr)

    def emptyPrinter(self, expr):
        return expr

    def _print_Vector(self, expr, **kwargs):
        return self._print(expr.to_matrix(
            kwargs['reference_frame'])[:kwargs['dim'],:], **kwargs)

    def _print_ImmutableMatrix(self, expr, **kwargs):
        assert expr.shape[1] == 1
        return dolfin.as_vector(
            tuple(self._print(e, **kwargs) for e in tuple(expr)))

    def _print_ImmutableDenseMatrix(self, *args, **kwargs):
        return self._print_ImmutableMatrix(*args, **kwargs)

    def doprint(self, expr, **kwargs):
        return self._print(expr, **kwargs)

def sympy_to_ufl_base(expr, syms, **kwargs):
    return DolfinPrinter(syms=syms).doprint(expr, **kwargs)

def sympy_to_ufl(R, mesh, expr, syms=None, **kwargs):
    syms = syms or dict()
    dim = mesh.geometry().dim()
    coord = dolfin.SpatialCoordinate(mesh)
    syms.update(zip(R.varlist, coord))
    kwargs.setdefault('dim', dim)
    kwargs.setdefault('reference_frame', R)
    return sympy_to_ufl_base(expr, syms, **kwargs)

class TestThis(unittest.TestCase):
    def test_print(self):
        from .expr import tuple_to_point

        mesh = dolfin.UnitSquareMesh(25, 25)
        muc = mesh.ufl_cell()

        R = ReferenceFrame('R')

        from sympy import exp, atan
        x, y = R[0], R[1]

        s_expr = x**2 + exp(y - DolfinConstant(0.5)) - atan(x - y)
        u_expr = sympy_to_ufl(R, mesh, s_expr)

        el1 = dolfin.FiniteElement('CG', muc, 1)
        W1 = dolfin.FunctionSpace(mesh, el1)

        u1 = dolfin.project(u_expr, W1)

        f = sympy.lambdify((x, y), DolfinConstant.release(s_expr))
        for x0 in np.linspace(0, 1, 10):
            for y0 in np.linspace(0, 1, 10):
                a, b = f(x0, y0), u1(tuple_to_point((x0, y0)))
                self.assertLess(abs(a/b - 1), 0.001)

        s2_expr = (y * R.x + x * y * R.y).to_matrix(R)[:2,:]
        u2_expr = sympy_to_ufl(R, mesh, s2_expr)

        el2 = dolfin.FiniteElement('BDM', muc, 1)
        W2 = dolfin.FunctionSpace(mesh, el2)
        u2 = dolfin.project(u2_expr, W2)

        self.assertLess(abs(u2(tuple_to_point((0.2, 0.2)))[1] - 0.04), 0.001)
