import sympy
import dolfin

from sympy.physics.vector import Vector

try:
    from sympy.printing.ccode import CCodePrinter
except ImportError:
    from sympy.printing.ccode import C99CodePrinter as CCodePrinter

class DolfinCCodePrinter(CCodePrinter):
    def _print_Pi(self, expr):
        return 'pi'

def ccode(expr, **settings):
    return DolfinCCodePrinter(settings).doprint(expr, None)

def to_coord_symbols(expr, R):
    return expr.subs({k:v for k,v in zip(
        R.varlist, sympy.symbols("x[0] x[1] x[2]"))})

def sympy_to_dolfin(expr, element, R, kwargs={}):
    expr = to_coord_symbols(expr, R)
    if isinstance(expr, Vector):
        dim = element.value_shape()[0]
        expr = tuple(ccode(e) for e in expr.to_matrix(R)[:dim,:])
    else:
        expr = ccode(expr)
    return dolfin.Expression(expr, element=element, **kwargs)
