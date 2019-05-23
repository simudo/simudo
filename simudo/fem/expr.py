
import unittest

import numpy as np
from numpy import array as npa

import dolfin
import ufl
from ufl.core.expr import Expr as ufl_Expr
from ufl.equation import Equation as ufl_Equation

__all__ = [
    'tuple_to_point',
    'point_to_array',
    'is_ufl_like',
    'force_ufl',
    'constantify_if_literal',
    'CellMidpointExpression',
    'scalar_function_to_Expression',
    'vector_function_to_Expression',
    'CellFunctionFunction',
    'dofs_coordinates',
    'mesh_bbox',
    'bbox_vecs',
]

def tuple_to_point(xs):
    xs = tuple(xs)
    l = 3 - len(xs)
    if l > 0:
        xs = xs + (0,)*l
    return dolfin.Point(np.array(xs))

def point_to_array(p):
    v = np.zeros(3)
    v[0] = p[0]
    v[1] = p[1]
    v[2] = p[2]
    return v

def is_ufl_like(expr):
    return isinstance(expr, (ufl_Expr, ufl_Equation))

def force_ufl(expr, zero=None):
    ''' Certain functions, like :py:func:`dolfin.project`, expect the
expression to be a UFL form, and crash if it's a plain old float. This
function converts such a constant value into a ufl expression. '''

    if hasattr(expr, 'magnitude'):
        return expr.units * force_ufl(expr.magnitude)

    if is_ufl_like(expr):
        return expr
    else:
        if zero is None:
            zero = dolfin.Constant(0.0)
        return zero + expr

def constantify_if_literal(value):
    '''If the argument is not a UFL expression, assume it is a
floating point number or array, and pass it to
:py:meth:`dolfin.Constant`. This avoids re-compilation when the
literal value changes.
'''
    if hasattr(value, 'magnitude'):
        u = value.units
        value = value.magnitude
    else:
        u = None

    if not hasattr(value, 'ufl_shape'):
        value = dolfin.Constant(value)

    if u is not None:
        value = u * value

    return value

UserExpression = dolfin.UserExpression

class CellMidpointExpression(UserExpression):
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh

    def eval_cell(self, values, x, cell):
        c = dolfin.Cell(self.mesh, cell.index)
        m = c.midpoint()
        for i in range(len(values)):
            values[i] = m[i]

def scalar_function_to_Expression(f):
    class MyExpression(UserExpression):
        def eval(self, values, x):
            values[0] = f(x)
    return MyExpression

def vector_function_to_Expression(f):
    class MyExpression(UserExpression):
        def eval(self, values, x):
            y = f(x)
            for i in range(len(values)):
                values[i] = y[i]
    return MyExpression

def _make_binary_search_tree(expr, mapping):
    mapping = tuple(sorted(mapping))
    max_key = mapping[-1][0]
    N = len(mapping)
    k = (N - 1).bit_length()
    def _tree(base, length):
        if base >= N:
            return 0.0
        if length == 1:
            return mapping[base][1]
        elif length > 1:
            l2 = length//2
            if base + l2 >= N: # no right side
                return _tree(base, l2)
            cut = (mapping[base + l2][0] + mapping[base + l2 - 1][0]) / 2
            return dolfin.conditional(
                dolfin.lt(expr, cut),
                _tree(base     , l2),
                _tree(base + l2, l2))
        else:
            raise AssertionError()
    return _tree(0, 2**k)

# print(_make_binary_search_tree(dolfin.Constant(0.5),
#                                [(i, dolfin.Constant(i))
#                                 for i in range(4)]))

def make_cell_function_expression(cell_function, **kwargs):
    class _CellFunctionCases(UserExpression):
        def eval_cell(self, values, x, cell):
            values[0] = cell_function[cell.index]
    return _CellFunctionCases(**kwargs)

def make_cases_expression(cell_function_function, mapping):
    return _make_binary_search_tree(cell_function_function, mapping)

class CellFunctionFunction(object):
    def __init__(self, cell_function, DG0_space):
        self.cell_function_value_set = frozenset(cell_function.array())
        expr = make_cell_function_expression(
            cell_function, element=DG0_space.ufl_element())
        self.function = dolfin.project(expr, DG0_space)
    def make_cases(self, mapping, default=0.0):
        mapping = dict(mapping)
        for k in self.cell_function_value_set:
            if k not in mapping:
                mapping[k] = default
        return make_cases_expression(self.function, mapping.items())

def dofs_coordinates(function_space):
    dim = function_space.mesh().geometry().dim()
    return function_space.tabulate_dof_coordinates().reshape((-1, dim))

def mesh_bbox(mesh):
    '''compute mesh bounding box'''
    coords = mesh.coordinates().view()
    def _lims(axis):
        return coords[:,axis].min(), coords[:,axis].max()
    return npa(tuple(_lims(i) for i in range(mesh.geometry().dim())))

def bbox_vecs(bbox):
    ''' r[i,j,k] == (bbox[i,j] if i==k else 0)

example use:

(left, right), (bottom, top) = bbox_vecs(bbox)
v = (bottom+top) / 2 + 1/3 * left + 2/3 * right
'''
    bbox = bbox.reshape(bbox.size//2, 2)
    r = np.stack([np.diagflat(bbox[:,i]) for i in (0, 1)], axis=0)
    return np.einsum('jik', r)

def interp(x, y0, y1=None):
    if y1 is None:
        y0, y1 = y0
    return (1-x)*y0 + x*y1

def value_shape(expr):
    # TODO: extent to expressions
    shape1 = expr.ufl_shape
    if hasattr(expr, 'value_dimension'):
        shape2 = tuple(int(expr.value_dimension(k))
                       for k in range(expr.value_rank()))
        assert shape1 == shape2
    return shape1

class CellEdgesExpression(UserExpression):
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh

    def eval_cell(self, values, x, cell):
        c = dolfin.Cell(self.mesh, cell.index)
        m = c.midpoint()
        for i in range(len(values)):
            values[i] = m[i]

def interpolate(expr, space, u=None):
    ''' `space` should be CG1

Note: This only exists because `dolfin.interpolate` doesn't work on
ufl expressions.
'''

    if u is None:
        u = dolfin.Function(space)

    v = dolfin.TestFunction(space)

    result = dolfin.assemble(dolfin.dot(expr, v)*dolfin.dP)
    u.vector()[:] = result.array()

    return u

class TestThisModule(unittest.TestCase):
    def maxdiff(self, u, v):
        return np.max(np.abs(
            u.vector().array() - v.vector().array()))

    def test_extract_nodal_CG1(self):
        mesh = dolfin.UnitSquareMesh(16, 16)
        el = dolfin.VectorElement("CG", mesh.ufl_cell(), 1)
        CG1 = dolfin.FunctionSpace(mesh, el)

        e_expr = dolfin.Expression(("x[0]-2*exp(x[1])", "x[0]*x[1]"), degree=1)
        x = dolfin.SpatialCoordinate(mesh)
        e_ufl = dolfin.as_vector((x[0]-2*dolfin.exp(x[1]), x[0]*x[1]))

        u0 = dolfin.interpolate(e_expr, CG1)

        u_nodal = interpolate(e_ufl, CG1) # nodal values
        u_proj  = dolfin.project(e_ufl, CG1) # projection/averaging

        maxdiff = self.maxdiff

        self.assertLess   (maxdiff(u0, u_nodal), 2*dolfin.DOLFIN_EPS)
        self.assertGreater(maxdiff(u0, u_proj ), 2*dolfin.DOLFIN_EPS)
