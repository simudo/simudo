import dolfin
import ufl
import ufl.algebra
import ufl.constantvalue
import ufl.indexed
import ufl.mathfunctions
import ufl.tensors
import ufl.tensoralgebra
from functools import reduce
import operator as op
from cached_property import cached_property
import numpy as np
from .expr import value_shape
from .extra_bindings import Function_eval_many

''' DO NOT USE '''

class UFLEvaluator():
    ''' somewhat efficient evaluator for a subset of ufl '''
    type_handlers = [
        (ufl.algebra.Sum, 'ufl_sum'),
        (ufl.algebra.Product,  'ufl_product'),
        (ufl.algebra.Division, 'ufl_division'),
        (ufl.mathfunctions.Exp, 'ufl_exp'),
        (ufl.mathfunctions.Ln, 'ufl_log'),
        (ufl.tensoralgebra.Inner, 'ufl_inner'),
        (ufl.tensors.ComponentTensor, 'ufl_ComponentTensor'),
        (dolfin.Constant, 'dolfin_Constant'),
        (dolfin.Function, 'dolfin_Function'),
        (ufl.constantvalue.ConstantValue, 'ufl_ConstantValue'),
        (ufl.indexed.Indexed, 'ufl_Indexed'),
    ]

    @cached_property
    def cache(self):
        return {}

    def get_cache_key(self, expr):
        return expr

    def ufl_sum(self, expr):
        return sum(self(x) for x in expr.ufl_operands)

    def ufl_product(self, expr):
        return reduce(op.mul, (self(x) for x in expr.ufl_operands))

    def ufl_division(self, expr):
        a, b = expr.ufl_operands
        return self(a) / self(b)

    def ufl_exp(self, expr):
        a, = expr.ufl_operands
        return np.exp(self(a))

    def ufl_log(self, expr):
        a, = expr.ufl_operands
        return np.log(self(a))

    def ufl_inner(self, expr):
        a, b = expr.ufl_operands
        return np.dot(self(a), self(b))

    def ufl_Indexed(self, expr):
        obj, index = expr.ufl_operands
        try:
            index = int(str(index))
        except Exception as e:
            raise TypeError(
                "we don't support non-constant indices yet, sorry") from e
        return self(obj)[:, index, ...]

    def ufl_ComponentTensor(self, expr):
        components = [self(x) for x in expr.ufl_operands]
        return np.stack(components, axis=-1)

    def ufl_ConstantValue(self, expr):
        return np.array(expr.value())[None, ...]

    def dolfin_Constant(self, expr):
        return expr.values().reshape(value_shape(expr))[None, ...]

    def Function_eval_default_value(self, expr):
        return np.nan

    def dolfin_Function(self, expr):
        XY = self.coordinates
        N = len(XY)

        values = np.full(
            expr.value_size() * N,
            self.Function_eval_default_value(expr), dtype='float64')

        Function_eval_many(expr, 2, N, values.ravel(), XY.ravel())

        return values.reshape((N,) + value_shape(expr))

    def __call__(self, expr):
        print("*** {0} {0!r}".format(expr))
        cache = self.cache
        k = self.get_cache_key(expr)
        v = cache.get(k, None)

        if v is None:
            cache[k] = v = self._eval(expr)

        return v

    def _eval(self, expr):
        ''' this does not use the cache. use __call__ instead '''
        for typ, handler in self.type_handlers:
            if isinstance(expr, typ):
                return getattr(self, handler)(expr)

        raise TypeError("unknown type {!r} for {!r}"
                        .format(type(expr), expr))

'''
above is a bad idea, because it amounts to reimplementing all of the
ufl logic

instead let's use the .evaluate() method and override only the
problematic classes
'''


class SimpleEvaluateOverride():
    def evaluate_helper(self, *args):
        raise NotImplementedError()

    def evaluate(self, x, mapping, component, index_values):
        os = [o.evaluate(self, x, mapping, component, index_values)
              for o in self.ufl_operands]
        return self.evaluate_helper(*os)

def create_simple_evaluate_override(ufl_cls, func):
    cls = type('uflwrap_' + ufl_cls.__name__,
               (SimpleEvaluateOverride, ufl_cls), {})
    return cls

class BetterEvaluate():
    simple_1arg_override_evaluate = [
        (ufl.mathfunctions.Exp, np.exp),
        (ufl.mathfunctions.Ln , np.log),
        (ufl.mathfunctions.Sin, np.sin),
        (ufl.mathfunctions.Cos, np.cos),
        (ufl.mathfunctions.Tan, np.tan),
    ]

    @cached_property
    def class_map(self):
        return [(ufl_cls, create_simple_evaluate_override(ufl_cls, func))]

    def _eval(self):
        pass

    def traverse_all(self):
        pass


