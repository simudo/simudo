
import operator
from functools import reduce

import numpy as np

from ufl.constantvalue import Zero as ufl_Zero

__all__ = ['DelayedForm']

def multiply1(*args):
    return reduce(operator.mul, args)

class DelayedForm(object):
    def __init__(self, *terms):
        terms = tuple((integrand, measure)
                      for (integrand, measure) in terms
                      if not (integrand is 0 or integrand is 0.0))
        self.terms = terms

    def _iszero(self, x):
        if hasattr(x, 'magnitude'):
            x = x.magnitude
        if isinstance(x, np.number):
            return x == 0
        if isinstance(x, ufl_Zero):
            return True
        return x is 0 or x is 0.0

    def __add__(self, other):
        if self._iszero(other): return self
        return DelayedForm(*(self.terms + other.terms))

    def __radd__(self, other):
        if self._iszero(other): return self
        return DelayedForm(*(other.terms + self.terms))

    def __pos__(self):
        return self

    def __neg__(self):
        return DelayedForm(*((integrand*-1, measure)
                             for integrand, measure in self))

    def __sub__(self, other):
        return self + (other*-1)

    def __mul__(self, coefficient):
        return DelayedForm(*((multiply1(integrand, coefficient), measure)
                             for integrand, measure in self))

    def __truediv__(self, coefficient):
        return DelayedForm(*((integrand/coefficient, measure)
                             for integrand, measure in self))

    def __rtruediv__(self, coefficient):
        return DelayedForm(*((coefficient/integrand, measure)
                             for integrand, measure in self))

    def __rmul__(self, coefficient):
        return DelayedForm(*((multiply1(coefficient, integrand), measure)
                             for integrand, measure in self))

    def __repr__(self):
        return '<DForm {}>'.format(
            ' + '.join(('({}, {})'.format(integrand, measure)
                        for integrand, measure in self)))

    def map(self, func):
        return DelayedForm(*(func(integrand, measure)
                             for integrand, measure in self))

    def __iter__(self):
        return iter(self.terms)

    def __call__(self, *args, **kwargs):
        return DelayedForm(*((integrand, measure(*args, **kwargs))
                             for integrand, measure in self))

    def to_ufl(self, units=None):
        l = self.terms
        if not len(l):
            raise ValueError("empty form")

        representative_qty = l[0][0]

        if isinstance(units, str):
            units = representative_qty._REGISTRY(units)

        if units is None:
            units = representative_qty.units

        return units*reduce(operator.add, (
            integrand.m_as(units)*measure for integrand, measure in l))

    def delete_units(self):
        l = self.terms
        if not len(l):
            return self

        dimensionless = l[0][0]._REGISTRY.dimensionless
        return self.map(lambda integrand, measure:
                        (integrand.magnitude * dimensionless, measure))

    @classmethod
    def from_sum(cls, iterable):
        v = cls()
        for x in iterable:
            v = v + x
        return v

    def abs(self):
        terms = []
        for integrand, measure in self.terms:
            x = integrand
            if hasattr(x, 'magnitude'):
                x = x.magnitude
            if (x == -1) is True:
                integrand = -integrand
            terms.append((integrand, measure))
        return DelayedForm(*terms)
