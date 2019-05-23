
from .stupid2d import *
from cached_property import cached_property

import ufl

example_form1 = dolfin.dot(w, example_testfunc) * dolfin.dx() + w[0] * dolfin.ds()
example_form2 = w[0] * dolfin.ds() + dolfin.dot(w, example_testfunc) * dolfin.dx()

e2 = w[0]**2 + dolfin.Constant(5)

print(hash(example_form1), hash(example_form2))

print(ufl.algorithms.analysis.extract_arguments_and_coefficients(example_form))
print(ufl.algorithms.analysis.extract_coefficients(example_form))

class Trans(ufl.algorithms.transformer.ReuseTransformer):
    @cached_property
    def terminal_set(self):
        return set()

    def terminal(self, o):
        self.terminal_set.add(o)
        return o

trans = Trans()
ufl.algorithms.transformer.apply_transformer(e2, trans)
print(trans.terminal_set)

a = trans.terminal_set.pop()

print(type(a))

