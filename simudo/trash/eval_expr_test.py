
import dolfin
import numpy as np
from .stupid2d import *
from ..util.eval_expr import UFLEvaluator

ev = UFLEvaluator()

ev.coordinates = np.array([
    [0.1, 0.2],
    [0.1, 0.3],
    [0.1, 0.5]])

a = ev(3 * w + dolfin.as_vector((1.0, 2.0)) * dolfin.exp(2 * w[1]))

print(a)
