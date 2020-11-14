import dolfin
import numpy as np
from scipy.interpolate import interp1d

import pytest

from ..fem.expr import dolfin_interp1d


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_interp(n):
    X = [0, 5, 6, 10, 11, 13.2, 14, 15, 19, 3][:n]
    Y = [-17, 20, 5.553, -100, 5, -1, -2, -13, 1][:n]
    fill_value = (-3.3, 7.7)

    c = dolfin.Constant(1.0)
    e = dolfin_interp1d(c, X, Y, fill_value=fill_value)

    def f(x):
        c.assign(x)
        return e(0.0)

    if len(Y) == 1:
        g = lambda x: fill_value[0] if x < X[0] else fill_value[1]
    else:
        g = interp1d(
            X, Y, kind="linear", fill_value=fill_value, bounds_error=False
        )

    for x in np.arange(-1, 20, 0.1):
        assert f(x) == pytest.approx(g(x))
