import pytest
import dolfin
import numpy as np
from itertools import product, chain
from ..util.probe import Probes

@pytest.fixture(scope="module")
def mesh1():
    return dolfin.UnitSquareMesh(10, 10)

pts = (0.0, 0.25, 0.5)
@pytest.mark.parametrize("x", chain(
    product(pts, pts),
    product(np.linspace(0, 1, 8), pts),
))
def test_probes(x, mesh1):
    el = dolfin.FiniteElement('BDM', mesh1.ufl_cell(), 2)
    W = dolfin.FunctionSpace(mesh1, el)
    z = dolfin.SpatialCoordinate(mesh1)
    expr = dolfin.as_vector((z[0], z[1]))
    w = dolfin.project(expr, W)
    ps = Probes(W)
    p = ps.probe(x)
    print(x, p(w), w(x))
    assert (p(w) == w(x)).all()

