
import numpy as np

import pytest

from ..mesh import interval1dtag as IT


def test_interval1dtag():
    # FIXME: not a proper unit test

    I = IT.Interval
    CI = IT.CInterval

    intervals = [
        I(0, 1, 'A'),
        I(0, 2, 'B'),
        I(1, 5, 'C'),
        I(4, 9, 'A'),
        I(6, 7, 'B'),
        CI(0, 9, edge_length=0.25),
        CI(3, 3.5, edge_length=0.02),
        CI(6, 6.2, edge_length=0.02),
    ]

    o = IT.Interval1DTag(intervals)

    o.minimum_coordinate_distance = 0.05

    # print(o.subdomains)
    # print(o.coordinates)

    o.product2d_Ys = (0, 0.2, 0.3, 1.0)
    pm = o.product2d_mesh
    # import dolfin
    # dolfin.plot(pm.mesh)
    # dolfin.plot(pm.cell_function)
    # dolfin.interactive()
