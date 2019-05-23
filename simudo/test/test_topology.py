
from itertools import chain, product

import numpy as np

import pytest

from ..mesh import CellRegion


def test_cell_regions():

    f = lambda A,B,C,D: ((A | B) - C) & D

    ids = dict(A = {1, 2, 3, 4},
               B = {1, 2, 5},
               C = {3, 6},
               D = {2, 3, 4})

    reg = f(**{k:CellRegion(k) for k in 'ABCD'})
    res = f(**ids)

    print(reg)

    assert reg.evaluate({'region_name_to_cvs': ids}) == res
