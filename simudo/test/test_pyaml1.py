
import pkg_resources
from cached_property import cached_property

import pytest

from .. import pyaml
from ..pyaml.helper import PyamlEnv, PyamlGetAttr
from ..pyaml.multidict import multidict_items
from ..pyaml.pyaml1 import CodeGenerator


def test_test1():
    mod = pyaml.load_res(__name__, 'pyaml1_test1.yaml')

    class Example(mod.Example):
        @cached_property
        def unit_registry(self):
            import pint
            return pint.UnitRegistry()

        @cached_property
        def w(self): return super().w

    ex = Example()

    # print(src)

    assert ex.w is ex.w # cached_property
    assert ex.qflmixed_w0 is not ex.qflmixed_w0 # no caching so different objs

    assert ex.qflmixed_w == ex.qw

    assert ex.qflmixed_expr.m_as('dimensionless') == 0.5 * ex.w.m_as('eV')

def test_multidict():
    d = {'x@1':1, "x@2":2, "y":3}

    assert frozenset(multidict_items(d)) == frozenset(
        (('x', 1), ('x', 2), ('y', 3)))
