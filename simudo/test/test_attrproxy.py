
import pytest

from ..util import AttrPrefixProxy


def test_common():
    class Classy(object):
        x = 1
        foo_x = 2

        def foo_bar(self):
            return self.foo_x

        def bar(self):
            return self.x

    obj = Classy()

    d = AttrPrefixProxy(obj, 'foo_')

    assert d.bar() == 2

    obj.x = 3

    assert d.bar() == 2

    d.x = 5

    assert obj.bar() == 3
    assert d.bar() == 5

    with pytest.raises(AttributeError):
        d.nonexistent()
