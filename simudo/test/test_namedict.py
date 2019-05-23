import pytest

from ..util import NameDict


def test_common():
    class StrNameDict(NameDict):
        def obj_to_key(self, obj):
            return obj.lower()

    d = StrNameDict()

    d.add('Hello')
    d.add('World')
    with pytest.raises(KeyError):
        d.add('hello')

    assert d['hello'] == 'Hello'

    d.add('hello', existing_raise=False)

    assert d['hello'] == 'Hello'

    d.replace('WORLD')

    assert d['world'] == 'WORLD'

    d[None] = 'HELLO'

    assert d['hello'] == 'HELLO'
