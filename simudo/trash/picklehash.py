from __future__ import division, absolute_import, print_function
from builtins import bytes, dict, int, range, str, super

import collections


def picklehash(obj):
    seent = dict()
    def _f(o):
        if   isinstance(o, str):
            out(b"s")
            _f(o.encode("utf8"))
        elif isinstance(o, bytes):
            out(b"b")
            _f(len(o))
            out(o)
        elif isinstance(o, int):
            out(b"i")
            out(str(o).encode('ascii'))
            out(b"e")
        elif isinstance(o, collections.Mapping):
            out(b"d")
            for k in sorted(o):
                _f(k)
                _f(o[k])
            out(b"e")
        elif isinstance(o, collections.Sequence):
            out(b"l")
            for x in o:
                _f(x)
            out(b"e")
        elif hasattr(o, '__getstate__'):
            out(b"o")
            out(type().encode('ascii'))

