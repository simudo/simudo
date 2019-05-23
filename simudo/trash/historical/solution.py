from __future__ import division, absolute_import, print_function
from builtins import bytes, dict, int, range, str, super

from ..util.raise_from import raise_from

from collections import namedtuple
import operator
from functools import reduce
from itertools import chain as ichain

import dolfin
import ufl
import pint

from ..pyaml import path_join

class Directory(object):
    def __init__(self, mapping, path):
        self._mapping_ = mapping
        self._path_ = path

    def __getattr__(self, key):
        return self[key.replace('__', '/')]

    def __getitem__(self, key):
        d = self._mapping_
        k = path_join(self._path_, key)
        if key.endswith('/'):
            return type(self)(mapping=d, path=k)
        else:
            return d[k]

    def __str__(self):
        return self._path_

    def __add__(self, other):
        return self[str(other).lstrip("/")]

_NO_SUCH_KEY = []

class Solution(object):
    def __init__(self, factory):
        self.dict = {}
        self.cdict = {}
        self.factory = factory

    def compute_key(self, key):
        if key.endswith('/'): # directory
            return Directory(self, key)

        try:
            func = self.factory.dict[key]
        except KeyError:
            return _NO_SUCH_KEY

        return func(self)

    def compute_key_wrapped(self, key, store=True):
        try:
            value = self.compute_key(key)
        except BaseException as exc:
            raise_from(
                RuntimeError("error in computing key {!r}".format(key)), exc)
        if store and value is not _NO_SUCH_KEY:
            self.cdict[key] = value
        return value

    def __getitem__(self, key):
        d  = self.dict
        v = d.get(key, None)
        if v is not None: return v

        cd = self.cdict
        v = cd.get(key, None)
        if v is not None: return v

        v = self.compute_key_wrapped(key)
        if v is _NO_SUCH_KEY:
            raise KeyError("{!r}".format(key))
        return v

    def keys(self):
        return frozenset(ichain.from_iterable(
            (self.dict.keys(),
             self.cdict.keys(),
             self.factory.dict.keys())))

    def __iter__(self):
        return iter(self.keys())

    def _debug_str(self):
        r = []
        for k in sorted(self.dict.keys()):
            v = self.dict[k]
            r.append(' {} = {!r}'.format(k, v))
        return '\n'.join(r)


class SolutionFactory(object):
    def __init__(self):
        self.dict = {}

    def __setitem__(self, name, value):
        self.dict[name] = value

    def new_solution(self, dictionary=None):
        sol = Solution(self)
        if dictionary:
            sol.dict.update(dictionary)
        return sol

