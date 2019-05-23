from __future__ import absolute_import, division, print_function

from builtins import bytes, dict, int, range, str, super

__all__ = ['DictAttrProxy', 'AttrPrefixProxy']

class DictAttrProxy(object):
    def __init__(self, mapping):
        super().__setattr__('_mapping_', mapping)

    def __getattr__(self, key):
        return self._mapping_[key]

    def __setattr__(self, key, value):
        self._mapping_[key] = value

class AttrPrefixProxy(object):
    def __init__(self, object, prefix):
        self.__dict__.update(
            _object=object,
            _prefix=prefix)

    def _transform_attrname(self, attr):
        return self._prefix + attr

    def __getattr__(self, attr):
        return getattr(self._object, self._transform_attrname(attr))

    def __setattr__(self, attr, value):
        setattr(self._object, self._transform_attrname(attr), value)
