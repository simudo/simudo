from cached_property import cached_property

# TODO: docs, unit test

__all__ = ['NameDict']

class NameDict(object):
    _NULL = ['*null*']

    def __init__(self, iterable=None):
        if iterable is not None:
            add = self.add
            for x in iterable:
                add(x)

    @cached_property
    def mapping(self):
        return dict()

    def __iter__(self):
        return iter(self.mapping.values())

    def __getitem__(self, key):
        return self.mapping[key]

    def __setitem__(self, key, value):
        ''' the `key` argument is ignored '''
        self.replace(value)

    def add(self, obj, existing_raise=True):
        NULL = self._NULL
        key = self.obj_to_key(obj)
        obj0 = self.mapping.get(key, NULL)
        if obj0 is not NULL: # already exists
            if existing_raise and not self.obj_eq(obj0, obj):
                raise KeyError("key with different value {!r}".format(key))
            return obj0
        else:
            self.mapping[key] = obj
            return obj

    def replace(self, obj):
        self.mapping[self.obj_to_key(obj)] = obj

    def obj_to_key(self, obj):
        return obj.name

    def obj_eq(self, obj0, obj1):
        return id(obj0) == id(obj1)

    def items(self):
        return self.mapping.items()

    def keys(self):
        return self.mapping.keys()

    def values(self):
        return self.mapping.values()

    def __repr__(self):
        return '{}({{{}}})'.format(
            self.__class__.__name__,
            ', '.join(repr(x) for x in self.mapping.values()))

    def update(self, data, existing_raise=True):
        if hasattr(data, 'values'):
            data = data.values()
        add = self.add
        for x in data:
            add(x, existing_raise=existing_raise)
        return self

    def copy(self):
        return type(self)(self)

    def __or__(self, other):
        copy = self.copy()
        copy.update(other)
        return copy
