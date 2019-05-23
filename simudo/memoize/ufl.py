
class EqualsToEq(object):
    '''
    This is necessary because ufl Forms override `__eq__` so that `a
    == b` creates an Equation object (to be passed to a solver for
    example). We need something that behaves correctly with respect to
    `__eq__` and `__hash__`, so we wrap ufl objects with this class.
    '''
    __just_a_stupid_wrapper__ = True

    def __init__(self, obj):
        self.object = obj

    def __hash__(self):
        return hash(self.object)

    def _eq_unwrap(self, obj):
        if hasattr(obj, '__just_a_stupid_wrapper__'):
            return obj.object
        return obj

    def __eq__(self, other):
        obj = self.object
        other = self._eq_unwrap(other)
        eq = getattr(obj, 'equals', None)
        if eq:
            return eq(other)
        return obj == other

    def __repr__(self):
        return repr(self.object)

    def __str__(self):
        return str(self.object)



