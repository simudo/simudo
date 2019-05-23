
from .ufl import EqualsToEq
from uuid import uuid4
from threading import Lock, RLock

class Memo():
    @cached_property
    def lock(self):
        return threading.RLock()

    def ufl_value_cache_key(self, expr):
        coeffs = ufl.algorithms.analysis.extract_coefficients(expr)
        return (expr, tuple(
            (coeff, self.ufl_coefficient_value_cache_key(coeff))))

    def ufl_coefficient_value_cache_key(self, coeff):
        fspace = getattr(coeff, 'function_space', None)
        if fspace is not None: fspace = fspace()
        mesh = fspace.mesh()
        return (self.mutable_cache_key(coeff),
                self.mutable_cache_key(mesh))

    def mutable_key(self, value):
        ''' construct key corresponding to mutable value `value` '''

         # HACK: duck typing to tell whether it's okay to do repr()
        if hasattr(value, 'ufl_id'):
            return repr(value)

        # as a backup, we could use id(object) instead of crashing ?
        raise NotImplementedError("don't know how to handle {!r}".format(value))

    def mutable_cache_key(self, value):
        return self.mutable_get(self.mutable_key(value))

    ## mutable_table accessors ##

    @cached_property
    def mutable_table(self):
        return defaultdict(uuid4)

    def mutable_get(self, key):
        with self.lock:
            return self.mutable_table[value]

    def mutable_touch(self, value):
        with self.lock:
            del self.mutable_table[value]
            return self.mutable_table[value]

