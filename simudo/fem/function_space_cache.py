from __future__ import print_function

import dolfin

__all__ = ['FunctionSpaceCache']

class FunctionSpaceCache(object):
    def __init__(self):
        self.spaces = {}

    def FunctionSpace(self, mesh, element, constrained_domain=None):
        k = (mesh.id(), element, constrained_domain)
        v = self.spaces.get(k, None)
        if v is None:
            v = self.spaces[k] = dolfin.FunctionSpace(
                mesh, element, constrained_domain=constrained_domain)
        return v
