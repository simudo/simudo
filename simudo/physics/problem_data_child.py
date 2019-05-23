
from cached_property import cached_property

from ..fem import MeshUtil

__all__ = ['DefaultProblemDataChildMixin']

class DefaultProblemDataChildMixin(object):
    @cached_property
    def mesh_util(self):
        return MeshUtil(
            mesh_data=self.mesh_data,
            function_space_cache=self.function_space_cache,
            function_subspace_registry=self.function_subspace_registry,
            unit_registry=self.unit_registry)

    @property
    def mesh_data(self):
        return self.problem_data.mesh_data

    @property
    def function_space_cache(self):
        return self.problem_data.function_space_cache

    @property
    def function_subspace_registry(self):
        return self.problem_data.function_subspace_registry

    @property
    def unit_registry(self):
        return self.problem_data.unit_registry
