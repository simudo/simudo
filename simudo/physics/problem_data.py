
import pint
from cached_property import cached_property

from ..fem import FunctionSpaceCache, FunctionSubspaceRegistry
from ..util import NameDict, SetattrInitMixin
from .optical import Optical
from .poisson_drift_diffusion import PoissonDriftDiffusion

__all__ = ['ProblemData']

class ProblemData(SetattrInitMixin):
    '''
Parameters
----------
goal: str
    Represents the goal of this problem. Must be
    ``"local charge neutrality"``, ``"thermal equilibrium"``,
    or ``"full"`` (representing full coupled solution).
unit_registry: :py:class:`pint.UnitRegistry`
    Unit registry to use.
mesh_data: :py:class:`~.mesh_data.MeshData`
    Mesh data to use by default in
    :py:class:`~.poisson_drift_diffusion.PoissonDriftDiffusion` and
    :py:class:`~.optical.Optical`.

Attributes
----------
pdd: :py:class:`~.poisson_drift_diffusion.PoissonDriftDiffusion`
    Poisson and electronic transport.
optical: :py:class:`~.optical.Optical`
    Optics.
'''
    @cached_property
    def pdd(self):
        return PoissonDriftDiffusion(
            problem_data=self)

    @cached_property
    def optical(self):
        return Optical(problem_data=self)

    @cached_property
    def function_space_cache(self):
        return FunctionSpaceCache()

    @cached_property
    def function_subspace_registry(self):
        return FunctionSubspaceRegistry()

    @property
    def goal_abbreviated(self):
        '''return abbreviated goal, for logging tag purposes'''
        goal = self.goal
        if goal == 'local charge neutrality':
            return 'ntrl'
        elif goal == 'thermal equilibrium':
            return 'thmq'
        elif goal == 'full':
            return goal
        else:
            raise ValueError('goal')
