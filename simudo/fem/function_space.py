
from collections import OrderedDict

from cached_property import cached_property

import dolfin
import ufl

from ..util import SetattrInitMixin

__all__ = [
    'MixedFunctionSpace',
    'MixedFunctionHelper',
    'WithSubfunctionsMixin',
]

class MixedFunctionLikeMixin(object):
    def __init__(self, mixed_function_space, function=None):
        ''' pass `function` to use an existing dolfin function instead
        of allocating a new one '''
        self.mixed_function_space = mixed_function_space
        if function is not None:
            # TODO: check function space matches
            self.function = function

    @cached_property
    def function(self):
        return self._make_dolfin_function[0](
            self.mixed_function_space.function_space)

    def split(self):
        return self.mixed_function_space.split(self.function)

class MixedFunction(MixedFunctionLikeMixin):
    # the tuple is there to prevent methodification
    _make_dolfin_function = (dolfin.Function,)

class MixedTestFunction(MixedFunctionLikeMixin):
    _make_dolfin_function = (dolfin.TestFunction,)

class MixedFunctionSpace(object):
    '''Convenience class for dealing with mixed function spaces.
'''

    MixedFunction = MixedFunction
    MixedTestFunction = MixedTestFunction

    def __init__(self, mesh, subspace_descriptors,
                 function_space_cache):
        '''
        Parameters
        ----------
        mesh: dolfin.Mesh
        subspace_descriptors: iterable
            List of dict with keys:
            element, trial_key, trial_units, test_key, test_units
        function_space_cache: FunctionSpaceCache
            Used to recycle existing FunctionSpace.
        '''
        self.mesh = mesh
        self.subspace_descriptors = tuple(subspace_descriptors)
        self.function_space_cache = function_space_cache

    @cached_property
    def subspace_descriptors_dict(self):
        ''' Like `subspace_descriptors`, except it's a dictionary
        where the keys are "trial_key" as well as "test_key". '''
        r = {}
        for desc in self.subspace_descriptors:
            r[desc['trial_key']] = desc
            r[desc['test_key' ]] = desc
        return r

    def get_element(self):
        ''' use `element` property instead '''
        descs = self.subspace_descriptors
        elements = [desc['element'] for desc in descs]
        n = len(elements)
        if n == 0: raise AssertionError()
        elif n == 1:
            return elements[0]
        else:
            return dolfin.MixedElement(elements)

    def get_function_space(self):
        ''' use `function_space` property instead '''
        return self.function_space_cache.FunctionSpace(
            self.mesh, self.element)

    @cached_property
    def element(self):
        return self.get_element()

    @cached_property
    def function_space(self):
        return self.get_function_space()

    @cached_property
    def function_subspaces(self):
        descs = self.subspace_descriptors
        W = self.function_space
        keys = [desc['trial_key'] for desc in descs]
        n = len(keys)
        if n == 1:
            return {keys[0]: W}
        else:
            return {key: W.sub(i) for i, key in enumerate(keys)}

    def make_function(self, **kwargs):
        return self.MixedFunction(self, **kwargs)

    def make_test_function(self, **kwargs):
        return self.MixedTestFunction(self, **kwargs)

    def split(self, dolfin_function, kind=None):
        '''
        returns OrderedDict of (key: subfunction*unit)
        '''

        if kind is None:
            if isinstance(dolfin_function, dolfin.Function):
                kind = 'trial'
            elif isinstance(dolfin_function, ufl.Argument):
                kind = 'test'
            else:
                raise TypeError(
                    "failed to identify whether trial or test function")

        descs = self.subspace_descriptors

        n = len(descs)
        if n == 1:
            subfunctions = (dolfin_function,)
        else:
            subfunctions = dolfin.split(dolfin_function)

        return OrderedDict((inf[kind+'_key'], inf[kind+'_units']*sub)
                           for (inf, sub)
                           in zip(descs, subfunctions))

class MixedFunctionHelper(SetattrInitMixin):
    '''Utility mixin for objects that hold a mixed function make up of
many subfunctions that need to be gathered from different objects, for
example :py:class:`.poisson_drift_diffusion.PoissonDriftDiffusion`.

Parameters
----------
mesh_util: :py:class:`.mesh_util.MeshUtil`
    Instance of MeshUtil.
subspace_descriptors_for_solution_space: sequence of dict
    Sequence of subspace descriptor dicts. This property must contain
    the subspace descriptors that will be used to build the solution
    space. Typically one would gather these by calling
    :py:attr:`WithSubfunctionsMixin.solution_subspace_descriptors`
    on all the objects that contain subspace descriptors.
'''

    @property
    def mesh(self):
        return self.mesh_util.mesh

    @property
    def function_space_cache(self):
        return self.mesh_util.function_space_cache

    @property
    def solution_function(self):
        '''Dolfin Function.'''
        return self.solution_mixed_function_data['trial'].function

    @property
    def solution_test_function(self):
        '''Dolfin TestFunction.'''
        return self.solution_mixed_function_data['test'].function

    @cached_property
    def solution_mixed_function_data(self):
        trial = self.make_solution_mixed_function()
        test = self.solution_mixed_space.make_test_function()
        split = OrderedDict(tuple(trial.split().items()) +
                            tuple(test.split().items()))
        return dict(trial=trial, test=test, split=split)

    def make_solution_mixed_function(self):
        ''' intercept this to use your own Function instead of
        allocating a new one '''
        return self.solution_mixed_space.make_function()

    @cached_property
    def solution_mixed_space(self):
        descs = self.subspace_descriptors_for_solution_space
        return MixedFunctionSpace(
            mesh=self.mesh,
            subspace_descriptors=descs,
            function_space_cache=self.function_space_cache)


class WithSubfunctionsMixin(SetattrInitMixin):
    '''Utility mixin for objects that need subfunctions
(:py:class:`.poisson_drift_diffusion.MixedPoisson`,
:py:class:`.poisson_drift_diffusion.MixedQflBand`, etc).

Parameters
----------
key: str
    Unique prefix to prevent clashes between trial/test function names
    across instances (for example, "CB" to distinguish between "CB/j"
    and "VB/j").
subfunctions_info:
    Sequence of dicts with keys "{trial,test}_{name,units}" and "element".
mixed_function_solution_object: :py:class:`MixedFunctionHelper`
    Where to register subfunctions.
'''

    @property
    def solution_subspace_descriptors(self):
        '''Creates properly namespaced subspace descriptors.

Returns
-------
:
    Returns properly namespaced subspace descriptors by prefixing the
    descriptors in :py:attr:`~subfunctions_info` with
    :code:`self.key + "/"`.
'''
        descs = []
        prefix = self.subfunction_prefix
        for inf in self.subfunctions_info:
            desc = inf.copy()
            desc['trial_key'] = prefix(desc['trial_key'])
            desc['test_key' ] = prefix(desc['test_key' ])
            descs.append(desc)
        return descs

    def get_subfunction(self, name):
        name = self.subfunction_prefix(name)
        return (self.mixed_function_solution_object
                .solution_mixed_function_data['split'][name])

    def subfunction_prefix(self, name):
        '''Apply prefix to this subfunction name.'''
        return '/'.join((self.key, name))
