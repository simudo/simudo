
import operator
from functools import reduce

import numpy as np
from cached_property import cached_property
from numpy import array as npa
from numpy import linalg

from .expr import value_shape
from .extra_bindings import Function_eval_many

__all__ = [
    'evaluate_function_at_coordinates',
    'BaseDolfinFunctionRenderFrame',
    'DolfinFunctionRenderFrame',
    'DolfinFunctionRenderLineCutFrame',
    'LineCutter',
    'DolfinFunctionRenderer',
    'DolfinFunctionLineCutRenderer']


def evaluate_function_at_coordinates(
        function, coordinates, value_outside_mesh=np.nan):
    '''Evaluate Dolfin function on `N` arbitrary coordinates.

Parameters
----------
function: :py:class:`dolfin.Function`
    Dolfin function to evaluate.
coordinates: :py:class:`numpy.ndarray`
    Array of `N` coordinates with shape :code:`(N, space_dimension)`
    on which to evaluate function. For 1D, the shape :code:`(N,)` is also
    accepted.
value_outside_mesh: value, optional
    For points that landed outside the function's range (mesh), use
    this value instead.

Returns
-------
values: :py:class:`numpy.ndarray`
    Array of function values. First shape component is `N`, and remaining shape components are given by the shape of the function value (:code:`function.value_shape()`).
'''

    if len(coordinates.shape) == 1:
        coordinates = coordinates.reshape((-1, 1))

    N, dim = coordinates.shape
    value_shape = tuple(function.value_shape())

    values = np.full(
        (N,) + value_shape,
        value_outside_mesh, dtype='float64')

    Function_eval_many(
        function, dim, N, values.ravel(), coordinates.ravel())

    return values


class BaseDolfinFunctionRenderFrame(object):
    def __init__(self, renderer, size, extent):
        self.size = size
        self.extent = extent
        self.renderer = renderer

    @cached_property
    def V(self):
        return self.compute_V()

    def evaluate_function_at_coordinates(
            self, function, coordinates, default_value):
        return evaluate_function_at_coordinates(
            function, coordinates, default_value)

    def compute_V(self):
        V = self.evaluate_function_at_coordinates(
            self.renderer.function,
            self.coordinates,
            self.renderer.default_value)

        return V

    def extract(self, params):
        ''' method to facilitate subclassing '''
        V = self.V
        shape = self.renderer.value_shape
        preprocess_callback  = params.get('preprocess_callback', lambda V: V)
        postprocess_callback = params.get('postprocess_callback', lambda V: V)
        V = preprocess_callback(V)
        if len(shape) == 0: # scalar
            return postprocess_callback(self.V)
        elif len(shape) == 1: # vector
            component = params.get('component', 0)
            if component == 'magnitude':
                return postprocess_callback(linalg.norm(V, axis=-1))
            else:
                return postprocess_callback(self.V[..., component])
        else:
            raise NotImplementedError(
                "value_shape == {!r}".format(V.shape))

    @cached_property
    def extract_params(self):
        return self.renderer.extract_params

    @cached_property
    def extracted_V(self):
        return self.extract(self.extract_params)

class DolfinFunctionRenderFrame(BaseDolfinFunctionRenderFrame):
    def compute_V(self):
        nX, nY = len(self.Xs), len(self.Ys)

        V = super().compute_V()

        V = V.reshape((nY, nX) + V.shape[1:])

        return V

    @cached_property
    def Xs(self):
        return np.linspace(self.extent[0], self.extent[1], self.size[0])

    @cached_property
    def Ys(self):
        return np.linspace(self.extent[2], self.extent[3], self.size[1])

    @cached_property
    def coordinates(self):
        # note: this is necessary because imshow expects array[y,x]
        # (i.e. fastest varying index is 'x')
        X, Y = np.meshgrid(self.Xs, self.Ys, indexing='xy',
                           sparse=False, copy=False)
        return np.stack((X.ravel(), Y.ravel()), axis=-1)

class DolfinFunctionRenderLineCutFrame(BaseDolfinFunctionRenderFrame):
    @cached_property
    def Ts(self):
        return np.linspace(self.extent[0], self.extent[1], self.size[0])

    @cached_property
    def coordinates(self):
        return self.extract_params['line_cutter'].coordinates(self.Ts)

    @property
    def extracted_T(self):
        return self.Ts

class LineCutter(object):
    def __init__(self, p0, p1):
        self.p0 = npa(p0)
        self.p1 = npa(p1)

    def coordinates(self, Ts):
        p0, p1 = self.p0, self.p1
        v = p1 - p0
        vnorm = v/linalg.norm(v)
        Ps = p0[None, ...] + vnorm[None, ...] * Ts[..., None]
        return Ps

    @cached_property
    def t_bounds(self):
        return (0, linalg.norm(self.p1 - self.p0))

class DolfinFunctionRenderer(object):
    frame_class = DolfinFunctionRenderFrame
    default_value = np.nan

    def __init__(self, function, extract_params):
        if not extract_params: extract_params = {}

        self.function = function
        self.extract_params = extract_params

    @cached_property
    def value_shape(self):
        return value_shape(self.function)

    def get_frame(self, size, extent):
        return self.frame_class(self, size, extent)

    def __call__(self, size, extent):
        frame = self.get_frame(size=size, extent=extent)
        return frame.extracted_V

class DolfinFunctionLineCutRenderer(DolfinFunctionRenderer):
    frame_class = DolfinFunctionRenderLineCutFrame

    def __call__(self, size, extent):
        frame = self.get_frame(size=size, extent=extent)
        return (frame.extracted_T, frame.extracted_V)


