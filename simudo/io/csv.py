
from __future__ import absolute_import, division, print_function

import os
import re
import shlex
from builtins import bytes, dict, int, range, str, super
from collections import defaultdict
from io import StringIO

import numpy as np
import pandas as pd
from cached_property import cached_property
from future.utils import PY2, PY3, native
from numpy import array as npa

import dolfin

from ..fem import DolfinFunctionLineCutRenderer, mesh_bbox, tuple_to_point
from .xdmf import PlotAddMixin, magnitude


class LineCutCsvPlot(PlotAddMixin):
    timestamp = 0.0
    resolution = 5001
    _empty = True

    def __init__(self, filename, function_subspace_registry=None):
        self.base_filename = filename
        self.function_subspace_registry = function_subspace_registry

    def new(self, timestamp):
        self.close()
        self.timestamp = timestamp

    @cached_property
    def _df(self):
        return pd.DataFrame()

    def _compute_coords(self, mesh=None):
        coords = getattr(self, '_coords', None)
        if coords is None:
            xl, yl = mesh_bbox(mesh)
            y0 = npa(yl).mean()
            t = np.linspace(0, 1, self.resolution)
            coords = (np.outer(1-t, npa((xl[0], y0))) +
                      np.outer(t,   npa((xl[1], y0))))
            self._coords = coords
        df = self._df
        df['coord_x'] = coords.view()[:,0]
        df['coord_y'] = coords.view()[:,1]
        return coords

    def _add_func(self, func):
        coords = self._compute_coords(func.function_space().mesh())

        renderer = DolfinFunctionLineCutRenderer(func, {})
        frame = renderer.get_frame(size=None, extent=None)
        frame.coordinates = coords

        vals = frame.extracted_V
        self._df[func.name()] = vals
        self._empty = False

    def close(self):
        if self._empty == True:
            return

        try:
            os.mkdir(os.path.dirname(self.base_filename))
        except OSError:
            pass
        self._df.to_csv(self.base_filename + '.{}'.format(self.timestamp),
                        index=False)
        del self._df

        self._empty = True

    def __del__(self):
        self.close()
