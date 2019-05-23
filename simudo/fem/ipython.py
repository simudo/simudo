from collections import ChainMap

import ipywidgets as widgets
import numpy as np
from cached_property import cached_property
from IPython.display import display
from ..plot import pyplot as plt

from mpl_render import RenderingImShow, RenderingPlot

from ..fem import (
    DolfinFunctionLineCutRenderer, DolfinFunctionRenderer, LineCutter)

# modify defaults in mpl_render classes
for cls in [RenderingImShow, RenderingPlot]:
    cls.force_single_thread = True

'''
possible features:
- autosearch expressions: not yet
- interpolate expression: text
- for vector expr, pick component or magnitude: text
- linear/log plot: text
- textually edit viewlim: text + widget

- line cut defined by end points
'''

class DisplayMixin():
    def display(self):
        for w in self.widgets:
            display(w)

def downproject(expr, scalar_space, vector_space):
    shape = function_value_shape(expr)
    r = len(shape)
    if r == 0:
        return dolfin.project(function, scalar_space)
    elif r == 1:
        return dolfin.project(function, vector_space)
    else:
        raise TypeError("function value_shape {!r}".format(shape))

def fchainmap(args):
    ''' construct chainmap with empty dict as first argument, and
    discard None values from args '''
    l = list(filter(lambda x:x is not None, args))
    l.insert(0, {})
    return ChainMap(*l)

class LineCutPlot():
    RenderingPlot = RenderingPlot

    def __init__(self, functions, line_cutter=None):
        self.functions = functions
        self.line_cutter = line_cutter
        self.init_all()

    def init_all(self):
        self.init_fig()
        self.init_renderers()
        self.init_plots()

    @cached_property
    def extent(self):
        t0, t1 = self.line_cutter.t_bounds
        return (t0, t1, 0, 1)

    def init_fig(self):
        self.fig, self.ax = plt.subplots()

    @cached_property
    def default_extract_params(self):
        return self.get_default_extract_params()

    def get_default_extract_params(self):
        return dict(
            line_cutter=self.line_cutter,
            component=0)

    def get_renderer_extract_params(self, k, v):
        return fchainmap((v.get('extract_params', None),
                          self.default_extract_params))

    def create_renderer(self, k, v):
        return DolfinFunctionLineCutRenderer(
            v['function'],
            extract_params=self.get_renderer_extract_params(k, v))

    def init_renderers(self):
        self.renderers = r = {k: self.create_renderer(k, v)
                              for k, v in self.functions.items()}

    def create_plot(self, k, v, render):
        return self.RenderingPlot(
            self.ax, kw=v['kw'],
            extent=self.extent,
            render_callback=render)

    def init_plots(self):
        self.plots = p = {
            k: self.create_plot(k, v, self.renderers[k])
            for k, v in self.functions.items()}

    def close(self):
        plt.close(self.fig)
