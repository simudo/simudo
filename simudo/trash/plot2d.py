
import dolfin
from ..util.xtimeit import xtimeit
from ..util.extra_bindings import Function_eval_many
from ..util.plot import figclosing, subplots, DolfinFunctionRenderer, DolfinFunctionLineCutRenderer, LineCutter
from mpl_render import RenderingImShow, RenderingPlot
from matplotlib import pyplot as plt
import numpy as np

from .stupid2d import *

render = DolfinFunctionRenderer(
    w, extract_params=dict(component='magnitude'))

with subplots() as (fig, ax):
    p = RenderingImShow(ax, kw=dict(cmap='inferno', aspect='auto'),
                        extent=(0, 1, 0, 1),
                        size=(400, 300), render_callback=render)
    p.force_single_thread = True
    ax.grid(True)
    plt.show()

lcrender = DolfinFunctionLineCutRenderer(
    w, extract_params=dict(line_cutter=LineCutter((0.2, 0), (0.2, 1)),
                           component=0))

with subplots() as (fig, ax):
    p = RenderingPlot(ax, kw=dict(),
                      extent=(0, 2**0.5, 0, 1),
                      render_callback=lcrender)
    p.force_single_thread = True
    ax.grid(True)
    plt.show()

