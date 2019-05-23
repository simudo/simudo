
from matplotlib import pyplot as plt
from ..util.ipython import LineCutPlot, LineCutter, DolfinFunctionRenderer
from ..util.expr import interp, bbox_vecs, mesh_bbox
from mpl_render import RenderingImShow

from .stupid2d import *

B = bbox_vecs(mesh_bbox(mesh))

p0 = LineCutPlot(dict(a=dict(function=w, kw=dict(label="hello"))),
                line_cutter=LineCutter(
                    interp(0, B[0]) + interp(0.2, B[1]),
                    interp(1, B[0]) + interp(0.2, B[1])))
p0.ax.legend()
p0.ax.grid(True)

p = LineCutPlot(dict(a=dict(function=w, kw=dict(label="hello"))),
                line_cutter=LineCutter((0.2, 0), (0.2, 1)))
p.ax.legend()
p.ax.grid(True)
plt.show()

fig, ax = plt.subplots()
p2 = RenderingImShow(ax, kw=dict(cmap='inferno', aspect='auto'),
                     extent=(0, 1, 0, 1),
                     size=(400, 300), render_callback=
                     DolfinFunctionRenderer(
                         w, extract_params=dict(component='magnitude')))
ax.grid(True)

