
import dolfin
from ..util.xtimeit import xtimeit
from .probe_slow import Probes as ProbesSlow
from ..util.extra_bindings import Function_eval_many
import numpy as np

mesh = dolfin.UnitSquareMesh(200, 200)
el = dolfin.VectorElement("DG", mesh.ufl_cell(), 1)
x = dolfin.SpatialCoordinate(mesh)
expr = dolfin.as_vector((1 - x[0], x[1]))
W = dolfin.FunctionSpace(mesh, el)
w = dolfin.project(expr, W)

px = np.array([0.55454, 0.34232, 0.0], dtype='double')
p = dolfin.Point(px)

r = np.array([0.0, 0.0], dtype='double')
r_flat = r.ravel()

cell_index = mesh.bounding_box_tree().compute_first_entity_collision(p)
cell = dolfin.Cell(mesh, cell_index)

probes = ProbesSlow(W)
probe = probes.probe(p)

em_N = 100
em_x = np.array(tuple(px)*em_N, dtype='double')
em_values = np.zeros(len(r_flat)*em_N, dtype='double')

def em_test():
    Function_eval_many(w, 3, em_N, em_values, em_x)

def xt(desc, func, factor=1):
    print("{:>40s}: {:9.3e} s per call".format(
        desc, xtimeit(func)/factor))

# xt("`w.eval` with cell", lambda: w.eval(r_flat, px, cell, cell))
xt("`w.eval` without cell", lambda: w.eval(r_flat, px))
xt("`w(p)`", lambda: w(p))
xt("`w.eval_cell`", lambda: w.eval_cell(r_flat, px, cell))
xt("`Function_eval_many`", lambda: em_test(), factor=em_N)
xt("`w.ufl_shape` call overhead", lambda: w.ufl_shape)

#          `w.eval` with cell: 1.324e-06 s per call
#       `w.eval` without cell: 1.101e-06 s per call
#                      `w(p)`: 1.575e-05 s per call
#               `w.eval_cell`: 1.332e-06 s per call
#        `Function_eval_many`: 6.413e-07 s per call
# `w.ufl_shape` call overhead: 2.290e-07 s per call

# apparently telling it specifically what cell to look at actually
# makes it slightly slower. wew

