
import dolfin
from dolfin import Vertex, Cell, MPI
from ..util.expr import tuple_to_point
import numpy as np
from numpy import infty
from cached_property import cached_property

# this is a pure python rewrite of Probe.cpp from
# https://github.com/mikaem/fenicstools

class Probes(object):
    def __init__(self, function_space):
        self.W = function_space

    @cached_property
    def _mesh(self):
        return self.W.mesh()

    @cached_property
    def _dim(self):
        return self._mesh.geometry().dim()

    @cached_property
    def _element(self):
        return self.W.element()

    @cached_property
    def _value_shape(self):
        # stupid
        e = self._element
        return tuple(int(e.value_dimension(k))
                     for k in range(e.value_rank()))

    def probe(self, x):
        return Probe(self, x)

class Probe(object):
    def __init__(self, probes, x):
        self.probes = probes
        if not isinstance(x, dolfin.Point):
            x = tuple_to_point(x)
        self.x = x

    @cached_property
    def _x_cell_id(self):
        x = self.x
        mesh = self.probes._mesh

        val = -1
        index = mesh.bounding_box_tree().compute_first_entity_collision(x)
        if (0 <= index < mesh.num_cells()):
            val = index

        return MPI.max(dolfin.mpi_comm_world(), val)

    @cached_property
    def _containing_cell(self):
        return Cell(self.probes._mesh, self._x_cell_id)

    @cached_property
    def _containing_cell_vertex_coordinates(self):
        return self._containing_cell.get_vertex_coordinates()

    @cached_property
    def _basis_coefficients(self):
        x = self.x
        xv = np.array(x, dtype='double')

        element = self.probes._element
        vertex_coordinates = self._containing_cell_vertex_coordinates

        # number of basis functions per cell
        N = element.space_dimension()

        shape = self.probes._value_shape
        r = np.zeros(shape=(N,)+shape, dtype='double')
        cell_orientation = 0
        for basis_index in range(N):
            element.evaluate_basis(
                basis_index, r[basis_index, ...].ravel(), xv,
                vertex_coordinates, cell_orientation)
        return r

    def __call__(self, function):
        # r[i, ...]: contribution from each basis function
        # c[i]: coefficient of basis function i in given dolfin.Function
        r = self._basis_coefficients
        c = np.zeros(shape=(r.shape[0],), dtype='double')
        cell = self._containing_cell
        vertex_coordinates = self._containing_cell_vertex_coordinates
        function.restrict(
            c, self.probes._element, cell, vertex_coordinates, cell)
        return np.einsum('i...,i...->...', c, r)
