
import copy
import unittest

import numpy as np
from cached_property import cached_property

import dolfin

__all__ = ['Product2DMesh']

class Product2DMesh(object):
    def __init__(self, Xs, Ys=None):
        if Ys is None:
            Ys = (0.0, 1.0)

        Xs = self.Xs = np.asarray(Xs, dtype='double')
        Ys = self.Ys = np.asarray(Ys, dtype='double')

    def copy(self):
        ''' note: does not deepcopy Xs and Ys '''
        return copy.copy(self)

    @cached_property
    def mesh(self):
        Xs, Ys = self.Xs, self.Ys
        nX, nY = self.nX, self.nY

        mesh = dolfin.Mesh()
        ed = dolfin.MeshEditor()
        ed.open(mesh, 'triangle', 2, 2)

        ed.init_vertices(nX*nY)
        av = ed.add_vertex
        i = 0
        for x in Xs:
            for y in Ys:
                av(i, [x, y])
                i += 1

        ed.init_cells((nX-1)*(nY-1)*2)
        ac = ed.add_cell
        i = 0
        for ix in range(nX-1):
            for iy in range(nY-1):
                # ^ y
                # |   (b+1)(c+1)
                # |    |   / |
                # |    |  /  |
                # |    | /   |
                # |   (b)---(c)
                # |
                # +---------------> x
                b = ix*nY + iy
                c = b + nY
                ac(i  , [b, c,   c+1])
                ac(i+1, [b, c+1, b+1])
                i += 2

        ed.close()

        mesh.init()
        mesh.order()

        return mesh

    @cached_property
    def nX(self):
        return len(self.Xs)

    @cached_property
    def nY(self):
        return len(self.Ys)

    def cells_at_ix(self, ix):
        nc = 2*(self.nY-1) # number of cells at particular x
        b = nc*ix
        return range(b, b+nc)

    def vertices_at_ix(self, ix):
        b = self.nY*ix
        return range(b, b+self.nY)

    def vertices_at_iy(self, iy):
        return range(iy, self.nY*self.nX, self.nY)

    def ix_from_vertex(self, v):
        return v // self.nY

    def iy_from_vertex(self, v):
        return v % self.nY

    @cached_property
    def element(self):
        return dolfin.FiniteElement("CG", self.mesh.ufl_cell(), 1)

    @cached_property
    def space(self):
        return dolfin.FunctionSpace(self.mesh, self.element)

    def create_function_from_x_values(self, values):
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        u = dolfin.Function(self.space)
        dofmap = self.space.dofmap()
        vec = u.vector()
        vertices = np.array(range(len(vec)), dtype='uintp')
        dofs = dofmap.entity_dofs(
            self.mesh, 0, vertices)
        vec[dofs] = values[self.ix_from_vertex(vertices)]
        return u

    def __getstate__(self):
        return dict(Xs=self.Xs, Ys=self.Ys)

class TestThis(unittest.TestCase):
    def test_cells(self):
        o = Product2DMesh([0, 1, 2, 3], [0, 1, 2])
        # dolfin.plot(o.mesh)
        # dolfin.interactive()
        self.assertEqual(list(o.cells_at_ix(1)),
                         [4, 5, 6, 7])

    def test_interp(self):
        from ..fem import tuple_to_point

        Xs = [0, 2, 3, 10, 12, 15, 17]
        Ys = [0, 4, 6.5, 10]
        Vs = [1, 2, 3, 4, 5, 6, 7]

        rect = Product2DMesh(Xs, Ys)
        u = rect.create_function_from_x_values(Vs)

        for ix in range(rect.nX):
            for vert in rect.vertices_at_ix(ix):
                self.assertEqual(rect.ix_from_vertex(vert), ix)
        for iy in range(rect.nY):
            for vert in rect.vertices_at_iy(iy):
                self.assertEqual(rect.iy_from_vertex(vert), iy)

        for x, v in zip(Xs, Vs):
            for y in (0.0, 0.1, 0.5, 1.0):
                self.assertLessEqual(abs(u(tuple_to_point((x, y))) - v), 1e-8)
