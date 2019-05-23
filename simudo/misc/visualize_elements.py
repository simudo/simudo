from __future__ import division, absolute_import, print_function
from builtins import bytes, dict, int, range, str, super

from ..util import plot as pplt

import itertools
import dolfin
import numpy as np

def create_mf_and_set(mesh, dimension, indices):
    mf = dolfin.MeshFunction('size_t', mesh, dimension, 0)
    for i in indices:
        mf.array()[i] = 1
    return mf

def create_mesh():
    mesh = dolfin.Mesh()
    ed = dolfin.MeshEditor()
    ed.open(mesh, 'triangle', 2, 2)

    ed.init_vertices(5)
    av = ed.add_vertex
    av(0, [0.0, 1.0])
    av(1, [0.0, 0.0])
    av(2, [1.0, 0.0])
    av(3, [1.0, 1.0])
    av(4, [0.5, 1.0])

    ed.init_cells(3)
    ac = ed.add_cell
    ac(0, [4, 1, 2])
    ac(1, [0, 1, 4])
    ac(2, [4, 2, 3])

    ed.close()

    return {'mesh': mesh,
            'marker0': create_mf_and_set(mesh, 0, {4}),
            'marker2': create_mf_and_set(mesh, 2, {0}),
    }

def iterindices(shape):
    return itertools.product(*(range(n) for n in shape))

def vectorize_dolfin(u):
    def wrapped(*XY):
        img = np.zeros(XY[0].shape + u.ufl_shape)
        p = dolfin.Point()
        n = len(XY)
        for ij in iterindices(XY[0].shape):
            for k, A in enumerate(XY):
                p[k] = A[ij]
            img[ij] = np.array(u(p))
        return img
    return wrapped

def evaluate_image(X, Y, u):
    return vectorize_dolfin(u)(X, Y)

def rect_meshgrid(Nx=None, Ny=None,
                  xmin=0.0, xmax=1.0,
                  ymin=0.0, ymax=1.0, N=None):
    if Nx is None: Nx = N
    if Ny is None: Ny = N
    Xs = np.linspace(xmin, xmax, num=Nx)
    Ys = np.linspace(ymin, ymax, num=Ny)
    return np.meshgrid(Xs, Ys)

def do_element(element_name, element_order, vector_element=False):
    m = create_mesh()['mesh']
    el = (dolfin.VectorElement if vector_element else dolfin.FiniteElement)(
        element_name, m.ufl_cell(), element_order)
    W = dolfin.FunctionSpace(m, el)
    u = dolfin.Function(W)

    N = 200
    xmin = ymin = 0
    xmax = ymax = 1
    quiver_decimation = N//30

    X, Y = rect_meshgrid(N=N)

    shape = u.ufl_shape
    assert len(shape) <= 1

    vec = u.vector()
    dofmap = W.dofmap()
    dofs = dofmap.cell_dofs(0)
    print("Element {!r} order {}".format(element_name, element_order))
    for dof_index, dof in enumerate(dofs):
        print("...dof {}".format(dof_index))
        vec[dof] = 1.0

        R = evaluate_image(X, Y, u)

        with pplt.subplots(figsize=(9, 9)) as (fig, ax):
            ax.set_title("element={!r} order={}: dof index {}"
                         .format(element_name, element_order, dof_index))
            if len(shape) == 0:
                vmax = R.max()
                vmin = R.min()
                assert vmax > 0
                vspan = max(abs(vmax), abs(vmin))
                im = ax.imshow(R, extent=(0, 1, 0, 1), origin='lower',
                               vmin=-vspan, vmax=vspan, cmap='bwr')
                fig.colorbar(im, ax=ax)
            else:
                magn = np.linalg.norm(R, axis=2)
                ax.imshow(magn, extent=(0, 1, 0, 1), origin='lower')
                k = quiver_decimation
                ax.quiver(X[::k,::k], Y[::k,::k], R[::k,::k,0], R[::k,::k,1],
                           pivot='middle', units='height',
                           zorder=4, color='white')
            fig.savefig("out/elements/{}_{}_dof{}.png".format(
                element_name, element_order, dof_index))
        vec[dof] = 0.0

def do_func(name, func):
    N = 100
    X, Y = rect_meshgrid(N=N)

    with pplt.subplots(figsize=(9, 9)) as (fig, ax):
        ax.imshow(evaluate_image(X, Y, func),
                  extent=(0, 1, 0, 1), origin='lower')
        fig.savefig("out/misc_{}.png".format(name))

def dP_test():
    d = create_mesh()
    mesh = d['mesh']

    el = dolfin.FiniteElement(
        'CG', mesh.ufl_cell(), 1)
    W = dolfin.FunctionSpace(mesh, el)
    u = dolfin.Function(W)
    v = dolfin.TestFunction(W)

    dP1 = dolfin.dP(subdomain_data=d['marker0'])(1)
    form = (u-1)*v*dP1 + u*v*dolfin.dx

    dolfin.solve(form == 0, u)

    do_func('dP1', u)

if __name__ == '__main__':
    if 0:
        dP_test()
    if 1:
        for o in range(1, 3+1):
            do_element("CG", o)
        for o in range(2+1):
            do_element("DG", o)
            if o >= 1:
                do_element("CG", o)
                do_element("N1curl", o)
                do_element("RT", o)
                do_element("BDM", o)
