
import dolfin

__all__ = ['make_1d_mesh_from_points']

def make_1d_mesh_from_points(Xs):
    N = len(Xs)

    mesh = dolfin.Mesh()
    ed = dolfin.MeshEditor()
    ed.open(mesh, 'interval', 1, 1)

    ed.init_vertices(N)
    av = ed.add_vertex
    for i, x in enumerate(Xs):
        av(i, [x])

    ed.init_cells(N-1)
    ac = ed.add_cell
    for i in range(N-1):
        ac(i, [i, i+1])

    ed.close()

    mesh.init()
    mesh.order()

    return mesh

