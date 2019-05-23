
from dolfin import *
import matplotlib.pyplot as plt
import pygmsh
import meshio
import numpy as np

geom = pygmsh.opencascade.Geometry(
   characteristic_length_min = 0.2,
   characteristic_length_max = 0.2,
   )

r0 = geom.add_rectangle([0, 0, 0], 1., 1.)
r1 = geom.add_rectangle([0, 1, 0], 1, 0.5)
r2 = geom.add_rectangle([0, -0.5, 0], 1, 0.5)

union = geom.boolean_union([r1, r0, r2])
union2 = geom.boolean_union([r0, r1])

geom.add_physical_surface([r0], 'core')
geom.add_physical_surface([r1], 'top')
geom.add_physical_surface([r0, r1], 'test1')
geom.add_physical_surface([r2], 'bottom')
geom.add_physical_surface([r0, r2], 'test2')
geom.add_physical_surface([union2], 'test3')

DIM = 2
points, cells, point_data, cell_data, field_data = pygmsh.generate_mesh(
    geom, dim=DIM, prune_z_0=True,
    extra_gmsh_arguments=['-string', 'Mesh.Algorithm=7;'])

from ...mesh.pygmsh import PygmshMakeRegions

pyreg = PygmshMakeRegions().process(
    cells=cells, cell_data=cell_data, field_data=field_data, dim=DIM)
tag_to_cell_values = pyreg['tag_to_cell_values']

print("tag_to_cell_values:", tag_to_cell_values)

# Convert the physical geometry data to uint format so it can be read to  a MeshFunction of type size_t
cell_data['triangle']['gmsh:physical'] = np.array(cell_data['triangle']['gmsh:physical'], dtype='uint')
meshio.write_points_cells('mesh.xml', points, cells, cell_data=cell_data)

mesh = Mesh('mesh.xml')
celldata = MeshFunction('size_t', mesh, 'mesh_gmsh:physical.xml')

with XDMFFile('killme.xdmf') as uf:
    uf.write(celldata)

