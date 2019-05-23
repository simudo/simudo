
from dolfin import *
import matplotlib.pyplot as plt
import pygmsh
import meshio
import numpy as np
from ...mesh.construction_helper import ConstructionHelperPygmsh

class CH(ConstructionHelperPygmsh):
    def user_define_pygmsh_geo(self):
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

        return geom

ch = CH()
ch.params = {}
ch.run()

with XDMFFile('killme.xdmf') as uf:
    uf.write(ch.cf)

