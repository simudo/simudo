from __future__ import absolute_import, division, print_function

from builtins import bytes, dict, int, range, str, super

import numpy as np

import dolfin
import mshr

from . import mesh_entity_predicate as mep
from .construction_helper import ConstructionHelper, LinearTransform


class Bob(ConstructionHelper):
    def user_define_mshr_regions(self):
        '''must return `{region_name: mshr_domain}`
        `regions['domain']` is overall domain'''
        def r(x0, y0, x1, y1):
            return mshr.Rectangle(dolfin.Point(x0, y0), dolfin.Point(x1, y1))

        A = r(0, 0, 10, 10)
        B = r(0, 0, 1, 1)
        C = r(1, 1, 2, 2)
        D = r(3, 3, 5, 5)
        E = r(4, 4, 7, 7)
        F = r(2, 3.5, 10.7, 4.5)
        domain = A + F
        keys = 'ABCDEF'

        _locals = locals()
        d = {k:_locals[k] for k in keys}
        d['domain'] = domain
        return d

    def user_extra_definitions(self):
        '''override me'''

        R = self.cell_regions
        F = self.facet_regions
        fc = self.facets

        X = R['E']
        Y = R['F'] | R['D']

        R['Z'] = Z = ((R['E'] | R['D']) - R['F']) ^ R['A']
        F.update(Z_internal=fc.internal(Z),
                 XXY_boundary=fc.boundary(X, Y, 0),
                 F_right=fc.boundary(R['F'], R['right']))

    def user_refinement(self):
        R = self.cell_regions

        pred1 = mep.MaxEdgeLengthCellPredicate(0.2, dim=self.dim)
        self.refine_subdomains((R['F'] - R['A']), pred1)

        pred2 = mep.DirectionalEdgeLengthPredicate(
            np.array([1.0, 0.0, 0.0]), 0.06)
        trans = LinearTransform([1.0, 0.2])
        trans.transform(self.mesh)
        self.refine_subdomains((R['D'] - (R['F'] | R['E'])), pred2)
        trans.untransform(self.mesh)

dolfin.parameters["refinement_algorithm"] = 'plaza_with_parent_facets'
h = Bob()
h.run()
h.save("out/poop")

ro = h.robjs

for k, v in h.facet_regions.items():
    dolfin.plot(h.debug_fvs_to_mf(v, signed=False))

dolfin.plot(ro['cf'])

dolfin.plot(h.debug_cvs_to_mf(h.cell_regions['Z']))

dolfin.interactive()
