from simudo.api import (ProblemData, CellRegions, FacetRegions,
                   PoissonDriftDiffusion)
from simudo.api.poisson_drift_diffusion import (
    BeerLambertAbsorption, RadiativeTrap)
from simudo.api.material import Material
from simudo.mesh.construction_helper import ConstructionHelperLayeredStructure
from simudo.mesh.interval1dtag import (Interval, CInterval)
from simudo.util.pint import make_unit_registry
import numpy as np

from simudo.util.newton_solver import NewtonSolverMaxDu

import dolfin
import logging

class Strandbergium(Material):
    name = 'strandbergium'
    def get_dict(self):
        d = super().get_dict()
        U = self.unit_registry

        d.update({
            # band parameters
            'CB/energy_level': U('2 eV'),
            'VB/energy_level': U('0 eV'),
            'IB/energy_level': U('1.2 eV'),
            'IB/degeneracy': 1,

            'CB/effective_density_of_states': U('2e19 1/cm^3'),
            'VB/effective_density_of_states': U('2e19 1/cm^3'),
            'IB/number_of_states': U('1e7 cm^-3'),

            # electrical properties
            'CB/mobility': U('500 cm^2/V/s'),
            'VB/mobility': U('500 cm^2/V/s'),
            'IB/mobility': U('500 cm^2/V/s'),

            #electro-optical properties
            'radiative_top/sigma_opt': U('3e-14 cm^2'),
            'radiative_bottom/sigma_opt': U('3e-14 cm^2'),

            # poisson
            'poisson/permittivity': U('11.7 vacuum_permittivity'),
        })
        return d

class IB_material(Strandbergium):
    '''Material for figure of merit study'''

    name = 'IB'
    def get_dict(self):
        d = super().get_dict()
        U = self.unit_registry

        d['IB/number_of_states'] = U('2.5e17 cm^-3')

        return d


def topology_standard_contacts(cell_regions, facet_regions):
    R, F = cell_regions, facet_regions

    F.left_contact  = lc = R.exterior_left.boundary(R.domain        )
    F.right_contact = rc = R.domain       .boundary(R.exterior_right)

    F.exterior = R.domain.boundary(R.exterior)
    F.contacts = lc | rc
    F.nonconductive = F.exterior - F.contacts.both()

def main():
    U = make_unit_registry(("mesh_unit = 1 micrometer",))

    layers = [dict(name='pfsf', material='strandbergium' , thickness=0.05),
              dict(name='p', material='strandbergium' , thickness=1.0),
              dict(name='I', material='IB', thickness=3.3233644921759677),
              dict(name='n', material='strandbergium' , thickness=1.0)]

    # -layer means relative to left endpoint of layer
    # +layer means relative to right endpoint of layer
    simple_overmesh_regions = [
        dict(x0=('+p', -0.05), x1=('+p', +0.05), edge_length=0.001),
        dict(x0=('+I', -0.1), x1=('+I', +0.1), edge_length=0.001),
    ]

    ls = ConstructionHelperLayeredStructure()
    ls.params = dict(edge_length=0.05, # default edge_length
                     layers=layers,
                     simple_overmesh_regions=simple_overmesh_regions,
                     mesh_unit=U.um)
    ls.run()
    mesh_data = ls.mesh_data


    ## topology
    R = CellRegions()
    F = FacetRegions()

    topology_standard_contacts(R, F)

    R.cell = R.p | R.n | R.I | R.pfsf


    F.p_contact   = F.left_contact
    F.pI_junction = R.p.boundary(R.I)
    F.In_junction = R.I.boundary(R.n)
    F.n_contact   = F.right_contact

    F.IB_junctions = F.pI_junction | F.In_junction
    ## end topology


    def create_problemdata(
            goal='full', V_ext=None, phi_cn=None):
        """
goal in {'full', 'local neutrality', 'thermal equilibrium'}
"""
        root = ProblemData(
            goal=goal,
            mesh_data=mesh_data,
            unit_registry=U)
        pdd = root.pdd

        CB = pdd.easy_add_band('CB')
        IB = pdd.easy_add_band('IB')
        VB = pdd.easy_add_band('VB')

        # material to region mapping
        spatial = pdd.spatial

        spatial.add_rule('temperature', R.domain, U('300 K'))

        spatial.add_rule(
            'poisson/static_rho', R.p   , U('-1e17 elementary_charge/cm^3'))
        spatial.add_rule(
            'poisson/static_rho', R.n   , U('+1e17 elementary_charge/cm^3'))
        spatial.add_rule(
            'poisson/static_rho', R.pfsf, U('-1e19 elementary_charge/cm^3'))

        IB_material(problem_data=root).register()
        Strandbergium(problem_data=root).register()

        mu = pdd.mesh_util

        zeroE = F.exterior

        if goal == 'full':
            RadiativeTrap.easy_add_two_traps_to_pdd( pdd, 'radiative',
                                                     CB, VB, IB)


        elif goal == 'thermal equilibrium' and True:
            # to match old method, use local charge neutrality phi as
            # the phi boundary condition for Poisson-only thermal
            # equilibrium
            spatial.add_BC('poisson/phi', F.p_contact | F.n_contact,
                           phi_cn)
            zeroE -= (F.p_contact | F.n_contact).both()

        spatial.add_BC('poisson/E', zeroE,
                       U('V/m') * mu.zerovec)

        return root

    problem = create_problemdata(goal='local charge neutrality')

    problem.pdd.easy_auto_pre_solve()

    problem0 = problem
    problem = create_problemdata(goal='thermal equilibrium',
                                 phi_cn=problem0.pdd.poisson.phi)
    problem.pdd.initialize_from(problem0.pdd)

    problem.pdd.easy_auto_pre_solve()

    V_ext = U.V * dolfin.Constant(0.0)
    problem0 = problem
    problem = create_problemdata(goal='full', V_ext=V_ext)
    problem.pdd.initialize_from(problem0.pdd)

    from simudo.util.adaptive_stepper import (
        VoltageStepper, OpticalIntensityAdaptiveStepper)
    from simudo.io.output_writer import OutputWriter


    stepper = VoltageStepper(
        solution=problem, constants=[V_ext],
        parameter_target_values=[1.2],
        parameter_unit=U.V,
        selfconsistent_optics=False)

    stepper.do_loop()

    return locals()

if __name__ == '__main__':
    main()
