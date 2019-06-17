
import glob
import logging
import os
from functools import partial
from os import path as osp

import numpy as np
import pandas as pd
from argh import ArghParser, arg
from cached_property import cached_property

from simudo.fem import setup_dolfin_parameters
from simudo.io import h5yaml
from simudo.io.output_writer import (MetaExtractorBandInfo,
                                     MetaExtractorIntegrals, OutputWriter)
from simudo.mesh import (CellRegions, CInterval,
                         ConstructionHelperLayeredStructure, FacetRegions,
                         Interval)
from simudo.physics import (Material, NonOverlappingTopHatBeerLambert,
                            NonOverlappingTopHatBeerLambertIB,
                            NonRadiativeTrap, PoissonDriftDiffusion,
                            ProblemData, SimpleSiliconMaterial,
                            SRHRecombination, VoltageStepper)
from simudo.util import DictAttrProxy, TypicalLoggingSetup, make_unit_registry
import dolfin


def topology_standard_contacts(cell_regions, facet_regions):
    ''' helper function to define typical regions in a 1d device '''

    R, F = cell_regions, facet_regions

    F.left_contact  = lc = R.exterior_left.boundary(R.domain        )
    F.right_contact = rc = R.domain       .boundary(R.exterior_right)

    F.exterior = R.domain.boundary(R.exterior)
    F.contacts = lc | rc
    F.nonconductive = F.exterior - F.contacts.both()

def run():
    ''' main function '''

    setup_dolfin_parameters()
    TypicalLoggingSetup(filename_prefix="output ").setup()

    U = make_unit_registry(("mesh_unit = 1 micrometer",))

    length = 0.500 # micrometers
    layers = [dict(name='pSi', material='Si' , thickness=length/2),
              dict(name='nSi', material='Si' , thickness=length/2)]

    # -layer means relative to left endpoint of layer
    # +layer means relative to right endpoint of layer
    simple_overmesh_regions = [
        # extra meshing near contacts
        dict(edge_length=length/100, x0=('-pSi',  0   ), x1=('-pSi', +0.05)),
        dict(edge_length=length/100, x0=('+nSi', -0.05), x1=('+nSi', +0   )),

        # extra meshing near junction
        dict(edge_length=length/100, x0=('+pSi', -0.05), x1=('+pSi', +0.05)),
    ]

    ls = ConstructionHelperLayeredStructure()
    ls.params = dict(edge_length=length/200, # default edge_length
                     layers=layers,
                     simple_overmesh_regions=simple_overmesh_regions,
                     mesh_unit=U.mesh_unit)
    ls.run()
    mesh_data = ls.mesh_data

    logging.getLogger('main').info("NUM_MESH_POINTS: {}".format(
        len(ls.interval_1d_tag.coordinates['coordinates'])))

    ### begin topology ###
    R = CellRegions()
    F = FacetRegions()

    topology_standard_contacts(R, F)

    R.Si = R.pSi | R.nSi

    F.p_contact   = F.left_contact
    F.pn_junction = R.pSi.boundary(R.nSi)
    F.n_contact   = F.right_contact
    ### end topology ###

    def create_problemdata(goal='full', V_ext=None, phi_cn=None):
        """
Create the ProblemData object for a given solver stage `goal`.

goal in {'full', 'local neutrality', 'thermal equilibrium'}
"""
        root = ProblemData(
            goal=goal,
            mesh_data=mesh_data,
            unit_registry=U)
        pdd = root.pdd

        CB = pdd.easy_add_band(name='CB')
        VB = pdd.easy_add_band(name='VB')

        # material to region mapping
        spatial = pdd.spatial

        spatial.add_rule(
            'temperature', R.domain, U('300 K'))

        doping = 1e18
        spatial.add_rule(
            'poisson/static_rho', R.pSi,
            -float(doping)*U('elementary_charge/cm^3'))
        spatial.add_rule(
            'poisson/static_rho', R.nSi,
            +float(doping)*U('elementary_charge/cm^3'))

        SimpleSiliconMaterial(problem_data=root).register()

        mu = pdd.mesh_util

        if goal == 'full':
            pdd.easy_add_electro_optical_process(
                SRHRecombination, dst_band=CB, src_band=VB)

            # no conduction through top and bottom surfaces
            spatial.add_BC('CB/j', F.nonconductive,
                           U('A/cm^2') * mu.zerovec)
            spatial.add_BC('VB/j', F.nonconductive,
                           U('A/cm^2') * mu.zerovec)

            # minority contact
            spatial.add_BC('CB/j', F.p_contact,
                           U('A/cm^2') * mu.zerovec)
            spatial.add_BC('VB/j', F.n_contact,
                           U('A/cm^2') * mu.zerovec)

            # majority contact
            spatial.add_BC('VB/u', F.p_contact,
                           VB.thermal_equilibrium_u)
            spatial.add_BC('CB/u', F.n_contact,
                           CB.thermal_equilibrium_u)

            phi0 = pdd.poisson.thermal_equilibrium_phi
            spatial.add_BC('poisson/phi', F.p_contact,
                           phi0)
            spatial.add_BC('poisson/phi', F.n_contact,
                           phi0 - V_ext)

        elif goal == 'thermal equilibrium':
            # to match old method, use local charge neutrality phi as
            # the phi boundary condition for Poisson-only thermal
            # equilibrium
            spatial.add_BC('poisson/phi', F.p_contact | F.n_contact, phi_cn)

        # apply zero perpendicular electric field BC
        zeroE = F.exterior - (F.p_contact | F.n_contact).both()
        spatial.add_BC('poisson/E', zeroE,
                       U('V/m') * mu.zerovec)

        spatial.add_rule('SRH/CB/tau', R.domain, U('1e-9 s'))
        spatial.add_rule('SRH/VB/tau', R.domain, U('1e-6 s'))
        spatial.add_rule('SRH/energy_level', R.domain,
                         U('0.5525628437189991 eV'))

        return root

    problem = create_problemdata(goal='local charge neutrality')
    problem.pdd.easy_auto_pre_solve()

    problem0 = problem
    problem = create_problemdata(goal='thermal equilibrium',
                                 phi_cn=problem0.pdd.poisson.phi)
    problem.pdd.initialize_from(problem0.pdd)
    problem.pdd.easy_auto_pre_solve()

    V_ext = U.V * dolfin.Constant(0.0) # applied bias
    problem0 = problem
    problem = create_problemdata(goal='full', V_ext=V_ext)
    problem.pdd.initialize_from(problem0.pdd)

    # things to evaluate on the solution and save to
    # `output_file.plot_meta.yaml`
    meta_extractors = (
        MetaExtractorBandInfo,
        partial(MetaExtractorIntegrals,
                facets=F[{'p_contact', 'n_contact', 'pn_junction'}],
                cells=R[{'pSi', 'nSi'}]))

    # define stepper
    stepper = VoltageStepper(
        solution=problem, constants=[V_ext],
        parameter_target_values=np.linspace(0, 1, 21),
        parameter_unit=U.V,
        output_writer=OutputWriter(
            filename_prefix="output V_stepper", plot_1d=True,
            format_parameter=(lambda solution, V: '{:.5f}'.format(V)),
            meta_extractors=meta_extractors),
        selfconsistent_optics=False)

    # run stepper!
    stepper.do_loop()

    return locals()

parser = ArghParser()
parser.add_commands([run])

if __name__ == '__main__':
    parser.dispatch()

