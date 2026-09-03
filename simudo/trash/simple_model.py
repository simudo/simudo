# Other modules
from functools import partial
import numpy as np
import dolfin

# Simudo
from simudo.physics import (ProblemData,
                            SRHRecombination, NonRadiativeTrap,
                            NonOverlappingTopHatBeerLambert,
                            NonOverlappingTopHatBeerLambertIB)
from simudo.mesh import CellRegion, CellRegions, FacetRegions
from simudo.mesh.construction_helper import ConstructionHelperLayeredStructure
from simudo.util.pint import make_unit_registry
from simudo.physics import VoltageStepper, OpticalIntensityAdaptiveStepper
from simudo.io.output_writer import (OutputWriter,
                                     MetaExtractorBandInfo,
                                     MetaExtractorIntegrals)
from simudo.fem import setup_dolfin_parameters

from simudo.util import TypicalLoggingSetup
from pathlib import Path
from datetime import date, datetime
import logging

# Local project
from silicon import SiliconMaterial


def topology_standard_contacts(cell_regions, facet_regions):
    R, F = cell_regions, facet_regions

    F.left_contact = lc = R.exterior_left.boundary(R.domain)
    F.right_contact = rc = R.domain.boundary(R.exterior_right)

    F.exterior = R.domain.boundary(R.exterior)
    F.contacts = lc | rc
    F.nonconductive = F.exterior - F.contacts.both()
    # F.ib_bounds = R.ib.boundary(R.blocking) | R.ib.boundary(R.base2)

def main():
    start_time = datetime.now()
    # # FIXME: this should go elsewhere
    # parameters = dolfin.parameters
    # parameters["refinement_algorithm"] = "plaza_with_parent_facets"
    # parameters["form_compiler"]["cpp_optimize"] = True
    # parameters["form_compiler"]["cpp_optimize_flags"] = '-O3 -funroll-loops'
    setup_dolfin_parameters()
    parameters = dolfin.parameters
    # parameters["form_compiler"]["cpp_optimize_flags"] = '-O3 -funroll-loops' #not our default option

    #File prefixes for saving outputs
    today= date.today()
    outdir = "out/" + today.strftime("%b%d") + "/a/"
    prefix = "sim" #+ 
    
    optics = False
    
    if not logging.getLogger().hasHandlers():
        logsetup = TypicalLoggingSetup(filename_prefix=outdir + prefix)
        logsetup.setup()
        # # remove the junk going to console (riot convo 7 July 2020)
        # # fmt: off
        console_filter = logsetup.stream_console.filters[-1]
        console_filter.name_levelno_rules.insert(0, ("newton.optical", logging.ERROR))
        # console_filter.name_levelno_rules.insert(0, ("newton.ntrl", logging.ERROR))
        # console_filter.name_levelno_rules.insert(0, ("newton.thmq", logging.ERROR))
        # console_filter.name_levelno_rules.insert(0, ("newton", logging.ERROR))
        # console_filter.name_levelno_rules.insert(0, ("stepper", logging.ERROR))
        # # fmt: on
    # TypicalLoggingSetup(filename_prefix=outdir + prefix).setup()

    U = make_unit_registry(("mesh_unit = 1 micrometer",))

    s = dict(
        type="constant",
        edge_length=0.05, 
    )  

    layers = [dict(name='n', material='Silicon', thickness=0.1, mesh=s),  
                dict(name='p', material='Silicon', thickness=0.1, mesh=s),            
              ]
    ls = ConstructionHelperLayeredStructure()
    ls.params = dict(edge_length=0.02, # max edge length #default coarseness, was 0.0005
                     layers=layers,
                    #  simple_overmesh_regions=simple_overmesh_regions,
                     extra_regions=[],
                     mesh_unit=U.mesh_unit)
    ls.run()
    mesh_data = ls.mesh_data

    logging.getLogger("main").info(
        "NUM_MESH_POINTS: {}".format(
            len(ls.interval_1d_tag.coordinates["coordinates"])
        )
    )

    ## topology
    R = CellRegions()
    F = FacetRegions()
    topology_standard_contacts(R, F)

    F.l_contact = F.left_contact
    F.r_contact = F.right_contact
    ## end topology

    def create_problemdata(goal='full', phi_cn=None):
        """
goal in {'full', 'local charge neutrality', 'thermal equilibrium'}
"""
        root = ProblemData(
            goal=goal,
            mesh_data=mesh_data,
            unit_registry=U)
        pdd = root.pdd

        CB = pdd.easy_add_band('CB', band_type='nondegenerate')
        VB = pdd.easy_add_band('VB', band_type='nondegenerate')

        # material to region mapping
        spatial = pdd.spatial
        spatial.add_rule('temperature', R.domain, U('300 K'))

        spatial.add_rule('poisson/static_rho', R.n,  U('1e18 elementary_charge/cm^3'))
        spatial.add_rule('poisson/static_rho', R.p,  U('-1e18 elementary_charge/cm^3'))
        
        spatial.add_rule('opt_cv/alpha', R.domain, U('1e4 /cm'))        

        
        
        SiliconMaterial(problem_data=root).register()
                    
        mu = pdd.mesh_util

        contact_facets = [F.l_contact, F.r_contact]
        contact_names = ['l_contact', 'r_contact']
        zeroE = F.exterior

        def define_voltage_contacts(facets, names, goal):
            if goal == 'full':
                bias_parameters = {}
                
                for cf, name in zip(facets, names):
                    Vc = U.V * dolfin.Constant(0.0)
                    #Ohmic contacts
                    spatial.add_BC('CB/u', cf, CB.thermal_equilibrium_u)
                    spatial.add_BC('VB/u', cf, VB.thermal_equilibrium_u)

                    spatial.add_BC('poisson/phi', cf, phi0 - Vc)
                    # Voltage bias at this contact can be applied by altering
                    # the value of bias_parameters[facet].
                    bias_parameters[cf] = {'name': name, 'facet': cf, 'bias':Vc}
                return bias_parameters

            elif goal == 'thermal equilibrium':
                for cf in contact_facets:
                    spatial.add_BC('poisson/phi', cf, phi_cn)
                return None

        if goal == 'full':
            for cf in contact_facets:
                zeroE -= cf.both()

            # pdd.easy_add_electro_optical_process(
            #     SRHRecombination, name='SRH', dst_band=CB, src_band=VB)

            if True:                
                pdd.easy_add_electro_optical_process(
                    NonOverlappingTopHatBeerLambert,
                    name="opt_cv",
                    dst_band=CB, src_band=VB)


            spatial.add_BC('CB/j', F.nonconductive,
                            U('A/cm^2') * mu.zerovec)
            spatial.add_BC('VB/j', F.nonconductive,
                            U('A/cm^2') * mu.zerovec)                        
            phi0 = pdd.poisson.thermal_equilibrium_phi
                          
        elif goal == 'thermal equilibrium':
            for cf in contact_facets:
                zeroE -= cf.both()

        root.contacts = define_voltage_contacts(contact_facets, contact_names, goal)
        spatial.add_BC('poisson/E', zeroE, U('V/m') * mu.zerovec)

        #MUMPS ICNTL settings
        pdd.mumps_icntl = ((6,5),(7,0),(8,77),(10,-2))

        return root

    # solver_params = {'extra_iterations': 2, 'maximum_du': None,
    #                  'relaxation_parameter': 1.0, 'absolute_tolerance': 1e-10}
    solver_params = {}


    V_target = np.linspace(0, 0.001, 2+1)
    V_target=[0.003]

    # checkpoint_write_values = [0.001]
    checkpoint_write_filename = outdir + prefix + "_checkpoint" 

    checkpoint_write_values = []        

    checkpoint_reload_yaml = "./out/Apr19/a/sim_checkpoint_V=0.001.yaml"
    # checkpoint_reload_yaml = None
    
    meta_extractors = (
    MetaExtractorBandInfo,
    partial(MetaExtractorIntegrals,
            facets=F[{'r_contact', 'l_contact'}],
            cells=R[{'n','p'}]
            ))

    if not checkpoint_reload_yaml:
        #If we don't have a checkpoint, do the initial steps

        lcn_problem = create_problemdata(goal='local charge neutrality')
        lcn_problem.pdd.easy_auto_pre_solve(solver_params)

        eqm_problem = create_problemdata(goal='thermal equilibrium',
                                        phi_cn=lcn_problem.pdd.poisson.phi)
        eqm_problem.pdd.initialize_from(lcn_problem.pdd)
        eqm_problem.pdd.easy_auto_pre_solve(solver_params)

        full_problem = create_problemdata(goal='full')
        full_problem.pdd.initialize_from(eqm_problem.pdd)

    if not checkpoint_reload_yaml:
            # optics = True
        if optics:
            stepper = OpticalIntensityAdaptiveStepper(
                solution=full_problem,
                step_size=1e-4,
                parameter_target_values=[0, 1],
                output_writer=OutputWriter(filename_prefix=outdir + prefix,
                                            parameter_name="I",
                                            meta_extractors=meta_extractors,
                                            plotdu = False,
                                        plot_1d=True, plot_mesh=False, plot_iv=False),
                selfconsistent_optics=True)
            stepper.do_loop()
    else:
        full_problem = create_problemdata(goal='full')


    step_size = 1e-4
    bias_contact = F.r_contact
    stepper = VoltageStepper(
        solution=full_problem, 
        constants=[full_problem.contacts[bias_contact]['bias']],
        step_size=step_size,
        parameter_target_values=V_target,
        parameter_unit=U.V,
        output_writer=OutputWriter(
            filename_prefix=outdir + prefix,
            parameter_name="V",
            meta_extractors=meta_extractors,
            plot_du = True,
            plot_1d=True, plot_mesh=True, plot_iv=True),
        selfconsistent_optics=optics,
        checkpoint_write_values=checkpoint_write_values,
        checkpoint_write_filename=checkpoint_write_filename,
        checkpoint_reload_yaml=checkpoint_reload_yaml,)

    stepper.do_loop()


    end_time = datetime.now()
    logging.info("Elapsed time: {}".format(end_time - start_time))
    return locals()

if __name__ == '__main__':
    main()
