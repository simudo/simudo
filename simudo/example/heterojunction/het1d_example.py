from simudo.physics import (
    ProblemData,
    SRHRecombination,
    ThermionicHeterojunction,
)
from simudo.mesh import CellRegion, CellRegions, FacetRegions

from simudo.mesh.construction_helper import ConstructionHelperLayeredStructure

from simudo.util.pint import make_unit_registry
import numpy as np

from simudo.physics import VoltageStepper, OpticalIntensityAdaptiveStepper
from functools import partial
from simudo.io.output_writer import (
    OutputWriter,
    MetaExtractorBandInfo,
    MetaExtractorIntegrals,
)

import dolfin
from simudo.util import TypicalLoggingSetup

from .silicon import SiliconMaterial
from .silicongermanium import SiliconGermaniumAlloy

from datetime import date


def topology_standard_contacts(cell_regions, facet_regions):
    R, F = cell_regions, facet_regions

    F.left_contact = lc = R.exterior_left.boundary(R.domain)
    F.right_contact = rc = R.domain.boundary(R.exterior_right)

    F.exterior = R.domain.boundary(R.exterior)
    F.contacts = lc | rc
    F.nonconductive = F.exterior - F.contacts.both()


def main():
    # FIXME: this should go elsewhere
    parameters = dolfin.parameters
    parameters["refinement_algorithm"] = "plaza_with_parent_facets"
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -funroll-loops"

    datestring = date.strftime(date.today(), "%b%d")
    filename_prefix = "out/" + datestring + "/het1d_0"

    TypicalLoggingSetup(filename_prefix=filename_prefix).setup()

    U = make_unit_registry(("mesh_unit = 1 micrometer",))

    length = 0.5
    std_mesh = {"type": "geometric", "start": 0.001, "factor": 1}
    layers = [
        dict(
            name="emitter",
            material="SiliconGermanium",
            thickness=length,
            mesh=std_mesh,
        ),
        dict(name="barrier", material="Silicon", thickness=0.1, mesh=std_mesh),
        dict(
            name="base",
            material="SiliconGermanium",  # change to Silicon to make just one interface
            thickness=length,
            mesh=std_mesh,
        ),
    ]

    ls = ConstructionHelperLayeredStructure()
    ls.params = dict(
        edge_length=0.02,  # default coarseness
        layers=layers,
        extra_regions=[],
        mesh_unit=U.mesh_unit,
    )
    ls.run()
    mesh_data = ls.mesh_data

    ## topology
    R = CellRegions()
    F = FacetRegions()

    topology_standard_contacts(R, F)
    F.p_contact = F.left_contact
    F.n_contact = F.right_contact

    ## end topology

    def create_problemdata(goal="full", phi_cn=None):
        """
        goal in {'full', 'local charge neutrality', 'thermal equilibrium'}
        """
        root = ProblemData(goal=goal, mesh_data=mesh_data, unit_registry=U)
        pdd = root.pdd

        CB = pdd.easy_add_band("CB", band_type="nondegenerate")
        VB = pdd.easy_add_band("VB", band_type="nondegenerate")

        # material to region mapping
        spatial = pdd.spatial

        spatial.add_rule("temperature", R.domain, U("300 K"))

        spatial.add_rule("MoleFractionX", R.domain, dolfin.Constant(U("0.5")))

        spatial.add_rule(
            "poisson/static_rho", R.emitter, U("1e18 elementary_charge/cm^3")
        )

        spatial.add_rule(
            "poisson/static_rho", R.base, U("1e18 elementary_charge/cm^3")
        )

        spatial.add_rule(
            "poisson/static_rho", R.barrier, U("1e18 elementary_charge/cm^3")
        )

        SiliconMaterial(problem_data=root).register()
        SiliconGermaniumAlloy(problem_data=root).register()

        mu = pdd.mesh_util

        contact_facets = [F.p_contact, F.n_contact]
        contact_names = ["p_contact", "n_contact"]

        zeroE = F.exterior

        def define_voltage_contacts(facets, names, goal):
            if goal == "full":
                bias_parameters = {}
                for cf, name in zip(facets, names):
                    Vc = U.V * dolfin.Constant(0.0)
                    spatial.add_BC("CB/u", cf, CB.thermal_equilibrium_u)
                    spatial.add_BC("VB/u", cf, VB.thermal_equilibrium_u)
                    spatial.add_BC("poisson/phi", cf, phi0 - Vc)
                    # Voltage bias at this contact can be applied by altering
                    # the value of bias[parameters[facet].
                    bias_parameters[cf] = {
                        "name": name,
                        "facet": cf,
                        "bias": Vc,
                    }
                return bias_parameters

            elif goal == "thermal equilibrium":
                for cf in contact_facets:
                    spatial.add_BC("poisson/phi", cf, phi_cn)
                return None

        if goal == "full":
            for cf in contact_facets:
                zeroE -= cf.both()

            pdd.easy_add_electro_optical_process(
                SRHRecombination, dst_band=CB, src_band=VB
            )

            spatial.add_BC("CB/j", F.nonconductive, U("A/cm^2") * mu.zerovec)
            spatial.add_BC("VB/j", F.nonconductive, U("A/cm^2") * mu.zerovec)

            phi0 = pdd.poisson.thermal_equilibrium_phi

            optical = root.optical
            ospatial = optical.spatial

        elif goal == "thermal equilibrium":
            for cf in contact_facets:
                zeroE -= cf.both()

        root.contacts = define_voltage_contacts(
            contact_facets, contact_names, goal
        )
        spatial.add_BC("poisson/E", zeroE, U("V/m") * mu.zerovec)

        if goal == "full":
            boundary1 = R.base.boundary(R.barrier)
            boundary2 = R.emitter.boundary(R.barrier)
            ThermionicHeterojunction(CB, boundary1).register()
            ThermionicHeterojunction(CB, boundary2).register()
            ThermionicHeterojunction(VB, boundary1).register()
            ThermionicHeterojunction(VB, boundary2).register()

        return root

    solver_params = {}

    lcn_problem = create_problemdata(goal="local charge neutrality")
    lcn_problem.pdd.easy_auto_pre_solve(solver_params)

    eqm_problem = create_problemdata(
        goal="thermal equilibrium", phi_cn=lcn_problem.pdd.poisson.phi
    )
    eqm_problem.pdd.initialize_from(lcn_problem.pdd)
    eqm_problem.pdd.easy_auto_pre_solve(solver_params)

    full_problem = create_problemdata(goal="full")
    full_problem.pdd.initialize_from(eqm_problem.pdd)

    meta_extractors = (
        MetaExtractorBandInfo,
        partial(
            MetaExtractorIntegrals,
            facets=F[{"p_contact", "n_contact"}],
            cells=R[{"emitter", "barrier", "base"}],
        ),
    )

    optics = False
    #    full_problem.optical.I_scale.assign(0)
    if optics:
        stepper = OpticalIntensityAdaptiveStepper(
            solution=full_problem,
            step_size=1e-5,
            parameter_target_values=[0, 1],
            output_writer=OutputWriter(
                filename_prefix="out/optramp",
                plot_1d=True,
                plot_mesh=True,
                plot_iv=False,
            ),
            selfconsistent_optics=True,
        )

        stepper.do_loop()

    bias_contact = F.p_contact

    V_target = np.linspace(0, -0.2, 7 + 1)

    stepper = VoltageStepper(
        solution=full_problem,
        constants=[full_problem.contacts[bias_contact]["bias"]],
        step_size=1e-1,
        parameter_target_values=V_target,
        parameter_unit=U.V,
        output_writer=OutputWriter(
            filename_prefix=filename_prefix,
            parameter_name="V",
            meta_extractors=meta_extractors,
            plot_1d=True,
            plot_mesh=False,
            plot_iv=True,
            line_cut_resolution=20001,
        ),
        selfconsistent_optics=False,
    )

    stepper.do_loop()

    print("starting second set of voltages")
    V_target2 = np.linspace(0, 0.2, 7 + 1)
    full_problem2 = create_problemdata(goal="full")
    full_problem2.pdd.initialize_from(eqm_problem.pdd)

    stepper = VoltageStepper(
        solution=full_problem2,
        constants=[full_problem2.contacts[bias_contact]["bias"]],
        step_size=1e-1,
        parameter_target_values=V_target2,
        parameter_unit=U.V,
        output_writer=OutputWriter(
            filename_prefix=filename_prefix,
            meta_extractors=meta_extractors,
            parameter_name="V",
            plot_1d=True,
            plot_mesh=False,
            plot_iv=True,
            line_cut_resolution=20001,
        ),
        selfconsistent_optics=False,
    )

    stepper.do_loop()

    return locals()


if __name__ == "__main__":
    main()
