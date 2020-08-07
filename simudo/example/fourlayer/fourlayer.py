# -*- coding: utf-8 -*-
r"""Set up four layer FSF-p-IB-n calculations
Adapted from ``marti20002_customizable_nonradiative.py``

The most important functions are:

- :py:func:`multiplex_setup` creates submit.yaml files corresponding
  to a combinatorial array of input parameters

- :py:func:`run` runs simudo with specified submit.yaml

:py:func:`run` receives a yaml submitfile with a dict ``P`` of parameters
and executes the Simudo run according to those parameters. It is
designed to treat a 1D layered device (though all Simudo runs are
technically 2D).  Physically, this device has four layers with
contacts at left and right. Light is incident from the right.  Layer
thicknesses are controlled by keys ``FSF_thickness``, ``p_thickness``,
``IB_thickness``, ``n_thickness``, in um.

Uniform properties
    There is a uniform band gap throughout the device. Band edges in
    eV are in ``VB_E``, ``CB_E`` and ``IB_E``.  VB and CB have effective DOS
    of ``NV``, ``NC`` in cm^-3

    Cell temperature is ``T`` and there is a uniform relative
    permittivity ``epsilon_r``.

    Electron and hole mobilities are uniform, set by ``mu_n``,
    ``mu_p`` in :math:`\mathrm{cm^2/Vs}`.

BC's
    Majority carriers have Ohmic BC's at the contacts and minority
    carriers have 0 SRV at the contacts.

FSF
    Zero optical absorption and doping `NFSF` in cm^-3.

p- and n- regions
    ``alpha_cv`` sets absorption coefficient in 1/cm. Constant for all E>Eg.
    The associated radiative recombination is the only recombination in those layers.
    Uniform doping is set by ``NA``, ``ND`` in cm^-3

IB region
    IB has a sharp energy ``IB_E`` with ``NI`` states in cm^-3

    Optical absorption and Shockley-Read trapping processes, set by
    ``sigma_opt_ci``, ``sigma_opt_iv``, ``sigma_th_ci``,
    ``sigma_th_iv`` in cm^2.

    The thermal (SR) processes also depend on ``vth_CB`` and ``vth_VB``,
    in cm/s.

    There are two models for IB mobility. ``simple_mobility_model=True`` uses
    :math:`j_I = mu_I n_I \nabla (\mathrm{qfl}_I)` with constant ``mu_I``.

    ``simple_mobility_model=False`` changes ``mu_I`` with filling
    fraction ``f`` in the IB, so
    mobility is ``mu_I`` when ``f=0`` and ``0`` when ``f=1``.

    For details, see ``IntermediateBand`` in
    ``simudo/physics/poisson_drift_diffusion1.py1``

Optical absorption
    Black body radiation with solar temperature ``Ts`` (K) with ``X``
    suns concentration in incident from the left.

    Absorptions are nonoverlapping, so each photon can only be
    absorbed by highest-energy transition (CV, CI, IV) consistent with
    conservation of energy.

    IB absorption processes depend on IB filling, so optical problem
    is solved self-consistently.

Numerical parameters
    Meshes geometrically expand away from interfaces with initial mesh
    size ``mesh_start`` (um) and expansion factor ``mesh_factor``. The
    maximum mesh spacing is ``mesh_max`` (um)

Results are output in the dark (I=0) and at whatever specified ``X``
(I=1) at the set of applied voltages ``V_exts`` (V)

"""

# simudo version a04dde44ef34d2843e624b99aee80dd02100ff51

from simudo.physics import (
    Material,
    ProblemData,
    PoissonDriftDiffusion,
    SRHRecombination,
    NonRadiativeTrap,
    NonOverlappingTopHatBeerLambert,
    NonOverlappingTopHatBeerLambertIB,
)

from simudo.mesh import (
    ConstructionHelperLayeredStructure,
    Interval,
    CInterval,
    CellRegions,
    FacetRegions,
)

from simudo.fem import setup_dolfin_parameters

from simudo.util import make_unit_registry, DictAttrProxy, TypicalLoggingSetup

from simudo.io import h5yaml

from functools import partial
from cached_property import cached_property
import pandas as pd
import numpy as np
import os
from os import path as osp
from pathlib import Path
from argh import ArghParser, arg, expects_obj
import itertools
from datetime import datetime
from atomicwrites import atomic_write
import attr

# import copy

import dolfin
import logging

__all__ = ["run", "multiplex_setup"]


def dirname(path):
    path = Path(path)
    return path.parent


# @attr.s
class Semiconductor(Material):
    # params can be a dictionary of parameters for the material.
    # Must contain a bunch of things I should really write in this line; see below
    #   params = attr.ib(default=None) #replaced by SetattrInitMixin, which sets all kwargs as attributes
    name = "semiconductor"

    def get_dict(self):
        # from run_intrinsic import IB_E
        d = super().get_dict()
        U = self.unit_registry
        params = self.params

        d.update(
            {
                # band parameters
                "CB/energy_level": U(str(params["CB_E"]) + " eV"),
                "IB/energy_level": U(str(params["IB_E"]) + " eV"),
                "VB/energy_level": U(str(params["VB_E"]) + " eV"),
                "IB/degeneracy": 1,
                "CB/effective_density_of_states": U(
                    str(params["NC"]) + " 1/cm^3"
                ),
                "VB/effective_density_of_states": U(
                    str(params["NV"]) + " 1/cm^3"
                ),
                # electrical properties
                "CB/mobility": U(str(params["mu_n"]) + " cm^2/V/s"),
                "VB/mobility": U(str(params["mu_p"]) + " cm^2/V/s"),
                "opt_cv/alpha": U(str(params["alpha_cv"]) + " cm^-1"),
                # poisson
                "poisson/permittivity": U(
                    str(params["epsilon_r"]) + " vacuum_permittivity"
                ),
            }
        )
        return d


class IBSemiconductor(Semiconductor):
    name = "IB"

    def get_dict(self):
        d = super().get_dict()
        U = self.unit_registry
        params = self.params

        if params is not None:
            d.update(
                {
                    # these cross sections give alpha=1e4/cm at half-filled IB
                    # `1e4 / (1e17/2.)`
                    "opt_ci/sigma_opt": U(
                        str(params["sigma_opt_ci"]) + " cm^2"
                    ),
                    "opt_iv/sigma_opt": U(
                        str(params["sigma_opt_iv"]) + " cm^2"
                    ),
                    "IB/number_of_states": U(str(params["NI"]) + " cm^-3"),
                }
            )

        return d


class FrontSurfaceField(Semiconductor):
    name = "fsf"

    def get_dict(self):
        d = super().get_dict()
        U = self.unit_registry

        d["opt_cv/alpha"] = U("0 cm^-1")

        return d


def topology_standard_contacts(cell_regions, facet_regions):
    R, F = cell_regions, facet_regions

    F.left_contact = lc = R.exterior_left.boundary(R.domain)
    F.right_contact = rc = R.domain.boundary(R.exterior_right)

    F.exterior = R.domain.boundary(R.exterior)
    F.contacts = lc | rc
    F.nonconductive = F.exterior - F.contacts.both()


def submitfile_to_title(submitfile_tree):
    p = submitfile_tree["parameters"]
    title = "{prefix}".format(**p)
    for key in p["title_keys"]:
        title += key + "=" + str(p[key])

    return title


def multiplex_setup(params, output_prefix="out", return_values=False):
    # params is a dict of parameters.
    # This method multiplexes over the items listed in params['multiplex_keys']
    # Outputs files of the form prefix/#/submit.yaml, where # iterates from 0
    # If params['multiplex_keys'] is False/None, files are placed in prefix/submit.yaml
    print("multiplex_setup")
    output_prefix = Path(output_prefix)
    outfiles = []

    def write(data, filename):
        dirname(filename).mkdir(parents=True, exist_ok=True)
        h5yaml.dump(data, str(filename))

    if not params["multiplex_keys"]:
        # Not multiplexing
        submitfile_tree = dict(parameters=params)
        title = submitfile_to_title(submitfile_tree)
        submitfile_tree["parameters"]["title"] = title
        outfile = output_prefix / "submit.yaml"
        outfiles.append(outfile)
        write(submitfile_tree, str(outfile))
        print("outfile=" + str(outfile))
    else:
        for counter, updateparams in enumerate(
            itertools.product(
                *(params[mkey] for mkey in params["multiplex_keys"])
            )
        ):

            curparams = params.copy()  # Shallow copy, be warned
            # Update curparams with the current multiplexed value for each appropriate key
            for i, key in enumerate(params["multiplex_keys"]):
                curparams.update({key: updateparams[i]})

            submitfile_tree = dict(parameters=curparams)

            title = submitfile_to_title(submitfile_tree)
            submitfile_tree["parameters"]["title"] = title

            outfile = output_prefix / str(counter) / "submit.yaml"
            outfiles.append(outfile)

            # Logic to avoid overwriting a submit.yaml file
            # if os.path.exists(outfile):
            #     print(str(outfile) + " already exists. Not writing new.")
            #     continue

            write(submitfile_tree, str(outfile))
            print("outfile=" + str(outfile))

    print("end multiplex_setup")
    return outfiles


@arg("submitfiles", nargs="*")
def fixup_titles(submitfiles):
    for submitfile in submitfiles:
        submitfile = str(submitfile)
        t = h5yaml.load(submitfile)
        p = t["parameters"]
        p.setdefault("IB_sigma_ci", "2e-13")
        p.setdefault("IB_sigma_iv", "2e-13")

        p["title"] = title = submitfile_to_title(t)

        with atomic_write(submitfile, overwrite=True) as f:
            h5yaml.dump(t, f)

        olddir = osp.dirname(submitfile)
        newdir = osp.join(osp.dirname(olddir), title)
        os.rename(olddir, newdir)


def run(submitfile):
    # submitfile as produced by multiplex_setup

    U = make_unit_registry(("mesh_unit = 1 micrometer",))
    if isinstance(submitfile, list) and len(submitfile) == 1:
        submitfile = submitfile[0]
    PREFIX = dirname(submitfile)
    print(PREFIX)
    submitfile_data = h5yaml.load(str(submitfile))
    P = submitfile_data["parameters"]

    # Set up logging if it hasn't already been set up
    if not logging.getLogger().hasHandlers():
        logsetup = TypicalLoggingSetup(filename_prefix=str(PREFIX) + os.sep)
        logsetup.setup()

        # remove the junk going to console (riot convo 7 July 2020)
        # fmt: off
        console_filter = logsetup.stream_console.filters[-1]
        console_filter.name_levelno_rules.insert(0, ("newton.optical", logging.ERROR))
        console_filter.name_levelno_rules.insert(0, ("newton.ntrl", logging.ERROR))
        console_filter.name_levelno_rules.insert(0, ("newton.thmq", logging.ERROR))
        console_filter.name_levelno_rules.insert(0, ("newton", logging.ERROR))
        console_filter.name_levelno_rules.insert(0, ("stepper", logging.ERROR))
        # fmt: on

    # boilerplate information
    from simudo.version import __version__ as simudo_ver

    logging.info(f"#### Simudo version {simudo_ver} ####")
    logging.info(f"started at: {datetime.now()}")
    logging.info(f"params:")
    for k, v in P.items():
        logging.info(f"   {k}: {v}")

    try:
        # if the simudo folder is a fossil repo, log its status
        from simudo import __file__ as simudo_file

        simudo_dir = pathlib.Path(simudo_file).parent
        fossil_status = subprocess.run(
            ["fossil", "status"],
            cwd=simudo_dir,
            capture_output=True,
            check=True,
        )
        logging.info("""Output of 'fossil status':""")
        logging.info(fossil_status.stdout.decode("utf-8"))
    except:
        pass

    setup_dolfin_parameters()

    # fmt: off
    s = dict(
        type="geometric",
        start=float(P["mesh_start"]),
        factor=float(P["mesh_factor"]),
    )
    layers = [
        dict(name="fsf", material="fsf", thickness=float(P["FSF_thickness"]), mesh=s),
        dict(name="p", material="semiconductor", thickness=float(P["p_thickness"]), mesh=s),
        dict(name="I", material="IB", thickness=float(P["IB_thickness"]), mesh=s),
        dict(name="n", material="semiconductor", thickness=float(P["n_thickness"]), mesh=s),
    ]
    # fmt: on

    ls = ConstructionHelperLayeredStructure()
    ls.params = dict(
        edge_length=float(P["mesh_max"]),  # maximum edge length
        layers=layers,
        extra_regions=[],
        mesh_unit=U.um,
    )
    ls.run()
    mesh_data = ls.mesh_data

    if True:
        Xs_df = pd.DataFrame(
            {"X": list(sorted(set(mesh_data.mesh.coordinates()[:, 0])))}
        )
        Xs_df.to_csv(PREFIX / "mesh_Xs.csv")

    if False:
        print("exiting early")
        return

    logging.getLogger("main").info(
        "NUM_MESH_POINTS: {}".format(
            len(ls.interval_1d_tag.coordinates["coordinates"])
        )
    )

    ## topology
    R = CellRegions()
    F = FacetRegions()

    topology_standard_contacts(R, F)

    R.cell = R.p | R.n | R.I | R.fsf

    F.p_contact = F.left_contact
    F.pI_junction = R.p.boundary(R.I)
    F.In_junction = R.I.boundary(R.n)
    F.n_contact = F.right_contact

    # F.IB_bounds = F.pI_junction | F.In_junction
    F.IB_bounds = R.IB.boundary(R.domain - R.IB) | R.IB.boundary(R.exterior)
    ## end topology

    def create_problemdata(goal="full", V_ext=None, phi_cn=None):
        """
goal in {'full', 'local neutrality', 'thermal equilibrium'}
"""
        root = ProblemData(goal=goal, mesh_data=mesh_data, unit_registry=U)
        pdd = root.pdd

        CB = pdd.easy_add_band("CB")
        IB = pdd.easy_add_band("IB", subdomain=R.I)
        VB = pdd.easy_add_band("VB")

        # material to region mapping
        spatial = pdd.spatial

        spatial.add_rule("temperature", R.domain, U(str(P["T"]) + " K"))

        spatial.add_rule(
            "poisson/static_rho",
            R.p,
            U(str(P["NA"]) + " elementary_charge/cm^3"),
        )
        spatial.add_rule(
            "poisson/static_rho",
            R.n,
            U(str(P["ND"]) + " elementary_charge/cm^3"),
        )
        spatial.add_rule(
            "poisson/static_rho",
            R.fsf,
            U(str(P["NFSF"]) + " elementary_charge/cm^3"),
        )
        spatial.add_rule(
            "poisson/static_rho",
            R.I,
            U(
                str(float(P["f_0"]) * float(P["NI"]))
                + " elementary_charge/cm^3"
            ),
        )

        # for non-radiative recombination (Shockley-Read trapping)
        spatial.add_rule(
            "nr_top/sigma_th", R.I, U(str(P["sigma_th_ci"]) + " cm**2")
        )
        spatial.add_rule(
            "nr_bottom/sigma_th", R.I, U(str(P["sigma_th_iv"]) + " cm**2")
        )
        spatial.add_rule("nr_top/v_th", R.I, U(str(P["vth_CB"]) + " cm/s"))
        spatial.add_rule("nr_bottom/v_th", R.I, U(str(P["vth_VB"]) + " cm/s"))

        ib_material = IBSemiconductor(problem_data=root, params=P)
        ib_material.dict["IB/simple_mobility_model"] = bool(
            P["simple_mobility_model"]
        )
        ib_material.dict["IB/mobility0"] = float(P["mu_I"]) * U("cm^2/V/s")
        for k in ["ci", "iv"]:
            ib_material.dict["opt_{}/sigma_opt".format(k)] = float(
                P["sigma_opt_{}".format(k)]
            ) * U("cm^2")

        FrontSurfaceField(problem_data=root, params=P).register()
        semiconductor_material = Semiconductor(problem_data=root, params=P)

        ib_material.register()
        semiconductor_material.register()

        if P["simple_mobility_model"]:
            IB.use_constant_mobility = True

        mu = pdd.mesh_util

        zeroE = F.exterior

        if goal == "full":
            optical = root.optical
            ospatial = optical.spatial

            def _ediff(lower, upper):
                d = semiconductor_material.dict
                return (
                    d[upper + "/energy_level"] - d[lower + "/energy_level"]
                ) + U("1e-10 eV")

            optical.easy_add_field(
                "forward_ci",
                photon_energy=_ediff("IB", "CB"),
                direction=(1.0, 0.0),
            )

            optical.easy_add_field(
                "forward_iv",
                photon_energy=_ediff("VB", "IB"),
                direction=(1.0, 0.0),
            )

            optical.easy_add_field(
                "forward_cv",
                photon_energy=_ediff("VB", "CB"),
                direction=(1.0, 0.0),
            )

            from simudo.util import Blackbody

            bb = Blackbody(temperature=float(P["Ts"]) * U.K)

            ofield_E_keys = ("forward_cv", "forward_ci", "forward_iv")
            for name, (lower, upper) in bb.non_overlapping_energy_ranges(
                {
                    name: optical.fields[name].photon_energy
                    for name in ofield_E_keys
                }
            ).items():
                flux = bb.photon_flux_integral_on_earth(lower, upper) * float(
                    P["X"]
                )
                flux = flux.to("1/cm^2/s")
                ospatial.add_BC(name + "/Phi", F.left_contact, flux)
                logging.getLogger("main").info(
                    "Input photon flux for field {!r}: {}".format(name, flux)
                )

            pdd.easy_add_electro_optical_process(
                NonOverlappingTopHatBeerLambertIB,
                name="opt_ci",
                dst_band=CB,
                src_band=IB,
                trap_band=IB,
            )

            pdd.easy_add_electro_optical_process(
                NonOverlappingTopHatBeerLambertIB,
                name="opt_iv",
                dst_band=IB,
                src_band=VB,
                trap_band=IB,
            )

            pdd.easy_add_electro_optical_process(
                NonOverlappingTopHatBeerLambert,
                name="opt_cv",
                dst_band=CB,
                src_band=VB,
            )

            # pdd.easy_add_electro_optical_process(
            #     SRHRecombination, dst_band=CB, src_band=VB, name='SRH')

            # for non-radiative recombination
            NonRadiativeTrap.easy_add_two_traps_to_pdd(pdd, "nr", CB, VB, IB)

            spatial.add_BC("CB/j", F.nonconductive, U("A/cm^2") * mu.zerovec)
            spatial.add_BC("VB/j", F.nonconductive, U("A/cm^2") * mu.zerovec)
            # spatial.add_BC('IB/j', F.exterior,
            #                U('A/cm^2') * mu.zerovec)
            spatial.add_BC(
                "IB/j", F.nonconductive | F.IB_bounds, U("A/cm^2") * mu.zerovec
            )

            # majority contact, Ohmic BC
            spatial.add_BC("VB/u", F.p_contact, VB.thermal_equilibrium_u)
            spatial.add_BC("CB/u", F.n_contact, CB.thermal_equilibrium_u)

            # # minority contact Ohmic BC
            # spatial.add_BC('CB/u', F.p_contact,
            #                 CB.thermal_equilibrium_u)
            # spatial.add_BC('VB/u', F.n_contact,
            #                VB.thermal_equilibrium_u)

            # minority contact, 0 SRV
            spatial.add_BC("VB/j", F.n_contact, U("A/cm^2") * mu.zerovec)
            spatial.add_BC("CB/j", F.p_contact, U("A/cm^2") * mu.zerovec)

            phi0 = pdd.poisson.thermal_equilibrium_phi
            spatial.add_BC("poisson/phi", F.p_contact, phi0 + V_ext)
            spatial.add_BC("poisson/phi", F.n_contact, phi0)

            zeroE -= (F.p_contact | F.n_contact).both()

        elif goal == "thermal equilibrium":
            # to match old method, use local charge neutrality phi as
            # the phi boundary condition for Poisson-only thermal
            # equilibrium
            spatial.add_BC("poisson/phi", F.p_contact | F.n_contact, phi_cn)
            zeroE -= (F.p_contact | F.n_contact).both()

        spatial.add_BC("poisson/E", zeroE, U("V/m") * mu.zerovec)

        return root

    problem = create_problemdata(goal="local charge neutrality")

    problem.pdd.easy_auto_pre_solve()

    problem0 = problem
    problem = create_problemdata(
        goal="thermal equilibrium", phi_cn=problem0.pdd.poisson.phi
    )
    problem.pdd.initialize_from(problem0.pdd)

    problem.pdd.easy_auto_pre_solve()

    V_ext = U.V * dolfin.Constant(0.0)
    problem0 = problem
    problem = create_problemdata(goal="full", V_ext=V_ext)
    problem.pdd.initialize_from(problem0.pdd)

    from simudo.physics import VoltageStepper, OpticalIntensityAdaptiveStepper
    from simudo.io.output_writer import (
        OutputWriter,
        MetaExtractorBandInfo,
        MetaExtractorIntegrals,
    )

    meta_extractors = (
        MetaExtractorBandInfo,
        partial(
            MetaExtractorIntegrals,
            facets=F[{"p_contact", "n_contact"}],
            cells=R[{"fsf", "p", "n", "I"}],
        ),
    )

    optics = float(P["X"]) != 0

    if optics:
        stepper = OpticalIntensityAdaptiveStepper(
            solution=problem,
            parameter_target_values=[0, 1],
            step_size=1e-25,
            stepper_rel_tol=1e-6,
            output_writer=OutputWriter(
                filename_prefix=str(PREFIX) + "/pd",
                parameter_name="I",
                plot_1d=True,
                plot_iv=False,
                plot_mesh=False,
                meta_extractors=meta_extractors,
            ),
            selfconsistent_optics=True,
        )

        stepper.do_loop()

    # #parameter changed from "a parameter" to "V parameter" 10Jul2020
    # stepper = VoltageStepper(
    #     solution=problem, constants=[V_ext],
    #     parameter_target_values=P['V_exts'],
    #     parameter_unit=U.V,
    #     output_writer=OutputWriter(
    #         filename_prefix=str(PREFIX)+"/V", plot_1d=True, plot_iv=True,
    #         meta_extractors=meta_extractors),
    #     selfconsistent_optics=optics)

    # Adapted from Matt, minutiae 13 Jul 2020
    stepper = VoltageStepper(
        solution=problem,
        # constants=[problem.contacts[bias_contact]['bias']],
        constants=[V_ext],
        step_size=1e-4,
        stepper_rel_tol=1e-6,
        parameter_target_values=P["V_exts"],
        parameter_start_value=P["V_exts"][0],
        parameter_unit=U.V,
        output_writer=OutputWriter(
            filename_prefix=str(PREFIX) + "/pd",
            parameter_name="V",
            meta_extractors=meta_extractors,
            line_cut_resolution=10000,
            plot_1d=True,
            plot_mesh=False,
            plot_iv=True,
        ),
        selfconsistent_optics=optics,
    )

    stepper.do_loop()

    return  # locals()


parser = ArghParser()
parser.add_commands([multiplex_setup, run, fixup_titles])

if __name__ == "__main__":
    parser.dispatch()
