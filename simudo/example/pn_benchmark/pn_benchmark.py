
import argparse
import logging
import os
from functools import partial

import numpy as np

import dolfin

from simudo.fem import setup_dolfin_parameters
from simudo.io import h5yaml
from simudo.io.output_writer import (
    MetaExtractorBandInfo, MetaExtractorIntegrals, OutputWriter)
from simudo.mesh import (
    CellRegions, CInterval, ConstructionHelperLayeredStructure, FacetRegions,
    Interval)
from simudo.physics import (
    NonOverlappingTopHatBeerLambert, OpticalIntensityAdaptiveStepper,
    PoissonDriftDiffusion, ProblemData, Material,
    SRHRecombination, VoltageStepper,
    MixedDensityNondegenerateBand)
from simudo.util import TypicalLoggingSetup, make_unit_registry


def topology_standard_contacts(cell_regions, facet_regions):
    R, F = cell_regions, facet_regions

    F.left_contact  = lc = R.exterior_left.boundary(R.domain        )
    F.right_contact = rc = R.domain       .boundary(R.exterior_right)

    F.exterior = R.domain.boundary(R.exterior)
    F.contacts = lc | rc
    F.nonconductive = F.exterior - F.contacts.both()


class SimpleSiliconMaterial(Material):
    '''Very simple silicon material. No temperature dependences.

Copy-pasted here so it's unaffected by people messing with
:py:mod:`.physics.material`.'''

    name = 'Si'

    def get_dict(self):
        d = super().get_dict()
        U = self.unit_registry

        d.update({
            # band parameters
            # source: http://www.ioffe.ru/SVA/NSM/Semicond/Si/bandstr.html
            'CB/energy_level': U('1.12 eV'),
            'VB/energy_level': U('0 eV'),
            'CB/effective_density_of_states': U('3.2e19 1/cm^3'),
            'VB/effective_density_of_states': U('1.8e19 1/cm^3'),

            # electrical properties
            # source: http://www.ioffe.ru/SVA/NSM/Semicond/Si/electric.html
            'CB/mobility': U('1400 cm^2/V/s'),
            'VB/mobility': U(' 450 cm^2/V/s'),

            # poisson
            # http://www.ioffe.ru/SVA/NSM/Semicond/Si/optic.html
            'poisson/permittivity': U('11.7 vacuum_permittivity'),
        })

        return d

class SentaurusComparisonSilicon(SimpleSiliconMaterial):

    name = 'Si'

    def get_dict(self):
        d = super().get_dict()
        U = self.unit_registry

        # override mobilities
        d['CB/mobility'] = d['VB/mobility'] = U(
            '773.6345825238105 centimeter ** 2 / second / volt')

        return d

def main():
    from pprofile import StatisticalProfile, Profile

    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='*')
    args = parser.parse_args()

    for input in args.inputs:
        with open(input, "rt") as h:
            tree = h5yaml.load(h)
        prefix = os.path.dirname(input).rstrip('/')+'/'
        def thunk():
            return submain(prefix=prefix, tree=tree)

        STATISTICAL_PROFILE = True

        if STATISTICAL_PROFILE:
            prof = StatisticalProfile()
            with_item = prof(period=0.001)
        else:
            prof = Profile()
            with_item = prof()

        with with_item:
            thunk()

        with open(prefix+'cachegrind.out.0', 'wt') as file:
            prof.callgrind(file)
        with open(prefix+'pprofile.txt', 'wt') as file:
            prof.annotate(file)

def submain(prefix, tree):
    PREFIX = prefix
    P = tree["parameters"]

    setup_dolfin_parameters()

    TypicalLoggingSetup(filename_prefix=PREFIX).setup()

    U = make_unit_registry(("mesh_unit = 1 micrometer",))

    length = P['device_length']
    layers = [dict(name='pSi', material='Si' , thickness=length/2),
              dict(name='nSi', material='Si' , thickness=length/2)]

    # -layer means relative to left endpoint of layer
    # +layer means relative to right endpoint of layer
    simple_overmesh_regions = [
        # dict(x0=('+nSi', -0.2), x1=('+ISi', +0.2), edge_length=0.01),
    ]

    meshp = P['mesh_points']
    if meshp > 0:
        mesh_points = meshp
        nonuniform = False
    elif meshp == -1:
        mesh_points = 100
        nonuniform = True
        close_to_contact = length/20
        close_to_contact_edge_length = length/10000
    else:
        raise ValueError()

    if nonuniform:
        for where in ('-pSi', '+nSi'):
            simple_overmesh_regions.append(dict(
                x0=(where, -close_to_contact), x1=(where, +close_to_contact),
                edge_length=close_to_contact_edge_length))

    ls = ConstructionHelperLayeredStructure()
    ls.params = dict(edge_length=length/mesh_points, # default edge_length
                     layers=layers,
                     simple_overmesh_regions=simple_overmesh_regions,
                     mesh_unit=U.mesh_unit)
    ls.run()
    mesh_data = ls.mesh_data

    logging.getLogger('main').info("NUM_MESH_POINTS: {}".format(
        len(ls.interval_1d_tag.coordinates['coordinates'])))

    ## topology
    R = CellRegions()
    F = FacetRegions()

    topology_standard_contacts(R, F)

    R.Si = R.pSi | R.nSi

    F.p_contact   = F.left_contact
    # F.pI_junction = R.pSi.boundary(R.ISi)
    # F.In_junction = R.ISi.boundary(R.nSi)
    F.pn_junction = R.pSi.boundary(R.nSi)
    F.n_contact   = F.right_contact

    F.IB_junctions = F.pI_junction | F.In_junction
    ## end topology

    # print(mesh_data.region_name_to_cvs)
    # print(mesh_data.facets_name_to_fvs)
    # print(mesh_data.evaluate_topology(F.pn_junction))

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

        transport_method = P['transport_method']
        if transport_method == 'mixed_qfl_j':
            band_cls = None # default
        elif transport_method == 'mixed_u_j':
            band_cls = MixedDensityNondegenerateBand
        else:
            raise ValueError()

        CB = pdd.easy_add_band(name='CB', band_type=band_cls)
        VB = pdd.easy_add_band(name='VB', band_type=band_cls)
        # IB = pdd.easy_add_band(
        #     'IB',
        #     restrict='IB/exists') # restrict based on material property

        pdd.poisson.mixed_debug_quad_degree_rho = P['quaddeg_rho']

        for b in (VB, CB):
            b.mixedqfl_debug_use_bchack = P['bchack']
            b.mixedqfl_debug_fill_from_boundary = P['fill_from_boundary']
            b.mixedqfl_debug_fill_thresholds = (
                float(P['fill_threshold_abs']),
                float(P['fill_threshold_rel']))
            b.mixedqfl_debug_fill_with_zero_except_bc = (
                P['fill_with_zero_except_bc'])
            b.mixedqfl_debug_quad_degree_super = P['quaddeg_super']
            b.mixedqfl_debug_quad_degree_g = P['quaddeg_g']
            b.mixed_debug_quad_degree_super = P['quaddeg_super']
            b.mixed_debug_quad_degree_g = P['quaddeg_g']

        # material to region mapping
        spatial = pdd.spatial

        spatial.add_rule(
            'temperature', R.domain, U('300 K'))

        spatial.add_rule(
            'poisson/static_rho', R.pSi,
            -float(P['doping'])*U('elementary_charge/cm^3'))
        spatial.add_rule(
            'poisson/static_rho', R.nSi,
            +float(P['doping'])*U('elementary_charge/cm^3'))

        #SimpleSiliconMaterial(problem_data=root).register()
        SentaurusComparisonSilicon(problem_data=root).register()

        mu = pdd.mesh_util

        zeroE = F.exterior

        if goal == 'full':
            optical = root.optical
            ospatial = optical.spatial

            pdd.easy_add_electro_optical_process(
                SRHRecombination, dst_band=CB, src_band=VB)

            if False:
                optical.easy_add_field('forward', photon_energy=U('3 eV'),
                                      direction=(1.0, 0.0))

                pdd.easy_add_electro_optical_process(
                    NonOverlappingTopHatBeerLambert, dst_band=CB, src_band=VB,
                    name='beer_lambert',
                    get_alpha=lambda wavelength: 1/U("0.1 um"))

                ospatial.add_BC('forward/Phi', F.left_contact,
                                1e20 * U('1/cm^2/s'))

            spatial.add_BC('CB/j', F.nonconductive,
                           U('A/cm^2') * mu.zerovec)
            spatial.add_BC('VB/j', F.nonconductive,
                           U('A/cm^2') * mu.zerovec)

            # minority contact
            if P['min_srv'] == 'inf':
                spatial.add_BC('CB/u', F.p_contact,
                               CB.thermal_equilibrium_u)
                spatial.add_BC('VB/u', F.n_contact,
                               VB.thermal_equilibrium_u)
            elif P['min_srv'] == '0':
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
            zeroE -= (F.p_contact | F.n_contact).both()

        elif goal == 'thermal equilibrium' and True:
            # to match old method, use local charge neutrality phi as
            # the phi boundary condition for Poisson-only thermal
            # equilibrium
            spatial.add_BC('poisson/phi', F.p_contact | F.n_contact,
                           phi_cn)
            zeroE -= (F.p_contact | F.n_contact).both()

        spatial.add_BC('poisson/E', zeroE,
                       U('V/m') * mu.zerovec)

        spatial.add_rule('SRH/CB/tau', R.domain, U('1e-9 s'))
        spatial.add_rule('SRH/VB/tau', R.domain, U('1e-6 s'))
        spatial.add_rule('SRH/energy_level', R.domain,
                         U('0.5525628437189991 eV'))

        # for t in ['continuity_j_term',
        #           'continuity_g_term',
        #           'drift_diffusion_j_term',
        #           'drift_diffusion_diffusion_term',
        #           'drift_diffusion_drift_term',
        #           'drift_diffusion_bc_term']:
        #     t = 'mixed_' + t
        #     v = getattr(CB, t, None)
        #     print('ZZZZZZZZ', t)
        #     print(v)
        #     if v is not None:
        #         print(v.to_ufl(units='dimensionless'))

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

    meta_extractors = (
        MetaExtractorBandInfo,
        partial(MetaExtractorIntegrals,
                facets=F[{'p_contact', 'n_contact'}],
                cells=R[{'pSi', 'nSi'}]))

    def sentaurus_debug_test():
        from simudo.misc import sentaurus_import as sen
        import lzma
        with lzma.open("/home/user/j/data/sentaurus/"
                       "diode_1d c=1e18 V=0.csv.xz", mode='rb') as handle:
            df, dfu = sen.read_df_unitful(
                handle=handle,
                unit_registry=U)
        xtest = create_problemdata(goal='full', V_ext=V_ext)
        # xtest.pdd.initialize_from(problem0.pdd)
        sen.jam_sentaurus_data_into_solution(df, dfu, xtest)

        ow = OutputWriter(
            PREFIX+"sen", plot_1d=True, plot_iv=False,
            meta_extractors=meta_extractors)
        ow.write_output(xtest, 0.0)

    # sentaurus_debug_test()
    # return

    optics = False

    if optics:
        stepper = OpticalIntensityAdaptiveStepper(
            solution=problem,
            # parameter_target_values=[1e-10, 0.1, 1.0],
            # parameter_target_values=[1.0],
            step_size=1e-10,
            # step_size=0.5,
            output_writer=OutputWriter(
                PREFIX+"I", plot_1d=True, plot_iv=False,
                meta_extractors=meta_extractors),
            selfconsistent_optics=True)

        stepper.do_loop()

    to_Vext_str = {d['Vext']: d['Vext_str']
                   for d in P['voltage_steps']}
    stepper = VoltageStepper(
        solution=problem, constants=[V_ext],

        # parameter_target_values=[1.0],
        # step_size=0.5,

        # parameter_target_values=np.linspace(0, 0.8, 4 + 1),
        parameter_target_values=[d['Vext'] for d in P['voltage_steps']],

        # parameter_target_values=np.arange(0, 12)*0.1,

        parameter_unit=U.V,
        output_writer=OutputWriter(
            filename_prefix=PREFIX+"a", plot_1d=True, plot_iv=False,
            format_parameter=(lambda solution, V: to_Vext_str[V]),
            meta_extractors=meta_extractors),
        selfconsistent_optics=optics)

    stepper.do_loop()

    # from simudo.trash.mrs2018_sentaurus_plot import main as plotmain

    # for a in ['0.2', '0.4', '0.6', '0.8']:
    #     plotmain(input=PREFIX+"a parameter="+a+" csvplot.csv.0",
    #              output_prefix=PREFIX+'xplot V='+a+' '+P['title']+' ',
    #              sentaurus_file="data/sentaurus/diode_1d c=1e18 V="+a+".csv.xz",
    #              title=P['title'])

    return locals()

if __name__ == '__main__':
    main()
