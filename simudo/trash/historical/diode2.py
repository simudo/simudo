from __future__ import division, absolute_import, print_function
from builtins import dict, int, str

from simudo.mesh.construction_helper import ConstructionHelperIntervalProduct2DMesh
from simudo.mesh.interval1dtag import Interval, CInterval
from simudo.util.solution import SolverBoilerplate
from simudo.util.assign import copy_solution, assign_subvar
from simudo.io import h5yaml
from simudo.io.csv import LineCutCsvPlot
from simudo.util.newton_solver import NewtonSolverMaxDu
from simudo.util.adaptive_stepper import (OpticalIntensityAdaptiveStepper,
                                            VoltageStepper)
from simudo.io.output_writer import solution_plot_1d, OutputWriter

import multiprocessing
import shutil
import os
import dolfin
import logging
import numpy as np
from contextlib import closing
from time import time
from datetime import timedelta, datetime
from math import floor

parameters = dolfin.parameters
parameters["refinement_algorithm"] = "plaza_with_parent_facets"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = '-O3 -funroll-loops'


def assert_(must_be_true):
    if not must_be_true:
        raise AssertionError()


class ConsIB(ConstructionHelperIntervalProduct2DMesh):
    def user_define_interval_regions(self):
        intervals = []
        coarseness_default = 0.01

        intervals.append(CInterval(-np.inf, np.inf,
                                   coarseness=coarseness_default))

        p_length = 1.0
        n_length = 1.0
        ib_length = 1.0

        x_Cp = 0.0
        x_pI = p_length
        x_In = p_length + ib_length
        x_nC = p_length + ib_length + n_length

        height = coarseness_default  # doesn't really matter
        self.product2d_Ys = (0.0, height)

        def mkinterval(*args, **kwargs):
            intervals.append(Interval(*args, **kwargs))

        domain = (x_Cp, x_nC)
        mkinterval(x_Cp, x_pI, ('pSi',))
        mkinterval(x_pI, x_In, ('ISi',))
        mkinterval(x_In, x_nC, ('nSi',))

        def create_overmesh_at(coarseness, x0, before, after=None):
            if after is None:
                after = before
            intervals.append(CInterval(
                x0-before, x0+after, coarseness=coarseness))

        # overmesh at junctions
        create_overmesh_at(0.002, x_pI-0.05, 0.1)
        create_overmesh_at(0.002, x_In-0.05, 0.1)
        create_overmesh_at(0.002, 0., 0.05)
        create_overmesh_at(0.002, 3.-0.05, 0.05)

        mkinterval(domain[0], domain[1], ('domain',))
        return (domain, intervals)

    def user_extra_definitions(self):
        R = self.cell_regions
        F = self.facet_regions
        fc = self.facets

        R['Si'] = R['pSi'] | R['nSi'] | R['ISi']

        R['has_IB'] = R['ISi']
        R['no_IB'] = R['domain'] - R['ISi']

        F.update(
            p_contact=fc.boundary(R['left'], R['pSi']),
            pI_contact=fc.boundary(R['pSi'], R['ISi']),
            In_contact=fc.boundary(R['ISi'], R['nSi']),
            n_contact=fc.boundary(R['nSi'], R['right']))

        F['top_bottom'] = fc.boundary(
            R['domain'], R['top'] | R['bottom'])
        F['contacts'] = F['p_contact'] | F['n_contact']
        F['IB_junctions'] = F['pI_contact'] | F['In_contact']


class DiodeExample(SolverBoilerplate):
    def diode_genmesh(self, params):
        meshfile = params['prefix']+'mesh.yaml'
        cons = ConsIB()
        cons.params = dict(some_parameter=42)
        cons.run()
        cons.save(meshfile)
        sol = self.loader.load(meshfile, override_fragments=(
            'minimal',))
        return sol

    def make_sf(self, other_sf, method=None):
        # FIXME: method arguments for both bands and poisson

        keys = [
            'minimal', 'pde',
            'solution_mixed_function_space',
            'geometry_param', 'electro_misc',
            'physics_common',
            'poisson',
            # 'poisson_CG',
            'poisson_mixed',
            'material_figureofmerit',
            # 'material_pfsf',
            'diode_recombination_1em9',
            'simple_nondegenerate_semiconductor',
            'base_CB', 'base_VB', 'base_IB',
            'nondegenerate_CB', 'nondegenerate_VB', 'intermediateband_IB',
            'diode_problem', 'mesh_unit_micrometer',

        ]

        def add_VB_CB(*bandk):
            keys.extend(a+'_'+band
                        for band in ('VB', 'CB') for a in bandk)

        if method == 'pre_po':
            keys[keys.index('poisson_mixed')] = 'poisson_charge_neutrality'
            keys.append('thermal_equilibrium_CB_VB_IB')
            keys.append('poisson_thmeq')
        elif method == 'po_thmeq_CG':
            keys.append('thermal_equilibrium_CB_VB_IB')
            keys.append('poisson_thmeq')
        elif method == 'qfl_mixed':
            i = len(keys)
            keys[i:i] = ['opt_strbg_msorte_iv',
                         'opt_strbg_msorte_ci',
                         'opt_strbg_msorte_cv',
                         'strb_generation']
            keys.append('poisson_thmeq')
            add_VB_CB(
                'nondegenerate_band_qfl_mixed_naive',
            )
            keys.append('intermediate_band_qfl_mixed_naive_IB')
        else:
            raise ValueError()

        return self.sff.make_solution_factory_from_fragment_names(
            other_sf, keys)

    def run_solver(self, params, solution, name):
        sl = solution

        solver = NewtonSolverMaxDu.from_solution(sl)
        solver.parameters.update(
            relaxation_parameter=0.5,
            maximum_iterations=30,
            extra_iterations=7,
            relative_tolerance=1e-5,
            absolute_tolerance=1e-5,
            )

        logging.getLogger(name).info("############ solver name: {}".format(name))

        if name == 'pre_po':
            solver.parameters.update(
                # relaxation_parameter=0.01,
                maximum_iterations=2000, # WOW
                extra_iterations=5,
                relative_tolerance=1e-10,
                absolute_tolerance=1e-5,
                maximum_du=0.02,
                omega_cb=lambda self: 0.2 if self.iteration < 100 else 0.5,
            )
        elif name == 'po_thmeq':
            solver.parameters.update(
                relaxation_parameter=0.5,
                maximum_iterations=500,
                extra_iterations=3,
                omega_cb=lambda self: 0.4 if self.iteration < 30 else 0.9,
            )
        solver.solve()

    def do_solve(self, params):
        # Show fewer messages from FFC and UFL
        logging.getLogger('FFC').setLevel('ERROR')
        logging.getLogger('UFL').setLevel('ERROR')
        logging.getLogger('assign').setLevel('DEBUG')
        logging.getLogger('').setLevel('INFO')
        # Show messages on console
        ch = logging.StreamHandler()

        # Also log to file
        prefix = params['prefix']
        logfile = os.path.join(prefix, 'simudo.log')
        fh = logging.FileHandler(logfile, mode='w')
        logging.basicConfig(handlers=(ch, fh))

        logging.info('**** Simudo *****')
        logging.info('Started at {}'.format(datetime.now()))
        logging.info('parameters: {}'.format(params))
        self.start_time = time()

        sl = self.diode_genmesh(params, )
        sl.dict['/run_params'] = params

        def _make_sol(sl0, *args, **kwargs):
            return self.make_sf(sl0.factory, *args, **kwargs).new_solution()

        def _assign_subvar_keys(sl0, sl, keys):
            for k in keys:
                try:
                    assign_subvar(self.fsr, source=sl0[k], target=sl[k])
                except Exception:
                    logging.info("failed to assign subvar {}".format(k))
#                    raise

        def _assign_extracted(sl0, sl, k0, k, space):
            if isinstance(k0, str):
                v0 = sl0[k0]
            else:
                v0 = k0
            sl.dict[k] = sl['/pde/project'](v0, space)

        # local charge neutrality solver
        sl, sl0 = _make_sol(sl, 'pre_po'), sl
        copy_solution(sl0, sl)

        assign_subvar(
            self.fsr,
            source=sl['/common/approx_equilibrium_mu_guess']
            / sl['/unit_registry'].elementary_charge,
            target=sl['/poisson/V/delta'])

        self.run_solver(params, sl, 'pre_po')

        # poisson-only thermal equilibrium
        sl, sl0 = _make_sol(sl, 'po_thmeq_CG'), sl
        copy_solution(sl0, sl)

        sl.dict['/common/approx_equilibrium_mu'] = (
            sl0['/poisson/V/delta']*sl['/unit_registry'].elementary_charge)

        assign_subvar(
            self.fsr,
            source=sl0['/poisson/V/delta'],
            target=sl['/poisson/V/delta'])

        self.run_solver(params, sl, 'po_thmeq')

        with closing(LineCutCsvPlot(prefix+'po_thmeq.csv', self.fsr)) as xp:
            solution_plot_1d(xp, sl, 0)


        to_copy_keys = [
            '/poisson/V/delta',
            '/poisson/E/delta',
            '/CB/qfl_mixed/delta_w', '/CB/qfl_mixed/base_w',
            '/VB/qfl_mixed/delta_w', '/VB/qfl_mixed/base_w',
            '/IB/qfl_mixed/delta_w', '/IB/qfl_mixed/base_w',
            '/o/ci/I',
            '/o/iv/I',
            '/o/cv/I',
        ]

        # coupled at thermal equilibrium
        sl, sl0 = _make_sol(sl, 'qfl_mixed'), sl
        copy_solution(sl0, sl)

        _assign_subvar_keys(sl0, sl, ['/poisson/V/delta'])
        for b in ['/CB', '/VB', '/IB']:
            _assign_extracted(sl0, sl,
                              sl[b+'/qfl_mixed/u2_to_w'](sl0[b+'/u']), b+'/qfl_mixed/base_w',
                              sl['/pde/space/DG0'])

        _assign_subvar_keys(sl0, sl, to_copy_keys)

        logging.info("*** optical ramp-up")

        for b in ['/CB', '/VB', '/IB']:
            sl[b + '/qfl_mixed/do_rebase_w']()

        writer = OutputWriter(prefix + 'optramp.csv')
        opt = OpticalIntensityAdaptiveStepper(initial_solution=sl,
                                              intensity_values=[1.],
                                              selfconsistent_optics=True,
                                              output_writer=writer)
        opt.do_loop()

        # We'll find solutions at these voltages.
        Vexts = np.linspace(0, 1.6, 16*2 + 1)
        logging.info("*** voltages: {}".format(Vexts))

        # stop solving i-v curve points once we pass Voc
        def break_if_positive_current(solution):
            from simudo.io.output_writer import get_contact_currents
            u = solution['/unit_registry']
            jn, jp = get_contact_currents(solution, 'n_contact')
            if jn + jp > 0: return True
            else: return False

        stepper = VoltageStepper(initial_solution = sl,
                                 contacts=['/common/p_contact_bias'],
                                 bias_values=Vexts,
                                 selfconsistent_optics=True,
                                 output_writer='iv_curve',
                                 break_criteria_cb = break_if_positive_current)
        stepper.do_loop()

        end_time = time()
        runtime = timedelta(seconds=(end_time - self.start_time))
        logging.info('Finished at: {}'.format(datetime.now()))
        logging.info('Runtime: {}'.format(runtime))


ex = DiodeExample()


def glob_params():
    return ex.glob_params('out/diode *.out.d/params.yaml')


def mp_trampoline(a):
    ex.multiprocessing_trampoline(*a[0], **a[1])


class Main(object):
    def __init__(self):
        self.init_argparser()

    def init_argparser(self):
        import argparse
        self.argparser = parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(title='subcommands', dest="subparser")
        parser.add_argument('-j', '--jobs', metavar='N', type=int, default=10,
                            help='allow N jobs at once')

        s = subparsers.add_parser('generate')
        s.set_defaults(callback=self.cmd_generate)

        s = subparsers.add_parser('solve')
        s.set_defaults(callback=self.cmd_solve)

    def main(self):
        parser = self.argparser
        args = parser.parse_args()
        callback = getattr(args, 'callback', None)
        if not callback:
            parser.error("missing command")

        args.callback(args)

    def cmd_generate(self, args):
        for d in ({'tau': 1e-6}, {'tau': 1e-9}):
            d.update({'m': 'mixed'})
            keydict = d
            key = 'diode tau={tau}'.format(**keydict)
            outdir = os.path.abspath('out/{}.out.d'.format(key))+'/'
            shutil.rmtree(outdir, ignore_errors=True)
            os.makedirs(outdir, exist_ok=True)

            params = dict(
                key=key,
                keydict=keydict,
                prefix=outdir,
                should_plot_solver=False,
                method=keydict['m'])

            with open(params['prefix']+'params.yaml', "wt") as h:
                h5yaml.dump(params, h)

    def mp_pool(self, args):
        return multiprocessing.Pool(args.jobs)

    def cmd_solve(self, args):
        with self.mp_pool(args) as p:
            p.map(mp_trampoline,
                  [(('do_solve', pf.params), {}) for pf in glob_params()])


def main():
    Main().main()


if __name__ == '__main__':
    main()
