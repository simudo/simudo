from __future__ import division, absolute_import, print_function
from builtins import bytes, dict, int, range, str, super

from ..util import jitlock

from . import sentaurus_import
from ..mesh.call_generator import call_generator_module
from ..util.solution import (SolutionFactoryFactory, SolutionLoader,
                             save_solution, SolverBoilerplate)
from ..util.assign import (
    copy_solution, assign_subvar, solution_add_key_to_dict)
from ..util.newton_solver import (
    NewtonSolver, NewtonSolution, PlottingNewtonSolverMixin)
from ..util.expr import interpolate as interpolate_expr, tuple_to_point
from ..io.xdmf import XdmfPlot
from ..io import h5yaml
from ..util import plot as uplot
from ..util.plot import figclosing
from ..util.pdftoppm import pdftoppm
from matplotlib import pyplot as plt

import multiprocessing
import copy
import pandas as pd
import yaml
import lzma
import shutil
import os
import numpy as np
import dolfin
import glob
import logging
from collections import ChainMap
from contextlib import closing, contextmanager
from functools import partial
from cached_property import cached_property

parameters = dolfin.parameters
parameters["refinement_algorithm"] = "plaza_with_parent_facets"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = '-O3 -funroll-loops'

class MySolver(PlottingNewtonSolverMixin, NewtonSolver):
    def get_omega(self):
        return self.parameters['relaxation_parameter']
    #     du_norm = self.solution.du_norm
    #     force_omega = self.force_omega
    #     if force_omega is not None:
    #         return force_omega
    #     if du_norm < 1e-3:
    #         return 1.0
    #     else: # look up devsim log damping
    #         if self.is_poisson:
    #             return 0.8
    #         else:
    #             return 0.9 # min(0.007/du_norm, 0.5)

    def do_before_first_iteration_hook(self):
        self.do_post_iteration_hook()


class AdvectionDiffusionExample(SolverBoilerplate):
    def get_solution_from_sentaurus_df(self, params):
        with lzma.open(params['sentaurus_import_filename'], mode='rb') as handle:
            df, df_units = sentaurus_import.read_df(
                handle=handle,
                unit_registry=self.ur,
                **params['sentaurus_import_kwargs'])
        with lzma.open(params['sentaurus_import_equilibrium_filename'],
                       mode='rb') as handle:
            df_eq, df_eq_units = sentaurus_import.read_df(
                handle=handle,
                unit_registry=self.ur,
                **params['sentaurus_import_kwargs'])

        if not df['x'].equals(df_eq['x']): raise AssertionError()
        df_eq.index = df.index

        meshfile = params['prefix']+'mesh.yaml'

        sentaurus_import.remove_close_x_values(
            df, min_distance=1e-10)

        md = params['meshing_density']
        sentaurus_import.df_enforce_sampling_density(
            df, df['x'],
            df['x'].searchsorted([0.0, 0.3, 0.7, 1.0]),
            [md['majority'], md['depletion'], md['minority']])

        Xs = df['x'].values
        average_x_step = (Xs[-1] - Xs[0])/len(Xs)

        Ys = np.arange(7) * average_x_step

        call_generator_module(meshfile, '.mesh.gen.product2d_diode',
                              dict(Xs=Xs, Ys=Ys))

        sol = self.loader.load(meshfile, override_fragments=(
            'minimal', 'mesh_from_product2d'))

        df_eq = df_eq.loc[df.index,:]
        sol.dict['/adt/sentaurus_df'] = df
        sol.dict['/adt/sentaurus_df_eq'] = df_eq
        sol.dict['/adt/sentaurus_df_units'] = df_units

        return sol

    def make_sf(self, other_sf, method='naive'):
        if method == 'naive':
            method_keys = ['adt_density_naive']
        elif method == 'SGS':
            method_keys = ['adt_density_naive', 'adt_density_naive_SGS']
        elif method == 'mixed_naive':
            method_keys = ['adt_density_mixed_naive']
        elif method == 'mixed_naive_thmeq':
            method_keys = ['adt_density_mixed_naive',
                           'adt_density_mixed_naive_thm_equilibrium']
        elif method == 'du_mixed_naive':
            method_keys = ['adt_density_mixed_naive',
                           'adt_delta_density_mixed_naive']
        elif method == 'qfl_mixed_naive':
            method_keys = ['adt_qfl_mixed_naive']
        else:
            raise KeyError()
        keys = [
            'minimal', 'mesh_from_product2d', 'pde',
            'solution_mixed_function_space',
            'geometry_param',
            'adt_problem_units',
            'adt_problem',
            'adt_exact_from_sentaurus'] + method_keys + [
            'adt_plot_debug']
        return self.sff.make_solution_factory_from_fragment_names(
            other_sf, keys)

    def run_solver(self, params, solution):
        prefix = params['prefix']

        sl = solution
        form  = sl['/common/form'].to_ufl().magnitude
        trial = sl['/solution_space/function/trial']
        bcs   = sl['/common/essential_bcs']

        def hook(solver):
            self.clear_iteration_variables(solution)

        solver = MySolver(form, trial, bcs)
        solver.user_post_iteration_hooks.append(hook)
        solver.parameters.update(
            relaxation_parameter=1.0,
            maximum_iterations=6,
            extra_iterations=1,
            relative_tolerance=1e-10,
            absolute_tolerance=1e-7)

        if params.get('should_plot_solver', False):
            with closing(XdmfPlot(prefix+'solver.xdmf', self.fsr)) as xp:
                solver.plotter = partial(
                    self.xdmf_solution_plot, xp, sl, solver=solver)
                solver.solve()
        else:
            solver.solve()

    def do_solve(self, params):
        logging.getLogger('assign').setLevel(logging.DEBUG)

        prefix = params['prefix']

        sl0 = self.get_solution_from_sentaurus_df(params)
        sl0.dict['/run_params'] = params

        if False:
            sl_eq = self.make_sf(
                sl0.factory, 'mixed_naive_thmeq').new_solution()
            copy_solution(sl0, sl_eq)

            self.run_solver(params, sl_eq)
        else:
            sl_eq = None

        sl = self.make_sf(sl0.factory, params['method']).new_solution()
        copy_solution(sl0, sl)

        adt = sl['/adt/']

        if sl_eq is not None:
            assign_subvar(self.fsr,
                          target=sl_eq['/adt/u'],
                          source=sl['/adt/u_eq_computed'])
            solution_add_key_to_dict(sl, '/adt/u_eq_computed')

        if params['method'] == 'qfl_mixed_naive':
            assign_subvar(
                self.fsr,
                target=sl['/adt/qfl_mixed/base_w'],
                source=sl['/adt/qfl_mixed/u_to_w'](sl['/adt/exact/u']))
            sl['/adt/qfl_mixed/do_rebase_w'](
                sl['/adt/sentaurus_df']['w'].values[0])

        if params['method'] == 'qfl_mixed_naive' and False:
            self.run_solver(params, sl)
            sl['/adt/qfl_mixed/do_rebase_w']()
        self.run_solver(params, sl)

        with closing(XdmfPlot(prefix+'final.xdmf', self.fsr)) as xp:
            self.xdmf_solution_plot(xp, sl, 0)

        # mark for save
        solution_add_key_to_dict(sl, '/solution_space/function/trial')

        save_solution(prefix+"solution.yaml", sl)

ex = AdvectionDiffusionExample()

class ParamsFile(object):
    def __init__(self, filename):
        self.filename = filename

    @cached_property
    def solution(self):
        sl_file = os.path.join(self.params['prefix'], 'solution.yaml')
        if os.path.exists(sl_file):
            return ex.loader.load(sl_file)
        return None

    @cached_property
    def params(self):
        with open(self.filename, 'rt') as h:
            return h5yaml.load(h)

    def __str__(self):
        return self.params['key']

def glob_params():
    for filename in glob.iglob('out/adt *.out.d/params.yaml'):
        yield ParamsFile(filename)

def do_solve(params_filename):
    ex.do_solve(ParamsFile(params_filename).params)

class LineCut(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)

    def copy_with(self, **kwargs):
        new = copy.copy(self)
        for k, v in kwargs.items(): setattr(new, k, v)
        return new

    def __str__(self):
        return 'linecut_y={:e}'.format(self.y)

def extract_line_cut_along_x(
        sl, expr, y=None, N=2000):
    fsr = sl['/function_subspace_registry']
    p2d = sl['/mesh/product_2d_mesh_phantom']
    expr = getattr(expr, 'magnitude', expr)

    if y is None:
        y0, y1 = p2d.Ys[[0,-1]]
        y = y1*0.5 + y0*0.5
    limits = p2d.Xs[[0,-1]]
    # limits = (0.55, 0.555)
    Xs = np.linspace(limits[0], limits[1], N)
    Ys = np.array([expr(tuple_to_point((x, y))) for x in Xs])
    return LineCut(Xs=Xs, Ys=Ys,
                   y=y, N=N)

def plot_linecut(ax, pf, name, value, line_cutter=None,
                 label=None, plot_kwargs={}, xscale=1.0):
    if isinstance(value, LineCut):
        lc = value
    else:
        lc = line_cutter(pf.solution, value)
    if name is not None:
        label_ = ('{!s} {!s} {:s}'.format(pf, lc, name) if label is None
                  else (None if label is False else label))
        result = ax.plot(lc.Xs*xscale, lc.Ys, label=label_,
                **plot_kwargs)
    else:
        result = None
    return (lc, result)

class Limits():
    def __init__(self, minimum=None, maximum=None):
        self.minimum = +np.inf if minimum is None else minimum
        self.maximum = -np.inf if maximum is None else maximum

    def update(self, values):
        self.minimum = min(self.minimum, min(values))
        self.maximum = max(self.maximum, max(values))

def diffplot(exa, our, debug, absP, relP,
             label="$j_{x,ours} - j_{x,ref}$",
             rel_label="$j_{x,ours}/j_{x,ref} - 1$",
             varname='j', neg_to_nan=False,
             relerr_abs=False):
    P, P2 = absP, relP
    def _neg_to_nan(x):
        return np.where(x > 0, x, np.nan)
    f = (lambda x:x) if not neg_to_nan else _neg_to_nan
    dp, plotres = P(varname+'_diff' if debug else None,
                   our.copy_with(Ys=f(our.Ys - exa.Ys)),
                   label=None if debug else label)
    if plotres:
        color = plotres[0].get_color()
    else:
        color = None
    dn, _ = P(varname+'_negdiff' if debug else None,
      our.copy_with(Ys=f(exa.Ys - our.Ys)),
      label=False,
      plot_kwargs=dict(color=color, linestyle='dotted'))

    expr = (our.Ys/exa.Ys - 1)
    expr = abs(expr) if relerr_abs else expr
    _, plotres = P2(varname+'_reldiff',
       dp.copy_with(Ys=expr),
       label=None if debug else rel_label,
       plot_kwargs=dict(color=color))
    if not relerr_abs:
        color = plotres[0].get_color()
        P2(varname+'_reldiff_neg',
           dp.copy_with(Ys=(exa.Ys/our.Ys - 1)),
           label=False,
           plot_kwargs=dict(color=color, linestyle='dotted'))

def stacked_subplots_sharex(fig, axs=None):
    axs = fig.axes if axs is None else axs
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in axs[:-1]], visible=False)

@contextmanager
def do_plot_density(line_cutter, pfs, debug=True):
    lim = Limits()
    xlim = Limits()
    X = None
    with uplot.subplots(nrows=2, figsize=(6, 8),
                        sharex=True) as (fig, (ax, axr)):
        for pf in pfs:
            params = pf.params
            sl = pf.solution

            sentaurus_df_units = sl['/adt/sentaurus_df_units']
            x_scale = sentaurus_df_units['x'  ].m_as('nanometre')

            P = partial(plot_linecut, ax, pf, line_cutter=line_cutter,
                        xscale=x_scale)
            P2 = partial(plot_linecut, axr, pf, line_cutter=line_cutter,
                        xscale=x_scale)

            our = P(None if debug else 'u_c', sl['/adt/u_proj'],
                    label=None if debug else '$u_{ours}$')[0]
            exa = P('u_e', sl['/adt/exact/u'],
                    label=None if debug else "$u_{ref}$",
                    plot_kwargs=dict(linestyle='dotted'))[0]
            lim.update(exa.Ys)
            xlim.update(exa.Xs[[0, -1]]*x_scale)
            diffplot(exa, our, debug=debug, absP=P, relP=P2,
                     label="$n_{ours} - n_{ref}$",
                     rel_label="$|n_{ours}/n_{ref} - 1|$",
                     varname='n', relerr_abs=not debug)

        ax.set_yscale('log')
        ax.set_ylim([lim.minimum/1e8, lim.maximum*2])
        axr.set_yscale('log')
        axr.set_ylim([None, 1e2])
        for a in (ax, axr):
            a.set_xlim([xlim.minimum, xlim.maximum])
            a.grid(True, which="both", ls="-", color='0.65')
        if debug:
            ax.set_title('carrier concentration')
            uplot.clever_legend(ax)
        else:
            p = pfs[0].params['keydict']
            # ax.set_title('carrier concentration, doping={c} bias={V}'
            #              .format(c=p['c'], V=p['V']))
            for a in (ax, axr):
                a.legend()
            axr.set_xlabel("position ($\mathrm{nm}$)")
            ax.set_ylabel("carrier density ($\mathrm{1/cm^3}$)")
            axr.set_ylabel("relative error")
        stacked_subplots_sharex(fig)
        fig.tight_layout()
        yield (fig, ax)

@contextmanager
def do_plot_current(line_cutter, pfs, current_key, debug=True):
    lim = Limits()
    xlim = Limits()
    X = None
    negated = False
    with uplot.subplots(nrows=2, figsize=(6, 8),
                        sharex=True) as (fig, (ax, axr)):
        for pf in pfs:
            params = pf.params
            sl = pf.solution

            sentaurus_df_units = sl['/adt/sentaurus_df_units']
            x_scale = sentaurus_df_units['x'  ].m_as('nanometre')
            j_scale = sentaurus_df_units['j_x'].m_as('mA/cm^2')

            P = partial(plot_linecut, ax, pf, line_cutter=line_cutter,
                        xscale=x_scale)
            P2 = partial(plot_linecut, axr, pf, line_cutter=line_cutter,
                         xscale=x_scale)

            exa = P(None, sl['/adt/exact/j_proj'])[0]
            our = P(None, sl[current_key])[0]
            if 0:
                exa2= P(None, sl['/adt/j_e_DG0proj'])[0]
                our2= P(None, sl['/adt/j_c_DG0proj'])[0]
            else:
                exa2 = None
                our2 = None

            exa.Ys = exa.Ys[:,0] * j_scale
            our.Ys = our.Ys[:,0] * j_scale
            if our2:
                exa2.Ys=exa2.Ys[:,0] * j_scale
                our2.Ys=our2.Ys[:,0] * j_scale

            sgn = ''
            if exa.Ys.mean() < 0:
                exa.Ys *= -1
                our.Ys *= -1
                if our2:
                    exa2.Ys *= -1
                    our2.Ys *= -1
                sgn = '-'

            if not debug:
                P('j_c', our, label=None if debug else sgn+'$j_{x,ours}$')
            P('j_e', exa, label=None if debug else sgn+'$j_{x,ref}$',
              plot_kwargs=dict(linestyle='dotted'))

            if False:
                P('j_c2', our2, label=None if debug else sgn+'$j_{x,ours,2}$')

            lim.update(exa.Ys)
            xlim.update(exa.Xs[[0, -1]]*x_scale)

            diffplot(exa, our, debug=debug, absP=P, relP=P2,
                     label="$j_{x,ours} - j_{x,ref}$",
                     rel_label="$|j_{x,ours}/j_{x,ref} - 1|$",
                     varname='j', relerr_abs=not debug)
            if our2:
                diffplot(exa2, our2, label='$j_{x,ours,2} - j_{x,ref,2}$')

        ymin = lim.minimum
        ymax = lim.maximum
        if debug:
            ax.set_ylim([ymin/1e6, ymax*20])
            ax.set_yscale('log')
        else:
            h = (ymax-ymin)*0.2
            ax.set_ylim([ymin-h, ymax+h])
            ax.set_yscale('linear')
        axr.set_yscale('log')
        axr.set_ylim([1e-6, 10])
        for a in (ax, axr):
            a.set_xlim([xlim.minimum, xlim.maximum])
            a.grid(True, which="both", ls="-", color='0.65')
        if debug:
            ax.set_title('current density')
            uplot.clever_legend(ax)
            uplot.clever_legend(axr)
        else:
            p = pfs[0].params['keydict']
            # ax.set_title('current density, doping={c} bias={V}'
            #              .format(c=p['c'], V=p['V']))
            for a in (ax, axr):
                a.legend()
            axr.set_xlabel("position ($\mathrm{nm}$)")
            ax.set_ylabel("current density ($\mathrm{mA/cm^2}$)")
            axr.set_ylabel("relative error")
        stacked_subplots_sharex(fig)
        fig.tight_layout()
        yield (fig, ax)

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

        s = subparsers.add_parser('analyze')
        s.set_defaults(callback=self.cmd_analyze)

        s = subparsers.add_parser('plots')
        s.set_defaults(callback=self.cmd_plots)

    def main(self):
        parser = self.argparser
        args = parser.parse_args()
        callback = getattr(args, 'callback', None)
        if not callback:
            parser.error("missing command")
        args.callback(args)

    def cmd_generate(self, args):
        for sentaurus_filename, sentaurus_dict in (
                sentaurus_import.list_sentaurus_diode_1d_data()):
            print('***', sentaurus_filename)
            keydict = dict(c=sentaurus_dict['c'],
                           V=sentaurus_dict['V'],
                           m='qfl_mixed_naive')

            if sentaurus_dict['c'] == '1e19': # missing V=0
                continue

            key = 'adt c={c} V={V} m={m}'.format(**keydict)
            outdir = os.path.abspath('out/{}.out.d'.format(key))+'/'
            shutil.rmtree(outdir, ignore_errors=True)
            os.makedirs(outdir, exist_ok=True)
            # to check which files have the same coordinates
            # for f in *.csv.xz; do a="$(unxz < $f | cut -d, -f1 | md5sum)"; echo "$a $f"; done
            params = dict(
                key=key,
                keydict=keydict,
                prefix=outdir,
                should_plot_solver=False,
                method=keydict['m'],
                sentaurus_import_filename=(
                    'data/sentaurus/diode_1d c={c} V={V}.csv.xz'.format(**keydict)),
                sentaurus_import_equilibrium_filename=(
                    'data/sentaurus/diode_1d c={c} V={V}.csv.xz'.format(
                        **ChainMap(dict(V='0'), keydict))),
                sentaurus_import_kwargs={},
                meshing_density=dict(majority=3000,
                                     depletion=3050,
                                     minority=3000))

            with open(params['prefix']+'params.yaml', "wt") as h:
                h5yaml.dump(params, h)

    def mp_pool(self, args):
        return multiprocessing.Pool(args.jobs)

    def cmd_solve(self, args):
        with self.mp_pool(args) as p:
            p.map(do_solve, [pf.filename for pf in glob_params()])

    def cmd_analyze(self, args):
        from tabulate import tabulate
        JV = []
        target_j_unit = 'A/cm^2'
        for params_file in glob_params():
            params = params_file.params
            sl = params_file.solution
            j_unit = sl['/adt/sentaurus_df_units']['j_x'].m_as(target_j_unit)
            kd = params['keydict']
            row = [float(kd['c']), float(kd['V'])]
            for pn, index in (('p', 0), ('n', -1)):
                sen_val = sl['/adt/sentaurus_df']['j_x'].values[index] * j_unit
                mcr_val = sl['/adt/{}_contact_avg_j3'.format(pn)] * j_unit
                row.extend((sen_val, mcr_val))
            JV.append(row)

        df = pd.DataFrame(JV, columns=['doping', 'Vext',
                                       'j_pc_sentaurus', 'j_pc_simudo',
                                       'j_nc_sentaurus', 'j_nc_simudo'])
        df.sort_values(['doping', 'Vext'], inplace=True)
        print(tabulate(df, headers='keys', tablefmt='psql'))

    @staticmethod
    def do_plots_single_pf(pf):
        debug = False
        if True:
            with do_plot_density(extract_line_cut_along_x,
                                 [pf],
                                 debug=debug) as (fig, ax):
                fn = pf.params['prefix']+"n.pdf"
                fig.savefig(fn)
                pdftoppm(fn)
        if True:
            with do_plot_current(extract_line_cut_along_x,
                                 [pf], '/adt/j_c_CG1proj',
                                 debug=debug) as (fig, ax):
                fn = pf.params['prefix']+"j.pdf"
                fig.savefig(fn)
                pdftoppm(fn)

    def cmd_plots(self, args):
        with self.mp_pool(args) as p:
            p.map(self.do_plots_single_pf, list(glob_params()))

def main(): Main().main()

if __name__=='__main__': main()
