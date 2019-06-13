
import simudo.plot # matplotlib config

import functools
import glob
import itertools
import lzma
import os
from os import path as osp

import numpy as np
import pandas as pd
from argh import ArghParser, arg
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.ndimage import filters
from tabulate import tabulate

from simudo.io import h5yaml
from simudo.misc import sentaurus_import
from simudo.plot import (
    df_add_drift_diffusion_terms, matplotlib_configure_CM_font, savefig)
from simudo.util import make_unit_registry

U = make_unit_registry()

def make_compatible_sentaurus(df):
    for b in ('VB', 'CB'):
        df.rename(columns={b+'_u': 'u_'+b,
                           b+'_j_x': 'j_'+b}, inplace=True)
    df.rename(columns={'x': 'coord_x'}, inplace=True)

def _lazy(df_simudo, df_sentaurus):
    df0 = df_simudo
    df1 = df_sentaurus
    Xs = df0.coord_x
    if max(Xs.values) <= 1:
        Xs = Xs * 1e3
        xunit = r'\mathrm{nm}'
    else:
        xunit = r'\mu\mathrm{m}'
    xlim = [min(Xs.values), max(Xs.values)]
    return (df0, df1, Xs, xunit, xlim)

def _text(ax, x, y, s, **kwargs):
    kwargs.setdefault('horizontalalignment', 'center')
    kwargs.setdefault('verticalalignment', 'center')
    kwargs.setdefault('transform', ax.transAxes)
    return ax.text(x, y, s, **kwargs)

def _subsp(fig, ax0, ax1, title=None, value_only=False, is_inset=False):
    if title:
        ax0.set_title(title)
        ax0.title.set_size(10)

    if not is_inset:
        vsz = 1.35 if value_only else 1.0
        fig.subplots_adjust(
            hspace=0.,
            left=(0.18 if 1 or title else 0.13), right=0.95,
            bottom=0.15*vsz, top=1 - (0.1 if title else 0.08)*vsz)
    else:
        fig.subplots_adjust(
            hspace=0.,
            left=0, right=1,
            bottom=0, top=1)

    if 0 and ax1:
        if 1 or title:
            labelx = -0.17
            ax0.yaxis.set_label_coords(labelx, 0.5)
            ax1.yaxis.set_label_coords(labelx, 0.3)
        else:
            labelx = -0.12
            ax0.yaxis.set_label_coords(labelx, 0.1)
            ax1.yaxis.set_label_coords(labelx, 0.1)


def plot_cmp_density(df_simudo, df_sentaurus, title):
    (df0, df1, Xs, xunit, xlim) = _lazy(df_simudo, df_sentaurus)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, sharex=True, gridspec_kw=dict(height_ratios=[1, 1]))

    for b, bname, col in (('CB', 'n', '#294171'), ('VB', 'p', '#5A1705')):
        uk = 'u_'+b
        for df, sname in ((df0, 'simudo'),
                          (df1, 'sentaurus')):
            ax0.plot(Xs, df[uk], label=r'${}_{{\mathrm{{{}}}}}$'
                     .format(bname, sname),
                     color=col, linestyle='-' if sname=='simudo' else 'dashed')
        reldiff = df0[uk] / df[uk] - 1
        #reldiff = filters.median_filter(reldiff, size=25)
        relerr = abs(reldiff)
        ax1.plot(Xs, relerr, label='rel. error in ${}_{{\mathrm{{{}}}}}$'
                 .format(bname, 'simudo'), color=col)

    ax1.set_xlabel(r'position (${}$)'.format(xunit))
    if 1 or title is not None:
        ax0.set_ylabel(r'$1/\mathrm{cm}^3$')
        # ax1.set_ylabel(r'rel. err.')
    else:
        _text(ax0, 0.04, 0.9, '(a)')
        _text(ax1, 0.04, 0.8, '(b)')
    ax0.set_yscale('log')
    ax1.set_yscale('log')
    ax0.set_xlim(xlim)
    ax1.set_ylim([None, 10])
    ax0.set_ylim([None, 1e19])
    ax0.legend()
    ax1.legend()
    ax0.grid(True)
    ax1.grid(True)
    _subsp(fig, ax0, ax1, title)
    return fig


def plot_cmp_current(
        df_simudo, df_sentaurus, title,
        relative_only=False, value_only=False,
        grid=True, is_inset=False,
        force_lims=None):
    (df0, df1, Xs, xunit, xlim) = _lazy(df_simudo, df_sentaurus)

    if relative_only:
        fig, ax1 = plt.subplots(figsize=(6.4, 2.4))
        ax0 = None
        axx = ax1
    elif value_only:
        fig, ax0 = plt.subplots(
            figsize=(6.4, 3.0) if not is_inset else (1.5, 0.5))
        ax1 = None
        axx = ax0
    else:
        fig, (ax0, ax1) = plt.subplots(
            2, 1, sharex=True, gridspec_kw=dict(
                height_ratios=[2, 1]),
            figsize=(6.4, 4.8))
        axx = ax1

    relmax = 0.0
    for b, bname, col in (('CB', 'n', '#294171'), ('VB', 'p', '#5A1705')):
        uk = 'j_'+b
        for df, sname in ((df0, 'simudo'),
                          (df1, 'sentaurus')):
            if ax0:
                ax0.plot(
                    Xs, df[uk], label=r'$j_{{{},\mathrm{{{}}}}}$'
                    .format(bname, sname),
                    color=col, linestyle='-' if sname=='simudo' else 'dashed')
        reldiff = df0[uk] / df[uk] - 1
        #reldiff = filters.median_filter(reldiff, size=25)
        relerr = abs(reldiff)
        if ax1:
            ax1.plot(Xs, relerr,
                     # label='$j_{{{},\mathrm{{{}}}}}$'.format(bname, 'simudo'),
                     label='$j_{{{}}}$'.format(bname),
                     color=col)
        relmax = max(relmax, np.max(relerr))

    axx.set_xlabel(r'position (${}$)'.format(xunit))
    if 1 or title is not None:
        if not is_inset:
            if ax0: ax0.set_ylabel(r'$\mathrm{mA}/\mathrm{cm}^2$')
            if ax1: ax1.set_ylabel(r'rel. err.')
    else:
        if ax0: _text(ax0, 0.04, 0.9, '(a)')
        if ax1: _text(ax1, 0.04, 0.8, '(b)')
    # ax0.set_yscale('log')
    if ax1:
        ax1.set_yscale('log')
        ax1.set_ylim([1e-4, 1.2 if relmax >= 1.0 else None])

    axx.set_xlim(xlim if not force_lims else force_lims[0])

    # ax0.ticklabel_format(axis='y', style='sci', scilimits=(-1, 3))
    mf = ScalarFormatter(useMathText=True)
    mf.set_powerlimits((-1, 3))
    if ax0:
        if force_lims: ax0.set_ylim(force_lims[1])
        ax0.yaxis.set_major_formatter(mf)
        # ax0.set_ylim([None, 1e19])
        if not is_inset: ax0.legend()
        ax0.grid(grid)
    if ax1:
        ax1.legend()
        ax1.grid(grid)
    if ax0:
        _subsp(fig, ax0, ax1, title, value_only=value_only,
               is_inset=is_inset)
    else:
        fig.subplots_adjust(
            hspace=0.,
            left=(0.18 if 1 or title else 0.13), right=0.95,
            bottom=0.25, top=(0.9 if title else 0.92))
    return fig

def plot_offset_partitioning(df_simudo, title):
    (df0, df1, Xs, xunit, xlim) = _lazy(df_simudo, None)

    fig, ax0 = plt.subplots()

    if 'w_CB_base' not in df0.columns:
        return fig

    D = df0
    a = {'linewidth': 3.0}
    ax0.plot(
        Xs, D.w_CB_base + D.w_CB_delta, label="$w_{n}$", **a)
    ax0.plot(
        Xs, D.w_VB_base + D.w_VB_delta, label="$w_{p}$", **a)
    ax0.plot(
        Xs, D.w_CB_base, label="$w_{n0}$")
    ax0.plot(
        Xs, D.w_VB_base, label="$w_{p0}$",
        linestyle="dashed")

    ax0.set_xlabel(r'position (${}$)'.format(xunit))
    # if title is not None:
    ax0.set_ylabel(r'energy ($eV$)')
    ax0.legend()
    ax0.grid(True)
    _subsp(fig, ax0, None, title)
    return fig

def _positivize(x):
    return (x, '') if x.sum() > 0 else (-x, '-')

def plot_drift_diffusion_currents(
        df0, title,
        grid=True,
        force_lims=None):
    (df0, _, Xs, xunit, xlim) = _lazy(df0, None)

    fig, ax0 = plt.subplots(
        figsize=(6.4, 3.0))

    min_j = np.inf
    for b, bname, col in (('CB', 'n', '#294171'), ('VB', 'p', '#5A1705'))[:1]:
        uk = 'j_'+b

        j, sgn = _positivize(df0[uk])
        ax0.plot(Xs, j, label=r'${}j_{{{}}}$'.format(sgn, bname),
                 color='#aaaaaa')
        min_j = min(min_j, j.min())

        for term, term_label, ls, col in (
                ('drift', 'drift', 'solid', '#748CBC'),
                ('diffusion', 'diff', 'dashed', '#834736')):
            j_term, sgn = _positivize(df0['_'.join(('j', term, b))])
            ax0.plot(
                Xs, j_term, linestyle=ls, color=col,
                label=r'${}j_{{{},\mathrm{{{}}}}}$'
                .format(sgn, bname, term_label))

    ax0.set_xlabel(r'position (${}$)'.format(xunit))
    ax0.set_ylabel(r'$\mathrm{mA}/\mathrm{cm}^2$')
    ax0.set_yscale('log')

    ax0.set_xlim(xlim if not force_lims else force_lims[0])
    ax0.legend()
    ax0.grid(grid)
    ax0.set_ylim([min_j*1e-1, None])

    fig.subplots_adjust(
        hspace=0.,
        left=(0.18 if 1 or title else 0.13), right=0.95,
        bottom=0.25, top=(0.9 if title else 0.92))
    return fig


def do_single_analysis_summary(
        df0, df1, title, meta_input=None,
        tree_voltage_step=None):

    fn = tree_voltage_step.get('ivcurve_filename', None)
    if fn:
        iv = pd.read_csv(fn)
        ivrow = iv[
            iv.index == tree_voltage_step[
                'ivcurve_row_index']].to_dict('records')[0]
    else:
        ivrow = None
        # j_VB:p_contact,j_VB:n_contact,j_CB:p_contact,j_CB:n_contact,j_tot:p_contact,j_tot:n_contact

    d = {}
    assert len(df0.coord_x.values) == len(df1.coord_x.values)
    n2 = max(round(len(df0.coord_x.values) / 40), 5)
    for where_index, where_name, should_use_meta in [
            (0, 'pI', True), (-1, 'nI', True),
            (0, 'p', False), (-1, 'n', False),
            (n2, 'p2', False), (-n2-1, 'n2', False),
            ('median', 'pM', False), ('median', 'nM', False),
            ]:
        j_tot = dict(sim=0, sen=0)
        for band_key, band_cname in zip(('VB', 'CB', 'tot'), 'pnt'):
            colk = 'j_'+band_key
            k = 'c{}_{}'.format(where_name, band_cname)
            if band_key != 'tot':
                if should_use_meta:
                    j_sim = (
                        meta_input['integrals']['avg_j_{}:{}_contact'.format(
                            band_key, where_name[0])])
                elif where_index == 'median':
                    j_sim = np.nan
                else:
                    j_sim = df0[colk].iloc[where_index]
                if ivrow:
                    j_sen = ivrow['j_{}:{}_contact'.format(
                        band_key, where_name[0])]
                else:
                    j_sen = df1[colk].iloc[where_index]
                j_tot['sim'] += j_sim
                j_tot['sen'] += j_sen
            else: # total current
                if where_index == 'median':
                    j_tot['sim'] = np.median(df0['j_CB'] + df0['j_VB'])
                j_sim, j_sen = j_tot['sim'], j_tot['sen']
            rel = j_sim/j_sen - 1.0
            d['j_sim__'+k] = float(j_sim)
            d['j_sen__'+k] = float(j_sen)
            d['jrel__'+k] = float(rel)

    return d

def do_main(
        input='out/stupidtest/a parameter=0.4 csvplot.csv.0',
        output_prefix='out/mrs2018_pnbench_',
        meta_input=None,
        sentaurus_file="data/sentaurus/diode_1d c=1e18 V=0.4.csv.xz",
        title=None, tree_voltage_step=None,
        production=False, reduce_vsize=False):
    df = pd.read_csv(input)

    if production:
        title = None

    if sentaurus_file is not None:
        with lzma.open(sentaurus_file, mode='rb') as handle:
            Sdf, Sdf_units = sentaurus_import.read_df_unitful(
                handle=handle,
                unit_registry=U)
        #df.sort_values('coord_x', inplace=True)
        #df.reset_index(drop=True, inplace=True)
        Sdf, Sdf_units = sentaurus_import.interpolate_df(
            Sdf, Sdf_units, df['coord_x'].values*U.micrometer)
        make_compatible_sentaurus(Sdf)
    else:
        Sdf, Sdf_units = None, None
        Sdf = df # to avoid a billion conditionals in the plot

    fig = plot_cmp_density(df, Sdf, title)
    savefig(fig, output_prefix+"u")
    plt.close(fig)

    fig = plot_cmp_current(df, Sdf, title)
    savefig(fig, output_prefix+"j")
    plt.close(fig)

    fig = plot_cmp_current(df, Sdf, title, value_only=True, grid=False)
    savefig(fig, output_prefix+"jv")
    plt.close(fig)

    if input.endswith(" c=1e18 Vd=-1 meshp=2000 bchack=0 ffill=1 fta=0 ftr=1e-10 fwz=0/a parameter=-0.2 csvplot.csv.0") and 0:
        fig = plot_cmp_current(
            df, Sdf, title, value_only=True, is_inset=True, grid=False,
            force_lims=[(40, 70), (-1.655e-6, -1.62e-6)])
        savefig(fig, output_prefix+"jv_inset")
        plt.close(fig)

    fig = plot_cmp_current(df, Sdf, title, relative_only=True)
    savefig(fig, output_prefix+"jr")
    plt.close(fig)

    fig = plot_offset_partitioning(df, title)
    savefig(fig, output_prefix+"op")
    plt.close(fig)

    df_add_drift_diffusion_terms(df)
    fig = plot_drift_diffusion_currents(df, title)
    savefig(fig, output_prefix+"jdd")
    plt.close(fig)

    if meta_input:
        meta_input = h5yaml.load(meta_input)

    d = do_single_analysis_summary(
        df, Sdf, title, meta_input=meta_input,
        tree_voltage_step=tree_voltage_step)
    d['voltage_step'] = tree_voltage_step
    h5yaml.dump(d, output_prefix+'sentaurus_plot_analysis.yaml')

    # return locals()

@arg('input', help='Input `submit.yaml` file.')
@arg('--production', help='Make production quality plots?')
def voltage_step_plots(input, production=False):
    tree = h5yaml.load(input)

    P = tree['parameters']

    for d in P['voltage_steps']:
        prefix = osp.dirname(input).rstrip('/')+'/'
        fn0 = prefix+'a parameter={} csvplot'.format(d['Vext_str'])
        fn = fn0 + '.csv.0'
        ttl = "V={} {} ".format(d['Vext_str'], P['title'])
        if not osp.exists(fn): continue
        do_main(
            input=fn,
            output_prefix=prefix+"xplot "+ttl,
            meta_input=fn0 + '.plot_meta.yaml',
            sentaurus_file=d['filename'],
            title=ttl,
            production=production,
            tree_voltage_step=d)

def sci_to_latex(string, suppress_mantissa_eq_1=True):
    mantissa, expt = string.lower().split('e')
    if mantissa[0] in '+-':
        sign = mantissa[0]
        mantissa = mantissa[1:]
    else:
        sign = ''

    if suppress_mantissa_eq_1 and mantissa == '1':
        mantissa = None

    if mantissa is not None:
        mantissa = mantissa + r'\times'
    else:
        mantissa = ''

    return (r"{}{}10^{{{}}}".format(sign, mantissa, expt))

@arg('basedirs', help="Directories under which "
     "to look for `sentaurus_plot_analysis.yaml` files.", nargs='+')
@arg('--output-prefix', help="Prefix output files with this.")
@arg('--pdf', help='Output pdf and png plots?')
@arg('--usetex', help='Use tex font rendering?')
def summary_analysis(basedirs, output_prefix=None, pdf=False,
                     usetex=False):

    matplotlib_configure_CM_font(size=16, usetex=usetex)
    savefig_ = functools.partial(savefig, pdf=pdf)

    if output_prefix is None:
        output_prefix = osp.join(basedirs[0], 'sen_summary ')
    rows = []
    for basedir in basedirs:
        for subdir_ in os.listdir(basedir):
            subdir = osp.join(basedir, subdir_)
            submit_file = osp.join(subdir, 'submit.yaml')
            if not osp.exists(submit_file): continue
            submit = h5yaml.load(submit_file)

            submitp = submit['parameters']
            for analysis_file in glob.iglob(
                    glob.escape(subdir) + '/xplot * sentaurus_plot_analysis.yaml'):
                ana = h5yaml.load(open(analysis_file))
                row = ana.copy()
                del row['voltage_step']
                row['doping_str'] = doping = submitp['doping']
                row['doping'] = float(doping)
                row['Vext_str'] = Vext = ana['voltage_step']['Vext_str']
                row['Vext'] = float(Vext)
                row['meshp'] = float(submitp['mesh_points'])
                row['meth'] = submitp['transport_method']
                row['fta'] = float(submitp['fill_threshold_abs'])
                row['ftr'] = float(submitp['fill_threshold_rel'])
                row['fwz'] = bool(submitp['fill_with_zero_except_bc'])
                for k in ('quaddeg_super', 'quaddeg_g', 'quaddeg_rho',
                          'min_srv'):
                    row[k] = submitp[k]
                rows.append(row)
    df = pd.DataFrame(rows)

    TRIAL_FLOODFILL = 0
    TRIAL_FWZ = 1
    TRIAL_CELLAVG = 2

    df['numtrial'] = TRIAL_FLOODFILL

    def where(c, x, y):
        return c*x + (1-c)*y

    df['numtrial'] = where(
        df['numtrial'] != TRIAL_FLOODFILL, df['numtrial'],
        where(df['fwz'] == True,
              TRIAL_FWZ, df['numtrial']))

    df['numtrial'] = where(
        df['numtrial'] != TRIAL_FLOODFILL, df['numtrial'],
        where(((df['fta'] == 0.0) & (df['ftr'] == 0.0)),
              TRIAL_CELLAVG, df['numtrial']))

    df.to_csv(output_prefix + 'table.csv')
    with open(output_prefix + 'table.txt', 'wt') as handle:
        print(tabulate(df, headers='keys', tablefmt='psql'), file=handle)

    def jrel_comparison_plot(df0, df1=None,
                             baseline_label='baseline',
                             trial_label=None,
                             filename_prefix=None,
                             total_current=False,
                             contact_suffix0='',
                             contact_suffix1='',
                             is_paper_plot=False):
        for which_carrier, which_contact in itertools.product(
                't' if total_current else 'pn', 'pn'):
            def _make_relvar(contact_suffix):
                return 'jrel__c{}{}_{}'.format(
                    which_contact, contact_suffix, which_carrier)

            relvar0 = _make_relvar(contact_suffix0)
            relvar1 = _make_relvar(contact_suffix1)

            figkw = dict(figsize=
                         (6.4, 4.8))

            fig, ax = plt.subplots(**figkw)

            # ax.set_prop_cycle(color=['#294171', '#5A1705', '#8E887C'])

            for k in sorted(set(df0['doping'])):
                xdf0 = df0[df0['doping'] == k]

                doping_str = xdf0.iloc[0]['doping_str']
                if is_paper_plot:
                    label = (r"$N_A=N_D={}\;\mathrm{{cm^{{-3}}}}$"
                             .format(sci_to_latex(doping_str)))
                else:
                    label = r"c={}".format(doping_str)
                p = ax.plot(xdf0['Vext'], abs(xdf0[relvar0]),
                            label=label, marker='.')

                if df1 is None: continue

                xdf1 = df1[df1['doping'] == k]
                color = p[0].get_color()
                ax.plot(xdf1['Vext'], abs(xdf1[relvar1]),
                        label='_nolegend_',
                        color=color, linestyle='dashed', marker='.')

            ax.set_yscale('log')
            if df1 is not None:
                ax.plot([], [], color='black', linestyle='solid',
                        label=baseline_label)
                ax.plot([], [], color='black', linestyle='dashed',
                        label=trial_label)

            ax.set_xlabel(r"$V_{\mathrm{ext}}$ ($\mathrm{V}$)")

            if is_paper_plot:
                ax.set_ylabel(r"Relative error in $J$")
                legend_kwargs = dict(prop=dict(size=14))
            else:
                ax.set_ylabel(r"$j_{{{},\mathrm{{rel}}}}$ at {} contact"
                              .format(which_carrier, which_contact))
                legend_kwargs = {}

            # ax.set_xlim([-2, 2])
            # ax.set_ylim([1e-8, 1e-2])

            ax.legend(**legend_kwargs)
            if False:
                fig.tight_layout()
            else:
                fig.subplots_adjust(
                    hspace=0.,
                    left=0.15, right=0.95,
                    bottom=0.13, top=0.92)
            savefig_(fig, filename_prefix + ' ' + _make_relvar(''))
            plt.close(fig)

    df, df_orig = df[abs(df['Vext']) > 1e-3], df
    df.sort_values('Vext', inplace=True)

    jrel_comparison_plot(df[df['numtrial'] == TRIAL_CELLAVG],
                         filename_prefix=output_prefix+"baseline_cellavg")

    if True:
        df = df[df['min_srv'] == '0']

    df2 = df[df['numtrial'] == TRIAL_CELLAVG]
    if False:
        jrel_comparison_plot(
            df2[df2['meshp'] > 0],
            df2[df2['meshp'] < 0],
            baseline_label='unif',
            trial_label='non-unif',
            filename_prefix=output_prefix+"mesh_compare",
            total_current=True,
        )

    if False:
        jrel_comparison_plot(
            df2[df2['quaddeg_g'] == 30],
            df2[df2['quaddeg_g'] == 8],
            baseline_label='$d_g=30$',
            trial_label='$d_g=6$',
            filename_prefix=output_prefix+"quaddeg_compare",
            total_current=True,
        )

    # mixed-u-j vs mixedqfl method comparison
    if True:
        jrel_comparison_plot(
            df2[df['meth'] == 'mixed_qfl_j'],
            df2[df['meth'] == 'mixed_u_j'],
            baseline_label='Simudo',
            trial_label='TTCB...CD',
            contact_suffix0='I',
            contact_suffix1='M',
            filename_prefix=output_prefix+"method_compare",
            total_current=True,
            is_paper_plot=True,
        )
        jrel_comparison_plot(
            df2[df['meth'] == 'mixed_u_j'],
            df2[df['meth'] == 'mixed_u_j'],
            baseline_label='TTCB...CD integrated',
            trial_label='TTCB...CD median',
            contact_suffix0='M',
            contact_suffix1='I',
            filename_prefix=output_prefix+"method_compare_extraction",
            total_current=True,
            is_paper_plot=True,
        )

    df4 = df2[df2['quaddeg_g'] == df2['quaddeg_g'].values.max()]
    if True:
        jrel_comparison_plot(
            df4, df4,
            baseline_label='line cut',
            trial_label='integrated',
            filename_prefix=output_prefix+"linecut_vs_integrated",
            total_current=True,
            contact_suffix0='',
            contact_suffix1='I',
        )

    df5 = df4[(df4['meshp'] == -1) &
              ((df4['doping_str'] == '1e15') |
               (df4['doping_str'] == '1e18'))]
    df6 = df4[(df4['meshp'] == -1)]
    if True:
        jrel_comparison_plot(
            df5, filename_prefix=output_prefix+"paper_plot",
            total_current=True,
            contact_suffix0='I',
            is_paper_plot=True,
        )
        jrel_comparison_plot(
            df6, filename_prefix=output_prefix+"paper_plot_all_curves",
            total_current=True,
            contact_suffix0='I',
        )


parser = ArghParser()
parser.add_commands([voltage_step_plots, summary_analysis])

if __name__ == '__main__':
    parser.dispatch()
