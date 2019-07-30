
# simudo=a04dde44ef34d2843e624b99aee80dd02100ff51

from simudo.plot import (
    arg_matplotlib, df_add_drift_diffusion_terms)
from simudo.util import (
    outdir_path_helper, mkdirp, TODO, with_default_kwargs,
    read_xcsv, to_xcsv)
from argh import expects_obj
from simudo.io import h5yaml
from tabulate import tabulate
import pandas as pd
import lzma
import os
import itertools
import functools
from functools import partial
import re
from os import path as osp
import glob
from argh import ArghParser, arg
import numpy as np
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator, MultipleLocator
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from collections import OrderedDict
import shutil
import tqdm

'''
how2use

# extract JV data
python3 marti2002_plot.py extract-jv out/ out/marti02\ */

# copy data and JV points at maximum power point on each JV curve
python3 marti2002_plot.py copy-mpp --out out.mpp/ --jv out/JV.csv

# plot spatial components of current
python3 marti2002_plot.py plot-detail-j --usetex --out out.plot/ --jv-mpp out.mpp/JV_mpp.csv --spatial-input-dir out.mpp/

# plot bands (spatially)
python3 marti2002_plot.py plot-bands --usetex --out out.plot/ --jv-mpp out.mpp/JV_mpp.csv --spatial-input-dir out.mpp/

# plot IB fill factors (spatially)
python3 marti2002_plot.py plot-fs --usetex --out out.plot/ --jv-mpp out.mpp/JV_mpp.csv --spatial-input-dir out.mpp/

# plot JV curves
python3 marti2002_plot.py plot-jv --usetex --out out.plot/ --jv out/JV.csv
'''

COLOR = { # thanks to emily for color scheme
    'blue' : '#817aab',
    'red'  : '#d33d3c',
    'green': '#78c583'}

BANDS = OrderedDict([
    ('CB', dict(color=COLOR['blue' ], sym='C')),
    ('IB', dict(color=COLOR['red'  ], sym='I')),
    ('VB', dict(color=COLOR['green'], sym='V'))])

ABSORPTIONS = OrderedDict([
    ('cv', dict(color='blue')),
    ('ci', dict(color='red')),
    ('iv', dict(color='orange'))])

def plot_current(
        df, x_unit):

    Jscale = 1e-3
    Junit = 'A'

    fig, ax = plt.subplots()
    for band_name, band in BANDS.items():
        ax.plot(df['x'], df['j_'+band_name] * Jscale,
                label='$j_{}$'.format(band['sym']),
                color=band['color'])

    ax.set_xlabel('position ({})'.format(x_unit))
    ax.set_ylabel(r'current density ($\mathrm{{{}}}/\mathrm{{cm}}^{{2}}$)'
                  .format(Junit))

    ax.legend()

    fig.tight_layout()

    return fig

def plot_subcurrents(
        df, x_unit, concentration_factor, symlog=True):

    conc_scale = 1/concentration_factor
    Jscale = 1 * conc_scale * -1
    Junit = 'mA'

    fig, ax = plt.subplots()
    for band_name, band in BANDS.items():
        ax.plot(df['x'], df['j_drift_'+band_name] * Jscale,
                label=r'$j_{{{},\mathrm{{drift}}}}$'.format(band['sym']),
                color=band['color'], alpha=0.7)
        ax.plot(df['x'], df['j_diffusion_'+band_name] * Jscale,
                label=r'$j_{{{},\mathrm{{diff}}}}$'.format(band['sym']),
                color=band['color'], linestyle='dashed')

    j_expr_str = '-J' if symlog else '|J|'
    ax.set_xlabel('position ({})'.format(x_unit))
    ax.set_ylabel(r'${}/X$ ($\mathrm{{{}}}/\mathrm{{cm}}^{{2}}$)'
                  .format(j_expr_str, Junit))

    if symlog:
        ax.set_yscale('symlog', linthreshy=1e-6)
    else:
        ax.set_yscale('log')
        ax.set_ylim([1e-3, None])

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    return fig

def plot_qfl(
        df, x_unit):

    fig, ax = plt.subplots()

    for band_name, band in BANDS.items():
        ax.plot(df['x'], df['Ephi_'+band_name],
                label='_nolegend_',
                color=band['color'], alpha=0.5)

    for band_name, band in BANDS.items():
        ax.plot(df['x'], df['qfl_'+band_name],
                label='$w_{{{}}}$'.format(band['sym']),
                color=band['color'], linestyle='dashed')

    ax.set_xlabel('position ({})'.format(x_unit))
    ax.set_ylabel(r'quasi-Fermi level ($\mathrm{eV}$)')

    ax.legend()

    fig.tight_layout()

    return fig

def plot_Phi(
        df, x_unit):

    fig, ax = plt.subplots()

    for ab_name, ab in ABSORPTIONS.items():
        ax.plot(df['x'], df['Phi_forward_'+ab_name],
                label='$\Phi_{{{}}}$'.format(ab_name),
                color=ab['color'])

    ax.set_xlabel('position ({})'.format(x_unit))
    ax.set_ylabel(r'photon flux ($1/\mathrm{cm}^{2}/\mathrm{s}$)')

    ax.legend()

    fig.tight_layout()

    return fig

def plot_g(
        df, x_unit):

    fig, ax = plt.subplots()

    for band_name, band in BANDS.items():
        ax.plot(
            df['x'], df['g_'+band_name],
            label='$g_{{{}}}$'.format(band_name),
            color=band['color'])

    ax.set_xlabel('position ({})'.format(x_unit))
    ax.set_ylabel(r'generation ($1/\mathrm{cm}^{2}/\mathrm{s}$)')

    ax.set_yscale('symlog', linthreshy=1e9)

    ax.legend()

    fig.tight_layout()

    return fig

def plot_main(basename, submitfile, savefig):

    df = pd.read_csv(basename + '.csv.0')

    df_add_drift_diffusion_terms(df)

    concentration_factor = float(
        submitfile['parameters']['concentration_factor'])

    df['x'] = df['coord_x'] * 1e3
    x_unit = r'$\mathrm{nm}$'

    plot_kwargs = dict(
        df=df,
        x_unit=x_unit)

    if concentration_factor > 0:
        fig = plot_subcurrents(
            concentration_factor=concentration_factor,
            **plot_kwargs)
        savefig(fig, basename + ' xplot js')
        plt.close(fig)

    fig = plot_current(**plot_kwargs)
    savefig(fig, basename + ' xplot j')
    plt.close(fig)

    fig = plot_qfl(**plot_kwargs)
    savefig(fig, basename + ' xplot qfl')
    plt.close(fig)

    fig = plot_Phi(**plot_kwargs)
    savefig(fig, basename + ' xplot Phi')
    plt.close(fig)

    fig = plot_g(**plot_kwargs)
    savefig(fig, basename + ' xplot g')
    plt.close(fig)


# FIXME: hardcoded because I'm stupid
IB_SPATIAL_RANGE = (1050.0, 1050.0 + 1300)
def df_mask_IB(df):
    df["IB_exists"] = exists = df['x'].between(*IB_SPATIAL_RANGE)
    df["IB_mask"] = mask = exists.replace({True: 1.0, False: np.nan})
    for k in ['j_IB', 'qfl_IB', 'Ephi_IB']:
        df[k] *= mask
def extract_IB_region_only(df):
    df = df.copy()
    df['x'] -= IB_SPATIAL_RANGE[0]
    df = df[df['x'].between(0, IB_SPATIAL_RANGE[1] - IB_SPATIAL_RANGE[0])]
    return df

@arg('input_dir')
@arg('V', default='1.25')
@arg_matplotlib()
@expects_obj
def make_fs_plot(args):
    savefig = args.matplotlib_savefig
    out = outdir_path_helper(args.out)

    V = args.V
    input_dir = args.input_dir

    input_dir = input_dir.rstrip('/') + '/'

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()

    x_unit = r'$\mathrm{nm}$'

    for mu_str in [
            '0.001', '0.01', '0.1', '1',
            '30', '100',
            '300']:
        mdir = 'marti02 X=1e3 mu={} s_ci=1e-12 s_iv=2e-13'.format(mu_str)
        base = os.path.join(
            input_dir,
            mdir,
            'a parameter={} csvplot'.format(V))
        df = pd.read_csv(base + '.csv.0')

        df['x'] = df['coord_x'] * 1e3

        df = extract_IB_region_only(df, None)

        df = df[df['x'].between(250, 1000)] # Jacob request

        label = r'$\mu={}$'.format(mu_str)

        N_IB = 1e17
        ax.plot(df['x'], df['u_IB'] / N_IB,
                    label=label,
                    alpha=0.6)

        grad_qfl = np.gradient(df['qfl_IB'], df['x'])
        ax2.plot(df['x'], grad_qfl,
                 label=label,
                 alpha=0.6)

        ax3.plot(df['x'], df['E'],
                 label=label,
                 alpha=0.6)

        ax4.plot(df['x'], df['qfl_IB'],
                 label=label,
                 alpha=0.6)

        ax5.plot(df['x'], df['qfl_IB'] - df['Ephi_IB'],
                 label=label,
                 alpha=0.6)

    for a in (ax, ax2, ax3, ax4, ax5):
        a.set_xlabel(r'position ({})'.format(x_unit))
        a.grid(True)

    ax.set_ylabel(r'IB filling fraction')
    ax2.set_ylabel(r'$\nabla w_{\mathrm{IB}}$ ($\mathrm{eV/nm}$)')
    ax3.set_ylabel(r'electric field ($\mathrm{V/cm}$)')
    ax4.set_ylabel(r'$w_{\mathrm{IB}}$ ($\mathrm{eV}$)')
    ax5.set_ylabel(
        r'$w_{\mathrm{IB}} + eV - \mathcal{E}_{\mathrm{IB}}$ ($\mathrm{eV}$)')

    ax2.set_yscale('symlog', linthreshy=1e-8)
    ax3.set_yscale('symlog', linthreshy=1e-8)
    # ax4.set_yscale('symlog', linthreshy=1e-8)

    # ax.set_ylim([0, 1])

    ax.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()

    fig.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    fig5.tight_layout()

    savefig(fig, input_dir + 'IBfs V={}'.format(V))
    savefig(fig2, input_dir + 'IBgradqfls V={}'.format(V))
    savefig(fig3, input_dir + 'IBE V={}'.format(V))
    savefig(fig4, input_dir + 'IBqfls V={}'.format(V))
    savefig(fig5, input_dir + 'IBphiqfl V={}'.format(V))

@arg('input_dir')
def voltage_step_plots(input_dir, usetex=False, pdf=False):
    r = []
    for f in os.listdir(input_dir):
        before, sep, after = f.rpartition('.plot_meta.yaml')
        if sep and not after:
            r.append(os.path.join(input_dir, before))

    submitfile = h5yaml.load(os.path.join(input_dir, 'submit.yaml'))

    for basename in r:
        if 'parameter=1.25' not in basename: continue
        plot_main(
            basename=basename,
            usetex=usetex, pdf=pdf, submitfile=submitfile)

voltage_step_re = re.compile(
    r'a parameter=(\S+) csvplot\.plot_meta\.yaml$')

@arg('out_dir')
@arg('input_dirs', nargs='*')
@expects_obj
def extract_jv(args):
    out_dir = outdir_path_helper(args.out_dir)

    input_dirs = args.input_dirs

    rows = []
    for input_dir in tqdm.tqdm(input_dirs):
        submit = h5yaml.load(os.path.join(input_dir, 'submit.yaml'))
        spar = submit['parameters']
        for f in os.listdir(input_dir):
            m = voltage_step_re.match(f)
            if not m:
                continue
            V_ext = m.group(1)

            filename = os.path.join(input_dir, f)
            tree = h5yaml.load(filename)

            integrals = tree['integrals']

            d = dict(
                IB_mobility=float(spar['IB_mobility']),
                IB_thickness=float(spar['IB_thickness']),
                IB_sigma_ci=float(spar['IB_sigma_ci']),
                IB_sigma_iv=float(spar['IB_sigma_iv']),
                concentration_factor=float(spar['concentration_factor']),
                V=float(V_ext),
                j_tot_nc=(
                    integrals['avg_j_CB:n_contact'] +
                    integrals['avg_j_VB:n_contact']))

            for k, v in integrals.items():
                d["integrals:"+k] = v

            d['filename'] = filename

            rows.append(d)

    df = pd.DataFrame(rows)

    to_xcsv(df, out_dir + "JV.csv")

@arg('--out', required=True)
@arg('--jv', required=True)
@arg_matplotlib()
@expects_obj
def plot_jv_old(args):
    savefig = args.matplotlib_savefig
    out = outdir_path_helper(args.out)

    df = read_xcsv(args.jv)[0]

    df = df[df['IB_sigma_ci'] == 1e-12]

    mus = list(df['IB_mobility'].unique())
    mus.sort(key=float)


    fig, ax = plt.subplots()

    j_min = 0
    for mu in mus:
        df2 = df[((df['IB_mobility'] == mu) &
                  (df['concentration_factor'] == 1e3))]
        j = df2['j_tot_nc'] / df2['concentration_factor']
        j_min = min(j_min, j.min())
        ax.plot(df2['V'], j,
                label=r"$\mu={}$".format(mu), marker='.')

    ax.set_xlabel(r'applied bias ($\mathrm{V}$)')
    ax.set_ylabel(r'$J_{\mathrm{tot}} / X$ ($\mathrm{mA/cm^2}$)')

    #ax.set_ylim([j_min*1.05, 0])
    ax.set_ylim([-55, -48])

    ax.legend()

    fig.tight_layout()

    savefig(fig, out + 'JV mus')

SIGMA_CI_TO_LABEL = {
    2e-13: r'$\sigma_{\mathrm{ci}} = \sigma_{\mathrm{iv}}$',
    1e-12: r'$\sigma_{\mathrm{ci}} = 5\,\sigma_{\mathrm{iv}}$'
}
SIGMA_CI_TO_LS = {
    2e-13: 'solid',
    1e-12: 'dotted',
}
SIGMA_CI_TO_MARKER = {
    2e-13: 'x',
    1e-12: '.',
}

@arg('--out', required=True)
@arg('--jv', required=True)
@arg_matplotlib(default_font_size=14)
@expects_obj
def plot_jv(args):
    savefig = args.matplotlib_savefig
    out = outdir_path_helper(args.out)

    df = read_xcsv(args.jv)[0]

    fig = plt.figure(figsize=(4.8, 6.4), dpi=100)

    gs0 = GridSpec(
        ncols=1, nrows=2, figure=fig,
        left=0.2, right=0.95, top=0.97, bottom=0.1,
        wspace=0.00, hspace=0.25,
        height_ratios=[1, 1])

    ax0 = fig.add_subplot(gs0[0, 0])
    ax1 = fig.add_subplot(gs0[1, 0])

    def _ll(ax, *args, **kwargs):
        ax.plot([], [], *args, **kwargs)

    sigma_ci_to_ls = SIGMA_CI_TO_LS
    sigma_ci_to_label = SIGMA_CI_TO_LABEL

    ax = ax0
    for sigma_ci, mu in itertools.product(
            [2e-13, 1e-12], [0.001, 300.0]):
        df1 = df[(df.IB_mobility == mu) &
                 (df.IB_sigma_ci == sigma_ci) &
                 (df.concentration_factor > 0)]
        color = COLOR['red' if mu == 300.0 else 'blue']
        ls = sigma_ci_to_ls[sigma_ci]
        j = df1['j_tot_nc'] / df1['concentration_factor']
        ax.plot(df1.V, j, color=color, linestyle=ls,
                label='_nolegend_'
                # label="${}$ ${}$".format(sigma_ci, mu)
        )
    # by Jacob's request
    # _ll(ax, color='black', linestyle='solid',
    #     label=sigma_ci_to_label[2e-13])
    # _ll(ax, color='black', linestyle='dotted',
    #     label=sigma_ci_to_label[1e-12])
    _ll(ax, color=COLOR['red'],
        label=r'$\mu_{I} = 300\;\mathrm{cm^2/V/s}$')
    _ll(ax, color=COLOR['blue'],
        label=r'$\mu_{I} = 0.001\;\mathrm{cm^2/V/s}$')
    ax.set_ylim([-55, -48.5])
    ax.set_xlim([0, 1.35])
    ax.legend()
    ax.set_xlabel(r'applied bias ($\mathrm{V}$)')
    ax.set_ylabel(r'$J_{\mathrm{tot}} / X$ ($\mathrm{mA/cm^2}$)')

    ax = ax1
    df1 = df[(df.V == 0.0) & (df.concentration_factor > 0)]
    # mus = list(sorted(df['IB_mobility'].unique()))
    for sigma_ci in [2e-13, 1e-12]:
        df2 = df1[df1['IB_sigma_ci'] == sigma_ci]
        df2 = df2.sort_values('IB_mobility')
        j = df2['j_tot_nc'] / df2['concentration_factor']
        ls = sigma_ci_to_ls[sigma_ci]
        ax.plot(df2.IB_mobility, j, linestyle=ls, color='black',
                label=sigma_ci_to_label[sigma_ci],
                marker=SIGMA_CI_TO_MARKER[sigma_ci])
    # ax.set_xlim([1e-3, 300])
    ax.set_ylim([-54.5, -53])
    ax.set_xscale('log')
    ax.legend()
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_xlabel(r'IB mobility ($\mathrm{cm^2/V/s}$)')
    ax.set_ylabel(r'$J_{SC} / X$ ($\mathrm{mA/cm^2}$)')

    savefig(fig, out + 'marti02_final_JVs')

CASE_KEYS = [
    'IB_mobility', 'IB_sigma_ci', 'IB_sigma_iv', 'concentration_factor']

@arg('--out', required=True)
@arg('--jv', required=True)
@expects_obj
def copy_mpp(args):
    out = outdir_path_helper(args.out)
    df = read_xcsv(args.jv, read_csv_kwargs=dict(index_col=0))[0]
    df.index.name = 'index'

    df['P'] = -df.j_tot_nc*df.V

    df = df[df.concentration_factor > 0]

    df_mpp = df.sort_values('P', ascending=False).drop_duplicates(CASE_KEYS)
    df_jsc = df[df.V.abs() < 1e-10].drop_duplicates(CASE_KEYS)
    # df1 = df1.sort_values('index')

    df1 = pd.concat([df_mpp, df_jsc])

    for index, row in df1.iterrows():
        filename = row['filename']
        base = osp.dirname(filename)
        for oldname in [
                filename,
                filename.rpartition('.plot_meta.yaml')[0] + '.csv.0',
                osp.join(base, 'submit.yaml')]:
            newname = osp.join(out, oldname)
            mkdirp(osp.dirname(newname))
            print(oldname, newname)
            shutil.copyfile(oldname, newname)

    to_xcsv(df_mpp, out + "JV_mpp.csv")
    to_xcsv(df1   , out + "JV_mpp_and_jsc.csv")


def common_load_spatial_data(spatial_input_dir, jv_filename):
    before, sep, after = jv_filename.rpartition('.plot_meta.yaml')
    if sep and not after:
        jv_filename = before + '.csv.0'

    df = pd.read_csv(osp.join(spatial_input_dir, jv_filename))

    N_IB = 1e17

    df['x'] = df['coord_x'] * 1e3
    df['f_IB'] = df['u_IB'] / N_IB

    return df

@arg('--out', required=True)
@arg('--jv', required=True)
@arg('--spatial-input-dir', required=True)
@arg_matplotlib(default_font_size=14)
@expects_obj
def plot_bands(args):
    savefig = args.matplotlib_savefig
    out = outdir_path_helper(args.out)
    spdir = args.spatial_input_dir

    # python3 -m ecd_thesis2018.marti2002_plot plot-bands --usetex --out out.marti2002_plot/ --jv out.marti2002_plot_datafiles/JV_mpp_and_jsc.csv --spatial-input-dir out.marti2002_plot_datafiles/
    # thesis2018=d0332dc0382ea2b5cd609c931e7b7970c218bd29
    # simudo=682f18db136c86b6342e6fbbd2ff7a4df3ea67c1

    jv = read_xcsv(args.jv)[0]

    jv = jv[jv.V.abs() < 1e-10] # short circuit

    fig = plt.figure(figsize=(4.8, 2.6), dpi=100)

    gs0 = GridSpec(
        ncols=1, nrows=1, figure=fig,
        left=0.14, right=0.95, top=0.97, bottom=0.22,
        wspace=0.00, hspace=0.00,
        height_ratios=[1])

    ax0 = fig.add_subplot(gs0[0, 0])

    jv1 = jv[(jv.IB_sigma_ci == 1e-12) & (jv['IB_mobility'] == 10.0)]

    (_, row), = list(jv1.iterrows())

    df = common_load_spatial_data(spdir, row['filename'])
    print(row['filename'])

    df_mask_IB(df)

    XS = 1e-3

    ax = ax0
    for band_name, band in BANDS.items():
        ax.plot(df['x']*XS, df['Ephi_'+band_name],
                label='_nolegend_',
                color=band['color'], alpha=0.5)
        ax.text(0.5, 0.5, r"$\mathcal{{E}}_{}$".format(band['sym']),
                color='black') # as my soul

    for band_name, band in BANDS.items():
        ax.plot(df['x']*XS, df['qfl_'+band_name],
                label='$w_{{{}}}$'.format(band['sym']),
                color=band['color'], linestyle='dashed')
        ax.text(0.5, 0.5, "$w_{}$".format(band['sym']),
                color=band['color'])

    # print(df['qfl_VB'])

    ax.set_xlabel(r'position ($\mathrm{\mu m}$)')
    ax.set_ylabel(r'Energy ($\mathrm{eV}$)')

    ax.set_xlim(np.array([0 - 10, (df['x'].max() + 10)])*XS)
    ax.set_ylim([-1.63, 1.76])

    # ax.legend()

    # fig.tight_layout()

    savefig(fig, out + 'marti02_final_bands')

@arg('--out', required=True)
@arg('--jv-mpp', required=True)
@arg('--spatial-input-dir', required=True)
@arg('--silly', action='store_true')
@arg_matplotlib(default_font_size=14)
@expects_obj
def plot_fs(args):
    savefig = args.matplotlib_savefig
    out = outdir_path_helper(args.out)
    spdir = args.spatial_input_dir

    SILLY = args.silly

    jv = read_xcsv(args.jv_mpp)[0]

    fig = plt.figure(figsize=(4.8, 3.6), dpi=100)

    gs0 = GridSpec(
        ncols=1, nrows=1, figure=fig,
        left=0.18, right=0.95, top=0.97, bottom=0.16,
        wspace=0.00, hspace=0.00,
        height_ratios=[1])

    ax0 = fig.add_subplot(gs0[0, 0])

    sigma_ci_to_ls = {
        2e-13: 'solid',
        1e-12: 'dashed',
    }
    mu_to_color = {
        0.001: COLOR['blue'],
        30.0: COLOR['red']
    }

    ax = ax0
    for sigma_ci in [2e-13, 1e-12]:

        jv1 = jv[jv.IB_sigma_ci == sigma_ci]
        jv1 = jv1.sort_values('IB_mobility', ascending=False)
        # if not SILLY:
        #     jv1 = jv1[jv1.IB_mobility.isin([0.001, 30])]

        ls = sigma_ci_to_ls[sigma_ci]

        # position doesn't matter, will be manually adjusted in inkscape
        ax.text(0.07, 0.93, SIGMA_CI_TO_LABEL[sigma_ci],
                horizontalalignment='left',
                verticalalignment='top', transform=ax.transAxes,
                # fontsize='small'
        )

        for index, row in jv1.iterrows():
            try:
                color = mu_to_color[row['IB_mobility']]
            except KeyError:
                continue

            df = common_load_spatial_data(spdir, row['filename'])

            df = extract_IB_region_only(df)

            df = df[df.x.between(200, 1000)] # Jacob request; was (250, 1000)

            ax.plot(df['x'], df['f_IB'],
                    color=color, linestyle=ls,
                    label='_nolegend_')
            if sigma_ci == 2e-13:
                mu_label = (
                    r'$\mu_{{I}} = {:.7g}'
                    '\;\mathrm{{cm^2/V/s}}$'.format(row['IB_mobility']))
                ax.plot([], [], label=mu_label, color=color)

        # ax.plot([], [], linestyle=ls, color='black',
        #         label=SIGMA_CI_TO_LABEL[sigma_ci])

    if not SILLY:
        ax.legend()
    ax.set_ylabel(r'IB filling fraction')
    ax.set_xlim([200, 1000])
    ax.set_ylim([0.23, 0.56])
    ax.set_xlabel(r"position ($\mathrm{nm}$)")

    savefig(fig, out + 'marti02_final_f_IB')

@with_default_kwargs()
def plot_detail_j_subplot(
        concentration_factor, df, ax, **kws):

    todo = TODO()

    cscale = 1/concentration_factor

    order = ['CB', 'IB', 'VB']

    for band_name in ['CB', 'VB', 'IB']:
        band = BANDS[band_name]
        ax.plot(df['x'], df['j_drift_'+band_name].abs()*cscale,
                # label=r'$j_{{{},\mathrm{{drift}}}}$'.format(band['sym']),
                label='_nolegend_',
                color=band['color'], linestyle='dashed')
        ax.plot(df['x'], df['j_diffusion_'+band_name].abs()*cscale,
                # label=r'$j_{{{},\mathrm{{diff}}}}$'.format(band['sym']),
                label='_nolegend_',
                color=band['color'], linestyle='solid')
        todo(order.index(band_name)+10, ax.plot, [], [],
             label=r'$j_{{{}}}$'.format(band['sym']),
             color=band['color'], linestyle='solid')

    todo.call()

    ax.set_xlabel('position ($\mathrm{nm}$)')
    ax.set_ylabel(r'$|j_x|/X$ ($\mathrm{{mA}}/\mathrm{{cm}}^{{2}}$)'
                  .format())

    ax.set_yscale('log')
    ax.set_ylim([1e-7, 1e2])
    ax.set_xlim(df['x'].iloc[[0, -1]])

    ax.legend()

def extract_max_abs_IB_current(spatial_input_dir, filename):
    df = common_load_spatial_data(spatial_input_dir, filename)
    return df.j_IB.abs().max()

@arg('--out', required=True)
@arg('--jv-mpp', required=True)
@arg('--spatial-input-dir', required=True)
@arg('--silly', action='store_true')
@arg_matplotlib(default_font_size=14)
@expects_obj
def plot_detail_j(args):
    savefig = args.matplotlib_savefig
    out = outdir_path_helper(args.out)
    spdir = args.spatial_input_dir

    SILLY = args.silly

    jv = read_xcsv(args.jv_mpp)[0]

    fig = plt.figure(figsize=(4.8, 6.4), dpi=100)

    gs0 = GridSpec(
        ncols=1, nrows=2, figure=fig,
        left=0.18, right=0.95, top=0.97, bottom=0.10,
        wspace=0.00, hspace=0.25,
        height_ratios=[1, 1])

    ax0 = fig.add_subplot(gs0[0, 0])
    ax1 = fig.add_subplot(gs0[1, 0])

    sigma_ci_to_ls = {
        2e-13: 'solid',
        1e-12: 'dashed',
    }
    mu_to_color = {
        0.001: COLOR['blue'],
        30.0: COLOR['red']
    }

    jv1 = jv.query(
        "(IB_sigma_ci == 2e-13) &"
        "(IB_mobility == 100.0)")
    assert len(jv1.index) == 1
    df = common_load_spatial_data(spdir, jv1['filename'].iloc[0])
    df_add_drift_diffusion_terms(df)
    ax = ax0
    plot_detail_j_subplot(
        locals(),
        ax=ax0,
        concentration_factor=jv1['concentration_factor'].iloc[0])

    ax = ax1

    jv["j_IB_max_abs"] = jv["filename"].apply(
        partial(extract_max_abs_IB_current, spdir))

    for sigma_ci in [2e-13, 1e-12]:
        jv1 = jv[jv.IB_sigma_ci == sigma_ci]
        jv1 = jv1.sort_values("IB_mobility")
        ax.plot(jv1.IB_mobility,
                jv1.j_IB_max_abs / jv1.concentration_factor,
                color='black',
                label=SIGMA_CI_TO_LABEL[sigma_ci],
                linestyle=SIGMA_CI_TO_LS[sigma_ci],
                marker=SIGMA_CI_TO_MARKER[sigma_ci])

    ax.set_ylim([0, 10.3])
    ax.set_xlabel(r'IB mobility ($\mathrm{cm^2/V/s}$)')
    ax.set_ylabel(r'$\max(|j_{I}| / X)$ ($\mathrm{cm^2/V/s}$)')
    ax.set_xscale('log')
    ax.legend()

    to_xcsv(jv, out+"marti02_final_Jdd_maxIB.csv")

    savefig(fig, out + 'marti02_final_Jdd')



parser = ArghParser()
parser.add_commands([
    voltage_step_plots,
    extract_jv,
    plot_jv,
    copy_mpp,
    plot_fs,
    plot_bands,
    plot_detail_j])

if __name__ == '__main__':
    parser.dispatch()




