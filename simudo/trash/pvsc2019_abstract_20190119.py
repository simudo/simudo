

from ..util.plot import plt, subplots
from ..util.pdftoppm import savefig
import pandas as pd
import lzma
import numpy as np

BANDS = (
    ('CB', 'blue'  , 'n', 'C'),
    ('IB', 'orange', 'I', 'I'),
    ('VB', 'red'   , 'p', 'V'),
)

def plot_bands_and_f(df0, df_illuminated, vlines, legend_loc='upper right'):

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 2, 2]),
        figsize=(8, 5.2))

    Xs, xlim, Xscale, xunit = _xlim(df0)

    for band, color, pn, E_what in BANDS:
        mask = df0['bandmask_'+band].values
        mu = df0['mu_'+band].values
        E = df0['edgeE_'+band].values - df0['V'].values

        color='black'

        ax0.plot(
            Xs, E*mask, color=color,
            label=r"${{E}}_{}$".format(pn))

        # ax0.plot(
        #     Xs, mu*mask, color=color, linestyle='dashed',
        #     label=r"${{\mu}}_{}$".format(pn))

    ax0.plot(Xs, df0['mu_CB'], linestyle='dashed',
             label="Fermi level", color='red')

    ax1.plot(Xs, df0['f_IB'],
             label=r"equil", color='black')
    ax1.plot(Xs, df_illuminated['f_IB'],
             label=r"illum", color='black',
             linestyle='dashed')

    for x in vlines:
        for a in (ax0, ax1, ax2):
            a.axvline(x=Xscale*x, alpha=0.2, color='darkgreen')
    ax1.axhline(y=1.0, alpha=0.3, color='black')

    for alpha_k, alpha_label, color in [
            ('alpha_ci', r'$\alpha_{{\mathrm{{CI,{}}}}}$', 'red'),
            ('alpha_iv', r'$\alpha_{{\mathrm{{IV,{}}}}}$', 'blue')]:
        for df_label, df, ls in [
                ('equil', df0, 'solid'),
                ('illum', df_illuminated, 'dashed')]:
            ax2.plot(Xs, df[alpha_k], linestyle=ls,
                     label=alpha_label.format(df_label),
                     color=color)

    ax0.set_xlim(xlim)
    ax1.set_ylim([0, 1.25])
    # ax.legend(loc='upper right')
    ax2.set_xlabel(r'position ({})'.format(xunit))
    ax0.set_ylabel(r"$E$ ($\mathrm{eV}$)")
    ax1.set_ylabel(r"$f_{\mathrm{IB}}$")
    ax2.set_ylabel(r'$\alpha$ ($1/\mu\mathrm{m}$)')
    # ax0.legend()
    ax1.legend(fontsize='small')
    ax2.legend(fontsize='small')
    fig.tight_layout()
    fig.subplots_adjust(
        hspace=0.,
        # left=0.15, right=0.95,
        # bottom=0.15, top=0.92
    )

    return fig

def _xlim(df):
    Xs = df.coord_x * 1
    xlim = [min(Xs.values), max(Xs.values)]
    #xunit = 'nm'
    xunit = r'$\mu\mathrm{m}$'
    return Xs, xlim, 1.0, xunit

def plot_alphas(df_equilibrium, df_illuminated):
    df0, df1 = df_equilibrium, df_illuminated
    Xs, xlim, Xscale, xunit = _xlim(df0)

    fig, ax = plt.subplots(figsize=(8, 3))

    for alpha_k, alpha_label, color in [
            ('alpha_ci', r'$\alpha_{{\mathrm{{CI,{}}}}}$', 'red'),
            ('alpha_iv', r'$\alpha_{{\mathrm{{IV,{}}}}}$', 'blue')]:
        for df_label, df, ls in [
                ('equil', df0, 'solid'),
                ('illum', df1, 'dashed')]:
            ax.plot(Xs, df[alpha_k], linestyle=ls,
                    label=alpha_label.format(df_label),
                    color=color)

    ax.set_xlabel(r'position ({})'.format(xunit))
    ax.set_ylabel(r'$\alpha$ ($1/\mu\mathrm{m}$)')
    ax.set_xlim(xlim)
    ax.legend(loc='center right')

    # fig.subplots_adjust(
    #     left=0.15, right=0.95,
    #     bottom=0.15, top=0.92)
    # fig.subplots_adjust(
    #     left=0.07, right=0.95,
    #     bottom=0.15, top=0.92)
    fig.tight_layout()

    return fig

def plot_photon():
    fig, ax = plt.subplots()
    Xs = np.linspace(0, 1, 2001)
    l = 1/15
    pi = np.pi
    stddev = 0.12
    Ys = 1.0 + np.sin(2*pi*Xs/l) * np.exp(-(Xs - 0.5)**2 / 2/(stddev)**2)
    ax.plot(Xs, Ys)
    ax.set_xlim([0, 1])
    return fig

def main():
    IB_N0 = 5e13 # cm^-3
    IB_sigma_opt = 3e-14 # cm^-2
    '''
50 nm window | 1 um p | 2.5 um IB | 1um n
'''

    with lzma.open("data/pvsc2019_abstract_20190119_po_thmeq.csv.0.xz",
                   mode='rb') as handle:
        df0 = pd.read_csv(handle)
    with lzma.open("data/pvsc2019_abstract_20190119_optics.csv.0.xz",
                   mode='rb') as handle:
        df1 = pd.read_csv(handle)

    for df in (df0, df1):
        df.sort_values('coord_x', inplace=True)
        df.reset_index(drop=True, inplace=True)

    assert (df0.coord_x == df1.coord_x).all()

    x = df0['coord_x']
    actual_IB = (1.050 <= x) & (x <= 1.050+2.5)
    vlines = [0.050, 1.050, 3.550]

    for df in (df0, df1):
        df.actual_IB = actual_IB
        f = (df['u_IB'] / IB_N0).clip(lower=0.0, upper=1.0)
        df['f_IB'] = f
        df['alpha_ci'] = actual_IB*IB_sigma_opt*IB_N0*f
        df['alpha_iv'] = actual_IB*IB_sigma_opt*IB_N0*(1-f)
        df['bandmask_VB'] = 1.0
        df['bandmask_CB'] = 1.0
        df['bandmask_IB'] = (actual_IB*1.0).replace(0.0, np.nan)


    # print(df1.iloc[::200][
    #     ['coord_x', 'actual_IB',
    #      'alpha_ci', 'alpha_ci2',
    #      'alpha_iv', 'alpha_iv2',
    #      'f_IB']])

    fig = plot_photon()
    fig.savefig("out/pvsc20190119_photon.svg")

    fig = plot_alphas(df0, df1)
    fig.savefig("out/pvsc20190119_alpha.svg")
    # savefig(fig, "out/pvsc20190119_alpha.pdf")

    fig = plot_bands_and_f(df0, df1, vlines)
    fig.savefig("out/pvsc20190119_bands.svg")
    # savefig(fig, "out/pvsc20190119_bands.pdf")

    return locals()

if __name__ == '__main__':
    main()

