
from ..util.plot import plt, subplots
from ..util.pdftoppm import pdftoppm
import numpy as np
import pandas as pd

bands = (('VB' , 'red'  , 'p'),
         ('CB' , 'blue' , 'n'),
         ('IB' , 'green', 'I'),
         ('tot', 'black', 'tot'))

def do_n_plot(out, df, xlim=None, ylim=None, legend_loc='upper right',
              title=None):
    with subplots(figsize=(8, 5.7), closefig=out is not None) as (fig, ax):
        x = df['coord_x'].values
        for band, color, pn in bands:
            u = df['u_'+band].values
            u_ref = df['u_'+band+'_ref'].values
            u_diff = abs(u - u_ref)

            ax.plot(
                x, u,      color=color,
                label=r"${}_{{\mathrm{{our}}}}$".format(pn))
            ax.plot(
                x, u_diff, color=color, linestyle='--',
                label=r"$|{0}_{{\mathrm{{our}}}}-{0}_{{\mathrm{{ref}}}}|$".format(pn))

        ax.set_yscale('log')
        if xlim:ax.set_xlim(xlim)
        if ylim:ax.set_ylim(ylim)

        if title:
            ax.set_title(title)
            ax.title.set_size(14)

        ax.grid(True)
        ax.legend(loc=legend_loc)
        ax.set_xlabel(r"$x$ ($\mu \mathrm{m}$)")
        ax.set_ylabel("carrier density ($\mathrm{cm^{-3}}$)")
        fig.tight_layout()

        if out is not None:
            fig.savefig(out)
            pdftoppm(out)
        return fig

def do_j_plot(out, df, xlim=None, ylim=None, legend_loc='upper right',
              title=None, yexp=None):
    with subplots(figsize=(8, 5.7), closefig=out is not None) as (fig, ax):
        x = df['coord_x'].values
        for band, color, pn in bands:
            try:
                j = df['j_'+band].values
            except KeyError:
                continue
            try:
                j_ref = df['j_'+band+'_ref'].values
            except KeyError:
                j_ref = None

            ax.plot(x, j,     color=color,
                    label=r"$j_{0}$".format(pn))
            if j_ref is not None:
                ax.plot(x, j_ref, color=color, linestyle='--',
                        label="$j_{{{0},\mathrm{{ref}}}}$".format(pn))

        # ax.set_yscale('log')
        # if xlim:ax.set_xlim(xlim)
        ax.set_xlim((min(x), max(x)))
        if ylim:ax.set_ylim(ylim)

        if title:
            ax.set_title(title)
            ax.title.set_size(14)

        if yexp:
            ax.ticklabel_format(style='sci', axis='y',
                                scilimits=(yexp, yexp))

        ax.grid(True)
        ax.legend(loc=legend_loc)
        ax.set_xlabel(r"$x$ ($\mu \mathrm{m}$)")
        ax.set_ylabel("current density ($\mathrm{mA/cm^2}$)")
        fig.tight_layout()

        if out is not None:
            fig.savefig(out)
            pdftoppm(out)
        return fig

def main():
    # plt.rc('text', usetex=True)
    def F(case, suf):
        d = 'out/diode {} m=mixed.out.d/'.format(case)
        return dict(out='out/poster_plot {} {}.pdf'.format(case, suf),
                    df=pd.read_csv(d+'final.csv.0'))
    tt = 'symmetric doping ${c}$ cm$^{{-3}}$, bias ${V}$ V'
    do_n_plot(ylim=(1e-4, 1.1e18), **F('c=1e18 V=-0.2', 'n'),
              title=tt.format(c='10^{18}', V='-0.2'))
    do_n_plot(ylim=(1e5, 1.1e18), **F('c=1e18 V=0.4', 'n'),
              title=tt.format(c='10^{18}', V='0.4'))

    do_j_plot(ylim=None, **F('c=1e18 V=-0.2', 'j'),
              title=tt.format(c='10^{18}', V='-0.2'), yexp=-6)
    do_j_plot(ylim=(0.22, 0.255), **F('c=1e18 V=0.4', 'j'),
              title=tt.format(c='10^{18}', V='0.4'))

if __name__=='__main__': main()
