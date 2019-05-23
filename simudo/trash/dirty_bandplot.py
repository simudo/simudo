
from ..util.plot import plt, subplots
from ..util.pdftoppm import pdftoppm

import pandas

bands = (('VB', 'red'  , 'p'),
         ('CB', 'blue' , 'n'),
         ('IB', 'green', 'i'))

def do_band_plot(out, df, xlim=None, ylim=None, legend_loc='upper right',
                 title=None):
    with subplots(figsize=(8, 5.7), closefig=out is not None) as (fig, ax):
        x = df['coord_x'].values
        for band, color, pn in bands:
            try:
                mu = df['qfl_'+band].values
            except KeyError:
                continue
            E = df['E_'+band].values - df['phi'].values

            ax.plot(
                x, E, color=color,
                label=r"${{E}}_{}$".format(pn))

            ax.plot(
                x, mu, color=color, linestyle='dashed',
                label=r"${{\mu}}_{}$".format(pn))

        if xlim:ax.set_xlim(xlim)
        if ylim:ax.set_ylim(ylim)

        if title:
            ax.set_title(title)
            ax.title.set_size(14)

        ax.grid(True)
        ax.legend(loc=legend_loc)
        ax.set_xlabel(r"$x$ ($\mu \mathrm{m}$)")
        ax.set_ylabel("E ($\mathrm{eV}$)")
        fig.tight_layout()

        if out is not None:
            fig.savefig(out)
            pdftoppm(out)
        return fig
