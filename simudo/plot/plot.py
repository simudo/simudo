
import os
import textwrap
from contextlib import contextmanager
from functools import wraps, partial

import numpy as np

try:
    import argh
except ImportError:
    pass

from ..util import string_system
from .matplotlib import pyplot as plt
from .pdftoppm import pdftoppm

@contextmanager
def figclosing(fig=None):
    if fig is None:
        fig = plt.figure()
    yield fig
    plt.close(fig)

@contextmanager
def subplots(*args, closefig=True, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    yield (fig, ax)
    if closefig:
        plt.close(fig)

@contextmanager
def plot01(filename):
    with plot() as (fig, ax):
        ax.grid(True)
        yield (fig, ax)
        ax.legend()
        fig.savefig(filename)

def matplotlib_configure_CM_font(size=16, usetex=False):
    if usetex:
        plt.rc('text', usetex=True)
        font = dict(
            family='serif',
            size=size,
            serif=['computer modern roman'])
        plt.rc('font',**font)
    else:
        plt.rc("mathtext", fontset="cm")
        plt.rc("font", family="serif", size=size)

def percentile_update_limits(limits=None, values=None, percentiles=(3, 97)):
    '''Update an array [ymin, ymax] with values `values` by using the
percentiles `percentiles`.

Note: this function modifies its first argument, and also returns it.
If the first argument is `None`, `[inf, -inf]` is returned.
'''
    if limits is None:
        limits = np.inf * np.array([1, -1])
    if values is not None:
        p = np.percentile(values, percentiles)
        limits[0] = min(limits[0], p[0])
        limits[1] = max(limits[1], p[1])
    return limits

def savefig(fig, basename, pdf=False, svg=True, png=None, optipng=True):
    if png is None:
        png = pdf

    if pdf is None:
        pdf = png

    if png and not pdf:
        warnings.warn(RuntimeWarning("pdf=True implies png=True"))
        pdf = png

    if svg:
        fig.savefig(basename + '.svg')

    if pdf:
        fig.savefig(basename + '.pdf')
        if png:
            pdftoppm(input =basename + '.pdf',
                     output=basename + '.png', optipng=optipng)

def arg_matplotlib(default_font_size=16, argument_prefix=""):
    def wrapper(func):
        p = '--'+argument_prefix
        arg = argh.arg
        @wraps(func)
        @arg(p+'pdf', dest='matplotlib_pdf', action='store_true',
             help='In addition to SVG, also output PDF and PNG files?')
        @arg(p+'usetex', dest="matplotlib_usetex", action='store_true',
             help='Tell matplotlib to use TeX to process text and math '
             'in figure. Slower but better.')
        @arg(p+'font-size', dest='matplotlib_font_size',
             help='Set default matplotlib font size.',
             default=default_font_size, type=int)
        def wrapped(args):
            matplotlib_configure_CM_font(
                size=args.matplotlib_font_size,
                usetex=args.matplotlib_usetex)
            args.matplotlib_savefig = partial(savefig, pdf=args.matplotlib_pdf)
            return func(args)
        return wrapped
    return wrapper

class BaseText(object):
    string = ...
    def __str__(self):
        return self.string

class TitleText(BaseText):
    def __init__(self, ax):
        self.ax = ax
    @property
    def string(self):
        return self.ax.title.get_text()
    @string.setter
    def string(self, value):
        self.ax.title.set_text(value)

class LegendText(BaseText):
    def __init__(self, ax, index):
        self.ax = ax
        self.index = index
    @property
    def legend_text_object(self):
        return self.ax.get_legend().get_texts()[self.index]
    @property
    def string(self):
        return self.legend_text_object.get_text()
    @string.setter
    def string(self, value):
        self.legend_text_object.set_text(value)
    @classmethod
    def from_axes(cls, ax):
        return [cls(ax, index) for index in
                range(len(ax.get_legend().get_texts()))]

def clever_legend(ax, title_wrap_width=65, title_font_size=12):
    ax.legend()

    title_text = TitleText(ax)
    texts = [title_text]
    texts.extend(LegendText.from_axes(ax))

    strings, defs = string_system.format_string_system(
        string_system.make_string_system(
            [str(t) for t in texts],
            min_length=6))

    for t, s in zip(texts, strings):
        t.string = s

    if title_font_size is not None:
        ax.title.set_size(title_font_size)

    title_text.string = '\n'.join(textwrap.wrap(
        title_text.string + '. ' + '; '.join(
            '{}:{{{}}}'.format(x[0], x[1]) for x in defs), title_wrap_width))

    return ax
