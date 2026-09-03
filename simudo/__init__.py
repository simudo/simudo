# The try/except lets `pip install --no-deps simudo` provide the GUI files on a machine with no working Simudo/FEniCS install.
from .version import __version__

try:
    from .util.ufl_import_hack import fix as _fix
except ImportError:
    # The numerical stack is absent. That is a legitimate install: the
    # graphical interface is served from simudo/gui/ and never imports simudo,
    # so a host machine may install this package with --no-deps purely to have
    # the files. Let `import simudo` succeed there.
    #
    # The dolfin check keeps that narrow. If FEniCS *is* present then the ufl
    # aliasing fix is mandatory and skipping it would surface much later as a
    # baffling UFL error, so a broken install still fails loudly here.
    import importlib.util as _importlib_util

    if _importlib_util.find_spec('dolfin') is not None:
        raise
else:
    _fix()
