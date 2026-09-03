# encoding: utf-8
from setuptools import setup, find_packages
import os
pkg = "simudo"

version_path = os.path.abspath(pkg + '/version.py')
version_info = {"__file__": version_path}
with open(version_path, 'rt', encoding="utf-8") as h:
    exec(compile(h.read(), version_path, 'exec'), version_info)
ver = version_info["__version__"].partition("~")[0]

with open("README.md", "rt", encoding="utf-8") as h:
    readme = h.read()

# For Debian Buster / Ubuntu 20.04:
# apt install python3-{numpy,scipy,pint,matplotlib,meshio,yamlordereddictloader,sortedcontainers,cached-property,h5py,petsc4py}
# Still need to install: mpl_render, suffix_trees, generic_escape

setup(
    name             = pkg,
    version          = ver,
    description      = (
        "SIMUlation of Devices with Optics / SIMulateur Université D'Ottawa"),
    long_description = readme,
    long_description_content_type = "text/markdown",
    author           = "Eduard Christian Dumitrescu",
    author_email     = "eduard.c.dumitrescu@gmail.com",
    license          = "LGPLv3",
    url              = "https://github.com/simudo/simudo",
    packages         = find_packages(),
    package_data     = {
        # pyaml sources, loaded at import time.
        'simudo.physics':   ['*.py1'],
        # pyaml sources, plus the C++ header dolfin JITs against:
        # fem/dolfin_parameters.py puts this directory on dolfin's include
        # path, so the header has to sit next to the installed module.
        'simudo.fem':       ['*.py1', '*.hpp'],
        # Custom pint unit definitions, and the ASTM G173 reference spectra
        # that util/am15.py loads relative to its own __file__.
        'simudo.util':      ['pint/*.txt', 'data/*.npz'],
        # NOT the alloy_optics/ tables.  0.7 ships measured optical data for
        # the binaries only; the composition-dependent alloy interpolation is
        # unfinished and nothing reads it, so the data stays in the source tree
        # rather than the wheel.  See materials/alloy_optics.py.
        # Example project the GUI falls back to when launched with no file.
        'simudo.gui':       ['fourlayer_example.yaml'],
        # Fixture for the pyaml round-trip test.
        'simudo.test':      ['*.yaml'],
        # Tutorial notebook and its figure.
        'simudo.example':   ['jupyter/*.ipynb', 'jupyter/*.svg'],
    },
    # NOT data_files: with a "" destination that installs these to the root of
    # sys.prefix, littering the environment.  license_files puts them in the
    # wheel's .dist-info/licenses/ where packaging tools expect them.
    license_files    = ["LICENSE", "COPYRIGHT"],
    # 3.10 for the `match` statement in materials/alloy_optics.py, which the
    # AlGaAs and GaNAs material modules import.  Ubuntu 22.04 ships 3.10 and
    # 24.04 ships 3.12, so this excludes no platform we document.
    python_requires  = ">=3.10",
    # Everything here is imported by `import simudo.physics` / `.io` / `.fem` /
    # `.mesh`, verified empirically -- see doc/release0_7/01_repo_map.md 5.4.
    # FEniCS (dolfin, ufl, ffc) is deliberately absent: it is not pip-installable
    # and is documented as a prerequisite.
    install_requires = [
        'numpy', 'scipy', 'pandas', 'Pint',
        'atomicwrites',
        'matplotlib', 'mpl_render',
        'yamlordereddictloader', 'suffix_trees', 'generic_escape',
        'sortedcontainers',
        'cached_property',
        'petsc4py',
        'PyYAML',        # io, pyaml
        'attrs',         # mesh, physics
        'sympy',         # fem.sympy_dolfin_printer
        'setuptools',    # pkg_resources, used by pyaml.load
        'future',        # io, util -- Python-2 shims, removable once those go
        'argh', 'tabulate', 'tqdm', 'cycler',   # example/, plot/
        'IPython', 'ipywidgets',                # fem.ipython widgets
    ],

    extras_require   = {
        # Alternative mesh generators.
        'mesh':     ['pygmsh', 'meshio', 'gmsh', 'mshr'],
        # The Panel-based graphical interface. Normally installed on the
        # user's own machine rather than alongside the solver -- see
        # doc/source/user/install.rst.
        'gui':      ['panel', 'param', 'bokeh', 'tornado'],
    },
    classifiers      = [
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries"])

