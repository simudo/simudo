
Installation
&&&&&&&&&&&&

Simudo has depends on some Python packages, as well as:

- FEniCS (finite element library)
- GMSH (mesh generator)

So far we've only tested Simudo on Debian Testing/Buster and Ubuntu
Linux 19.04.

Ubuntu Linux 19.04
==================

First, you need to install the dependencies that are available in
Ubuntu's repositories. Copy-paste this entire command in your
terminal, and run it::

  sudo apt install build-essential zip unzip parallel cython3 \
    python3-{argh,atomicwrites,cached-property,dolfin,future,h5py} \
    python3-{matplotlib,meshio,pandas,petsc4py,pint,pprofile,pytest} \
    python3-{scipy,sortedcontainers,sphinx,sphinx-rtd-theme,tabulate} \
    python3-{tqdm,yaml,yamlordereddictloader} python3-pip \
    optipng poppler-utils meshio-tools gmsh

Finally, install the remaining dependencies as well as Simudo itself
from PyPI::

  pip3 install suffix_trees mpl_render generic_escape simudo

That's it! Simudo is now installed.

To update to the latest version of Simudo on PyPI, use ``pip3``::

  $ pip3 install --upgrade simudo

