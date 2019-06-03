#!/bin/bash
set -e
set -x

# Packages available in Ubuntu's repositories
sudo apt install build-essential zip unzip parallel cython{,3} \
    python3-{argh,atomicwrites,cached-property,dolfin,future,h5py} \
    python3-{matplotlib,meshio,pandas,petsc4py,pint,pprofile,pytest} \
    python3-{scipy,sortedcontainers,sphinx,sphinx-rtd-theme,tabulate} \
    python3-{tqdm,yaml,yamlordereddictloader} \
    optipng poppler-utils meshio-tools gmsh \
    python3-pip

# Packages available through pip via PyPI
pip3 install suffix_trees mpl_render generic_escape simudo

