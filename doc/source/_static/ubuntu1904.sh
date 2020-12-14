#!/bin/bash
set -e

if ! [ "$(id -u)" -eq 0 ]; then
    echo "you need to run this as root"
    exit 1
fi

# Packages available in Ubuntu's repositories
apt install build-essential wget zip unzip parallel cython{,3} \
    python3-{argh,atomicwrites,cached-property,dolfin,future,h5py} \
    python3-{matplotlib,meshio,pandas,petsc4py,pint,pprofile,pytest} \
    python3-{scipy,sortedcontainers,sphinx,sphinx-rtd-theme,tabulate} \
    python3-{tqdm,yaml,yamlordereddictloader} \
    optipng poppler-utils meshio-tools gmsh

# Packages available through pip (if you don't trust PyPI, don't use this!)
apt install python3-pip
pip3 install --system suffix_trees mpl_render generic_escape

