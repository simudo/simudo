# encoding: utf-8
from setuptools import setup, find_packages
import os
pkg = "simudo"

version_path = os.path.abspath(pkg + '/version.py')
version_info = {"__file__": version_path}
with open(version_path, 'rt', encoding="utf-8") as h:
    exec(compile(h.read(), version_path, 'exec'), version_info)
ver = version_info["__version__"].partition("~")[0]

# For Debian Buster / Ubuntu 20.04:
# apt install python3-{numpy,scipy,pint,matplotlib,meshio,yamlordereddictloader,sortedcontainers,cached-property,h5py,petsc4py}
# Still need to install: mpl_render, suffix_trees, generic_escape

setup(
    name             = pkg,
    version          = ver,
    description      = (
        "SIMUlation of Devices with Optics / SIMulateur Université D'Ottawa"),
    long_description = (
        "Semiconductor device model, including intermediate band materials "
        "and self-consistent optics."),
    author           = "Eduard Christian Dumitrescu",
    author_email     = "eduard.c.dumitrescu@gmail.com",
    license          = "LGPLv3",
    url              = "https://github.com/simudo/simudo",
    packages         = find_packages(),
    package_data     = {pkg: [
        'physics/*.py1',
        'fem/*.py1',
        'util/pint/*.txt']},
    data_files       = [("", ["LICENSE", "COPYRIGHT", "README.md"])],
    install_requires = [
        'numpy', 'scipy', 'pandas', 'Pint',
        # 'meshio', # optional dependency
        'matplotlib', 'mpl_render',
        'yamlordereddictloader', 'suffix_trees', 'generic_escape',
        'sortedcontainers',
        'cached_property',
        'h5py',
        # 'h5dedup',
        'petsc4py'],
    classifiers      = [
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries"])

