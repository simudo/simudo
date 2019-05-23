# encoding: utf-8
from setuptools import setup, find_packages
pkg = "simudo"
ver = '0.1.0'

with open(pkg+'/version.py', 'wt') as h:
    h.write('__version__ = "{}"\n'.format(ver))

setup(
    name             = pkg,
    version          = ver,
    description      = (
        "SIMUlation of Devices with Optics / SIMulateur Université D'Ottawa"),
    long_description = (
        "Semiconductor device model, including intermediate band materials "
        "and self-consistent optics."),
    author           = "Eduard Christian Dumitrescu",
    license          = "LGPLv3",
    url              = "https://github.com/simudo/simudo",
    packages         = find_packages(),
    package_data     = {pkg: ['example/*.ipynb']},
    install_requires = [
        'numpy', 'scipy', 'pandas',
        'matplotlib', 'mpl_render',
        'yamlordereddictloader', 'suffix_trees',
        'sortedcontainers',
        'cached_property',
        'h5py',
        # 'h5dedup',
        'petsc4py'],
    classifiers      = ["Programming Language :: Python :: 3 :: Only"])

