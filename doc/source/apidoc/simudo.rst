simudo package
==============

Organization
------------

The code is logically organized into several subpackages.

- :ref:`physics <package-physics>` contains all the code *specific* to
  the semiconductor device modeling problem, including optics and
  problem-specific solution methods.
- :ref:`fem <package-fem>` contains utility code that enhances
  or wraps FEniCS objects and methods, and implements *general*
  solution methods (like Newton's method) that are not specific to the
  device modeling problem. This code is used extensively by the
  ``physics`` package.
- :ref:`mesh <package-mesh>` contains mesh generation and manipulation
  code, as well as the topology specification module (e.g.,
  :py:class:`~simudo.mesh.topology.CellRegion`).
- :ref:`io <package-io>` contains data import/export code.
- :ref:`plot <package-plot>` contains plotting helpers, intended to be
  used to postprocess the exported data and turn it into pretty
  plots. This package can be used without importing dolfin.
- :ref:`util <package-util>` contains a variety of small utility
  modules that didn't fit anywhere else, like the logging filter
  setup.
- :ref:`pyaml <package-pyaml>` implements the source translator for
  "py1" files, and turns a YAML-based format into Python code.

Subpackages
-----------

.. toctree::

    simudo.physics
    simudo.mesh
    simudo.example
    simudo.io
    simudo.plot
    simudo.fem
    simudo.pyaml
    simudo.util

Submodules
----------

simudo.matplotlib\_backend module
---------------------------------

.. automodule:: simudo.matplotlib_backend
    :members:
    :undoc-members:
    :show-inheritance:


Module contents
---------------

.. automodule:: simudo
    :members:
    :undoc-members:
    :show-inheritance:
