"""Library of :class:`~simudo.physics.material.Material` subclasses and their
optical property data.

Intentionally empty of imports: each material module imports
``simudo.physics``, which pulls in dolfin, and the GUI deliberately inspects
these files with :mod:`ast` instead of importing them. Import the individual
modules directly (e.g. ``simudo.materials.galliumarsenide``) if you need them.
"""
