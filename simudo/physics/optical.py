
import functools
import itertools
import logging
import re
from functools import partial
from itertools import chain

import numpy
from cached_property import cached_property

import dolfin
import ufl

from .. import pyaml
from ..fem import (
    MixedFunctionHelper, Spatial, WithSubfunctionsMixin, opportunistic_assign)
from ..util import NameDict, SetattrInitMixin
from .poisson_drift_diffusion import GetToSaveMixin, SaveItem
from .problem_data_child import DefaultProblemDataChildMixin

pyaml_optical = pyaml.load_res(__name__, 'optical1.py1')

__all__ = [
    'Optical',
    'OpticalField',
    'OpticalFieldMSORTE',
    'AbsorptionRangesHelper']

class Optical(DefaultProblemDataChildMixin, GetToSaveMixin, SetattrInitMixin):
    '''Contains optical part of the problem. Note that unlike PDD, the
optical fields are solved independent (self-consistently), so we do
not have a big mixed space made up of all the photon flux fields.

Parameters
----------
problem_data:
    See :py:attr:`problem_data`.
mesh_data: optional
    See :py:attr:`mesh_data`.

Attributes
----------
problem_data: :py:class:`~.problem_data.ProblemData`
    Parent problem.
mesh_data: :py:class:`~.mesh_data.MeshData`
    Mesh data. Taken from :py:attr:`problem_data` if not specified.
fields: :py:class:`.NameDict`
    Dictionary of :py:class:`OpticalField` objects.
spatial: :py:class:`.spatial.Spatial`
    Object managing boundary conditions and spatial quantities.
Phi_scale: :py:class:`dolfin.Constant`
    This controls the scaling factor applied to light photon flux
    boundary conditions. This is particularly useful when doing a
    light ramp-up.
'''

    @cached_property
    def Phi_pddproj_space(self):
        return self.problem_data.pdd.mesh_util.space.CG2

    @cached_property
    def Phi_scale(self):
        return dolfin.Constant(1.0) * self.unit_registry.dimensionless

    @cached_property
    def fields(self):
        return NameDict()

    @cached_property
    def spatial(self):
        return Spatial(parent=self)

    def easy_add_field(self, key, photon_energy, direction, solid_angle=None):
        if solid_angle is None:
            solid_angle = 4*numpy.pi
        self.fields.add(OpticalFieldMSORTE(
            optical=self,
            key=key, photon_energy=photon_energy,
            direction=direction, solid_angle=solid_angle))

    def get_to_save(self):
        r = super().get_to_save()
        for x in self.fields:
            r.update(x.get_to_save())
        return r

class OpticalField(SetattrInitMixin):
    '''A single optical field subproblem. This represents a scalar
photon flux at a single wavelength and direction of propagation.

Parameters
----------
optical: :py:class:`Optical`
    Parent object.
key: str
    Name of optical field. Must be unique.
photon_energy: :py:class:`pint.Quantity`
    Photon energy. Magnitude must be floating point constant.
direction: :py:class:`numpy.ndarray`
    Direction of propagation. Must be a normalized vector.
solid_angle: float
    Solid angle spanned by this optical field. The sum of all solid
    angles at a given wavelength must sum up to :math:`4\\pi`.

Attributes
----------
Phi: :py:class:`pint.Quantity` wrapping :py:class:`dolfin.Function`
    Photon flux field. This function is on the optical mesh.
alpha: :py:class:`pint.Quantity` wrapping expression
    Absorption/extinction coefficient. This quantity is on the optical
    mesh (and may be a result of a projection/interpolation from the
    PDD mesh).
g: :py:class:`pint.Quantity` wrapping expression
    Additional generation (excluding :code:`-alpha*Phi`). This quantity
    is on the optical mesh (and may be a result of a
    projection/interpolation from the PDD mesh).
Phi_pddproj: :py:class:`pint.Quantity` of :py:class:`dolfin.Function`
    Projected/interpolated version of :py:attr:`Phi` on the PDD
    mesh. To update the projection, call :py:meth:`update_output`.
Phi_pddproj_clipped: :py:class:`pint.Quantity` wrapping expression
    Clipped version of :py:class:`Phi_pddproj` that is always nonnegative.

Notes
-----
TODO: write down radiative transfer equation
'''
    _MixedFunctionHelper = MixedFunctionHelper

    def __init__(self, **kwargs):
        direction = kwargs.get('direction', None)
        if direction is not None:
            kwargs['direction'] = numpy.array(direction)
        super().__init__(**kwargs)

    @cached_property
    def mixed_function_helper(self):
        m = self._MixedFunctionHelper()
        m.mesh_util = self.mesh_util
        m.subspace_descriptors_for_solution_space = (
            self.get_subspace_descriptors_for_solution_space())
        self.function_subspace_registry.register(
            m.solution_mixed_space.function_space)
        return m

    @property
    def mixed_function_solution_object(self):
        return self.mixed_function_helper

    def get_subspace_descriptors_for_solution_space(self):
        return self.solution_subspace_descriptors

    def __str__(self):
        return self.key

    @property
    def name(self):
        return self.key

    @property
    def mesh_util(self):
        return self.optical.mesh_util

    @property
    def spatial(self):
        return self.optical.spatial

    @property
    def pdd(self):
        return self.optical.problem_data.pdd

    @property
    def unit_registry(self):
        return self.optical.unit_registry

    @property
    def function_subspace_registry(self):
        return self.optical.function_subspace_registry

    @cached_property
    def _hc_constant(self):
        u = self.unit_registry
        return (u.planck_constant * u.speed_of_light)

    @property
    def vacuum_wavelength(self):
        '''Computes vacuum wavelength based on
:py:attr:`photon_energy`. Can also be used to initialize that
property, with the conversion being done automatically.'''
        return (self._hc_constant / self.photon_energy).to('nm')

    @vacuum_wavelength.setter
    def vacuum_wavelength(self, value):
        self.photon_energy = (self._hc_constant / value).to('eV')

    @property
    def _Phi_scale(self):
        return self.optical.Phi_scale

    @property
    def Phi_pddproj_clipped(self):
        Phi = self.Phi_pddproj
        return dolfin.conditional(dolfin.lt(Phi.m, 0.0), 0.0, Phi.m) * Phi.u

    @cached_property
    def Phi_pddproj(self):
        return (dolfin.Function(self.optical.Phi_pddproj_space) *
                self.unit_registry("1/mesh_unit^2/s"))

    @cached_property
    def alpha(self):
        return (dolfin.Function(self.Phi_space) *
                self.unit_registry('1/mesh_unit'))

    @cached_property
    def g(self):
        return (dolfin.Function(self.Phi_space) *
                self.unit_registry('1/mesh_unit^3/s'))

    def get_alpha_pdd_expr(self):
        '''
This represents the extinction coefficient on the PDD mesh, as
extracted from the
:py:class:`.poisson_drift_diffusion.ElectroOpticalProcess` instances
in
:py:attr:`~.poisson_drift_diffusion.PoissonDriftDiffusion.electro_optical_processes`. '''
        alpha = self.pdd.mesh_util.zero * self.unit_registry('1/cm')
        for proc in self.pdd.electro_optical_processes:
            a = proc.get_alpha_by_optical_field(self)
            if a is not None:
                alpha = alpha + a
        return alpha

    def get_g_pdd_expr(self):
        '''
This represents the optical generation on the PDD mesh (excluding the
loss through the extinction coefficient), as extracted from the
:py:class:`.poisson_drift_diffusion.ElectroOpticalProcess` instances
in
:py:attr:`~.poisson_drift_diffusion.PoissonDriftDiffusion.electro_optical_processes`. '''
        g = self.pdd.mesh_util.zero * self.unit_registry('1/cm^3/s')
        for proc in self.pdd.electro_optical_processes:
            a = proc.get_optical_generation_by_optical_field(self)
            if a is not None:
                g = g + a
        return g

    def update_output(self):
        '''Update output quantities (e.g. by projecting onto PDD mesh).

This updates :py:attr:`Phi_pddproj`.
'''
        fsr = self.function_subspace_registry
        # TODO: don't assume pdd mesh = optical mesh
        opportunistic_assign(
            source=self.Phi,
            target=self.Phi_pddproj,
            function_subspace_registry=fsr)

    def update_input(self):
        '''Update input quantities (e.g. by projection onto optical mesh).

This updates :py:attr:`alpha` and :py:attr:`g`.
'''
        fsr = self.function_subspace_registry
        # TODO: don't assume pdd mesh = optical mesh
        opportunistic_assign(
            source=self.get_alpha_pdd_expr(),
            target=self.alpha,
            function_subspace_registry=fsr)
        opportunistic_assign(
            source=self.get_g_pdd_expr(),
            target=self.g,
            function_subspace_registry=fsr)

class OpticalFieldMSORTE(
        pyaml_optical.OpticalFieldMSORTE,
        WithSubfunctionsMixin,
        OpticalField,
        GetToSaveMixin):

    def get_weak_form(self):
        return self.msorte_form

    def get_essential_bcs(self):
        return self.essential_bcs

    def get_solution_function(self):
        return self.mixed_function_helper.solution_function

    def get_to_save(self):
        r = super().get_to_save()
        k = 'optical/' + self.name
        r.add(SaveItem(name=k+'/mixed_solution',
                       value=self.mixed_function_helper.solution_function,
                       solver_save=True))
        return r

class AbsorptionRangesHelper(SetattrInitMixin):
    '''
Parameters
----------
problem_data:
    Problem_data object.
inf_upper_bound:
    Upper bound on absorption to use instead of literal 'inf'. By
    default 100 eV.
energy_unit:
    By default, eV.
'''
    @property
    def unit_registry(self):
        return self.problem_data.unit_registry

    @property
    def energy_unit(self):
        return self.unit_registry('eV')

    @property
    def inf_upper_bound(self):
        return self.unit_registry('eV') * 100

    @property
    def bands(self):
        return self.problem_data.pdd.bands

    def _abs(self, x):
        return dolfin.conditional(x >= 0, x, -x)

    def get_transition_lower_bounds(self):
        '''Returns a dictionary of minimum transition lower
bounds. The keys in the dictionary are frozensets
:code:`{source_band_key, destination_band_key}`.'''
        r = {}
        Eunit = self.energy_unit
        abs = self._abs
        for a, b in itertools.combinations(self.bands,2):
            E_a = a.energy_level.m_as(Eunit)
            E_b = b.energy_level.m_as(Eunit)
            r[frozenset((a.name, b.name))] = abs(E_a - E_b) * Eunit
        return r

    def get_transition_bounds(self):
        '''Returns dictionary of `(lower, upper)` bounds for energy
transitions. See :py:meth:`get_transition_lower_bounds` for more.
'''
        lowers = self.get_transition_lower_bounds()
        upper_bound = ufl.as_ufl(self.inf_upper_bound.m_as(self.energy_unit))

        r = {}
        for k, lower in lowers.items():
            lst = []
            for k2, lower2 in lowers.items():
                if k == k2: continue

                # FIXME: this is WRONG if lower is exactly equal to lower2
                lst.append(dolfin.conditional(
                    ufl.as_ufl(lower2.m) < lower.m, upper_bound, lower2.m))

            if not lst:
                lst.append(upper_bound)

            upper = self._ufl_minimum(lst)
            r[k] = (lower, upper * lower.units)

        return r

    def _ufl_minimum(self, lst):
        if len(lst) > 10:
            raise NotImplementedError(
                "INEFFICIENT: rewrite this as a binary tree")
        elif len(lst) == 1:
            return lst[0]
        return functools.reduce(ufl.Min, lst)
