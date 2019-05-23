
import logging
import re
from functools import partial
from itertools import chain

from cached_property import cached_property

import dolfin

from .. import pyaml
from ..fem import (
    MixedFunctionHelper, NewtonSolverMaxDu, Spatial, WithSubfunctionsMixin,
    opportunistic_assign)
from ..util import NameDict, SetattrInitMixin
from .problem_data_child import DefaultProblemDataChildMixin

pyaml_pdd = pyaml.load_res(__name__, 'poisson_drift_diffusion1.py1')

__all__ = [
    'SaveItem', 'GetToSaveMixin',
    'PoissonDriftDiffusion',
    'Poisson',
    'MixedPoisson',
    'Band',
    'NondegenerateBand',
    'IntermediateBand',
    'MixedQflNondegenerateBand',
    'MixedQflIntermediateBand',
    'MixedDensityNondegenerateBand']

class SaveItem(SetattrInitMixin):
    name = None
    value = None
    file_save = True
    solver_save = False

class GetToSaveMixin(object):
    def get_to_save(self):
        '''Get dict of functions needed to restore a solution from
file, or to save in a backtracking/adaptive solver.

The default implementation returns an empty NameDict.

Returns
-------
:py:class:`~.NameDict`:
    NameDict of :py:class:`SaveItem`.
'''
        return NameDict()

class PoissonDriftDiffusion(DefaultProblemDataChildMixin,
                            GetToSaveMixin, SetattrInitMixin):
    '''
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
bands: :py:class:`~.NameDict`
    Dictionary of band objects.
poisson: :py:class:`Poisson`
    Poisson part of the problem.
electro_optical_processes: :py:class:`~.NameDict`
    Dictionary of :class:`ElectroOpticalProcess` instances, including
    dark generation/recombination mechanisms like SRH.
mesh_util: :py:class:`~.mesh_util.MeshUtil`
    PDE utilities, many of which are (needlessly) mesh-specific.
mixed_function_helper: MixedFunctionHelper
    Mixed function and related registry.
'''
    _MixedFunctionHelper = MixedFunctionHelper

    def get_weak_form(self):
        return sum(x.get_weak_form() for x in self.get_subproblem_children())

    def get_essential_bcs(self):
        return list(chain.from_iterable(
            x.get_essential_bcs() for x in self.get_subproblem_children()))

    def get_solution_function(self):
        return self.mixed_function_helper.solution_function

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
    def mixed_function_space(self):
        return self.mixed_function_helper.solution_mixed_space

    def get_subproblem_children(self):
        return (self.poisson,) + tuple(self.bands)

    def get_subspace_descriptors_for_solution_space(self):
        '''Used by ``self.mixed_function_helper`` to build the mixed
        function space.'''
        return chain.from_iterable(
            x.solution_subspace_descriptors for x in
            self.get_subproblem_children())

    @property
    def kT(self):
        '''Shortcut for :math:`k_B T`, where :math:`k_B` is the
Boltzmann constant, and :math:`T` is the temperature
(:py:attr:`temperature`).'''
        u = self.unit_registry
        return (self.temperature * u.boltzmann_constant).to('eV')

    @property
    def temperature(self):
        '''Temperature as taken from :py:attr:`spatial`.'''
        return self.spatial.get("temperature")

    def easy_add_band(self, name, band_type=None, sign="auto"):
        '''Shortcut for instantiating and adding a band.

Parameters
----------
name: str
    Name of the band.

band\_type: str or class or None
    Can be a string, or a band class (inheriting from
    :py:class:`Band`). If it is a string, it serves as an alias as
    defined below:

    - "nondegenerate", "boltzmann" -> :py:class:`MixedQflNondegenerateBand`
    - "degenerate", "parabolic" -> :py:class:`MixedQflDegenerateBand`
      (not implemented yet)
    - "intermediate", "sharp" -> :py:class:`MixedQflIntermediateBand`

    If `None`, the type will be deduced from the band name:

    - "CB", "VB" -> "nondegenerate"
    - "IB" -> "intermediate"

sign: +1, -1, None, or "auto"
    The sign of the charge carrier for this band (negative for
    electrons, positive for holes).

    - If "auto", the sign will be deduced from the name of the band.
    - If `None`, the `sign` keyword argument will not be passed to be
      band object constructor.

Returns
-------
band: :py:class:`Band`
    Band object.
'''
        cls = band_type

        # deduce type from name
        if cls is None:
            if name == "CB" or name == "VB":
                cls = 'nondegenerate'
            elif name == 'IB':
                cls = 'intermediate'

        # deduce sign from name
        if sign == "auto":
            if name == "CB" or name == "IB":
                sign = -1
            elif name == 'VB':
                sign = 1
            else:
                raise ValueError(
                    "Don't know how to automatically set sign for this band, "
                    "either set it manually or pass sign=None to not set the "
                    "attribute on the Band object")

        # resolve string cls to class
        if cls == "nondegenerate" or cls == "boltzmann":
            cls = MixedQflNondegenerateBand
        elif cls == "degenerate" or cls == "parabolic":
            raise NotImplementedError(
                "degenerate parabolic bands not implemented yet")
        elif cls == "intermediate" or cls == "sharp":
            cls = MixedQflIntermediateBand

        if not issubclass(cls, Band): raise TypeError()

        kwargs = dict(key=name)
        if sign is not None:
            kwargs['sign'] = sign

        return self.add_band(cls, kwargs)

    def add_band(self, cls, kwargs):
        """Instantiate and add band. You probably want to use
:py:meth:`easy_add_band` instead.
"""
        kwargs.setdefault("pdd", self)

        goal = self.problem_data.goal
        if goal != 'full':
            cls = cls._base_band_class

        band = cls(**kwargs)
        self.bands.add(band)
        return band

    def easy_add_electrostatic_potential_BC(
            self, facet_region, value):
        self.bcs.add(...)
        raise NotImplementedError()

    def easy_add_electro_optical_process(self, cls, **kwargs):
        inst = cls(pdd=self, optical=self.problem_data.optical, **kwargs)
        self.electro_optical_processes.add(inst)
        return inst

    @cached_property
    def bands(self):
        return NameDict()

    @cached_property
    def spatial(self):
        return Spatial(parent=self)

    @cached_property
    def electro_optical_processes(self):
        return NameDict()

    @cached_property
    def poisson(self):
        goal = self.problem_data.goal
        if goal == 'local charge neutrality':
            cls = LocalChargeNeutralityPoisson
        elif goal == 'thermal equilibrium' or goal == 'full':
            cls = MixedPoisson
        else:
            raise AssertionError("bad goal")
        return cls(pdd=self)

    def easy_create_newton_solver(self):
        goal = self.problem_data.goal

        if goal == 'local charge neutrality':
            cls = LocalChargeNeutralitySolver
        elif goal == 'thermal equilibrium':
            cls = ThermalEquilibriumSolver
        else:
            raise AssertionError('bad goal')

        return cls.from_nice_obj(self)

    def easy_auto_pre_solve(self):
        solver = self.easy_create_newton_solver()
        name = self.problem_data.goal_abbreviated
        solver.logger = logging.getLogger("newton." + name)
        solver.solve()

    def initialize_from(self, other_pdd):
        fsr = self.function_subspace_registry.from_union(
            (self.function_subspace_registry,
             other_pdd.function_subspace_registry))

        self.poisson.initialize_from(
            other_pdd.poisson, function_subspace_registry=fsr)
        for key in self.bands.keys():
            self.bands[key].initialize_from(
                other_pdd.bands[key],
                function_subspace_registry=fsr)

    def get_to_save(self):
        r = super().get_to_save()
        for x in self.get_subproblem_children():
            r.update(x.get_to_save())
        r.add(SaveItem(
            name='pdd/mixed_solution',
            value=self.mixed_function_helper.solution_function,
            solver_save=True))
        return r

class LocalChargeNeutralitySolver(
        NewtonSolverMaxDu):
    ''' Newton solver for local charge neutrality '''
    def do_init_parameters(self):
        super().do_init_parameters()
        self.parameters.update(
            relaxation_parameter=0.5,
            maximum_iterations=2000, # WOW
            extra_iterations=5,
            relative_tolerance=1e-10,
            absolute_tolerance=1e-5,
            maximum_du=0.02,
            omega_cb=lambda self: 0.2 if self.iteration < 100 else 0.5,
        )

class ThermalEquilibriumSolver(
        NewtonSolverMaxDu):
    ''' Newton solver for thermal equilibrium '''
    def do_init_parameters(self):
        super().do_init_parameters()
        self.parameters.update(
            relaxation_parameter=0.5,
            maximum_iterations=500,
            relative_tolerance=1e-5,
            absolute_tolerance=1e-5,
            extra_iterations=15,
            omega_cb=lambda self: 0.4 if self.iteration < 30 else 0.9,
        )

class TypicalFromPDDMixin(object):
    '''Common shortcuts.'''

    @property
    def spatial(self):
        return self.pdd.spatial

    @property
    def unit_registry(self):
        return self.pdd.unit_registry

    @property
    def mesh_util(self):
        return self.pdd.mesh_util

    @property
    def mixed_function_solution_object(self):
        return self.pdd.mixed_function_helper

    def initialize_from(self, other, **kwargs):
        ''' by default, do nothing '''

class Poisson(pyaml_pdd.Poisson, WithSubfunctionsMixin, GetToSaveMixin,
              SetattrInitMixin, TypicalFromPDDMixin):
    '''Base class for Poisson part of the problem. You should instead look at one of the subclasses, such as :class:`MixedPoisson`.

Attributes
----------
key: str
    Unique key used to prefix subfunctions.
pdd: PoissonDriftDiffusion
    Parent object.
phi: pint.Quantity
    Electrostatic potential.
E: pint.Quantity
    Electric field.
rho: pint.Quantity
    Charge density.
thermal_equilibrium_phi: pint.Quantity
    Electrostatic potential at thermal equilibrium (i.e. when all qfls
    are equal to zero).
'''
    key = "poisson"

    @property
    def permittivity(self):
        return self.pdd.spatial.get("poisson/permittivity")

    @property
    def at_thermal_equilibrium(self):
        return self.pdd.problem_data.goal == 'thermal equilibrium'

    @cached_property
    def thermal_equilibrium_phi(self):
        if self.at_thermal_equilibrium:
            return self.phi # we *are* the thermal equilibrium
        else:
            mu = self.mesh_util
            return mu.unit_registry('V') * dolfin.Function(mu.space.DG2)

    def get_to_save(self):
        r = super().get_to_save()
        if not self.at_thermal_equilibrium:
            tphi = self.thermal_equilibrium_phi
            if tphi is not None:
                 r.add(SaveItem(
                     name='poisson/thermal_equilibrium_phi',
                     value=tphi,
                     solver_save=False))
        return r

class InitializeFromByAttributesMixin(object):

    def initialize_from(self, other, function_subspace_registry, **kwargs):
        mu = self.mesh_util
        log = logging.getLogger('assign.initialize_from')
        assign = partial(
            opportunistic_assign,
            function_subspace_registry=function_subspace_registry)
        for attr in self.initialize_from_attributes:
            source = getattr(other, attr)
            target = getattr(self, attr)
            if source is None: continue
            log.debug("{!r} assigning attribute {!r}, src={!r}, dst={!r}"
                      .format(self, attr, source, target))
            custom_assign_attr = "_initialize_from_custom_" + attr
            custom_assign = getattr(self, custom_assign_attr, None)
            if custom_assign is None:
                assign(source=source, target=target)
            else:
                custom_assign(
                    assign=assign,
                    source=source, target=target,
                    other=other,
                    function_subspace_registry=function_subspace_registry,
                    kwargs=kwargs)

class LocalChargeNeutralityPoissonMixin(
        pyaml_pdd.LocalChargeNeutralityPoissonMixin):
    initialize_from_attributes = ('phi',)

    def get_weak_form(self):
        return self.local_charge_neutrality_weak_form

    def get_essential_bcs(self):
        return self.essential_bcs

    @property
    def thermal_equilibrium_phi(self):
        return None

class MixedMethodPoissonMixin(pyaml_pdd.MixedPoissonMixin):
    initialize_from_attributes = ('phi', 'E', 'thermal_equilibrium_phi')

    mixed_debug_quad_degree_rho = 20

    def get_weak_form(self):
        return self.mixed_weak_form

    def get_essential_bcs(self):
        return self.essential_bcs

class LocalChargeNeutralityPoisson(
        InitializeFromByAttributesMixin,
        LocalChargeNeutralityPoissonMixin,
        Poisson):
    pass

class MixedPoisson(InitializeFromByAttributesMixin,
                   MixedMethodPoissonMixin,
                   Poisson):
    pass

class Band(pyaml_pdd.Band, WithSubfunctionsMixin, GetToSaveMixin,
           TypicalFromPDDMixin, SetattrInitMixin):
    '''Represents a band where its carriers are at thermal equilibrium
    with each other (such that a quasifermi level is well defined).

Attributes
----------
name: str
    Name of this band. Must be unique among
    :py:attr:`PoissonDriftDiffusion.bands` as it is used as a key in
    that :py:class:`~.NameDict`. By default simply an
    alias for :py:attr:`key`, so **you do not need to set it**.
key: str
    Unique key used to prefix subfunctions.
pdd: PoissonDriftDiffusion
    Parent object.
u:
    Carrier density in this band.
qfl:
    Quasi-Fermi level, aka imref, of carriers in this band.
j:
    Current density through this band.
sign:
    Sign of the charge carriers; ``-1`` for electrons, and ``+1`` for holes.
mobility:
    Band mobility. By default taken from ``.spatial``.
'''
    @property
    def name(self):
        return self.key

    # FIXME: call this by its proper name, which is probably one of "chemical potential" or "electrochemical potential"
    def phiqfl_to_u(self, phi_plus_qfl):
        raise NotImplementedError()

    def u_to_phiqfl(self, u):
        raise NotImplementedError()

    def qfl_to_u(self, qfl):
        return self.phiqfl_to_u(self.e_phi + qfl)

    def u_to_qfl(self, u):
        return self.u_to_phiqfl(u) - self.e_phi

    subfunctions_info = ()

    def get_weak_form(self):
        return 0

    def get_essential_bcs(self):
        return ()

    @property
    def g(self):
        return sum(proc.get_generation(self) for proc in
                   self.pdd.electro_optical_processes)

class NondegenerateBand(pyaml_pdd.NondegenerateBand, Band):
    r'''This represents a nondegenerate band obeying Boltzmann
statistics (instead of Fermi-Dirac as it would be for a proper
degenerate band), with an effective density of states
:py:attr:`effective_density_of_states` (:math:`N_0`) at an energy
level :py:attr:`energy_level` (:math:`E_0`).

The defining relationship is

.. math::

   u = N_0 \exp\Big[s\cdot\big(E_0 - (w + q\phi)\big) / kT\Big]

where

- :code:`s` is :py:attr:`Band.sign`,

- :code:`w` is :py:attr:`Band.qfl`,

- :code:`q` is the elementary charge,

- :code:`kT` is :py:attr:`PoissonDriftDiffusion.kT`,

- :math:`\phi` is :py:attr:`Poisson.phi`.

Attributes
----------
effective_density_of_states:
    Band effective density of states. By default taken from
    :py:attr:`~PoissonDriftDiffusion.spatial`.
energy_level:
    Band effective energy level. By default taken from
    :py:attr:`~PoissonDriftDiffusion.spatial`.
'''

    @property
    def effective_energy_level(self):
        ''' In a nondegenerate band, by default the
:py:attr:`energy_level` is assumed to include degeneracy
effects, so this just returns that attribute. '''
        return self.energy_level

class DegeneracyMixin(object):
    r'''This mixin implements state degeneracy for traps and
degenerate bands, by defining :py:attr:`effective_energy_level` in terms of
:py:attr:`energy_level` and :py:attr:`degeneracy`. The relationship is

.. math::

   E_{0} = E_{0}^{\textnormal{true}} - s\cdot kT\ln\frac{m_{0}}{m_{1}}

where

- :math:`E_{0}` is the effective energy level
  (:py:attr:`effective_energy_level`),
- :math:`E_{0}^{\textnormal{true}}` is the true energy level
  (:py:attr:`energy_level`),
- :math:`\frac{m_{0}}{m_{1}}` is the degeneracy factor
  (:py:attr:`degeneracy`), where
   - :math:`m_{0}` is the degeneracy factor for an empty band,
   - :math:`m_{1}` is the degeneracy factor for a full band.
- :math:`s` is the :py:attr:`~Band.sign`, ``+1`` for holes and ``-1``
  for electrons.

This change simplifies many recombination mechanisms by letting them
assume that the degeneracy factor :math:`m = 1` in the Fermi-Dirac
distribution :math:`f_1(x) = \frac{1}{m e^{x} + 1}`.

For the derivation, see `doc/ib.lyx`.

Parameters
----------
energy_level:
    True energy level of the trap, as defined for example in
    :py:attr:`IntermediateBand.energy_level`.
degeneracy:
    Trap degeneracy factor :math:`m_0/m_1` (see
    :py:class:`DegeneracyMixin`). By default taken from
    :py:attr:`~PoissonDriftDiffusion.spatial`.

Attributes
----------
effective_energy_level:
    Effective band energy level. Computed from
    :py:attr:`energy_level` and :py:attr:`degeneracy`.
'''

    @property
    def degeneracy(self):
        '''(parameter, see :py:class:`DegeneracyMixin`)'''
        return self.spatial.get(self.spatial_prefix + "degeneracy")

    @property
    def effective_energy_level(self):
        mu = self.mesh_util
        return (self.energy_level - self.sign * self.pdd.kT *
                mu.ln(self.degeneracy))

class IntermediateBand(DegeneracyMixin, pyaml_pdd.IntermediateBand, Band):
    r'''This represents an intermediate band with an energetically
sharp density of states, where all :py:attr:`number_of_states`
(:math:`N_I`) states are concentrated at an energy level
:py:attr:`energy_level` (:math:`E_I`).

The number of carriers obeys Fermi-Dirac statistics. The defining
relationship is therefore

.. math::

   u = N_I f_1\Big(s\cdot\big((w + q\phi) - E_I\big)/kT\Big)

where

- :code:`s` is :py:attr:`Band.sign`,

- :code:`w` is :py:attr:`Band.qfl`,

- :code:`q` is the elementary charge,

- :code:`kT` is :py:attr:`PoissonDriftDiffusion.kT`,

- :math:`\phi` is :py:attr:`Poisson.phi`, and

- :math:`f_1(x) = \frac{1}{e^{x} + 1}`.

Note that state degeneracy is handled through :py:class:`DegeneracyMixin`.

Attributes
----------
number_of_states:
    Number of states in the intermediate band. The carrier
    concentration in this band can never be higher than this
    number. By default taken from
    :py:attr:`~PoissonDriftDiffusion.spatial`.
energy_level:
    Band energy level. By default inherited from
    :py:attr:`DegeneracyMixin.energy_level`.
'''

class MixedQflBandMixin(pyaml_pdd.MixedQflBand):
    '''Mixed method for the drift-diffusion and continuity equations
using quasi-fermi level and current density as the dynamical
variables.
'''
    initialize_from_attributes = ('qfl', 'j')

    # debug default parameters
    mixedqfl_debug_fill_thresholds = (0., 0.)
    mixedqfl_debug_fill_from_boundary = True
    mixedqfl_debug_fill_with_zero_except_bc = False
    mixedqfl_debug_use_bchack = False
    mixedqfl_debug_quad_degree_super = 20
    mixedqfl_debug_quad_degree_g = 8

    @cached_property
    def mixedqfl_base_w(self):
        p = self.mesh_util
        u = self.unit_registry
        return dolfin.Function(p.space.DG0) * u.electron_volt

    def get_weak_form(self):
        return self.mixedqfl_weak_form

    def get_essential_bcs(self):
        return self.essential_bcs

    def get_to_save(self):
        r = super().get_to_save()
        k = self.name
        r.add(SaveItem(name=k+'/mixedqfl_base_w',
                       value=self.mixedqfl_base_w.magnitude,
                       solver_save=True))
        return r

    def _initialize_from_custom_qfl(
            self, assign, source, function_subspace_registry, **kwargs):
        function_subspace_registry.assign_scalar(
            self.mixedqfl_base_w.m, 0.0)
        assign(source=source,
               target=self.mixedqfl_delta_w)

    def _initialize_from_custom_j(
            self, assign, source, function_subspace_registry, **kwargs):
        assign(source=source,
               target=self.mixedqfl_j)

class MixedQflNondegenerateBand(
        InitializeFromByAttributesMixin,
        MixedQflBandMixin, NondegenerateBand):
    _base_band_class = NondegenerateBand

class MixedQflIntermediateBand(
        InitializeFromByAttributesMixin,
        MixedQflBandMixin, IntermediateBand):
    _base_band_class = IntermediateBand


class MixedDensityBandMixin(pyaml_pdd.MixedDensityBand):
    '''**DO NOT USE THIS!**

Mixed method for the drift-diffusion and continuity equations
using carrier density and current density as the dynamical
variables.
'''
    initialize_from_attributes = ('u', 'j')

    # debug default parameters
    mixed_debug_quad_degree_super = 20
    mixed_debug_quad_degree_g = 8

    def get_weak_form(self):
        return self.mixed_weak_form

    def get_essential_bcs(self):
        return self.essential_bcs

    def _initialize_from_custom_u(
            self, assign, source, function_subspace_registry, **kwargs):
        assign(source=source, target=self.mixed_u)

    def _initialize_from_custom_j(
            self, assign, source, function_subspace_registry, **kwargs):
        assign(source=source, target=self.mixed_j)

class MixedDensityNondegenerateBand(
        InitializeFromByAttributesMixin,
        MixedDensityBandMixin, NondegenerateBand):
    _base_band_class = NondegenerateBand

