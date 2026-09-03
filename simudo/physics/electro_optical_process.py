
from cached_property import cached_property

import math

import dolfin
import ufl

import logging

from ..fem import expm1
from ..util import SetattrInitMixin

__all__ = [
    'ElectroOpticalProcess',

    'BeerLambert',
    'IBBeerLambert',
    'NonOverlappingTopHatBeerLambert',
    'NonOverlappingTopHatBeerLambertIB',

    'SRHRecombination',
    'NonRadiativeTrap',
    'ShockleyReadBand2BandTrap',
    'ShockleyReadTrap2Trap',

    'StaticGeneration',

    'DarkEOPMixin',
    'TrapEnergyLevelMixin',
    'TrapEOPMixin',
    'TwoBandEOPMixin',
    'RadiativeEOPMixin',
    'AbsorptionAndRadiativeRecombinationEOPMixin',
]

# Reference thermal energy used only to judge whether an SVR quadrature grid is
# too coarse to be accurate.  Not used in any physics: the actual kT is spatial.
_KT_REFERENCE_EV = 0.025852   # k_B * 300 K, in eV


class ElectroOpticalProcess(SetattrInitMixin):
    '''This class exists to represent both electro-optical process and
purely electronic (dark) processes, such as nonradiative
recombination.

Attributes
----------
pdd: PoissonDriftDiffusion
    Instance of :class:`.PoissonDriftDiffusion`.
optical: Optical
    Instance of :class:`.Optical`.
pre_iteration_hook: callable
    Implement this method in order to run code before each PDD Newton
    iteration. Useful for processes that need procedural code (cannot
    be expressed as UFL form).
pre_first_iteration_hook: callable
    This method is called before the *first* PDD Newton iteration.
post_iteration_hook: callable
    This method is called after each PDD Newton iteration.
'''
    pre_iteration_hook = None
    pre_first_iteration_hook = None
    post_iteration_hook = None

    # Spatial rule key suffixes (relative to the process name) that must be
    # set per layer for this process to work.  The GUI prepends the process
    # instance name to form the full key, e.g. ``nr_top`` + ``sigma_th`` →
    # ``nr_top/sigma_th``.
    #
    # Entries may contain ``{dst_band}`` or ``{src_band}`` as placeholders
    # that the GUI substitutes with the actual configured band names
    # (e.g. ``SRHRecombination`` needs ``{dst_band}/tau`` = ``CB/tau``).
    #
    # Subclasses override this with {suffix: unit} mapping.
    # Suffixes may contain {dst_band} or {src_band} as placeholders;
    # the GUI substitutes the actual band names and prepends the process
    # instance name to form the full spatial key, e.g. "nr_ci/sigma_th".
    required_spatial_params: dict = {}

    # Spatial callable rule suffixes (relative to the process name) that must
    # be registered via ``spatial.add_callable_rule`` for this process to work.
    # Mirrors ``required_spatial_params`` but for callables (functions of
    # photon energy, etc.) rather than scalar values.
    #
    # Entries are ``{suffix: {arg_units, return_units, description}}``.  The
    # GUI prepends the process instance name to ``suffix`` to form the full
    # spatial key, e.g. ``opt_cv/alpha_function``.
    #
    # A callable is considered "provided" when either
    #   (a) a material applied to a layer the process is active in includes
    #       an entry under the full key in its ``get_dict()`` with a
    #       recognised ``type`` tag (e.g. ``top_hat``, ``energy_table``), or
    #   (b) the process's own YAML config has an inline spec under the
    #       shorthand (e.g. an ``alpha:`` block on a BeerLambert process).
    required_spatial_callables: dict = {}

    # Band type this process is derived for, checked at construction:
    # True = must be a trap band, False = must not be, None = unconstrained.
    dst_band_is_trap = None
    src_band_is_trap = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.check_band_types()

    def check_band_types(self):
        '''Warn if a band is not of the type this process is derived for.

Bands whose ``is_trap_band`` is ``None`` are left alone: the class never
declared one, so there is nothing to check against.'''
        for role in ('dst_band', 'src_band'):
            want = getattr(self, role + '_is_trap')
            band = getattr(self, role, None)
            got = getattr(band, 'is_trap_band', None)
            if want is None or got is None or got == want:
                continue
            self.logger.warning(
                "%s: %s %r is %sa trap band; this process is derived for "
                "%s bands. Results may be physically meaningless.",
                type(self).__name__, role, band.name,
                "" if got else "not ",
                "trap (sharp)" if want else "dispersing")

    @cached_property
    def logger(self):
        return logging.getLogger('physics.' + self.name)

    @property
    def unit_registry(self):
        return self.mesh_util.unit_registry

    @property
    def mesh_util(self):
        return self.pdd.mesh_util

    def get_optical_generation_by_optical_field(self, optical_field):
        '''Exists as a separate method to make anisotropy possible.

By default just calls :code:`self.get_optical_generation`, and
multiplies by :py:attr:`~.optical.OpticalField.solid_angle`.'''

        return (self.get_optical_generation(optical_field.photon_energy) *
                optical_field.solid_angle)

    def get_optical_generation(self, photon_energy):
        ''' Override for luminescent coupling. By default returns zero. '''
        return 0 * self.unit_registry("1/cm^3/s")

    def get_alpha_by_optical_field(self, optical_field):
        '''Exists as a separate method to make anisotropy possible.

        By default just calls ``self.get_alpha``.'''

        return self.get_alpha(optical_field.photon_energy)

    def get_alpha(self, photon_energy):
        '''This method is called to get Beer-Lambert absorption
coefficient alpha at a particular photon energy. Must return a UFL
quantity on the PDD mesh, or :code:`None`.

**User class must implement this.**
        '''
        raise NotImplementedError()

    def get_quantum_yield_by_optical_field(self, optical_field):
        '''Exists as a separate method to make anisotropy possible.

        By default just calls ``self.get_quantum_yield``.'''
        return self.get_quantum_yield(optical_field.photon_energy)

    def get_quantum_yield(self, photon_energy):
        ''' Default ``quantum_yield=1``. '''
        return 1.0

    def get_band_generation_sign(self, band):
        '''As the generation process intensifies (e.g., light
intensity increases), does :code:`band` lose or gain more carriers?

The default implementation always returns :code:`None`. User classes
are expected to override this method.

Note: If you are implementing a two-band process, you almost certainly
want to inherit from :py:class:`TwoBandEOPMixin`.

Returns
-------
sign: object
    If the band participates in the process, returns :code:`+1` or
    :code:`-1` if it gains or loses carriers,
    respectively. Otherwise, return :code:`None`.'''
        return None

    @cached_property
    def _zero_generation(self):
        '''Constant for zero generation, with the right units.'''
        return self.unit_registry("1/mesh_unit^3/s") * 0.0

    def get_generation(self, band):
        return self.get_generation_user(band)

    def get_generation_user(self, band):
        '''This method is called to get the generation contribution to
band :code:`band`. Must return a UFL quantity on the PDD mesh.

**User class may override this.** By default, this method just calls
:py:meth:`get_generation_optical`.

Sign convention: **The returned value is positive or negative if
the band gains or loses carriers respectively.**

This method will be called for *every* band in the system. You are strongly
encouraged to use :py:meth:`get_band_generation_sign` to determine
whether the process causes a gain or a loss of carriers in band
:code:`band`, and whether the band participates in the process at all.
'''

        return self.get_generation_optical(band)

    def get_generation_optical(self, band):
        '''Compute carrier generation due to optical absorption, using
absorption coefficient given by :py:meth:`get_alpha_by_optical_field`
and quantum yield given by :py:meth:`get_quantum_yield`.'''

        sign = self.get_band_generation_sign(band)
        if sign is None:
            return self._zero_generation

        g = self._zero_generation
        for field in self.optical.fields:
            Phi = self.get_photon_flux_on_pdd_mesh(field)
            alpha = self.get_alpha_by_optical_field(field)

            # verbose = getattr(self, 'verbose', False)
            # if verbose:
            #     mu = self.mesh_util
            #     probe_y = 0.5
            #     probe_x = [0.01, 0.2, 0.5]

            #     # Phi
            #     # debug_probe_Phi = mu.get_debug_probe(Phi, "DG1")

            #     # alpha
            #     debug_probe_alpha = mu.get_debug_probe(alpha, "CG1")

            #     for xx in probe_x:
            #         # logging.info(f"{self.name} | Phi({xx})  | {debug_probe_Phi(xx, probe_y).to('1/cm^2/s')}")
            #         logging.info(f"{self.name} | abs alpha({xx})| {debug_probe_alpha(xx, probe_y).to('1/cm')} / cm")



            if alpha is not None:
                g = g + Phi*alpha*self.get_quantum_yield(field)*sign

        return g

    def get_photon_flux_on_pdd_mesh(self, optical_field):
        '''Returns optical photon flux, adapted onto PDD mesh. Use this
accessor instead of reaching inside :py:class:`.optical.OpticalField`
yourself.

Parameters
----------
optical_field: :py:class:`.optical.OpticalField`
    Optical field whose photon flux (clipped to be nonnegative) to get.
'''
        return optical_field.Phi_pddproj_clipped


class DarkEOPMixin():
    def get_alpha(self, photon_energy):
        return None


class TwoBandEOPMixin():
    '''Convenience mixin representing a generation process across
two bands.

Sign convention: **As the generation process intensifies, the
destination band gains more carriers.** The source band may gain or
lose carriers depending on its carrier type.

Attributes
----------
src_band: Band
    Source band (which may gain or lose a carrier depending on the sign).
dst_band: Band
    Destination band (which always gains a carrier).
'''

    def get_band_generation_sign(self, band):
        '''Provides a non-trivial implementation of
:py:meth:`ElectroOpticalProcess.get_band_generation_sign()` in the
case of two active bands.

See :py:class:`TwoBandEOPMixin` for the sign convention.'''
        if band == self.dst_band:
            return +1 # destination band gains carriers
        elif band == self.src_band:
            # what happens to source band depends on bands' carrier types
            return -(self.dst_band.sign * self.src_band.sign)
        else: # irrelevant band
            return None


class TrapEnergyLevelMixin():
    '''Minimal mixin providing trap energy level and SRH u1 statistics.

Shared by TrapEOPMixin (explicit trap band) and SRHRecombination
(implicit trap from spatial parameters, no Band object).

Methods
-------
get_trap_energy_level():
    Returns the trap energy level, either from the explicit trap_band
    (if set) or from the spatial parameter ``<name>/energy_level``.

get_u1(band):
    Quasi-Fermi level equilibrium carrier concentration in ``band``
    evaluated at the trap energy level.  Used in the SRH denominator.
'''

    def get_trap_energy_level(self):
        trap_band = self.trap_band
        if trap_band is None:
            return self.pdd.spatial.get('/'.join((self.name, 'energy_level')))
        else:
            return trap_band.effective_energy_level  # handles trap degeneracy

    def get_u1(self, band):
        E_I = self.get_trap_energy_level()
        if band.is_degenerate_band:
            u1 = band.phiqfl_to_u_nondegenerate(E_I)
        else:
            u1 = band.phiqfl_to_u(E_I)
        return u1


class TrapEOPMixin(TrapEnergyLevelMixin):
    '''Calculates properties involved in trapping between a trap band and
dispersing band, such as u1, tau, etc.

Requires an explicit trap_band Band object.

GUI cross-reference: any class that directly inherits this mixin will have
the "Trap band" selector shown for it in code/gui/panels/processes.py.
(SRHRecombination inherits TrapEnergyLevelMixin instead — it does not use
an explicit trap band and should NOT show the trap band selector.)

Attributes
----------
trap_band: Band
    Band doing the trapping.

reg_band: Band
    Non-trap band. e.g. CB or VB.'''

    def check_band_types(self):
        '''These processes need one trap band and one dispersing band, and
``reg_band`` identifies the dispersing one by elimination. Anything leaving it
undefined raises rather than warns: the process cannot be assembled at all.'''
        trap = getattr(self, 'trap_band', None)
        dst = getattr(self, 'dst_band', None)
        src = getattr(self, 'src_band', None)
        if trap is None:
            raise ValueError(
                f"{type(self).__name__} requires an explicit trap_band Band "
                f"object; got None.")
        if trap != dst and trap != src:
            raise ValueError(
                f"{type(self).__name__}: trap_band {trap.name!r} must be "
                f"either dst_band ({dst.name if dst else None!r}) or src_band "
                f"({src.name if src else None!r}); reg_band is undefined "
                f"otherwise.")
        if getattr(trap, 'is_trap_band', None) is False:
            self.logger.warning(
                "%s: trap_band %r is not a trap band; this process is designed for "
                "trapping into sharp states. Results may be physically meaningless.",
                type(self).__name__, trap.name)
        reg = self.reg_band
        if getattr(reg, 'is_trap_band', None) is True:
            self.logger.warning(
                "%s: reg_band %r is a trap band too; this process is derived "
                "for one trap band and one dispersing band. Results may be "
                "physically meaningless.", type(self).__name__, reg.name)

    @classmethod
    def easy_add_two_traps_to_pdd(
            cls, pdd, name_prefix, top_band, bottom_band, trap_band, **kwargs):

        CB = top_band
        VB = bottom_band
        IB = trap_band

        pdd.easy_add_electro_optical_process(
            cls, dst_band=CB, src_band=IB, trap_band=IB,
            name=name_prefix + '_top', **kwargs)
        pdd.easy_add_electro_optical_process(
            cls, dst_band=IB, src_band=VB, trap_band=IB,
            name=name_prefix + '_bottom', **kwargs)

    @cached_property
    def reg_band(self):
        if self.dst_band == self.trap_band:
            return self.src_band
        elif self.src_band == self.trap_band:
            return self.dst_band
        else:
            return None

    def get_trap_process_name(self, band):
        if band is None:
            name = self.name
        elif band == self.src_band:
            name = self.name + '_top'
        elif band == self.dst_band:
            name = self.name + '_bottom'
        else:
            raise ValueError()
        return name

    def trap_spatial_get(self, name, band=None):
        pname = self.get_trap_process_name(band)
        return self.pdd.spatial.get('/'.join((pname, name)))
    
    def band_spatial_get(self, name, band):
        return self.pdd.spatial.get('/'.join((band.name, name)))

    def get_trap_concentration(self, band=None):
        trap_band = self.trap_band
        if trap_band is None: # pull from spatial parameters
            return self.pdd.spatial.get('/'.join((self.name, 'N_t')))
        else: # just use trap band number of states
            return trap_band.number_of_states

    def get_tau(self, band=None):
        N_t = self.get_trap_concentration(band)
        c = self.get_capture_coefficient(band)
        return 1/(c * N_t)

    def get_sigma_th(self, band=None):
        return self.trap_spatial_get('sigma_th', band=band)

    def get_v_th(self, band=None):
        # Canonical spelling is 'vth'; 'v_th' is aliased onto it by
        # fem.spatial.canonicalize_key, so existing material files and project
        # YAML continue to resolve.
        try:
            return self.trap_spatial_get('vth', band=band)
        except AssertionError:
            # Thermal velocity is really a band property, and that is how the
            # material files supply it (CB/vth, VB/vth). Fall back to the band
            # when no process-scoped rule exists.
            return self.band_spatial_get('vth', band=band)

    def get_capture_coefficient(self, band=None):
        '''See [Shockley1952a] (3.5).'''
        sigma_th = self.get_sigma_th(band)
        v_th = self.get_v_th(self.reg_band) #replaced from `band`. v_th should be a band property, not a trap property
        return sigma_th * v_th

    def get_shockley_read_trap_generation(self, capture_coeff):

        IB = self.trap_band
        RB = self.reg_band

        if IB.sign == RB.sign:
            # same signs for trap and regular band, need other type of
            # carrier in the trap because we're look for empty states
            # to get trapped in
            trap_filling_factor = IB.number_of_states - IB.u
        else:
            # opposite signs, just return trap carrier concentration
            trap_filling_factor = IB.u

        kT = self.pdd.kT
        x = (RB.sign * (RB.qfl - IB.qfl)/kT).m_as('dimensionless')
        r = (-expm1(x)) * RB.u * trap_filling_factor * capture_coeff

        # `-r` corresponds to the generation rate in the reg_band.
        # However, the sign convention for generation in
        # `TwoBandEOPMixin` requires the destination band to always gain
        # carriers as the process intensifies (for generation
        # processes).
        if self.dst_band != RB:
            r = r * -(self.dst_band.sign * RB.sign)

        return -r


class RadiativeEOPMixin():
    ''' Calculates properties involved in radiative processes, such as
trapping or recombination.

Only works in non-degenerate condition
:code:`exp((E_photon_min - mu_fi)/kT) >> 1`. See [Strandberg2011],
page 3, under Eq. 6.
'''

    def get_strandberg_I(self):
        u = self.unit_registry
        h = u('1 planck_constant')
        c = u('1 speed_of_light')
        kT = self.pdd.kT
        mu = self.pdd.mesh_util

        n_r = 1
        K = (8 * dolfin.pi * n_r **2) / (h**3 * c**2)

        E_L, E_U = self.get_absorption_bounds()

        # verbose = getattr(self, 'verbose', False)
        # if verbose:
        #     mu = self.mesh_util
        #     probe_y = 0.5
        #     probe_x = [0.01, 0.2, 0.5]

        #     # E_L
        #     debug_probe_E_L = mu.get_debug_probe(E_L, "DG1")

        #     # E_U
        #     debug_probe_E_U = mu.get_debug_probe(E_U, "DG1")

        #     for xx in probe_x:
        #         logging.info(f"{self.name} | E_L({xx}) | {debug_probe_E_L(xx, probe_y).to('eV')}")
        #         logging.info(f"{self.name} | E_U({xx}) | {debug_probe_E_U(xx, probe_y).to('eV')}")

        # equivalent to flipping integration bounds if E_U < E_L
        sign = dolfin.conditional(dolfin.gt(
            E_U.magnitude,
            E_L.m_as(E_U.units)), +1, -1)

        # antiderivative of `E^2 exp(E/kT)`
        def F(E):
            return -kT*(2*kT**2 + 2*E*kT + E**2) * mu.exp(-E/kT)

        I = F(E_U) - F(E_L)

        ans = K * I * sign

        # to evaluate integrals
        # ans = (ans.m((0.0, 0.0)) * ans.units).to("1 / centimeter ** 2 / second")
        # print(self.src_band.key, self.dst_band.key, ans)

        return ans

    def get_absorption_bounds(self):
        from .optical import AbsorptionRangesHelper
        ab = AbsorptionRangesHelper(problem_data=self.pdd.problem_data)

        keyname = frozenset((self.src_band.key, self.dst_band.key))

        return ab.get_transition_bounds().get(keyname)

    def inside_absorption_bounds_conditional(self, E, value):
        E_L, E_U = self.get_absorption_bounds()
        E_L = E_L.m_as('eV')
        E_U = E_U.m_as('eV')

        E = E.m_as('eV')
        return value.units * dolfin.conditional(
            ufl.And(E_L <= E, E_U > E), value.magnitude, 0)


class AbsorptionAndRadiativeRecombinationEOPMixin():
    # The GUI detects classes that directly inherit this mixin via AST analysis
    # and shows the "Include radiative recombination: Yes/No" toggle for them.
    # If detection misses a class (e.g. indirect inheritance), the toggle won't
    # appear and no key is written to the YAML — the runner then uses this
    # class-level default (True), which is the physically correct fallback.
    enable_radiative_recombination = True

    def get_generation_user(self, band):
        '''Call :py:meth:`get_generation_optical` and
:py:meth:`get_radiative_recombination` and add together their results
accordingly.

Only include the radiative recombination process if
:py:attr:`enable_radiative_recombination` is true.
'''
        sign = self.get_band_generation_sign(band)

        if sign is None:
            return self._zero_generation

        g_abs = self.get_generation_optical(band) # absorption

        if self.enable_radiative_recombination:
            g_rad = self.get_radiative_recombination() * sign
        else:
            g_rad = self._zero_generation

        # probe the generations
        verbose = getattr(self, 'verbose', False)
        if verbose:
            mu = self.mesh_util
            probe_y = 0.5
            probe_x = [0.01, 0.2, 0.5]

            # g_rad
            debug_probe_rad = mu.get_debug_probe(g_rad, "DG1")

            # g_abs
            debug_probe_abs = mu.get_debug_probe(g_abs, "DG1")

            for xx in probe_x:
                logging.info(f"{self.name} | {band.name} | radiative ({xx}) | {debug_probe_rad(xx, probe_y).to('1/cm^3/s')}")
                logging.info(f"{self.name} | {band.name} | absorption({xx}) | {debug_probe_abs(xx, probe_y).to('1/cm^3/s')}")

        return g_abs - g_rad


class BeerLambert(
        AbsorptionAndRadiativeRecombinationEOPMixin,
        TwoBandEOPMixin,
        ElectroOpticalProcess):
    '''Beer-Lambert absorption with energy-dependent, per-region alpha(E).

Alpha is registered via
:py:meth:`~.fem.spatial.Spatial.add_callable_rule` with key
``<name>/alpha_function``, where ``<name>`` is the instance name of
this EOP.  Multiple ``BeerLambert`` instances can share the same
optical fields (overlapping absorption is physically correct).

Radiative recombination is disabled in v1 (``enable_radiative_recombination = False``).

Generation is pre-accumulated into a :class:`dolfin.Function` before
each Newton iteration so that Newton assembly sees a single function
rather than an N-term UFL sum over optical fields, where N is the number
of optical fields.

Example usage::

    spatial.add_callable_rule(
        'opt_cv/alpha_function', R.absorber,
        make_alpha_from_table(energies_eV, alphas))
    pdd.easy_add_electro_optical_process(
        BeerLambert, dst_band=CB, src_band=VB, name='opt_cv')
'''

    gui_hint = (
        "Beer-Lambert absorption with tabulated or analytic alpha(E). "
        "Register alpha per region with "
        "spatial.add_callable_rule('<name>/alpha_function', region, func). "
        "Overlapping absorption across multiple instances is supported."
    )

    name = 'beer_lambert'
    dst_band_is_trap = False
    src_band_is_trap = False
    required_spatial_params = {}
    required_spatial_callables = {
        'alpha_function': {
            'arg_units':    'eV',
            'return_units': '1/cm',
            'description':  'optical absorption coefficient α(E)',
        },
    }
    enable_radiative_recombination = True

    # Full 4π steradians by default (isotropic emission from device).
    # Override on an instance to change (e.g. 2π for single-hemisphere emission).
    svr_solid_angle = 4.0 * math.pi

    # Shockley-van Roosbroeck integration grid.  These are ordinary public
    # attributes: set them on the class, on an instance, or as keyword
    # arguments to the constructor (EOPs take kwargs via SetattrInitMixin).
    # The YAML runner is just one caller among others.
    #
    #   svr_integration_bins : number of uniform quadrature bins
    #   svr_E_min / svr_E_max: bounds; None means 0 eV / 4 eV.
    #
    # The integrand carries a blackbody weight exp(-E/kT), which peaks exactly
    # at the absorption threshold, so the bin width relative to kT is what sets
    # the accuracy.  Two errors, in order of size:
    #
    # 1. Misalignment.  Alpha turns on at the threshold, but the grid does not
    #    know where that is, so the edge falls partway through a bin and that
    #    bin is counted or dropped whole.  It is the largest bin in the sum, so
    #    the error is large and of either sign: sliding the threshold across one
    #    bin of the default grid swings the result by up to 78% for a top hat
    #    and 33% for a continuous alpha.  The defaults span 4 eV in 100 bins --
    #    dE/kT = 1.55 at 300 K -- and land wherever they land, so the shipped
    #    error is not a stable few percent but a lottery on the band gap.
    # 2. Midpoint bias.  What remains once the grid is aligned: a few percent,
    #    one-signed, second order in dE/kT.
    #
    # Set svr_E_min *at* the absorption threshold, which removes (1) outright,
    # and svr_E_max about 23 kT above it -- 0.6 eV at 300 K, proportionally more
    # when hotter -- which shrinks (2) by narrowing the bins.  Nothing beyond
    # that contributes: the discarded tail is 1e-8 of the integral or less, and
    # that holds for any alpha, since no physical absorption grows faster than
    # exp(+E/kT).  At the default 100 bins the two together take a silicon cell
    # from -8.5% to -0.2% (top hat) or +6.4% to +0.7% (continuous), at no extra
    # cost.  Raising svr_integration_bins helps both, at proportional expense,
    # which matters most for subclasses with svr_static_alpha = False.
    svr_integration_bins = 100
    svr_E_min = None
    svr_E_max = None

    # svr_static_alpha : if True, the SVR accumulator is computed once and cached.
    #                    Override to False in subclasses with state-dependent alpha.
    svr_static_alpha = True

    @cached_property
    def _svr_grid(self):
        """``(midpoint energies, bin width)`` for the SVR integral.

Computed on first use rather than in ``__init__``: it needs
:py:attr:`unit_registry`, which comes from ``pdd``, and it must see whatever
``svr_*`` values the caller set, in whatever order they were set.
"""
        U = self.unit_registry
        E0 = self.svr_E_min if self.svr_E_min is not None else 0.0 * U('eV')
        E1 = self.svr_E_max if self.svr_E_max is not None else 4.0 * U('eV')
        N = int(self.svr_integration_bins)
        if N < 1:
            raise ValueError(
                f"{type(self).__name__} {self.name!r}: svr_integration_bins "
                f"must be at least 1, got {N}")
        if E1 <= E0:
            raise ValueError(
                f"{type(self).__name__} {self.name!r}: svr_E_max ({E1}) must be "
                f"greater than svr_E_min ({E0})")
        dE = (E1 - E0) / N

        # The quadrature error is set by dE/kT, not by N alone: a wide window
        # at a fixed bin count is inaccurate, not just slow.  Warn rather than
        # adjust anything, since only the caller knows where alpha turns on.
        #
        # `pdd.kT` is a UFL expression -- temperature comes from the spatial
        # rules and may differ between regions -- so there is no scalar to
        # divide by here.  Judge the grid against a fixed 300 K reference and
        # say so.  That errs the safe way: below 300 K the true ratio is larger
        # than the reference, so a grid that warrants a warning still gets one.
        ratio = dE.m_as('eV') / _KT_REFERENCE_EV
        if ratio > 0.3:
            self.logger.warning(
                f"{type(self).__name__} {self.name!r}: SVR quadrature is "
                f"coarse -- dE/kT = {ratio:.2f} at 300 K over "
                f"E = [{E0.m_as('eV'):.3f}, {E1.m_as('eV'):.3f}] eV in {N} "
                f"bins. Bins this wide straddle the absorption threshold, "
                f"where the blackbody weight peaks, so the radiative rate can "
                f"be wrong by tens of percent in either direction. Put "
                f"svr_E_min *at* the threshold and svr_E_max ~23 kT above it "
                f"(0.6 eV at 300 K). Raising svr_integration_bins also works, "
                f"at proportional cost.")

        return tuple(E0 + (i + 0.5) * dE for i in range(N)), dE

    def get_alpha(self, photon_energy):
        try:
            return self.pdd.spatial.get_callable(
                '{}/alpha_function'.format(self.name), photon_energy)
        except AssertionError:
            return None

    @cached_property
    def _G_function(self):
        return (dolfin.Function(self.mesh_util.space.DG2)
                * self.unit_registry('1/mesh_unit^3/s'))

    @cached_property
    def _R_accumulator(self):
        return (dolfin.Function(self.mesh_util.space.DG2)
                * self.unit_registry('1/mesh_unit^3/s'))

    def get_refractive_index(self):
        """Return the spatially-varying refractive index from the spatial registry.

        Raises a clear error (instead of the low-level "no value rules for key"
        AssertionError) if ``refractive_index`` has not been set anywhere, since
        the Shockley-van Roosbroeck radiative-recombination integral cannot run
        without it.
        """
        try:
            return self.pdd.spatial.get('refractive_index')
        except AssertionError:
            raise ValueError(
                "Process {!r} ({}) has radiative recombination enabled "
                "(Shockley-van Roosbroeck), which requires a 'refractive_index' "
                "value, but none is set in any region. Set 'refractive_index' "
                "(dimensionless) in the domain region — or per layer — e.g. "
                "n_r ~ 3.5 for a typical semiconductor; or disable radiative "
                "recombination on this process."
                .format(getattr(self, 'name', '?'), type(self).__name__)
            ) from None

    def _compute_R_ufl(self):
        """Return UFL expression for B_0(x) = Σ_k α(E_k,x) · Φ_bb(E_k, T(x)) · ΔE.

        Uses the Boltzmann approximation Φ_bb ≈ K·n_r²·E²·exp(-E/kT)·ΔE, valid
        when E >> kT (E > 3kT ≈ 0.08 eV at 300 K — always true for semiconductor
        band-gap photons).  K = 2·Ω/(h³c²) where Ω = svr_solid_angle; the
        refractive index enters separately as the n_r² factor above, with
        n_r = spatial 'refractive_index'.  (K does *not* contain n_r².)
        """
        E_grid, dE = self._svr_grid
        U = self.unit_registry
        mu = self.mesh_util
        kT = self.pdd.kT
        n_r = self.get_refractive_index()
        h = U('planck_constant').to('eV*s')
        c = U('speed_of_light').to('cm/s')
        # K has units 1/(eV³·cm²·s);  n_r² is dimensionless UFL; K·n_r²·E²·exp(-E/kT)·ΔE → 1/(cm²·s)
        K = 2.0 * self.svr_solid_angle / (h ** 3 * c ** 2)
        r = self._zero_generation
        for E_k in E_grid:
            alpha_k = self.get_alpha(E_k)
            if alpha_k is None:
                continue
            phi_bb_k = K * n_r ** 2 * E_k ** 2 * mu.exp(-E_k / kT) * dE
            r = r + alpha_k * phi_bb_k
        return r

    def _update_G(self, solver):
        from ..fem.assign import opportunistic_assign
        fsr = self.mesh_util.function_subspace_registry
        g = self._zero_generation
        g_has_field = False
        for field in self.optical.fields:
            alpha = self.get_alpha_by_optical_field(field)
            if alpha is not None:
                Phi = self.get_photon_flux_on_pdd_mesh(field)
                qy = self.get_quantum_yield_by_optical_field(field)
                g = g + Phi * alpha * qy
                g_has_field = True
        if g_has_field:
            opportunistic_assign(
                source=g, target=self._G_function,
                function_subspace_registry=fsr)
        else:
            # No field contributed -- a dark run, or no alpha anywhere. Zero the
            # accumulator directly: dolfin.project rejects a pure-scalar zero
            # source with "expected a linear form for L". Same guard as
            # IBBeerLambert._update_sigma_phi.
            self._G_function.magnitude.vector()[:] = 0.0

        if self.enable_radiative_recombination:
            if not self.svr_static_alpha or not getattr(self, '_svr_accumulated', False):
                opportunistic_assign(
                    source=self._compute_R_ufl(),
                    target=self._R_accumulator,
                    function_subspace_registry=fsr)
                self._svr_accumulated = True

    pre_iteration_hook = _update_G
    pre_first_iteration_hook = _update_G

    def get_generation_optical(self, band):
        sign = self.get_band_generation_sign(band)
        if sign is None:
            return self._zero_generation
        return self._G_function * sign

    def get_radiative_recombination(self):
        UB = self.dst_band
        LB = self.src_band
        kT = self.pdd.kT
        x = ((UB.qfl - LB.qfl) / kT).m_as('dimensionless')
        return self._R_accumulator * expm1(x)


class IBBeerLambert(TrapEOPMixin, BeerLambert):
    '''Beer-Lambert absorption / radiative recombination for a transition
involving an intermediate band (IB), where alpha is state-dependent:

    α(E, x) = σ(E, x) · u_active(x) = σ(E, x) · N_IB(x) · f_IB(x)

where σ(E, x) is the optical cross-section (cm²), registered via
:py:meth:`~.fem.spatial.Spatial.add_callable_rule` with key
``<name>/sigma_function``, and f_IB = u_IB / N_IB.

Which states are active depends on the direction of the transition (live UFL):

- u_active = IB.u                          when IB and the regular band carry
                                            the same carrier sign (eg, IB and
                                            CB both electron-like): the
                                            *occupied* IB states are the ones a
                                            photon can excite out of
- u_active = IB.number_of_states - IB.u    otherwise (eg, IB electron-like and
                                            VB hole-like): the *empty* IB states
                                            are the ones that can receive a
                                            carrier

Both generation and radiative recombination are **Picard-frozen**: σΦ and
u_active are each projected into a :py:class:`dolfin.Function` and treated as
constant by UFL, so the Jacobian sees no contribution from these terms.
That is a fixed-point rather than a Newton linearisation — convergence is
linear rather than quadratic — but it is robust across the full intensity
range, whereas a live-UFL formulation gives a negative diagonal entry
∂G/∂(u_IB) = −σΦ that destabilises Newton once σΦ grows large enough to
overwhelm the stabilising terms.

``trap_band`` identifies the intermediate band; ``dst_band`` / ``src_band``
follow the usual BeerLambert convention (higher-energy band = dst_band).

Example usage::

    spatial.add_callable_rule(
        'opt_ib_cv/sigma_function', R.absorber,
        make_sigma_top_hat(0.7*U('eV'), 1.1*U('eV'), 1e-15*U('cm^2')))
    pdd.easy_add_electro_optical_process(
        IBBeerLambert,
        dst_band=CB, src_band=IB, trap_band=IB,
        name='opt_ib_cv')
'''

    # GUI hint displayed when this class is selected in the Processes panel.
    gui_hint = (
        "Beer-Lambert absorption / radiative recombination for a transition "
        "involving a sharp IB, with a state-dependent "
        "alpha = sigma(E) * N_IB * f_IB that tracks the band's occupancy. "
        "Lower E band and Higher E band are the two bands of the transition, "
        "one of which is the IB; Trap band = the IB. "
        "Requires an optical cross-section '<name>/sigma_function' per region: "
        "a top hat set below, a tabulated sigma(E) in the project file, or a "
        "callable provided by the material."
    )

    name = 'beer_lambert_ib'
    required_spatial_params = {}
    # IBBeerLambert needs σ(E) instead of α(E); α = σ · u_active is built internally.
    required_spatial_callables = {
        'sigma_function': {
            'arg_units':    'eV',
            'return_units': 'cm^2',
            'description':  'IB optical cross-section σ(E)',
        },
    }

    # Alpha depends on IB occupancy — B_0 must be updated each Newton step.
    svr_static_alpha = False

    # Radiative recombination is physically meaningful for IB transitions and
    # defaults to True (overrides BeerLambert's conservative False default).
    enable_radiative_recombination = True

    # Generation and recombination are  _u_active_function · (frozen scalar field),
    # with BOTH factors projected into dolfin Functions and held constant by UFL.
    #
    # The frozen fields (_SigmaPhi_function for generation, _B0norm_function for
    # SVR) are re-projected only before the FIRST Newton iteration of each solve
    # -- pre_iteration_hook is None and only pre_first_iteration_hook is set,
    # unlike the parent BeerLambert which refreshes every iteration.
    #
    # TODO: this is deliberate -- refreshing per iteration is believed to have
    # caused a convergence problem -- but the specifics were not recorded.
    # Worth re-testing whether per-iteration refresh now works, or improves
    # convergence, once there is a regression case to measure it against.
    # Not blocking for the 0.7 release.
    #
    # Keeping the sum over optical bins in a projected field also costs O(1) UFL
    # per quadrature point rather than O(N_bins).  (A naive live-UFL sum would be
    # O(N_bins) per quadrature point, scaling badly for large bin counts.)

    def _get_u_active(self):
        '''Density of IB states this transition can act on, cm⁻³ (live UFL).

        Occupied states when the IB and the regular band carry the same sign
        (a photon excites a carrier out of the IB), empty states otherwise (the
        IB receives one).
        '''
        IB = self.trap_band
        RB = self.reg_band
        if IB.sign == RB.sign:
            return IB.u
        else:
            return IB.number_of_states - IB.u

    def _get_f_IB(self):
        '''IB filling fraction u_active / N_IB (live UFL).'''
        return self._get_u_active() / self.trap_band.number_of_states

    @cached_property
    def _SigmaPhi_function(self):
        '''Frozen projection of Σ_k σ(E_k)·Φ_k.
        Units: [cm²] · [cm⁻²·s⁻¹] = [s⁻¹] (rate per unit carrier density).
        Multiply by u_active [cm⁻³] to obtain generation [cm⁻³·s⁻¹].'''
        return (dolfin.Function(self.mesh_util.space.DG2)
                * self.unit_registry('1/s'))

    @cached_property
    def _B0norm_function(self):
        '''Frozen projection of Σ_k σ(E_k)·φ_bb(E_k)·ΔE.
        Units: [cm²] · [cm⁻²·s⁻¹] = [s⁻¹] (B₀ per unit carrier density).
        Multiply by u_active [cm⁻³] to obtain recombination prefactor [cm⁻³·s⁻¹].'''
        return (dolfin.Function(self.mesh_util.space.DG2)
                * self.unit_registry('1/s'))

    @cached_property
    def _u_active_function(self):
        '''Frozen projection of u_active = IB.u or N_IB - IB.u.
        Units: cm⁻³.  Frozen before each Newton solve so the Jacobian sees zero contribution from the optical
        term (Picard linearization), which stabilises convergence when
        _SigmaPhi grows large enough to make the live Jacobian diagonal
        negative.'''
        return (dolfin.Function(self.mesh_util.space.DG2)
                * self.unit_registry('1/cm^3'))

    def _update_sigma_phi(self, solver):
        '''Project Σ σ·Φ and Σ σ·φ_bb into frozen Functions before each Newton step.'''
        from ..fem.assign import opportunistic_assign
        fsr = self.mesh_util.function_subspace_registry
        U = self.unit_registry
        zero_rate = 0.0 * U('1/s')  # accumulator unit: [s⁻¹] = σ[cm²] · Φ[cm⁻²s⁻¹]

        # ---- generation weight: Σ_k σ(E_k) · Φ_pddproj(E_k) ----
        g = zero_rate
        g_has_field = False
        for field in self.optical.fields:
            sigma = self.get_sigma(field.photon_energy)
            if sigma is None:
                continue
            Phi = self.get_photon_flux_on_pdd_mesh(field)
            qy = self.get_quantum_yield(field.photon_energy)
            g = g + Phi * sigma * qy
            g_has_field = True
        if g_has_field:
            opportunistic_assign(source=g, target=self._SigmaPhi_function,
                                 function_subspace_registry=fsr)
        else:
            # No optical field overlaps σ(E); set the frozen accumulator to
            # zero directly.  dolfin.project would reject a pure-scalar zero
            # source ("expected a linear form for L").
            self._SigmaPhi_function.magnitude.vector()[:] = 0.0

        # ---- SVR weight: Σ_k σ(E_k) · φ_bb(E_k, T) · ΔE ----
        if self.enable_radiative_recombination:
            mu = self.mesh_util
            kT = self.pdd.kT
            n_r = self.get_refractive_index()
            h = U('planck_constant').to('eV*s')
            c = U('speed_of_light').to('cm/s')
            K = 2.0 * self.svr_solid_angle / (h ** 3 * c ** 2)
            E_grid, dE = self._svr_grid
            r = zero_rate
            r_has_field = False
            for E_k in E_grid:
                sigma_k = self.get_sigma(E_k)
                if sigma_k is None:
                    continue
                phi_bb_k = K * n_r ** 2 * E_k ** 2 * mu.exp(-E_k / kT) * dE
                r = r + sigma_k * phi_bb_k
                r_has_field = True
            if r_has_field:
                opportunistic_assign(source=r, target=self._B0norm_function,
                                     function_subspace_registry=fsr)
            else:
                self._B0norm_function.magnitude.vector()[:] = 0.0

        # ---- Picard freeze: project current u_active into a frozen Function ----
        opportunistic_assign(source=self._get_u_active(),
                             target=self._u_active_function,
                             function_subspace_registry=fsr)

    pre_iteration_hook = None
    pre_first_iteration_hook = _update_sigma_phi

    def get_generation_optical(self, band):
        '''Picard-frozen generation: _u_active_function · _SigmaPhi_function.

        Both _SigmaPhi = Σ_k σ(E_k)·Φ_k and u_active are projected once before
        the first Newton iteration and held frozen during the solve.  The
        Jacobian therefore sees zero contribution from this term (Picard /
        fixed-point linearisation), which prevents the negative diagonal
        entry ∂G/∂(u_IB) = -_SigmaPhi from destabilising Newton when
        _SigmaPhi grows large enough to overwhelm the NR/relaxation
        stabilising contributions.  Convergence is slower (linear rather
        than quadratic) but robust across the full intensity range.
        '''
        sign = self.get_band_generation_sign(band)
        if sign is None:
            return self._zero_generation
        return self._SigmaPhi_function * self._u_active_function * sign

    def get_radiative_recombination(self):
        '''SVR recombination, Picard-frozen: _u_active_function · _B0norm · expm1(ΔqFL/kT).

        u_active is frozen (Picard) to remove the destabilising ∂u_active/∂u_IB
        Jacobian contribution.  UFL still differentiates through expm1(x),
        giving the full QFL-split Jacobian for the radiative term.
        '''
        UB = self.dst_band
        LB = self.src_band
        kT = self.pdd.kT
        x = ((UB.qfl - LB.qfl) / kT).m_as('dimensionless')
        return self._B0norm_function * self._u_active_function * expm1(x)

    def get_sigma(self, photon_energy):
        '''Return σ(E) from the spatial callable registry, or None.'''
        try:
            return self.pdd.spatial.get_callable(
                f'{self.name}/sigma_function', photon_energy)
        except AssertionError:
            return None

    def get_alpha(self, photon_energy):
        sigma = self.get_sigma(photon_energy)
        if sigma is None:
            return None
        # sigma * u_active, NOT sigma * N_IB * f_IB: the two are equal only
        # where N_IB > 0. Outside a limited-extent IB, N_IB is zero and the
        # cancelling division gives 0*(0/0) = NaN, which reaches the optical
        # matrix and makes the solve fail with DIVERGED_PC_FAILED.
        return sigma * self._get_u_active()


class NonOverlappingTopHatBeerLambert(
        AbsorptionAndRadiativeRecombinationEOPMixin,
        RadiativeEOPMixin,
        TwoBandEOPMixin,
        ElectroOpticalProcess):
    '''Absorption or radiative recombination between src_band and dst_band.
For absorption, src_band=VB and dst_band=CB, and vice versa for
recombination.

Includes radiative recombination by default.
'''

    # GUI hint displayed when this class is selected in the Processes panel.
    gui_hint = (
        "Photon absorption / radiative recombination between two standard bands "
        "(e.g. VB → CB). "
        "Lower E band = valence-like band (e.g. VB); "
        "Higher E band = conduction-like band (e.g. CB). "
        "Requires spatial parameter ‘alpha’ per region."
    )

    name = 'beer_lambert'
    dst_band_is_trap = False
    src_band_is_trap = False
    required_spatial_params = {'alpha': '1/cm'}

    def get_constant_alpha(self):
        return self.pdd.spatial.get('/'.join((self.name, 'alpha')))

    def get_alpha(self, photon_energy):
        '''Return constant alpha, clipped by absorption bounds.'''
        return self.inside_absorption_bounds_conditional(
            photon_energy, self.get_constant_alpha())

    def get_radiative_recombination(self):
        alpha = self.get_constant_alpha()

        strb_I = self.get_strandberg_I()
        beta = 1/self.pdd.kT

        CB = self.dst_band
        VB = self.src_band

        x = (beta * (CB.qfl - VB.qfl)).m_as('dimensionless')

        # debug probes
        verbose = getattr(self, 'verbose', False)
        # if verbose:
        #     mu = self.mesh_util

        #     # alpha
        #     # debug_probe_alpha = mu.get_debug_probe(alpha, "CG1")

        #     # strb_I
        #     # debug_probe_strb_I = mu.get_debug_probe(strb_I, "DG1")

        #     # x
        #     # debug_probe_x = mu.get_debug_probe(x, "DG1")

        #     #expm1(x)
        #     # debug_probe_expm1x = mu.get_debug_probe(expm1(x), "DG1")

        #     probe_x = [0.01, 0.2, 0.5]
        #     probe_y = 0.5
        #     # for xx in probe_x:
        #         # logging.info(f"{self.name} | rad alpha({xx})      | {debug_probe_alpha(xx, probe_y).to('1/cm')}")
        #         # logging.info(f"{self.name} | strb_I({xx})    | {debug_probe_strb_I(xx, probe_y).to('1/cm^2/s')}")
        #         # logging.info(f"{self.name} | beta * delta_qfl ({xx})         | {debug_probe_x(xx, probe_y)} dimensionless")
        #         # logging.info(f"{self.name} | expm1(x)({xx})  | {debug_probe_expm1x(xx, probe_y)} dimensionless")

        return alpha * strb_I * expm1(x)
        # return alpha * strb_I * expm1x


class NonOverlappingTopHatBeerLambertIB(
        AbsorptionAndRadiativeRecombinationEOPMixin,
        RadiativeEOPMixin,
        TrapEOPMixin,
        TwoBandEOPMixin,
        ElectroOpticalProcess):
    '''Absorption between intermediate band and regular band.

Includes radiative recombination by default.

Based on [Strandberg2011].'''

    # GUI hint displayed when this class is selected in the Processes panel.
    gui_hint = (
        "Photon absorption / radiative recombination between a standard band and a "
        "sharp IB. "
        "Lower E band and Higher E band are the regular bands on each side "
        "(e.g. VB and CB); Trap band = the IB. "
        "Requires spatial parameter 'sigma_opt' per region."
    )

    name = 'beer_lambert_IB'
    required_spatial_params = {'sigma_opt': 'cm^2'}

    def get_radiative_recombination(self):

        # pseudo capture coefficient
        capture_coeff = self.get_radiative_pseudo_capture_coefficient()
        g = self.get_shockley_read_trap_generation(capture_coeff)

        return -g

    def get_constant_sigma_opt(self):
        return self.pdd.spatial.get('/'.join((self.name, 'sigma_opt')))

    def get_sigma_opt(self, photon_energy):
        '''Return constant sigma_opt, clipped by absorption bounds.'''
        return self.inside_absorption_bounds_conditional(
            photon_energy, self.get_constant_sigma_opt())

    def get_alpha(self, photon_energy):

        IB = self.trap_band
        RB = self.reg_band

        sigma = self.get_sigma_opt(photon_energy)

        if IB.sign == RB.sign:
            # same signs for trap and regular band, the more carriers
            # in trap band the more this absorption process will
            # happen
            trap_filling_factor = IB.u
        else:
            # opposite signs, need complementary carrier type for trap
            trap_filling_factor = IB.number_of_states - IB.u

        return sigma*trap_filling_factor

    def get_radiative_pseudo_capture_coefficient(self):
        ''' See `eq:rad-u-form` in `doc/ib.lyx`. '''

        RB = self.reg_band

        sigma_opt = self.get_constant_sigma_opt()
        strandberg_I = self.get_strandberg_I()

        u1 = self.get_u1(RB)

        return sigma_opt * strandberg_I / u1


class SRHRecombination(
        DarkEOPMixin,
        TrapEnergyLevelMixin,
        TwoBandEOPMixin,
        ElectroOpticalProcess):
    '''SRH recombination. The destination band is the one *losing*
carriers through recombination.

Typically, dst_band=CB and src_band=VB.

Uses TrapEnergyLevelMixin (not TrapEOPMixin) — SRH has no explicit trap
band object; the trap energy level and lifetime tau come from spatial
parameters.  The GUI therefore does not show a "Trap band" selector for
this class.'''

    # GUI hint displayed when this class is selected in the Processes panel.
    gui_hint = (
        "Shockley-Read-Hall recombination through midgap traps with no explicit trap band. "
        "Lower E band = valence-like band (e.g. VB); "
        "Higher E band = conduction-like band (e.g. CB). "
        "Trap band should be left as (none) — SRH uses per-band lifetime parameters instead. "
        "Requires spatial parameters '{dst_band}/tau' and '{src_band}/tau' per region."
    )

    name = 'SRH'
    dst_band_is_trap = False
    src_band_is_trap = False
    # Keys are {proc_name}/{band_name}/tau; placeholders substituted by GUI.
    required_spatial_params = {
        '{dst_band}/tau': 's',
        '{src_band}/tau': 's',
        'energy_level': 'eV',   # trap energy level; midgap ≈ (E_CB + E_VB) / 2
    }
    trap_band = None  # SRH has no explicit trap Band; energy_level comes from spatial

    def get_tau(self, band):
        return self.pdd.spatial.get('/'.join((self.name, band.name, 'tau')))

    def get_degeneracy_factor(self, band):
        '''Degeneracy factor ``gamma = u / u_nondegenerate`` for `band`.

        Returns ``1`` for a nondegenerate band.

        Guarded against underflow: the Boltzmann density underflows to
        exactly zero in double precision once ``(E_F - E_C)/kT`` falls
        below roughly -745, which is reachable for a wide-gap band under
        reverse bias. Gamma tends to 1 in that limit -- Fermi-Dirac and
        Boltzmann statistics coincide when carriers are sparse -- so fall
        back to 1 rather than dividing by zero.
        '''
        if not band.is_degenerate_band:
            return 1
        u_nondegenerate = band.u_nondegenerate
        return dolfin.conditional(
            dolfin.gt(u_nondegenerate.m, 0.0),
            (band.u / u_nondegenerate).m_as('dimensionless'),
            1.0)

    def get_generation_user(self, band):
        sign = self.get_band_generation_sign(band)
        if sign is None:
            return self._zero_generation

        CB = self.dst_band
        VB = self.src_band

        if CB.sign == VB.sign:
            self.logger.warning('SRH recombination between two bands of the same sign is not physical')

        # When one of the bands doesn't exist in the full domain, these taus
        # come back as 0 outside the proper domain. In those cases, we want
        # r to be zero (as happens in the equivalent case for SR trapping),
        # but the expression divides by zero. Put in a conditional to resolve.
        CB_tau = self.get_tau(CB)
        VB_tau = self.get_tau(VB)

        CB_u1 = self.get_u1(CB)
        VB_u1 = self.get_u1(VB)

        gamma_C = self.get_degeneracy_factor(CB)
        gamma_V = self.get_degeneracy_factor(VB)

        # In the nondegenerate limit, we can use n_i^2 = CB.thermal_equilibrium_u * VB.thermal_equilibrium_u
        # But when at least one of the bands is degenerate, we need to use the Boltzmann definition of ni^2
        # Eg = CB.energy_level - VB.energy_level
        # kT = self.pdd.kT
        # mu = self.mesh_util
        # n_i_squared = CB.effective_density_of_states * VB.effective_density_of_states * mu.exp(-Eg/kT)
        n_i_squared = CB_u1 * VB_u1  #both calculated correctly with nondegenerate statistics

        rr = ((VB.u*CB.u - gamma_C*gamma_V*n_i_squared)/
             ((CB.u + gamma_C*CB_u1)*VB_tau +
              (VB.u + gamma_V*VB_u1)*CB_tau))

        # Return zero when tau's are all zero.
        mu = self.mesh_util
        r_units = rr.units
        r = dolfin.conditional(
                dolfin.gt((mu.abs(CB_tau) + mu.abs(VB_tau)).m, 0),
                rr.m,
                0
            )*r_units

        g = -r

        return g * sign


class NonRadiativeTrap(
        DarkEOPMixin,
        TrapEOPMixin,
        TwoBandEOPMixin,
        ElectroOpticalProcess):
    '''Shockley Read trapping process. Typically, trap_band = IB.
If you do not have an explicit :py:class:`Band` object for the trap
concentration, use :py:class:`SRHRecombination` instead.

See `doc/ib.lyx`.
'''

    # GUI hint displayed when this class is selected in the Processes panel.
    gui_hint = (
        "Non-radiative trapping between a standard band (SB) and a sharp IB. "
        "Use when you have an explicit Band object for the IB (use SRHRecombination otherwise). "
        "Trap band = IB. One of lower or higher E band must also be the IB. "
        "Requires spatial parameter 'sigma_th' per region; thermal velocity comes from the band (CB/vth, VB/vth), usually from the material. "
        "To implement a full recombination, you need two NonRadiativeTrap processes "
        "(eg, for CB-IB and IB-VB) "
    )

    name = 'nonradiative'
    # No 'vth' here: thermal velocity is a band property (CB/vth, VB/vth),
    # which is how the material files supply it. get_v_th() checks for a
    # process-scoped rule and falls back to the band, so prompting for one
    # per process asks the user for something they do not need to set.
    required_spatial_params = {'sigma_th': 'cm^2'}

    def get_generation_user(self, band):
        g_sign = self.get_band_generation_sign(band)
        if g_sign is None:
            return self._zero_generation

        # capture coefficient
        capture_coeff = self.get_capture_coefficient()
        g = self.get_shockley_read_trap_generation(capture_coeff)

        return g * g_sign


class ShockleyReadBand2BandTrap(
        TwoBandEOPMixin,
        DarkEOPMixin,
        ElectroOpticalProcess):
    ''' Shockley-Read-like trapping process between the receiving (R) and sending (D) bands.
    Similar to :py:class:`NonRadiativeTrap` (which is Shockley-Read trapping from a band to trap states with a single energy).

    If the signs of both bands are the same, s_D = s_R, then the rate is

        U = [1 - exp(s_D (w_D - w_R)/kT)] u_D/tau

    w_k is the qfl of band k
    u_D is the carrier concentration in band D
    where we approximate that the number of receiving states is independent of w_R
    Derivation assumes that the receiving band (R) is essentially entirely empty,
    so only the population of the sending band (D) enters the rate.

    The energy-averaged (usually phenomenological) capture time tau must be defined.
    Convention: only the value of tau in the **dst_band** will be used.
    If the instance of this EOP has :py:attr:`name` `SR_B2B` and the D band :py:attr:`name` is `DB`
    tau is found from `spatial["SR_B2B/DB/tau"]`


    If s_D = -s_R, then the rate is

        U = [1 - exp(s_D (w_D - w_R)/kT)] u_D u_R <c>

    <c> is the average capture rate [volume^-1 time^-1] for the process.
    If the instance of this EOP has :py:attr:`name` `SR_B2B` and the D band :py:attr:`name` is `DB`
    <c> is found from `spatial["SR_B2B/DB/capture_rate"]`

    (In the previous case, 1/tau = <c> u_pR
    where u_pR is the concentration of carriers of opposite sign in band R)

    Following the standard sign convention, dst_band = D, src_band = R
    '''

    # GUI hint displayed when this class is selected in the Processes panel.
    gui_hint = (
        "Shockley-Read-like interband transfer between two standard bands (no explicit or implicit trap band). "
        "Lower E band = lower-energy band "
        "Higher E band = higher-energy band "
        "Requires '{dst_band}/tau' (for same-sign bands) or "
        "'{dst_band}/capture_rate' (for opposite-sign bands)."
    )

    name = 'SR_B2B'
    dst_band_is_trap = False
    src_band_is_trap = False
    # same-sign bands use {dst_band}/tau; opposite-sign bands use {dst_band}/capture_rate
    required_spatial_params = {'{dst_band}/tau': 's', '{dst_band}/capture_rate': 'cm^3/s'}

    def get_generation_user(self, band):
        g_sign = self.get_band_generation_sign(band)
        if g_sign is None:
            return self._zero_generation

        if self.dst_band.sign == self.src_band.sign:
            g = self.get_SR_generation_same_sign()
        else:
            g = self.get_SR_generation_diff_sign()

        return g * g_sign

    def get_SR_generation_same_sign(self):
        tau = self.pdd.spatial.get('/'.join((self.name, self.dst_band.name, "tau")))
        RB = self.src_band
        DB = self.dst_band

        kT = self.pdd.kT
        x = (DB.sign * (DB.qfl - RB.qfl)/kT).m_as("dimensionless")
        rr = -expm1(x) * DB.u / tau

        # Return zero when tau is zero.
        mu = self.mesh_util
        r_units = rr.units
        r = dolfin.conditional(
                dolfin.gt(mu.abs(tau).m, 0),
                rr.m,
                0
            )*r_units

        return -r

    def get_SR_generation_diff_sign(self):
        c = self.pdd.spatial.get('/'.join((self.name, self.dst_band.name, "capture_rate")))
        RB = self.src_band
        DB = self.dst_band

        kT = self.pdd.kT
        x = (DB.sign * (DB.qfl - RB.qfl)/kT).m_as("dimensionless")
        r = -expm1(x) * DB.u * RB.u * c

        return -r


class ShockleyReadTrap2Trap(
        TwoBandEOPMixin,
        DarkEOPMixin,
        ElectroOpticalProcess):
    ''' Shockley-Read-like trapping process between the receiving (R) and sending (D) bands
    where both are trap (ie, sharp) bands.
    Similar to :py:class:`NonRadiativeTrap` (which is Shockley-Read trapping from a
    dispersing band to trap states with a single energy).

    Regardless of the signs of the band, the net transfer rate from D to R is

        U = [1 - exp( -(w_D - w_R)/kT)] f_D (1-f_R) n_R N_D c

    where
        c is the capture coefficient [vol/time] for the trapping process
        N_D is the number of states [1/vol] in the D band
        N_R is the number of states [1/vol] in the R band

    If the instance of this EOP has :py:attr:`name` `SR_T2T` and the D band :py:attr:`name` is `DB`
    c is found from `spatial["SR_T2T/DB/capture_rate"]`

    Following the standard sign convention, dst_band = D, src_band = R
    '''

    # GUI hint displayed when this class is selected in the Processes panel.
    gui_hint = (
        "Shockley-Read-like inter-trap transfer between two sharp IB states. "
        "Both Lower E band and Higher E band must be sharp (IB-type) bands. "
        "Lower E band = source / receiving (R) band; "
        "Higher E band = destination / sending (D) band. "
        "Requires '{dst_band}/capture_rate' per region."
    )

    name = 'SR_T2T'
    dst_band_is_trap = True
    src_band_is_trap = True
    required_spatial_params = {'{dst_band}/capture_rate': 'cm^3/s'}

    def get_generation_user(self, band):
        g_sign = self.get_band_generation_sign(band)
        if g_sign is None:
            return self._zero_generation

        g = self.get_SR_generation()

        return g * g_sign

    def get_SR_generation(self):
        c = self.pdd.spatial.get('/'.join((self.name, self.dst_band.name, "capture_rate")))
        RB = self.src_band
        DB = self.dst_band
        s_R = RB.sign
        s_D = DB.sign
        N_DB = DB.number_of_states
        N_RB = RB.number_of_states
        kT = self.pdd.kT
        x = ((DB.qfl - RB.qfl)/kT).m_as("dimensionless")

        # N_D f_D = u_D if s_D = -1 (electrons) and N_D - u_D if s_D = +1 (holes).
        # Together, those can be written as:
        #     N_D f_D = (1+s_D)/2 * N_DB - s_D * u_D

        N_Df_D = (1+s_D)/2 * N_DB - s_D * DB.u
        N_R_fpR = (1-s_R)/2 * N_RB + s_R * RB.u
        # one_minus_f_R = (1-s_R)/2 + s_R * RB.u/RB.number_of_states

        r = -expm1(-x) * N_Df_D * N_R_fpR * c

        return -r


class StaticGeneration(TwoBandEOPMixin, ElectroOpticalProcess):
    '''Static generation process. Typically, dst_band=CB and src_band=VB.
    Uses a generation rate that does not change during the simulation.
    
    Generation rate is defined in `pdd.spatial['name/generation']`,
    where the default `name` is `static`.
    '''

    # GUI hint displayed when this class is selected in the Processes panel.
    gui_hint = (
        "Not usable in the GUI. Consider using BeerLambert or IBBeerLambert. "
        "If using Python, you can apply arbitrary static absorption profiles "
        "using UniformGenerationAdaptiveStepper"
    )

    name = 'static'

    def get_generation_user(self, band):
        sign = self.get_band_generation_sign(band)
        if sign is None:
            return self._zero_generation

        return self.pdd.spatial.get('/'.join((self.name, 'generation'))) * sign