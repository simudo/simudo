
from functools import partial

import numpy as np
from cached_property import cached_property
from scipy.integrate import quad

__all__ = ['Blackbody']

class Blackbody():
    def __init__(self, temperature):
        unit_registry = temperature._REGISTRY

        self.unit_registry = unit_registry
        self.temperature = temperature

    @property
    def u(self):
        return self.unit_registry

    def blackbody_I_E(self, energy_unit):
        u = self.u
        h, c = u.h, u.c

        fac = (2*energy_unit**3/h**3/c**2)

        return (fac, lambda T, E: E**3/np.expm1(E/T))

    def blackbody_I_E_solar(self):
        ''' (energy unit, dimensionful prefactor, B_E in energy_unit) '''
        eu = self.u.eV
        fac, f = self.blackbody_I_E(eu)
        fac = fac.to('1/m**2/s')
        T = (self.temperature * self.u.boltzmann_constant).m_as(eu)
        return (eu, fac, partial(f, T))

    def blackbody_I_E_solar_on_earth(self):
        eu, factor, f = self.blackbody_I_E_solar()
        return (eu, factor*self.distance_factor*self.geometric_factor, f)

    def photon_flux_integral(self, eu_prefactor_B, E0, E1):
        eu, factor, f = eu_prefactor_B

        r = quad(lambda E: f(E)/E, E0.m_as(eu), E1.m_as(eu))[0]

        return (r*factor).to('1/m^2/s')

    def photon_flux_integral_on_earth(self, E0, E1):
        '''Assuming an observer at normal incidence on Earth, compute
the photon flux coming from a black sun in a given energy range.

Parameters
----------
E0:
    Lower bound on photon energy.
E1:
    Upper bound on photon energy.

Returns
-------
photon_flux: quantity
    Photon flux.
'''
        return self.photon_flux_integral(
            self.blackbody_I_E_solar_on_earth(), E0, E1).to('1/cm^2/s')

    def total_intensity_on_earth(self):
        eu, factor, f = self.blackbody_I_E_solar_on_earth()
        r = quad(f, 0, 20.)[0]
        return (r*factor*eu).to('W/m^2')

    @cached_property
    def geometric_factor(self):
        # because we're emitting into half of a hemisphere worth of solid angle
        factor = 2*np.pi

        # because there's a cosine factor for intensity that we're integrating over
        factor *= 0.5

        return factor

    @cached_property
    def distance_factor(self):
        earth_sun_distance = 149.6e6
        sun_radius = 695508
        return (sun_radius / earth_sun_distance)**2

    def non_overlapping_energy_ranges(self, energies, inf=None):
        '''Helper method for computing non-overlapping energy ranges.

Parameters
----------
energies: dict
    Mapping where keys are arbitrary (typically: names of transitions
    or optical fields), and values are energy range lower bounds.
inf: optional
    Upper limit on energy. By default, 20 eV.

Returns
-------
ranges: dict
    Mapping with the same keys as the :code:`energies` argument, and
    where the values are tuples :code:`(lower, upper)` such that the
    lower bound is equal to :code:`energies[k]`, and the upper bound
    is the smallest energy that is above :code:`lower`. If none
    exists, then :code:`upper` is the :code:`inf` argument.
'''
        if inf is None:
            inf = 20 * self.u.eV

        Es = list(energies.items())
        Es.sort(reverse=True, key=lambda x: x[1])

        result = {}
        prev = inf
        for k, E in Es:
            result[k] = (E, prev)
            prev = E

        return result
