
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

__all__ = ['AM15Spectrum']

_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'astm_g173_03.npz')


class AM15Spectrum:
    '''ASTM G-173-03 reference spectrum (AM1.5G global or AM1.5D direct).

    Wraps tabulated photon flux data (280–4000 nm) with a pint-aware interface
    consistent with the rest of Simudo.

    Parameters
    ----------
    unit_registry : pint.UnitRegistry
        Simudo unit registry (from ``make_unit_registry``).
    variant : str
        ``'global'`` for AM1.5G (global tilt, ~1000 W/m²) or
        ``'direct'`` for AM1.5D (direct + circumsolar, ~900 W/m²).
    concentration : float
        Dimensionless multiplier on the photon flux.  ``1.0`` = 1 sun.

    Examples
    --------
    >>> spec = AM15Spectrum(U, variant='global', concentration=1.0)
    >>> flux = spec.photon_flux_integral(U('1.1 eV'), U('4.0 eV'))
    >>> edges, E_rep, flux_per_bin = spec.bin_parameters(U('1.1 eV'), U('4.0 eV'), N=50)
    '''

    def __init__(self, unit_registry, variant='global', concentration=1.0):
        if variant not in ('global', 'direct'):
            raise ValueError(
                "variant must be 'global' or 'direct', got {!r}".format(variant))
        self.unit_registry = unit_registry
        self.variant = variant
        self.concentration = concentration
        self._load_data()

    @property
    def u(self):
        return self.unit_registry

    def _load_data(self):
        u = self.u
        d = np.load(_DATA_PATH)
        wl = d['wavelength_nm'] * u.nm

        key = 'global_tilt_W_m2_nm' if self.variant == 'global' else 'direct_W_m2_nm'
        irr = d[key] * u('W/m^2/nm')

        # Convert W/m²/nm → photons/cm²/s/eV.
        #
        # Number flux per unit wavelength:
        #   N(λ) = I(λ) / (hc/λ)     [photons/m²/s/nm]
        #
        # Change of variable  λ → E = hc/λ,  |dλ/dE| = hc/E²:
        #   N(E) = N(λ) · |dλ/dE| = I(λ) · λ / E²   [photons/m²/s/eV]

        hc = u.planck_constant * u.speed_of_light
        E = (hc / wl).to('eV')
        flux_density = (irr * wl / E**2).to('1/cm^2/s/eV')

        # Sort by ascending energy; strip to plain arrays for interp1d
        order = np.argsort(E.magnitude)
        self._E_eV = E.magnitude[order]
        self._flux_density = flux_density.magnitude[order]

        self._flux_interp = interp1d(
            self._E_eV, self._flux_density,
            kind='linear', bounds_error=False, fill_value=0.0)

        # Precompute CDF and energy-weighted CDF via cumulative trapezoid.
        # Avoids scipy.integrate.quad and associated integration warnings
        # on a piecewise-linear interpolant with thousands of kinks.
        dx = np.diff(self._E_eV)
        avg_f  = 0.5 * (self._flux_density[:-1] + self._flux_density[1:])
        avg_Ef = 0.5 * (self._E_eV[:-1] * self._flux_density[:-1] +
                        self._E_eV[1:]  * self._flux_density[1:])

        cdf  = np.concatenate([[0.0], np.cumsum(avg_f  * dx)])
        wcdf = np.concatenate([[0.0], np.cumsum(avg_Ef * dx)])

        kw = dict(kind='linear', bounds_error=False)
        self._cdf_interp  = interp1d(self._E_eV, cdf,
                                     fill_value=(0.0, float(cdf[-1])),  **kw)
        self._wcdf_interp = interp1d(self._E_eV, wcdf,
                                     fill_value=(0.0, float(wcdf[-1])), **kw)

    def _cdf(self, E_eV):
        return float(self._cdf_interp(E_eV))

    def _wcdf(self, E_eV):
        return float(self._wcdf_interp(E_eV))

    def photon_flux_density(self, E):
        '''Spectral photon flux density at photon energy ``E``.

        Returns a pint Quantity in ``1/cm^2/s/eV``, scaled by ``concentration``.
        '''
        val = float(self._flux_interp(E.m_as('eV'))) * self.concentration
        return val * self.u('1/cm^2/s/eV')

    def photon_flux_integral(self, E0, E1):
        '''Integrated photon flux over [E0, E1].

        Returns a pint Quantity in ``1/cm^2/s``, scaled by ``concentration``.
        '''
        val = max(self._cdf(E1.m_as('eV')) - self._cdf(E0.m_as('eV')), 0.0)
        return val * self.concentration * self.u('1/cm^2/s')

    def photon_flux_weighted_energy(self, E0, E1):
        '''Flux-weighted mean photon energy in [E0, E1].

        Returns a pint Quantity in ``eV``.  Used as the representative energy
        for a spectral bin.
        '''
        e0, e1 = E0.m_as('eV'), E1.m_as('eV')
        num = self._wcdf(e1) - self._wcdf(e0)
        den = self._cdf(e1)  - self._cdf(e0)
        if den <= 0.0:
            return 0.5 * (E0 + E1)
        return (num / den) * self.u.eV

    def equal_flux_bin_edges(self, E_min, E_max, N):
        '''Compute energy bin edges so each bin contains equal photon flux.

        Parameters
        ----------
        E_min, E_max : pint Quantity  — energy range
        N : int                       — number of bins

        Returns
        -------
        edges : pint Quantity array, shape (N+1,), in eV
        '''
        e0, e1 = E_min.m_as('eV'), E_max.m_as('eV')
        total_cdf = self._cdf(e1) - self._cdf(e0)

        if total_cdf == 0.0:
            return np.linspace(e0, e1, N + 1) * self.u.eV

        cdf0 = self._cdf(e0)
        target_per_bin = total_cdf / N

        edges_eV = [e0]
        for i in range(1, N):
            target = cdf0 + i * target_per_bin
            lo = edges_eV[-1]
            try:
                edge = brentq(
                    lambda E: self._cdf(E) - target,
                    lo, e1, xtol=1e-7 * (e1 - e0), maxiter=100)
            except ValueError:
                remaining = N - i
                edge = lo + (e1 - lo) / (remaining + 1)
            edges_eV.append(edge)
        edges_eV.append(e1)

        return np.array(edges_eV) * self.u.eV

    def bin_parameters(self, E_min, E_max, N):
        '''Return bin edges, representative energies, and per-bin photon flux.

        Parameters
        ----------
        E_min, E_max : pint Quantity
        N : int

        Returns
        -------
        edges     : pint Quantity array, shape (N+1,)  — bin edges in eV
        E_rep     : pint Quantity array, shape (N,)    — flux-weighted energy per bin
        flux      : pint Quantity array, shape (N,)    — photon flux per bin (1/cm²/s)
        '''
        edges = self.equal_flux_bin_edges(E_min, E_max, N)
        E_rep = np.array([
            self.photon_flux_weighted_energy(edges[i], edges[i+1]).m_as('eV')
            for i in range(N)]) * self.u.eV
        flux = np.array([
            self.photon_flux_integral(edges[i], edges[i+1]).m_as('1/cm^2/s')
            for i in range(N)]) * self.u('1/cm^2/s')
        return edges, E_rep, flux
