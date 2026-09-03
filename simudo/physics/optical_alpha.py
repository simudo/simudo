
import numpy as np
from scipy.interpolate import interp1d

import dolfin

__all__ = [
    'make_alpha_from_table',
    'make_alpha_from_wavelength_table',
    'make_alpha_from_formula',
    'make_alpha_top_hat',
    'make_sigma_top_hat',
    'make_sigma_from_table',
    'make_sigma_from_wavelength_table',
    'make_sigma_from_formula',
]

# hc in eV·nm, used for wavelength ↔ energy conversion
_HC_EV_NM = 1239.8419843320028


def make_alpha_from_table(
        energies_eV,
        alphas,
        alpha_unit='1/cm',
        extrapolate_zero=True):
    '''Return a callable ``alpha_func(E: pint.Quantity) -> pint.Quantity``
that interpolates a tabulated absorption coefficient.

Parameters
----------
energies_eV : array-like of float
    Photon energies in eV, monotonically increasing.
alphas : array-like of float
    Absorption coefficient values at the corresponding energies,
    in units given by ``alpha_unit``.
alpha_unit : str, optional
    Pint unit string for the alpha values.  Default ``'1/cm'``.
extrapolate_zero : bool, optional
    If True (default), alpha is zero outside the table range.
    If False, the interpolant extrapolates linearly.

Returns
-------
callable
    ``alpha_func(E: pint.Quantity) -> pint.Quantity``

    At each call, evaluates the interpolant at the scalar photon energy
    ``E`` and returns ``dolfin.Constant(alpha_value) * unit``.
    The returned value is suitable as the result of ``spatial.get_callable``.
'''
    energies = np.asarray(energies_eV, dtype=float)
    values = np.asarray(alphas, dtype=float)

    fill = (0.0, 0.0) if extrapolate_zero else 'extrapolate'
    _f = interp1d(energies, values, kind='linear',
                  bounds_error=False, fill_value=fill)

    def alpha_func(E):
        U = E._REGISTRY
        E_eV = float(E.m_as('eV'))
        alpha_val = float(_f(E_eV))
        return dolfin.Constant(alpha_val) * U(alpha_unit)

    return alpha_func


def make_alpha_from_wavelength_table(
        wavelengths_nm,
        alphas,
        alpha_unit='1/cm',
        extrapolate_zero=True):
    '''Return a callable ``alpha_func(E: pint.Quantity) -> pint.Quantity``
that interpolates a tabulated absorption coefficient given by wavelength.

Parameters
----------
wavelengths_nm : array-like of float
    Vacuum wavelengths in nm.  Need not be sorted; conversion to energy
    will sort them correctly (long wavelength = low energy).
alphas : array-like of float
    Absorption coefficient values at the corresponding wavelengths,
    in units given by ``alpha_unit``.
alpha_unit : str, optional
    Pint unit string.  Default ``'1/cm'``.
extrapolate_zero : bool, optional
    Zero outside table range if True (default).

Returns
-------
callable
    ``alpha_func(E: pint.Quantity) -> pint.Quantity``

Notes
-----
Wavelength ``λ`` (nm) is converted to photon energy ``E`` (eV) via
``E = hc / λ``, where ``hc ≈ 1239.84 eV·nm``.
'''
    wl = np.asarray(wavelengths_nm, dtype=float)
    al = np.asarray(alphas, dtype=float)

    # long wavelength → low energy; sort by ascending energy
    order = np.argsort(wl)[::-1]   # descending wavelength = ascending energy
    energies_eV = _HC_EV_NM / wl[order]
    alphas_sorted = al[order]

    return make_alpha_from_table(
        energies_eV, alphas_sorted,
        alpha_unit=alpha_unit,
        extrapolate_zero=extrapolate_zero)


def make_alpha_from_formula(func):
    '''Wrap an analytic callable so it is documented as an alpha function.

Parameters
----------
func : callable
    ``f(E: pint.Quantity) -> pint.Quantity``
    Returns alpha at photon energy E.  May return a plain pint Quantity
    (wrapping a float or dolfin.Constant) or a live UFL expression for
    state-dependent alpha.

Returns
-------
callable
    The same callable, unchanged.

Examples
--------
>>> import dolfin
>>> def gaas_alpha(E):
...     U = E._REGISTRY
...     E_eV = float(E.m_as('eV'))
...     E_g = 1.42  # eV
...     A = 1.8e4   # cm^-1 eV^-0.5
...     val = A * (max(E_eV - E_g, 0.0) ** 0.5)
...     return dolfin.Constant(val) * U('1/cm')
>>> alpha_func = make_alpha_from_formula(gaas_alpha)
'''
    return func


# ---------------------------------------------------------------------------
# Sigma (optical cross-section) helpers for IBBeerLambert
# ---------------------------------------------------------------------------
# σ(E) has units cm² — it multiplies the IB filling factor (cm⁻³) to give α (cm⁻¹).
# The helpers below mirror the make_alpha_* API; pass sigma_unit='cm^2' by default.

def make_sigma_top_hat(E_low, E_high, sigma_0, sigma_unit='cm^2'):
    '''Return σ(E) = sigma_0 for E ∈ [E_low, E_high], 0 otherwise.

Parameters
----------
E_low, E_high : pint.Quantity or float
    Photon energy bounds (eV if plain float).
sigma_0 : pint.Quantity or float
    Cross-section value.  If a pint Quantity, its magnitude is extracted in
    ``sigma_unit``; if a plain float it is taken to be already in sigma_unit.
sigma_unit : str, optional
    Pint unit string for the cross-section.  Default ``'cm^2'``.

Returns
-------
callable
    ``sigma_func(E: pint.Quantity) -> pint.Quantity``

Notes
-----
For GUI default suggestions, good choices for E_low / E_high are the
energy-level difference between the two bands participating in the
transition.
'''
    # Pre-compute scalar bounds at construction time (E is always a scalar Quantity).
    if hasattr(E_low, 'm_as'):
        E_low_eV = float(E_low.m_as('eV'))
    else:
        E_low_eV = float(E_low)
    if hasattr(E_high, 'm_as'):
        E_high_eV = float(E_high.m_as('eV'))
    else:
        E_high_eV = float(E_high)

    if hasattr(sigma_0, 'm_as'):
        sigma_val = float(sigma_0.m_as(sigma_unit))
    else:
        sigma_val = float(sigma_0)

    def sigma_func(E):
        U = E._REGISTRY
        E_eV = float(E.m_as('eV'))
        val = sigma_val if E_low_eV <= E_eV <= E_high_eV else 0.0
        return dolfin.Constant(val) * U(sigma_unit)

    return sigma_func


def make_alpha_top_hat(E_low, E_high, alpha_0, alpha_unit='1/cm'):
    '''Return α(E) = alpha_0 for E ∈ [E_low, E_high], 0 otherwise.

The absorption-coefficient counterpart of :func:`make_sigma_top_hat`, for
processes that take α(E) directly (:class:`.BeerLambert`) rather than a
cross-section (:class:`.IBBeerLambert`).  A band-to-band process, for instance,
switches on at the band gap::

    spatial.add_callable_rule(
        'opt_cv/alpha_function', R.domain,
        make_alpha_top_hat(E_gap, 5.0, 1e4))

Parameters
----------
E_low, E_high : pint.Quantity or float
    Photon energy bounds (eV if plain float).
alpha_0 : pint.Quantity or float
    Absorption coefficient.  If a pint Quantity, its magnitude is extracted in
    ``alpha_unit``; if a plain float it is taken to be already in alpha_unit.
alpha_unit : str, optional
    Pint unit string.  Default ``'1/cm'``.

Returns
-------
callable
    ``alpha_func(E: pint.Quantity) -> pint.Quantity``
'''
    return make_sigma_top_hat(E_low, E_high, alpha_0, sigma_unit=alpha_unit)


# Sigma-from-table, wavelength-table, and formula are identical in structure
# to their alpha counterparts — the only difference is the default unit.
# Provide them as thin wrappers so callers don't have to remember to pass
# sigma_unit='cm^2'.

def make_sigma_from_table(energies_eV, sigmas, sigma_unit='cm^2',
                          extrapolate_zero=True):
    '''Like :func:`make_alpha_from_table` but for optical cross-sections (cm²).'''
    return make_alpha_from_table(energies_eV, sigmas,
                                 alpha_unit=sigma_unit,
                                 extrapolate_zero=extrapolate_zero)


def make_sigma_from_wavelength_table(wavelengths_nm, sigmas, sigma_unit='cm^2',
                                     extrapolate_zero=True):
    '''Like :func:`make_alpha_from_wavelength_table` but for cross-sections (cm²).'''
    return make_alpha_from_wavelength_table(wavelengths_nm, sigmas,
                                            alpha_unit=sigma_unit,
                                            extrapolate_zero=extrapolate_zero)


def make_sigma_from_formula(func):
    '''Like :func:`make_alpha_from_formula` but documents the callable as a
cross-section function σ(E) → pint.Quantity [cm²].'''
    return func
