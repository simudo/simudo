"""Helpers shared by the material files.

:py:func:`thermal_velocity` is used by every material that defines a ``vth``;
:py:class:`Alloy` by the alloys only.

Composition interpolation
-------------------------

Every alloy in this directory interpolates its parameters between two parent
binaries using the same two formulas, and until now each file carried its own
copy of both.  The copies had drifted: only one of them tolerated a zero bowing
parameter, only one had a fallback for parameters missing from a parent, and
only one guarded the division in :py:meth:`Alloy.mobility_bowing`.

The two formulas, with ``x`` the mole fraction of the ``at_x1`` parent:

**Vegard's law** -- linear interpolation plus a quadratic bowing term::

    P(x) = P1 * x + P0 * (1 - x) + C * x * (1 - x)

``C = 0`` means no bowing.

**Mobility bowing** (Palankovski eq. 3.132) -- harmonic, with the bowing
parameter in the *denominator*::

    mu(x) = [ (1 - x) / mu0 + x / mu1 + x * (1 - x) / C ]^-1

Here **large ``C`` means negligible bowing** and ``C = 0`` is division by zero
-- the opposite convention to Vegard's law.  Mixing the two up is easy and
silent, so :py:meth:`Alloy.mobility_bowing` rejects ``C = 0`` explicitly.
"""

import numbers
from math import pi

__all__ = ['Alloy', 'thermal_velocity']


def thermal_velocity(U, T, m):
    """Mean thermal speed of carriers of effective mass *m*, in units of m_e.

    .. math::  \\langle v \\rangle = \\sqrt{8 k_B T / \\pi m^*}

    This is the **mean speed** of a Maxwell-Boltzmann distribution in a
    parabolic band, which is the average that belongs in an SRH capture
    coefficient :math:`c = \\sigma_{th} \\langle v \\rangle`
    ([Shockley1952a] 3.5) -- the capture rate goes as cross-section times how
    fast carriers move, irrespective of direction.

    Several other averages of the same distribution are also called "thermal
    velocity", and they differ only in prefactor -- all scale as
    :math:`\\sqrt{T/m^*}`:

    ===========================  ==========================  ================
    quantity                     value                       vs sqrt(kT/m*)
    ===========================  ==========================  ================
    mean speed (this function)   sqrt(8 kT / pi m*)          1.596
    RMS speed                    sqrt(3 kT / m*)             1.732
    most probable speed          sqrt(2 kT / m*)             1.414
    mean \\|v_z\\|                 sqrt(2 kT / pi m*)          0.798
    one-sided flux (Richardson)  sqrt(kT / 2 pi m*)          0.399
    ===========================  ==========================  ================

    The last is the one thermionic emission across a heterojunction needs, and
    it is exactly ``thermal_velocity(...) / 4``.  See
    :py:meth:`simudo.physics.heterojunction.ThermionicHeterojunction.emission_velocity`,
    which applies that factor rather than having the material files export a
    second ``vth``.

    Valid in the non-degenerate (Boltzmann) limit only; for a degenerate band
    the mean speed rises toward the Fermi velocity and the sqrt(T) law fails.

    Parameters
    ----------
    U : pint unit registry
    T : temperature, with units
    m : effective mass as a multiple of the free electron mass
        Which mass is correct depends on the band: the DOS mass for a single
        isotropic valley, the conductivity mass for an anisotropic one.
    """
    return (8 * U.boltzmann_constant * T / (pi * m * U.electron_mass)) ** 0.5


def _is_zero(value):
    """True for a zero of any dimension, including a pint ``Quantity``.

    A bare ``U("0")`` is dimensionless, and pint refuses to add a dimensionless
    zero to a quantity in eV or m/s.  Most call sites spell "no bowing" that
    way, so zero is detected by magnitude and the term dropped rather than
    added.
    """
    magnitude = getattr(value, 'magnitude', value)
    return isinstance(magnitude, numbers.Number) and magnitude == 0


class Alloy:
    """Interpolates material parameters between two parent binaries.

    Parameters
    ----------
    U : pint unit registry
        The alloy's registry, used only for the dimensionless ``1``.
    x : mole fraction
        Fraction of the *at_x1* parent.  May be a plain number, a pint
        ``Quantity``, or a UFL expression wrapped in one.
    at_x0, at_x1 : dict
        ``get_dict()`` of the parent recovered at ``x = 0`` and ``x = 1``
        respectively.  Getting these the wrong way round inverts the alloy
        silently, so name them at the call site.
    missing : {'raise', 'fallback'}
        What :py:meth:`vegard` does when *param* is absent from a parent.
        ``'raise'`` propagates the ``KeyError``; ``'fallback'`` returns the
        value from whichever parent does define it, uninterpolated, preferring
        *at_x0*.

    Examples
    --------
    ::

        alloy = Alloy(U, X, at_x0=GalliumArsenide, at_x1=AluminumArsenide)
        vegard = alloy.vegard
        mobility_bowing = alloy.mobility_bowing
    """

    def __init__(self, U, x, at_x0, at_x1, missing='raise'):
        if missing not in ('raise', 'fallback'):
            raise ValueError(
                f"missing must be 'raise' or 'fallback', not {missing!r}")
        self.U = U
        self.x = x
        self.at_x0 = at_x0
        self.at_x1 = at_x1
        self.missing = missing

    def vegard(self, param, C=0):
        """Interpolate *param* between the parents, bowing by *C*.

        ``C = 0`` means no bowing and is accepted whatever the units of
        *param*: the bowing term is dropped rather than added.
        """
        one = self.U("1")
        x = self.x
        try:
            value_0 = self.at_x0[param]
            value_1 = self.at_x1[param]
        except KeyError:
            if self.missing == 'raise':
                raise
            return self._single_parent_value(param)

        linear = value_1 * x + value_0 * (one - x)
        if _is_zero(C):
            return linear
        return linear + C * (one - x) * x

    def mobility_bowing(self, param, C):
        """Harmonic mobility interpolation, Palankovski eq. 3.132.

        *C* is the alloy-scattering parameter and appears in the
        **denominator**, so a large *C* means negligible bowing.  ``C = 0`` is
        the Vegard convention carried over by mistake and is rejected rather
        than left to divide by zero.
        """
        if _is_zero(C):
            raise ValueError(
                f"mobility_bowing({param!r}, C=0): the bowing parameter "
                f"appears in the denominator, so zero is division by zero. "
                f"For negligible bowing pass a large value instead -- the "
                f"satellite valleys in these files use 1e6 cm^2/V/s. Note that "
                f"C=0 means 'no bowing' only for vegard().")
        one = self.U("1")
        x = self.x
        return ((one - x) / self.at_x0[param]
                + x / self.at_x1[param]
                + (one - x) * x / C) ** -1

    def _single_parent_value(self, param):
        try:
            return self.at_x0[param]
        except KeyError:
            return self.at_x1[param]
