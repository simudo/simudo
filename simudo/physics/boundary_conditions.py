# Author: Claude Sonnet 4.6
"""
Robin-type boundary conditions for Simudo carrier bands.

Background
----------
A Robin (mixed) boundary condition specifies the normal current flux as a
linear function of the carrier density deviation from equilibrium:

    j·n = s * q * (u - u_eq)

where *s* is the surface recombination velocity [cm/s], *q* is the
elementary charge, *u* is the local carrier density, and *u_eq* is the
thermal-equilibrium carrier density.  The two existing Simudo BCs are the
limiting cases: ohmic (s → ∞, enforces u = u_eq as an essential BC) and
zero_current (s = 0, enforces j·n = 0 as an essential BC on j).

Weak-form derivation
--------------------
The carrier continuity equation, tested with v, contains ∫ v ∇·j dx.  On
an exterior boundary face where no essential BC is imposed on j, the BDM
normal flux is a free degree of freedom.  The Robin condition is imposed by
adding the surface integral

    ∫_{Γ_SRV} v * s * q * (u - u_eq) ds

directly to the residual — the standard FEniCS treatment of a Robin
condition (see Langtangen & Dokken, FEniCSx tutorial ch. 3, eq. for F(u,v)).

Because u is a nonlinear function of delta_w in Simudo's mixed method,
this term contributes to both the residual and the Newton Jacobian.
dolfin's automatic differentiation (via derivative()) handles the
Jacobian contribution automatically when it differentiates the full
residual with respect to the mixed function.

No essential BC on j or u should be registered for an SRV face; doing so
would override this weak-form condition.
"""

import logging
import numbers

import attr
import dolfin

from ..mesh.topology import FacetRegion

__all__ = ['SurfaceRecombination']


#: Allowed values of ``SurfaceRecombination.method``.
SRV_METHODS = ('continuity', 'continuity_corrected', 'drift_diffusion')


@attr.s
class SurfaceRecombination:
    """Robin BC imposing a finite surface recombination velocity.

    Imposes ``j·n = band.sign * s * q * (u - u_eq)`` weakly on an exterior
    face, leaving the BDM normal-flux DOF free (no essential BC).  Three
    weak-form placements are available, selectable via :py:attr:`method`:

    ``method='drift_diffusion'`` (**option 2 — default, recommended**)
        Impose the equivalent boundary QFL in the drift-diffusion
        (``ξ``-tested) row via the band's equation of state inverse
        (``u_to_qfl``).  Full Newton Jacobian via autodiff, no Picard.
        Documented in ``gui/notes/ROBIN_BOUNDARIES.md`` (Finding 3).

    ``method='continuity_corrected'``
        Add ``∫_Γ v · (srv_flux − j·n) ds`` to the continuity (``v``-tested)
        row, with ``srv_flux = band.sign · s · q · (u − u_eq)``.  Cancels the
        boundary flux that ``∫v∇·j`` already supplies and substitutes the
        Robin flux.  Converges; agrees with option 2 to within ~1% on the
        fourlayer SRV case but is several times slower.

    ``method='continuity'`` (**option 1, retained for documentation only**)
        Naive primal-Robin trick `∫_Γ v·srv_flux ds` in the v-row.  Double-
        counts the boundary flux in the BDM mixed method (``∫v∇·j`` already
        contains ``j·n``), leaving ``j·n`` under-constrained.  Does **not**
        converge — kept only to document the failure mode.

    All three placements use ``dolfin``'s automatic differentiation for a
    full Newton Jacobian; none uses a Picard freeze.  Call :py:meth:`register`
    after constructing.

    Parameters
    ----------
    band :
        The carrier band (must be a MixedQflBand instance).
    boundary : FacetRegion
        Exterior facet region where SRV is applied.
    srv :
        Surface recombination velocity as a pint Quantity with velocity
        units (e.g. ``1000 * U('cm/s')``).  A literal ``0`` is the
        no-recombination limit and is registered as a zero-current
        (``j·n = 0``) essential BC instead of a Robin term; see
        :py:meth:`register`.
    method : str
        ``'drift_diffusion'`` (default, option 2), ``'continuity_corrected'``,
        or ``'continuity'`` (option 1; does not converge — see above).

    Notes
    -----
    Do **not** call ``spatial.add_BC`` for the same band/face when using
    this class.  An essential BC on ``j`` or ``u`` registered for the same
    facet would suppress the Robin contribution.
    """

    band: object = attr.ib()
    boundary: FacetRegion = attr.ib()
    srv: object = attr.ib()
    method: str = attr.ib(default='drift_diffusion')

    def _srv_is_zero(self):
        """True if :py:attr:`srv` is a plain numeric zero.

        ``srv`` is a pint Quantity whose magnitude need not be a number. Only
        an immutable scalar counts as zero here; everything else falls through
        to the normal Robin treatment:

        * a spatially varying SRV wraps a UFL expression or
          :py:class:`dolfin.Function`, which has no value at registration time;
        * a :py:class:`dolfin.Constant` is *not* treated as zero even when it
          currently holds 0.0, because it is mutable — a stepper ramping it up
          from zero would otherwise be silently pinned to the zero-current BC
          chosen at registration.  :py:meth:`register` warns about that case.
        """
        magnitude = getattr(self.srv, 'magnitude', self.srv)
        return isinstance(magnitude, numbers.Number) and float(magnitude) == 0.0

    def _srv_is_zero_valued_constant(self):
        """True if :py:attr:`srv` is a ``dolfin.Constant`` currently holding 0.

        Distinct from :py:meth:`_srv_is_zero`, which covers immutable scalars
        only.  A zero-valued Constant is left on the Robin path -- see that
        method -- but is worth complaining about.
        """
        magnitude = getattr(self.srv, 'magnitude', self.srv)
        if not isinstance(magnitude, dolfin.Constant):
            return False
        values = magnitude.values()
        return len(values) == 1 and float(values[0]) == 0.0

    def register(self):
        """Inject the SRV Robin term into the band's weak form."""
        if self.method not in SRV_METHODS:
            raise ValueError(
                "SurfaceRecombination.method must be one of {}, got {!r}"
                .format(SRV_METHODS, self.method))

        band = self.band

        # s = 0 is the no-surface-recombination limit, and it is exactly the
        # existing zero_current BC: j·n = 0.  Handle it before dispatching on
        # method, because every placement mishandles it in its own way --
        # 'drift_diffusion' divides by s (UFL raises "Division by zero!"),
        # and 'continuity' silently contributes nothing, leaving the BDM
        # normal-flux DOF unconstrained.
        if self._srv_is_zero():
            mu = band.pdd.mesh_util
            zero_j = mu.unit_registry('A/cm^2') * mu.zerovec
            band.pdd.spatial.add_BC(
                '{}/j'.format(band.name), self.boundary, zero_j)
            return

        # A Constant holding zero stays on the Robin path, because it is
        # mutable and may be updated during the solve.  Warn anyway: the solve
        # it produces fails inside DOLFIN with a division by zero that says
        # nothing about srv, and the log is where the user will be looking.
        if self._srv_is_zero_valued_constant():
            logging.getLogger('physics.SurfaceRecombination').warning(
                "%s on band %r: srv is a dolfin.Constant holding 0. It is "
                "being kept as a live (mutable) value rather than treated as "
                "the no-surface-recombination limit, so the %r weak form will "
                "divide by zero and the solve is expected to fail. If you "
                "meant no surface recombination, pass a plain 0 or use the "
                "zero_current BC.",
                type(self).__name__, getattr(band, 'name', band), self.method)

        mu = band.pdd.mesh_util
        U = mu.unit_registry

        s_srv = self.srv
        q = U.elementary_charge
        u = band.u
        u_eq = band.thermal_equilibrium_u

        # Robin flux density [A / mesh_unit^2]: j·n = s_srv * q * (u - u_eq).
        # band.sign converts the outward flux to the correct sign convention:
        # electrons (sign=-1) flowing out reduce n, holes (sign=+1) flowing
        # out reduce p; both are recombination-like losses when u > u_eq.
        srv_flux = band.sign * s_srv * q * (u - u_eq)

        # Build the ds measure restricted to the SRV boundary facets.
        fvss = mu.mesh_data.evaluate_topology(self.boundary)
        ds_srv = sum(mu.ds(subdomain_id=fv) for fv, sign in fvss)

        if self.method == 'continuity':
            # Option 1: continuity (v-tested) row.
            # NOTE: structurally wrong for a flux BC in the BDM mixed method —
            # the boundary j·n already enters via ∫v∇·j, so this surface term
            # double-counts it and the j·n DOF is left under-constrained.
            # Retained only for comparison; does not converge.
            v = band.mixedqfl_v
            band.mixedqfl_surface_recombination_term += (
                ds_srv * v * srv_flux
            )
        elif self.method == 'continuity_corrected':
            # Option 1 with the double-count removed.  ∫_K v ∇·j already
            # supplies the boundary flux ∫_Γ v (j·n) (for the DG component),
            # so subtract it and substitute the desired Robin flux:
            #     v-row += ∫_Γ v (srv_flux − j·n) ds
            # The conservation row then enforces j·n|_Γ = srv_flux directly,
            # without an essential BC.  Both terms are nonlinear in the
            # solution (u via δw, and the live trial flux j·n), so dolfin's
            # autodiff supplies the full Newton Jacobian.
            v = band.mixedqfl_v
            j = band.mixedqfl_j
            n = mu.n
            jn = mu.dot(j, n)
            band.mixedqfl_surface_recombination_term += (
                ds_srv * v * (srv_flux - jn)
            )
        else:
            # Option 2: drift-diffusion (ξ-tested) row, natural-BC placement.
            #
            # In the BDM dual-mixed method the ξ-row's natural boundary term
            # ∫_Γ δw_Γ (ξ·n) ds imposes the *quasi-Fermi level* (a potential,
            # units eV), NOT the normal flux.  A finite-SRV Robin condition is
            # a *flux* condition, so it cannot be dropped into this slot
            # directly (that fails dimensionally).  Instead impose the
            # equivalent boundary QFL: invert the band equation of state so
            # that the carrier density the QFL produces yields the required
            # Robin flux,
            #
            #     j·n = band.sign * s * q * (u - u_eq)
            #     ⟹  u_target = u_eq + (j·n) / (band.sign * s * q)
            #     ⟹  w_target = u_to_qfl(u_target)         (FD inverse)
            #         δw_Γ     = w_target - base_w
            #
            # band.u_to_qfl is the (Fermi–Dirac) inverse, so this is valid for
            # degenerate bands too.  j·n is the live BDM trial flux, so the
            # term is nonlinear in the solution and dolfin's autodiff supplies
            # the full Newton Jacobian.
            xi = band.mixedqfl_xi
            j = band.mixedqfl_j
            n = mu.n
            base_w = band.mixedqfl_base_w

            jn = mu.dot(j, n)
            u_target = u_eq + jn / (band.sign * s_srv * q)
            w_target = band.u_to_qfl(u_target)
            delta_w_gamma = (w_target - base_w).to('eV')

            # Sign matches the existing drift_diffusion_bc_term
            # (−δw_bc·(ξ·n)), which imposes δw = δw_bc weakly.
            band.mixedqfl_surface_recombination_term += (
                -ds_srv * delta_w_gamma * mu.dot(xi, n)
            )
