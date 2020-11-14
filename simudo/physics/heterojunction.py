import attr
import dolfin
import numpy as np

from .poisson_drift_diffusion import Band, NondegenerateBand
from ..mesh.topology import FacetRegion


# TODO: rename this to the type of heterojunction that it is (what is
# the name of the model?)
@attr.s
class ThermionicHeterojunction:
    """
    Thermionic emission heterojunction BC valid with parabolic bands in the 
    Boltzmann approximation.

    Parameters
    ----------
    band: :py:class:`.Band`
        Semiconductor band on which the BC is applied. Only
        MixedQflBand is currently supported.
    boundary: :py:class:`.FacetRegion`
        Boundary on which to implement heterojunction boundary condition.

    Notes
    -----

    See V. Palankovski (2004),   eq. 3.72 and
    K. Yang, J. R. East, G. I. Haddad, Solid State Electronics v.36 (3) p.321-330 (1993)
    K. Horio, H. Yanai, IEEE Trans. Elec. Devices v.37(4) p.1093-1098 (1990)

    For a conduction band BC, band.spatial must have attribute 
    "CB/vth" in the barrier region. Similarly for other bands.

    Unlimited carrier flow from low to barrier region can be resolved, but due
    to precision issues, cannot resolve Delta_w producing carrier flows from 
    barrier to low region with Delta_w larger than 
    |ln(1e-16)| * kT in double precision

    TODO: Add warning/comment that only works for nondegenerate bands
    """

    band: Band = attr.ib()
    boundary: FacetRegion = attr.ib()

    @property
    def vth(self):
        return self.band.spatial.get(self.band.spatial_prefix + "vth")

    def register(self):
        """
        Apply boundary condition onto the band.
        """

        band = self.band
        mu = band.pdd.mesh_util
        U = mu.unit_registry

        if not isinstance(band, NondegenerateBand):
            raise NotImplementedError("Degenerate bands not yet supported")

        # delta: correction for tunneling through the barrier. not implemented.
        delta = 0.0
        sign = band.sign
        # Thermal velocities
        # These should be calculated from DOS effective mass (Palakovski 3.75)
        vth = self.vth
        # Eb is barrier height. should be positive. Use its sign to determine which
        # side is the low and which side is the barrier
        Eb = -sign * (mu.pside(band.energy_level) - mu.mside(band.energy_level))

        def Eb_fconditional(fun1, fun2):
            """Return a function conditional on the sign of Eb """

            def func(expr):
                u = expr.units
                return u * dolfin.conditional(
                    dolfin.gt(Eb.m, 0), fun1(expr).m_as(u), fun2(expr).m_as(u)
                )

            return func

        lowside = Eb_fconditional(mu.mside, mu.pside)
        barside = Eb_fconditional(mu.pside, mu.mside)

        kT = lowside(band.kT)  # T is assumed continuous
        qe = U.elementary_charge

        u_bar = barside(band.u)
        vth_bar = barside(vth)
        # mu.n points out from whichever surface. So to get current
        # from lowside to barside, need lowside(mu.n)
        j_band = lowside(mu.dot(band.j, mu.n))

        # Use the ln function down to where we can't resolve it anyway,
        # then use a linear extrapolation of the log, which doesn't give
        # nan for negative argument
        # Apply an arcsinh on that linear extrapolation, to avoid its
        # growing too large
        shift = j_band / (sign * qe * vth_bar * (1 + delta) * u_bar)
        shift = shift.m_as(U.dimensionless)
        argument = 1 + shift
        # argument = argument.m_as(U.dimensionless)
        # Use ln1p if shift is close to 0, for better precision
        large_target = dolfin.conditional(
            dolfin.gt(mu.dless(mu.abs(shift)), 1e-6),
            mu.ln(argument),
            mu.ln1p(shift),
        )
        eps = dolfin.DOLFIN_EPS
        small_target = mu.asinh((argument - eps) / eps) + np.log(eps)
        # Target for Delta_w
        Delta_w_BC = (
            kT
            / sign
            * dolfin.conditional(
                dolfin.gt(argument, eps), large_target, small_target
            )
        )

        w0 = band.mixedqfl_base_w
        # Actual Delta_w
        Delta_w = barside(w0) - lowside(w0)
        boundary = self.boundary

        dS = mu.region_oriented_dS(boundary, orient=False)[1]
        xi = band.mixedqfl_xi

        # Remove the standard current due to w-changes from the interfaces
        band.mixedqfl_drift_diffusion_heterojunction_facet_region |= (
            boundary.both()
        )
        # Add the boundary condition
        band.mixedqfl_drift_diffusion_heterojunction_bc_term += (
            -dS * (Delta_w - Delta_w_BC) * lowside(mu.dot(xi, mu.n))
        )
