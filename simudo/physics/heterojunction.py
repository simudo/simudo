import attr
import dolfin
import numpy as np

from .poisson_drift_diffusion import Band, NondegenerateBand
from ..mesh.topology import FacetRegion
from ..fem import expm1


@attr.s
class ThermionicHeterojunction:
    """
    Thermionic emission heterojunction BC valid with parabolic bands. 
    Treats both nondegenerate (Boltzmann) and degenerate bands

    Parameters
    ----------
    band: :py:class:`.Band`
        Semiconductor band on which the BC is applied. Only
        MixedQflBand is currently supported.
    boundary: :py:class:`.FacetRegion`
        Boundary on which to implement heterojunction boundary condition.

    Notes
    -----

    See V. Palankovski (2004), eq. 3.72 and
    K. Yang, J. R. East, G. I. Haddad, Solid State Electronics v.36 (3) p.321-330 (1993)
    K. Horio, H. Yanai, IEEE Trans. Elec. Devices v.37(4) p.1093-1098 (1990)

    For degenerate conditions, see Sentaurus sdevice manual

    For a conduction band BC, band.spatial must have attribute
    "CB/vth" in the barrier region. Similarly for other bands.

    Unlimited carrier flow from low to barrier region can be resolved, but due
    to precision issues, cannot resolve Delta_w producing carrier flows from
    barrier to low region with Delta_w larger than
    ``|ln(1e-16)| * kT`` in double precision
    
    DX 20260813: make use_alternative_BC = True the default. Alternative BC is using the exponential of delta_w, or lambda, instead of delta_w as
    the HJBC requirement.
    
    """

    band: Band = attr.ib()
    boundary: FacetRegion = attr.ib()
    interface_quadrature_degree: int = attr.ib(default=20)
    # factory, not default: a bare default would be evaluated once at import and
    # shared by every default-constructed instance.
    HJBC_enhancement: dolfin.Constant = attr.ib(factory=lambda: dolfin.Constant(1.0))
    use_alternative_BC: bool = attr.ib(default=True)
    use_nondegen: bool = attr.ib(default=False)

    @property
    def vth(self):
        return self.band.spatial.get(self.band.spatial_prefix + "vth")

    @property
    def emission_velocity(self):
        r"""Richardson emission velocity, :math:`\sqrt{k_B T / 2\pi m^*}`.

        The material files export ``<band>/vth`` as the **mean speed**
        :math:`\langle v\rangle = \sqrt{8k_BT/\pi m^*}`, which is the average
        an SRH capture coefficient needs.  Thermionic emission is a *flux*
        across the interface, so it needs the one-directional flux average
        instead -- only the carriers moving toward the barrier, weighted by
        how fast they cross:

        .. math::
            v_R = \langle v_z \rangle_{v_z>0}
                = \sqrt{\frac{k_B T}{2\pi m^*}}
                = \frac{\langle v \rangle}{4}

        The factor is exactly 4, since
        :math:`\sqrt{8/\pi} \big/ \sqrt{1/2\pi} = \sqrt{16}`.  Equivalently
        :math:`v_R = A^* T^2 / (q N_C)` with
        :math:`A^* = 4\pi q m^* k_B^2 / h^3`.

        Applying the conversion here, rather than exporting a second velocity
        from every material file, keeps one ``vth`` per band and puts the
        choice of average next to the physics that fixes it.
        """
        return self.vth / 4

    @property
    def N_eff(self):
        return self.band.spatial.get(self.band.spatial_prefix + "effective_density_of_states")

    def register(self, debug_output=False):
        """
        Apply boundary condition onto the band.
        """

        band = self.band
        mu = band.pdd.mesh_util
        U = mu.unit_registry

        
        # delta: correction for tunneling through the barrier. not implemented.
        delta = 0.0
        sign = band.sign
        # Emission velocity: the material files' vth is the mean speed, which
        # is a factor of 4 larger than the flux average thermionic emission
        # needs. See ThermionicHeterojunction.emission_velocity.
        # The underlying mass should be the DOS effective mass (Palankovski 3.75).
        v_R = self.emission_velocity
        # Eb is barrier height. should be positive. Use its sign to determine which
        # side is the low and which side is the barrier
        Eb = -sign * (mu.pside(band.energy_level) - mu.mside(band.energy_level))

        # 29 June 2022 -JK
        # Articles don't actually treat the case of holes explicitly. Previous version of HJBC
        # may have been using the wrong sense of the boundary when holes are considered.
        # Sentaurus' manual (I think) uses a convention where the band offset 
        # Delta_E = E_2-E_1 >0, so side "2" is the barrier for electrons but side "1" 
        # is the barrier for holes.
        # We can fix that by calling side the 2 the "high" side instead of the "bar" side 
        # (which was the old notation), and still keeping "low" for the "1" side.
        # Eb = mu.pside(band.energy_level) - mu.mside(band.energy_level)
        # If Eb>0, then pside=high. Else, pside=low

        # 18 Nov 2022 - JK & DX
        # We have determined that the proper sign convention is that Delta_E=E_b>0
        # for both electrons and holes. So we should use the "low" and "bar" 
        # labels for the code.

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
        v_R_bar = barside(v_R)
        N_eff_bar = barside(self.N_eff)
        # j_band is current from low to hi side.
        # mu.n points out from whichever surface. So to get current
        # from lowside to barside, need lowside(mu.n)
        j_band = lowside(mu.dot(band.j, mu.n))

        #w0 = band.mixedqfl_base_w
        w0 = band.qfl #Use correct qfl, so ufl derivative may work better with the BC
        phiqfl = band.phiqfl
        
        ##### nondegenerate expression
        # Use the ln function down to where we can't resolve it anyway,
        # then use a linear extrapolation of the log, which doesn't give
        # nan for negative argument
        # Apply an arcsinh on that linear extrapolation, to avoid its
        # growing too large
        shift = j_band / (sign * qe * v_R_bar * (1 + delta) * u_bar)
        shift = shift.m_as(U.dimensionless)
        argument = 1 + shift
        # Use ln1p [= ln(1+x)] if shift is close to 0, for better precision
        large_target = dolfin.conditional(
            dolfin.gt(mu.dless(mu.abs(shift)), 1e-6),
            mu.ln(argument),
            mu.ln1p(shift),
        )
        eps = dolfin.DOLFIN_EPS
        small_target = mu.asinh((argument - eps) / eps) + np.log(eps)
        # Target for Delta_w
        Delta_w_BC_nondegen = (
            kT / sign
            * dolfin.conditional(
                dolfin.gt(argument, eps), large_target, small_target
            )
        )
        
        
        ##### degenerate
        # Notes 17 June 2022 & 28 June 2022
        # See code/doc/heterojunction_boundary_condition.lyx
        # Following Sentaurus sdevice manual, Eqs 879...
        # Results for degenerate HJ are not cited to any paper
        # a = 2 # Parameter from the Sentaurus manual
                # Don't see any reason to use the a=2 from the manual.
                # Its reference (Schroeder94 p 167) doesn't support the factor
                # Reduces to above Nondegenerate formulation when a=1
        a = 1
        B = j_band / (-sign * qe * a * v_R_bar * N_eff_bar)
        B = B.m_as("dimensionless")
        E_bar = barside(band.energy_level)
        
        # 28 June version
        eta_2 = sign * ((E_bar - barside(phiqfl))/kT)
        # Delta_w_BC = kT/sign * mu.ln(
        #     mu.exp(-B) + mu.exp(-(B + eta_2)) - mu.exp(-eta_2)
        # )
        
        Delta_w_BC_degen = kT/sign * (-B + 
                            #  mu.ln(1 + mu.exp(-eta_2)*(1 - mu.exp(B))))
                            # mu.ln1p(mu.exp(-eta_2)*(1 - mu.exp(B))))
                            mu.ln1p(mu.exp(-eta_2)*
                                    (-expm1(B))
                                ))
     


        if isinstance(band, NondegenerateBand):           

            Delta_w_BC = Delta_w_BC_nondegen

        else: # Degenerate band
            
            # determine whether to use degenerate HJBC
            use_nondegen = getattr(self, 'use_nondegen', False)
            if use_nondegen:
                import logging
                logging.info("Use nondegenerate HJBC")
                Delta_w_BC = Delta_w_BC_nondegen
            else:
                Delta_w_BC = Delta_w_BC_degen
                 

        # Actual Delta_w
        Delta_w = barside(w0) - lowside(w0)
        boundary = self.boundary
        
        # probe degenerate and nondegenerate Delta_w_BC
        # Delta_w_BC_degen_probe = mu.get_debug_facet_probe(Delta_w_BC_degen, boundary)
        # Delta_w_BC_nondegen_probe = mu.get_debug_facet_probe(Delta_w_BC_nondegen, boundary)
        # print(f"band: {band.name}, Delta_w_BC_degen: {Delta_w_BC_degen_probe()}")
        # print(f"band: {band.name}, Delta_w_BC_nondegen: {Delta_w_BC_degen_probe()}")

        dS0 = mu.region_oriented_dS(boundary, orient=False)[1]
        dS = dS0(metadata = {"quadrature_degree": self.interface_quadrature_degree})
        xi = band.mixedqfl_xi

        # Remove the standard current due to w-changes from the interfaces
        band.mixedqfl_drift_diffusion_heterojunction_facet_region |= (
            boundary.both()
        )


        if not self.use_alternative_BC:
            BC_term = (Delta_w - Delta_w_BC).m_as("eV") # we shall put in the correct dimension later
            HJBC_enhancement = self.HJBC_enhancement # if we do not scale HJBC_enhancement
        else:
            # from ..fem import ufl_extra as ufle
            # use alternative boundary condition

            eta = sign * Delta_w / kT
            eta = eta.m_as(U.dimensionless)
            # see Daisy's note 20231116
            # exponetiate both sides
            
            # lam1 term for nondegenerate
            lam1_BC_nondegen = ((j_band / (sign * qe * v_R_bar * (1 + delta) * u_bar)).m_as("dimensionless") + 1) # "shift + 1" as above
            lam1_nondegen = mu.exp(eta)
            
            # lam1 term for degenerate
            x = (1 + mu.exp(-1 * eta_2.m_as("dimensionless")) * (-expm1(B)))

            lam1_degen = mu.exp(eta)
            lam1_BC_degen = (
                x * mu.exp(-B)
            )
            
            # determine whether to use degenerate HJBC 
            if isinstance(band, NondegenerateBand) or self.use_nondegen:
                lam1_BC = lam1_BC_nondegen
                lam1 = lam1_nondegen
            else:
                lam1_BC = lam1_BC_degen
                lam1 = lam1_degen
            
            BC_term_lam1 = lam1 - lam1_BC
            
            ##################################################
            # standard alternative BC
            #-------------------------------------------------
            BC_term = BC_term_lam1
            ##################################################
            
            
            ##################################################   
            # give HJBC enhancement a boost when lam1 is small
            #-------------------------------------------------
            HJBC_enhancement = self.HJBC_enhancement * dolfin.conditional(
                dolfin.gt(lam1, 1e-6),
                1,
                1e-6 / lam1
            )
            
            ################################################## 
            
        # add boundary condition
        band.mixedqfl_drift_diffusion_heterojunction_bc_term += (
            -1 * dS * HJBC_enhancement * BC_term * U("eV") * lowside(mu.dot(xi, mu.n))
        )
        
        #debug code
        if hasattr(band.pdd, "debug_HJBC_reltol"):

            if not hasattr(band, 'Delta_w_BC'):
                band.Delta_w_BC = []
            if not hasattr(band, 'Delta_w_BC_degen'):
                band.Delta_w_BC_degen = []
            if not hasattr(band, 'Delta_w'):
                band.Delta_w = []
            if not hasattr(band, 'j_band'):
                band.j_band = []
            if not hasattr(band, 'phiqfl_facet'):
                band.phiqfl_facet = []
                
            if self.use_alternative_BC:
                if not hasattr(band, 'lam1'):
                    band.lam1 = []
                if not hasattr(band, 'lam1_BC'):
                    band.lam1_BC = []

            Delta_w_BC_probe = mu.get_debug_facet_probe(Delta_w_BC, boundary)
            band.Delta_w_BC.append(Delta_w_BC_probe)
            
            Delta_w_BC_degen_probe = mu.get_debug_facet_probe(Delta_w_BC_degen, boundary)
            band.Delta_w_BC_degen.append(Delta_w_BC_degen_probe)

            Delta_w_probe = mu.get_debug_facet_probe(Delta_w, boundary)
            band.Delta_w.append(Delta_w_probe)

            j_band_probe = mu.get_debug_facet_probe(j_band, boundary)
            band.j_band.append(j_band_probe)

            phiqfl_facet_probe = mu.get_debug_facet_probe(barside(phiqfl), boundary)
            band.phiqfl_facet.append(phiqfl_facet_probe)
            
            if self.use_alternative_BC:
            
                lam1_probe = mu.get_debug_facet_probe(lam1, boundary)
                band.lam1.append(lam1_probe)

                lam1_BC_probe = mu.get_debug_facet_probe(lam1_BC, boundary)
                band.lam1_BC.append(lam1_BC_probe)
