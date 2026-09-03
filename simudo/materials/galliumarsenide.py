from math import pi
import numpy as np
from simudo.physics import Material

from .helpers import thermal_velocity


class GalliumArsenideMaterial(Material):
    """ GaAs material data based on Palankovski

        V. Palankovski and R. Quay, "Analysis and Simulation of Heterostructure Devices",
       Springer-Verlag (2004).
        Both gamma and X valley properties are included here for later use in the valleyPV InGaAs alloy.
        The CB energy level is still taken to be that of the lowest valley
    """

    name = "GalliumArsenide"

    def get_dict(self):
        d = super().get_dict()
        U = self.unit_registry

        d.update(
            {
                # Static dielectric constant, Palankovski Table 3.3
                "poisson/permittivity": U("13.1 vacuum_permittivity"),
                # Springer's handbook (2017) , Table 30.12
                "CB/Eg_300K": U("1.43 eV"),
                "CBX/Eg_300K": U("1.91 eV"),
                "CBL/Eg_300K": U("1.72 eV"),
                # Table 3.15 band edge alignment
                "VB/E_off": U("-0.712 eV"),
                # Table 3.21 Parameter for energy minima in the DOS model
                "CB/MC": U("1"),
                "CBX/MC": U("3"),
                "CBL/MC": U("4"),
                # Springer handbook Table 30.17 Effective electron masses
                "CB/mn": U("0.067"),
                "CBX/mn": U("0.85"),
                "CBL/mn": U("0.56"),
                # Springer Table 3.20 Effective hole masses
                "VB/mp": U("0.55"),
                
                # conductivity mass, Springer Table 30.17
                "CBX/mc": U("0.32"),
                "CBL/mc": U("0.11"),
                # Include other parameters for SRH etc. here.
                # SRH recombination lifetimes, Table 3.38
                "SRH/CB/tau": U("0.3e-9 s"),
                "SRH/CBX/tau": U("0.3e-9 s"),
                "SRH/CBL/tau": U("0.3e-9 s"), #TODO - find appropriate lifetimes for these valleys
                "SRH/VB/tau": U("10e-9 s"),

                # Springer 2017, Table 30.1
                "a": U("5.6533e-10 m"), # Lattice parameter

                #  Palankovski Table 3.22, lattice scattering mobility, not L-valley
                "CB/mu_L": U("8500 cm^2/V/s"),
                "CBX/mu_L": U("410 cm^2/V/s"),
                "CBL/mu_L": U("410 cm^2/V/s"),
                "VB/mu_L": U("470 cm^2/V/s"),

                # Palankovski Table 3.22, mobility temperature dependence power term
                "CB/y_0": U("-2.2"),
                "CBX/y_0": U("-2.2"),
                "CBL/y_0": U("-2.2"),
                "VB/y_0": U("-0.9"),

                # Palankvoski Table 3.23, mobility parameters to factor in ionized impurities
                "CB/mu_min" : U("800 cm^2/V/s"),
                "CBX/mu_min": U("40 cm^2/V/s"),
                "VB/mu_min" : U("40 cm^2/V/s"),

                "CB/C_ref" : U("1e17 elementary_charge/cm^3"),
                "CBX/C_ref": U("1e17 elementary_charge/cm^3"),
                "VB/C_ref" : U("1e17 elementary_charge/cm^3"),

                "CB/alpha_mu" : U("0.5 dimensionless"),
                "CBX/alpha_mu" : U("0.5 dimensionless"),
                "VB/alpha_mu" : U("1.0 dimensionless"),
            }
        )

        T = self.temperature
        E_off = d["VB/E_off"]
        # Effective electron and hole mass
        mn = d["CB/mn"]
        mnX = d["CBX/mn"]
        mnL = d["CBL/mn"]
        mp = d["VB/mp"]

        # Effective DOS. Palankovski eq. 3.111 & 3.112
        m_e = U.electron_mass
        k_B = U.boltzmann_constant
        h = U.planck_constant
        DOS_term = lambda m : (2 * pi * m * m_e * k_B * T / h ** 2) ** (3 / 2)
        NC = 2 * d["CB/MC"] * DOS_term(mn)
        NCX = 2 * d["CBX/MC"] * DOS_term(mnX)
        NCL = 2 * d["CBL/MC"] * DOS_term(mnL)
        NV = 2 * DOS_term(mp)

        # Thermal velocity
        vth = lambda m : thermal_velocity(U, T, m)
        #Gamma valley
        mtc = d["CB/mn"]
        vth_c = vth(mtc)  
        #X valley
        mtcX = d["CBX/mc"]
        vth_cX = vth(mtcX)
        #L valley
        mtcL = d["CBL/mc"]
        vth_cL = vth(mtcL)
        # For valence band, just use the DOS effective mass
        mtv = d["VB/mp"]
        vth_v = vth(mtv)

        Eg = d["CB/Eg_300K"]
        EgX = d["CBX/Eg_300K"]
        EgL = d["CBL/Eg_300K"]

        # Caughey-Thomas impurity scattering, Palankovski eq. 3.116
        def mu_LI(band: str):
            '''Band is either `CB`, `CBX`, or `VB`'''
            doping = abs(self.pdd.spatial.get('poisson/static_rho'))
            mu_min = d[band+'/mu_min']; mu_L = d[band+'/mu_L']; C_ref = d[band+'/C_ref']; alpha = d[band+'/alpha_mu']
            return mu_min + (mu_L - mu_min) / (1 + (doping/C_ref)**alpha)
        
        # Palankovski eq. 3.115 incorporates temperature dependence with mobility
        mu_T = lambda mu, y_0 : mu * (T/U("300 K"))**y_0

        d.update(
            {
                # Palankovski Table 3.22 Mobility
                "CB/mobility": mu_T(d["CB/mu_L"], d["CB/y_0"]), #Gamma valley
                "CBX/mobility": mu_T(d["CBX/mu_L"], d["CBX/y_0"]), #X valley
                "CBL/mobility": mu_T(d["CBL/mu_L"], d["CBL/y_0"]),
                "VB/mobility": mu_T(d["VB/mu_L"], d["VB/y_0"]),
                # "CB/mobility" : mu_LI('CB'),
                # "CBX/mobility" : mu_LI('CBX'),
                # "VB/mobility" : mu_LI('VB'),

                "CB/mDOS": mn,
                "CBX/mDOS": mnX,
                "CBL/mDOS": mnL,
                "VB/mDOS": mp,
                "CB/energy_level": E_off + Eg,
                "CBX/energy_level": E_off + EgX,
                "CBL/energy_level": E_off + EgL,
                "VB/energy_level": E_off,
                # Midgap trap level for SRH.  The material sets the SRH
                # lifetimes above, so it should set the level they go with;
                # without it a project using SRHRecombination has to supply
                # <proc>/energy_level by hand.  Midgap is the usual default.
                # TODO it should be source- and destination-band dependent.
                "SRH/energy_level": E_off + Eg / 2,
                "VB/effective_density_of_states": NV,
                "CB/effective_density_of_states": NC,
                "CBX/effective_density_of_states": NCX,
                "CBL/effective_density_of_states": NCL,
                "CB/vth": vth_c,
                "CBX/vth": vth_cX,
                "CBL/vth": vth_cL,
                "VB/vth": vth_v,
                "opt_cv/alpha": U("1e4 cm^-1"),
            }
        )

        # alpha(E) for a BeerLambert process named 'opt_cv', from the measured
        # table below.  ("opt_cv/alpha" above is the flat value used instead by
        # the NonOverlappingTopHat* classes.)  A top-hat set on a layer or
        # overlay overrides this: material rules have the lowest priority.
        # Keep this a literal dict -- the GUI detects it by AST scan.
        d["opt_cv/alpha_function"] = {
            "type":       "wavelength_table",
            "data":       self.optical_properties_table()[:, 0:2],
            "arg_unit":   "nm",
            "value_unit": "1/cm",
        }

        return d

    # Optical data sourced from:
    # K. Papatryfonos, T. Angelova, A. Brimont, B. Reid, S. Guldin, P. R. Smith, M. Tang, K. Li, A. J. Seeds, H. Liu, D. R. Selviah. Refractive indices of MBE-grown AlxGa1-xAs ternary alloys in the transparent wavelength region. AIP Adv. 11, 025327 (2021)
    # Data was collated into CSV files from:
    # https://refractiveindex.info/?shelf=maxin&book=GaAs&page=Papatryfonos
    def optical_properties_table(self):
        '''Array of arrays, where each element is of the form [wl, alpha, n, k].'''
        return np.array([
            [260.49, 1.78492730e+06, 3.43000, 3.70000e+00],
            [267.23, 1.56591753e+06, 3.74000, 3.33000e+00],
            [274.32, 1.35595133e+06, 3.96000, 2.96000e+00],
            [281.80, 1.12374925e+06, 4.02000, 2.52000e+00],
            [289.70, 9.54298079e+05, 3.90000, 2.20000e+00],
            [298.06, 8.55858966e+05, 3.76000, 2.03000e+00],
            [306.91, 7.98423730e+05, 3.64000, 1.95000e+00],
            [316.31, 7.62778021e+05, 3.57000, 1.92000e+00],
            [326.30, 7.39424811e+05, 3.53000, 1.92000e+00],
            [336.94, 7.27263688e+05, 3.52000, 1.95000e+00],
            [348.29, 7.17995852e+05, 3.55000, 1.99000e+00],
            [360.44, 7.18197854e+05, 3.63000, 2.06000e+00],
            [373.47, 7.20058723e+05, 3.78000, 2.14000e+00],
            [387.48, 7.36184095e+05, 4.18000, 2.27000e+00],
            [402.57, 6.80495018e+05, 4.47000, 2.18000e+00],
            [418.89, 5.90984509e+05, 4.96000, 1.97000e+00],
            [436.59, 4.08718621e+05, 4.88000, 1.42000e+00],
            [455.86, 2.28248911e+05, 4.60000, 8.28000e-01],
            [476.89, 1.54678428e+05, 4.39000, 5.87000e-01],
            [499.97, 1.13858149e+05, 4.22000, 4.53000e-01],
            [525.39, 8.75405250e+04, 4.09000, 3.66000e-01],
            [553.54, 6.78784697e+04, 3.99000, 2.99000e-01],
            [584.87, 5.39292326e+04, 3.90000, 2.51000e-01],
            [619.96, 4.21608666e+04, 3.84000, 2.08000e-01],
            [659.54, 3.29620966e+04, 3.78000, 1.73000e-01],
            [704.50, 2.22966122e+04, 3.72000, 1.25000e-01],
            [756.05, 1.59562407e+04, 3.67000, 9.60000e-02],
            [815.74, 1.04136937e+04, 3.63000, 6.76000e-02],
            [885.66, 5.74642651e+02, 3.54000, 4.05000e-03],
            [968.69, 0.00000000e+00, 3.49000, 0.00000e+00],
            [1068.90, 0.00000000e+00, 3.45000, 0.00000e+00],
            [1192.24, 0.00000000e+00, 3.42000, 0.00000e+00],
            [1347.75, 0.00000000e+00, 3.40000, 0.00000e+00],
            [1549.91, 0.00000000e+00, 3.38000, 0.00000e+00],
            [1823.42, 0.00000000e+00, 3.37000, 0.00000e+00],
        ])

