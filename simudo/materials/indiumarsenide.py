from math import pi
import numpy as np 
from simudo.physics import Material

from .helpers import thermal_velocity 


class IndiumArsenideMaterial(Material):
    """ InAs material data based on Palankovski

        V. Palankovski and R. Quay, "Analysis and Simulation of Heterostructure Devices",
       Springer-Verlag (2004).
        Both gamma and X valley properties are included here for later use in the valleyPV InGaAs alloy.
        The CB energy level is still taken to be that of the lowest valley
    """

    name = "IndiumArsenide"

    def get_dict(self):
        d = super().get_dict()
        U = self.unit_registry

        d.update(
            {
                # Static dielectric constant, Palankovski Table 3.3
                "poisson/permittivity": U("14.6 vacuum_permittivity"),
                # Springer's handbook (2017) , Table 30.12
                "CB/Eg_300K": U("0.359 eV"),
                "CBX/Eg_300K": U("1.37 eV"),
                "CBL/Eg_300K": U("1.07 eV"),
                # Table 3.15 band edge alignment
                "VB/E_off": U("-0.286 eV"),
                # Table 3.21 Parameter for energy minima in the DOS model
                "CB/MC": U("1"),
                "CBX/MC": U("3"),
                "CBL/MC": U("4"),
                # Springer handbook Table 30.17 Effective electron masses. Temperature dependence can be later
                "CB/mn": U("0.024"),
                "CBX/mn": U("0.98"),
                "CBL/mn": U("0.94"),
                # Springer Table 3.20 Effective hole masses
                "VB/mp": U("0.36"),
                # Table 3.22 Mobility
                "CB/mobility": U("32500 cm^2/V/s"), #Gamma valley
                "CBX/mobility": U("480 cm^2/V/s"), #X valley
                "CBL/mobility": U("480 cm^2/V/s"), #TODO - update L valley mobility
                "VB/mobility": U("510 cm^2/V/s"), 
                # conductivity mass, Springer Table 30.17
                "CBX/mc": U("0.38"),
                "CBL/mc": U("0.18"),
                # Include other parameters for SRH etc. here.    
                # SRH recombination lifetimes, Table 3.38
                "SRH/CB/tau": U("0.3e-9 s"),
                "SRH/CBX/tau": U("0.3e-9 s"),
                "SRH/CBL/tau": U("0.3e-9 s"), #TODO - find appropriate lifetimes for these valleys
                "SRH/VB/tau": U("10e-9 s"),
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

        d.update(
            {
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
            }
        )

        # alpha(E) for a BeerLambert process named 'opt_cv', from the measured
        # table below. A top-hat set on a layer or overlay overrides this:
        # material rules have the lowest priority.
        # Keep this a literal dict -- the GUI detects it by AST scan.
        d["opt_cv/alpha_function"] = {
            "type":       "wavelength_table",
            "data":       self.optical_properties_table()[:, 0:2],
            "arg_unit":   "nm",
            "value_unit": "1/cm",
        }

        return d

    # Optical data sourced from:
    # S. Adachi. Optical dispersion relations for GaP, GaAs, GaSb, InP, InAs, InSb, AlxGa1−xAs, and In1−xGaxAsyP1−y. J. Appl. Phys. 66, 6030-6040 (1989)
	# Data was collated into CSV files from:
    # https://refractiveindex.info/?shelf=main&book=InAs&page=Adachi
    def optical_properties_table(self):
        '''Array of arrays, where each element is of the form [wl, alpha, n, k].'''
        return np.array([
            [207.00, 9.59172250e+05, 1.27000, 1.58000e+00],
            [210.00, 9.75389719e+05, 1.28000, 1.63000e+00],
            [213.00, 9.97050063e+05, 1.29000, 1.69000e+00],
            [216.00, 1.01229097e+06, 1.30000, 1.74000e+00],
            [220.00, 1.03386958e+06, 1.31000, 1.81000e+00],
            [223.00, 1.05940703e+06, 1.33000, 1.88000e+00],
            [227.00, 1.07948999e+06, 1.34000, 1.95000e+00],
            [231.00, 1.10975740e+06, 1.36000, 2.04000e+00],
            [234.00, 1.14923218e+06, 1.39000, 2.14000e+00],
            [238.00, 1.18799722e+06, 1.42000, 2.25000e+00],
            [243.00, 1.22560899e+06, 1.46000, 2.37000e+00],
            [247.00, 1.28207506e+06, 1.51000, 2.52000e+00],
            [251.00, 1.34675446e+06, 1.60000, 2.69000e+00],
            [256.00, 1.41862543e+06, 1.72000, 2.89000e+00],
            [261.00, 1.50218683e+06, 1.92000, 3.12000e+00],
            [266.00, 1.58260683e+06, 2.23000, 3.35000e+00],
            [271.00, 1.62296299e+06, 2.70000, 3.50000e+00],
            [276.00, 1.55258420e+06, 3.29000, 3.41000e+00],
            [282.00, 1.34130410e+06, 3.78000, 3.01000e+00],
            [288.00, 1.08646746e+06, 3.95000, 2.49000e+00],
            [294.00, 8.93323625e+05, 3.88000, 2.09000e+00],
            [300.00, 7.74926188e+05, 3.74000, 1.85000e+00],
            [307.00, 7.04044217e+05, 3.61000, 1.72000e+00],
            [314.00, 6.64336790e+05, 3.50000, 1.66000e+00],
            [321.00, 6.42020181e+05, 3.42000, 1.64000e+00],
            [328.00, 6.28318531e+05, 3.37000, 1.64000e+00],
            [336.00, 6.20838548e+05, 3.35000, 1.66000e+00],
            [345.00, 6.15570039e+05, 3.35000, 1.69000e+00],
            [353.00, 6.15858956e+05, 3.38000, 1.73000e+00],
            [362.00, 6.10961665e+05, 3.42000, 1.76000e+00],
            [372.00, 6.08050191e+05, 3.47000, 1.80000e+00],
            [382.00, 6.05291150e+05, 3.55000, 1.84000e+00],
            [393.00, 6.01139358e+05, 3.64000, 1.88000e+00],
            [405.00, 5.95739051e+05, 3.75000, 1.92000e+00],
            [417.00, 5.90649554e+05, 3.88000, 1.96000e+00],
            [429.00, 5.82915560e+05, 4.04000, 1.99000e+00],
            [443.00, 5.73003807e+05, 4.21000, 2.02000e+00],
            [458.00, 5.62468554e+05, 4.41000, 2.05000e+00],
            [473.00, 5.55258236e+05, 4.61000, 2.09000e+00],
            [490.00, 5.53946133e+05, 4.78000, 2.16000e+00],
            [508.00, 4.67528356e+05, 4.71000, 1.89000e+00],
            [527.00, 3.02832840e+05, 4.44000, 1.27000e+00],
            [548.00, 1.68315986e+05, 4.22000, 7.34000e-01],
            [570.00, 1.54985238e+05, 4.12000, 7.03000e-01],
            [594.00, 1.40895671e+05, 4.02000, 6.66000e-01],
            [621.00, 1.26068420e+05, 3.93000, 6.23000e-01],
            [650.00, 1.11357377e+05, 3.85000, 5.76000e-01],
            [681.00, 9.65082501e+04, 3.79000, 5.23000e-01],
            [716.00, 8.17867138e+04, 3.72000, 4.66000e-01],
            [755.00, 6.75754499e+04, 3.67000, 4.06000e-01],
            [798.00, 5.40133474e+04, 3.62000, 3.43000e-01],
            [847.00, 4.15417210e+04, 3.58000, 2.80000e-01],
            [902.00, 3.03710509e+04, 3.55000, 2.18000e-01],
            [964.00, 2.11177597e+04, 3.52000, 1.62000e-01],
            [1040.00, 1.42579974e+04, 3.49000, 1.18000e-01],
            [1120.00, 1.05804353e+04, 3.48000, 9.43000e-02],
            [1220.00, 1.00427962e+04, 3.46000, 9.75000e-02],
            [1330.00, 9.82633492e+03, 3.45000, 1.04000e-01],
            [1470.00, 9.40340658e+03, 3.45000, 1.10000e-01],
            [1650.00, 7.84446166e+03, 3.45000, 1.03000e-01],
            [1870.00, 7.72798193e+03, 3.44000, 1.15000e-01],
            [2160.00, 7.44673814e+03, 3.46000, 1.28000e-01],
            [2550.00, 6.60350456e+03, 3.49000, 1.34000e-01],
            [3130.00, 4.29585194e+03, 3.56000, 1.07000e-01],
            [4030.00, 5.95577367e+01, 3.51000, 1.91000e-03],
            [5670.00, 3.01415591e+01, 3.48000, 1.36000e-03],
            [9560.00, 1.06340940e+01, 3.47000, 8.09000e-04],
        ])