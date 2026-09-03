
from math import pi
from simudo.physics import Material
import numpy as np

from .galliumnitride import GalliumNitrideMaterial
from .galliumarsenide import GalliumArsenideMaterial
from .helpers import Alloy

class GalliumArsenideNitrideAlloy(Material):
    """Gallium Arsenide Nitride material based off of Gallium Arsenide material data based on Palankovski
       and the band anticrossing model, with parameters from 
       R. Kudrawiec et al, "Electronic Band Structure of GaNxPyAs1−x−y Highly Mismatched Alloys: 
       Suitability for Intermediate-Band Solar Cells" Phys Rev App 2014

       V. Palankovski and R. Quay, "Analysis and Simulation of Heterostructure Devices",
       Springer-Verlag (2004).

       Band structure comes from BAC but other parameters are found with Vegard's law from GaAs and GaN
       
       MoleFractionX is the Nitrogen fraction

    """

    name = "GalliumArsenideNitride"

    def get_dict(self):
        d = super().get_dict()
        U = self.unit_registry
        X = self.problem_data.pdd.spatial.get("MoleFractionX")
        sqrt = self.problem_data.pdd.mesh_util.sqrt
        GalliumNitride = GalliumNitrideMaterial(problem_data=self.problem_data).get_dict()
        GalliumArsenide = GalliumArsenideMaterial(problem_data=self.problem_data).get_dict()

        # X is the nitrogen fraction, so x = 0 recovers GaAs. Parameters absent
        # from one parent fall back to the other rather than raising.
        alloy = Alloy(U, X, at_x0=GalliumArsenide, at_x1=GalliumNitride,
                      missing='fallback')
        vegard = alloy.vegard
        # Implements eq.3.132 from Palankovski
        mobility_bowing = alloy.mobility_bowing

        d.update(
            {
                "Em_300K" : GalliumArsenide["CB/Eg_300K"], # Conduction band minimum
                "En_300K" : U("1.65 eV"), # Impurity energy level
                "Vmn_300K" : U("2.7 eV"), # Coupling term
            }
        )

        #from doi: 10.1103/PhysRevApplied.1.034007, equation 2
        def band_anticrossing_level(E_plus: bool) -> float:
            Em = d["Em_300K"]; En = d["En_300K"]
            # print(f"\nX = {X}\n")
            crossingTerm = ((En-Em)**2 + 4 * d["Vmn_300K"]**2 * X)**0.5 * (1 if E_plus else -1)
            return (Em + En + crossingTerm)/2

        d.update(
            {
                "poisson/permittivity": vegard("poisson/permittivity", U("0")),
                #Springer Table 30.13 # JB - Previous values for these were for GaInAs, for some reason.
                # Note also this table says to assume linear variation where no bowing parameter is given
                "CB/Eg_300K": band_anticrossing_level(E_plus=True),
                "IB/Eg_300K": band_anticrossing_level(E_plus=False),
                "CBL/Eg_300K": vegard("CBL/Eg_300K", U("0 eV")), # JB: was -0.50
                # Table 3.21
                "CB/MC": U("1"),
                "IB/MC": U("1"),
                # "CBL/MC": U("4"),

                # mobility, Palankovski Table 3.28   
                # JB - These are from either InGaAs or InGaP. This table does not include GaNAs             
                "CB/mobility": mobility_bowing("CB/mobility", U("1e6 cm^2/V/s")),
                # "IB/mobility0": vegard("IB/mobility0", U("1e6 cm^2/V/s")),
                # "CBL/mobility": vegard("CBL/mobility", U("0")),
                "VB/mobility": mobility_bowing("VB/mobility", U("1e6 cm^2/V/s")),

                #thermal velocity
                "CB/vth": vegard("CB/vth", U("0")),
                # "IB/vth": vegard("IB/vth", U("0")),
                # "CBL/vth": vegard("CBL/vth", U("0")),
                "VB/vth": vegard("VB/vth", U("0")),

                # SRH recombination lifetimes, Table 3.38
                # JB - These are assumed from an instance of In_0.53 Ga_0.47 As
                "SRH/CB/tau": U("0.3e-9 s"),
                # "SRH/IB/tau": U("3e-9 s"),
                # "SRH/CBL/tau": U("0.3e-9 s"),
                "SRH/VB/tau": U("10e-9 s"),

                # Direct recombination coefficient, Palankovski Table 3.41
                "C_dir" : U("3e-10 cm^3/s"), # This is assuming the same value as for GaAs

                # TODO - absorpton coefficients, update these values to reflect the actual material
                # "opt_lv/alpha" : U("1e4 cm^-1"),
                # "opt_gv/alpha" : U("1e4 cm^-1"),
                # "opt_lg/alpha" : U("1e4 cm^-1"),
                
                #These are just guesses
                "opt_cv/alpha":     U("1e4 cm^-1"),  # source: https://aip-scitation-org.proxy.bib.uottawa.ca/doi/pdf/10.1063/1.3000451
                "opt_ci/sigma_opt": U("1e-16 cm^2"),
                "opt_iv/sigma_opt": U("1e-15 cm^2"),
                "opt_ci/alpha":     U("1e3 cm^-1"),
                "opt_iv/alpha":     U("1e4 cm^-1"),

                "IB/degeneracy": U("1"),
                "AB/degeneracy": U("1"),

                # Springer 2017, Table 30.1
                "a": vegard("a", U('0')) # Lattice parameter
            }
        )

        T = self.temperature
        EG = d["Em_300K"]
        EN = d["En_300K"] #isolated Nitrogen level
        V = d["Vmn_300K"] #matrix element

        # Effective carrier masses Table 3.20
        # JB - This table does not include GaNAs. Bowing parameters are taken from InGaAs
        # mn = vegard("CB/mDOS", U("-0.012"))
        # New mn from Skierbiszewski et al. 2004, "The electron effective mass at the bottom of the GaNAs conduction band"
        mn = 2*GalliumArsenide["CB/mDOS"] / (1 - (EG - EN)/((EG - EN)**2 + 4*X* V**2)**0.5)
        # mi = vegard("IB/mDOS", U("-0.018"))
        # mnL = vegard("CBL/mDOS", U("0"))
        mp = vegard("VB/mDOS", U("-0.03"))
        m_e = U.electron_mass
        k_B = U.boltzmann_constant
        h = U.planck_constant
        DOS_term = lambda m : (2 * pi * m * m_e * k_B * T / h ** 2) ** (3 / 2)
        NC = 2 * d["CB/MC"] * DOS_term(mn)
        NV = 2 * DOS_term(mp)
        
        # Eg = d["CB/Eg_300K"]
        # # EgL = d["CBL/Eg_300K"]

        # Energy offset, Palankovski Eqn (3.99). 
        # Using the offset energy from the standard alloy formula, but E+/E- levels from BAC
        E_off = (
            GalliumNitride["VB/E_off"] * (d["CB/Eg_300K"] - GalliumArsenide["CB/Eg_300K"])
            - GalliumArsenide["VB/E_off"] * (d["CB/Eg_300K"] - GalliumNitride["CB/Eg_300K"])
        ) / (GalliumNitride["CB/Eg_300K"] - GalliumArsenide["CB/Eg_300K"])
        # print(f"Calculated GaNAs E_off as: {E_off}")

        #IB total density of states -- Element HMA room 19 Mar 2021
        G14 = 3.625609908221 #Gamma(1/4)
        mc = GalliumArsenide["CB/mDOS"]
        hbar = h/2/pi
        kd = sqrt(2*mc*m_e*(EN-EG))/hbar
        z = V**2 * X / (EN-EG)**2
        NI = kd**3/2/pi**2 * (2/3 + z**0.75 * G14**2/(3*sqrt(2*pi)) )
        

        d.update(
            {
                "CB/mDOS": mn,
                # "IB/mDOS": mi,
                # "CBL/mDOS": mnL,
                "VB/mDOS": mp,
                "CB/energy_level": E_off + d["CB/Eg_300K"],
                "IB/energy_level": E_off + d["IB/Eg_300K"],
                # "CBL/energy_level": E_off + EgL,
                "IB/mobility": d["CB/mobility"], #Totally unknown -- just set equal to CB for now.
                #"IB/mobility0": d["CB/mobility"], #Totally unknown -- just set equal to CB for now.                
                "VB/energy_level": E_off,
                "CB/effective_density_of_states": NC,
                "IB/effective_density_of_states": NI,
                # "IB/number_of_states": NI,
                # "CBL/effective_density_of_states": NCL,
                "VB/effective_density_of_states": NV,
                "mole_fraction": X,
                "SRH/energy_level": E_off + d["IB/Eg_300K"] / 2, #TODO it has to be source- and destination-band dependent
            }
        )
        ## Mobility from alloy scattering, from eqn. 15.1 of"Dilute III-V Nitride Semiconductors and Material Systems", Springer 2008
        # NOTE: This is assuming n-type Si doping of GaInNAs, though the formula should still hold with no indium
        # muAS_1 = np.pi * d['a']**3 * X * np.sqrt(3*vegard("CB/mn", 0)*U("300 kelvin * electron_mass*boltzmann_constant"))
        # muAS_2 = 2*np.pi*U("hbar")**2 / (vegard("CB/mn", 0) * U("electron_mass"))
        # muAS_3 = d["Vmn_300K"]**2 / ((d["En_300K"] - d["Em_300K"])**2 - 4*X*d["Vmn_300K"]**2)**0.5
        # muAS = U("elementary_charge")/muAS_1 * (muAS_2**2) / (muAS_3**2)
        # print(f"{muAS.m_as('cm^2 / V / s')} cm^2/Vs")

        return d
