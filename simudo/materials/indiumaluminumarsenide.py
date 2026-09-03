from math import pi
from simudo.physics import Material

from .indiumarsenide import IndiumArsenideMaterial
from .aluminumarsenide import AluminumArsenideMaterial
from .helpers import Alloy

from ufl.operators import conditional


class IndiumAluminumArsenideAlloy(Material):
    """Indium-Aluminum-Arsenide material data based on Palankovski.

       This is for Al_x In_(1-x) As
       mole fraction should be specified by the MoleFractionX spatial rule.

       V. Palankovski and R. Quay, "Analysis and Simulation of Heterostructure Devices",
       Springer-Verlag (2004).

    """

    name = "IndiumAluminumArsenide"

    def get_dict(self):
        d = super().get_dict()
        U = self.unit_registry
        X = self.problem_data.pdd.spatial.get("MoleFractionX")
        IndiumArsenide = IndiumArsenideMaterial(problem_data=self.problem_data).get_dict()
        AluminumArsenide = AluminumArsenideMaterial(problem_data=self.problem_data).get_dict()
        # print(X)

        alloy = Alloy(U, X, at_x0=IndiumArsenide, at_x1=AluminumArsenide)
        vegard = alloy.vegard

        def linear_interp(param):
            return IndiumArsenide[param] * (U("1") - X) + AluminumArsenide[param] * X

        d.update(
            {
                "poisson/permittivity": vegard("poisson/permittivity", U("0")),
                # Springer Table 30.13
                "CB/Eg_300K": vegard("CB/Eg_300K", U("-0.720 eV")),
                "CBL/Eg_300K": vegard("CBL/Eg_300K", U("0")), #TODO - find bowing parameter
                "CBX/Eg_300K": vegard("CBX/Eg_300K", U("-0.30 eV")), #palankovski

                # Table 3.21
                "CBX/MC": U("3"),
                "CB/MC": U("1"),
                "CBL/MC": U("4"),

                # mobility, Table 3.28
                "CB/mobility": vegard("CB/mobility", U("1.5e8 cm^2/V/s")),
                "CBX/mobility": vegard("CBX/mobility", U("1e6 cm^2/V/s")),
                "CBL/mobility": vegard("CBL/mobility", U("0")), #TODO- find bowing parameter
                "VB/mobility": vegard("VB/mobility", U("95 cm^2/V/s")),

                "CB/vth": vegard("CB/vth", U("0")),
                "CBX/vth": vegard("CBX/vth", U("0")),
                "CBL/vth": vegard("CBL/vth", U("0")),
                "VB/vth": vegard("VB/vth", U("0")),

                # SRH recombination lifetimes, Table 3.38
                "SRH/CB/tau": U("100e-9 s"),
                "SRH/CBX/tau": U("100e-9 s"), #TODO - find appropriate lifetimes for these valleys
                "SRH/CBL/tau": U("100e-9 s"),
                "SRH/VB/tau": U("100e-9 s"),
            }
        )

        T = self.temperature

        # Table 3.20
        mn = vegard("CB/mDOS", U("0")) 
        mnX = vegard("CBX/mDOS", U("0"))
        mnL = vegard("CBL/mDOS", U("0"))
        mp = vegard("VB/mDOS", U("0"))
        m_e = U.electron_mass
        k_B = U.boltzmann_constant
        h = U.planck_constant
        NC = 2 * d["CB/MC"] * (2 * pi * mn * m_e * k_B * T / h ** 2) ** (3 / 2)
        NCX = 2 * d["CBX/MC"] * (2 * pi * mnX * m_e * k_B * T / h ** 2) ** (3 / 2)
        NCL = 2 * d["CBL/MC"] * (2 * pi * mnL * m_e * k_B * T / h ** 2) ** (3 / 2)
        NV = 2 * (2 * pi * mp * m_e * k_B * T / h ** 2) ** (3 / 2)

        # Energy offset, Eqn (3.99)
        EgX_300K = vegard("CBX/Eg_300K", U("0 eV"))
        E_off = (
            IndiumArsenide["VB/E_off"] * (EgX_300K - AluminumArsenide["CB/Eg_300K"])
            - AluminumArsenide["VB/E_off"] * (EgX_300K - IndiumArsenide["CB/Eg_300K"])
        ) / (IndiumArsenide["CB/Eg_300K"] - AluminumArsenide["CB/Eg_300K"])

        # Determine which is the lowest valley
        def Uconditional(cond, x, y):
            x = U.dimensionless * x
            y = U.dimensionless * y
            return x.u * conditional(cond, x.m, y.m_as(x.u))

        EgX = d["CBX/Eg_300K"]
        Eg = d["CB/Eg_300K"]
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
                "CB/effective_density_of_states": NC,
                "CBX/effective_density_of_states": NCX,
                "CBL/effective_density_of_states": NCL,
                "VB/effective_density_of_states": NV,
                "mole_fraction": X,
                "SRH/energy_level": E_off + Eg / 2, #TODO it has to be source- and destination-band dependent
            }
        )

        return d
