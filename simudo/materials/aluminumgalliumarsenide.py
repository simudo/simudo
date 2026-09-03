import dolfin
from math import pi
from simudo.physics import Material

from .aluminumarsenide import AluminumArsenideMaterial
from .galliumarsenide import GalliumArsenideMaterial
from .helpers import Alloy

class AluminumGalliumArsenideAlloy(Material):
    """Aluminum-Gallium-Arsenide material data based on Palankovski.

       This is for Ga_(1-x) Al_x As
       mole fraction should be specified by the MoleFractionX spatial rule.

       V. Palankovski and R. Quay, "Analysis and Simulation of Heterostructure Devices",
       Springer-Verlag (2004).

       Default CB properties are with the Gamma valley. For usage in valley PV, need to update the dictionary
       for CB -> X valley, and IB -> Gamma valley
    """

    name = "AluminumGalliumArsenide"
    required_spatial_params = ['MoleFractionX']

    def get_dict(self):
        d = super().get_dict()
        U = self.unit_registry
        X = self.problem_data.pdd.spatial.get("MoleFractionX")
        AluminumArsenide = AluminumArsenideMaterial(problem_data=self.problem_data).get_dict()
        GalliumArsenide = GalliumArsenideMaterial(problem_data=self.problem_data).get_dict()

        # AlGaAs is direct-gap (Gamma) below X = 0.45 and indirect (X valley) above, so the
        # conduction-band quantities switch valley there; UFL needs the bare magnitude.
        X_m = X.m_as(U("dimensionless")) if hasattr(X, "m_as") else X

        def conditional(gamma, x, set_units = ''):
            """AlGaAs-specific constructor for alloy composition-based `dolfin.conditional` that handles units."""
            if hasattr(gamma, 'm'):
                u = gamma.units
                return dolfin.conditional(dolfin.lt(X_m, 0.45), gamma.m_as(u), x.m_as(u)) * u
            return dolfin.conditional(dolfin.lt(X_m, 0.45), gamma, x) * U(set_units)

        alloy = Alloy(U, X, at_x0=GalliumArsenide, at_x1=AluminumArsenide)
        vegard = alloy.vegard
        # Implements eq.3.132 from Palankovski
        mobility_bowing = alloy.mobility_bowing

        d.update(
            {
                "poisson/permittivity": vegard("poisson/permittivity", U("0")),

                # These three values have been updated according to Springer Table 30.13
                "CB/Eg_300K": vegard("CB/Eg_300K", U("-0.37 eV")),
                "CBL/Eg_300K": vegard("CBL/Eg_300K", U("-0.055 eV")), # JB - Previously C = -0.50 eV
                "CBX/Eg_300K": vegard("CBX/Eg_300K", U("-0.245 eV")), # JB - Previously C = -0.70 eV
                # Table 3.21
                "CBX/MC": U("3"),
                "CB/MC": U("1"),
                "CBL/MC": U("4"),

                # mobility, Table 3.28
                "CB/_mobility": mobility_bowing("CB/mobility", U("-250 cm^2/V/s")),  #actual value, from Palankovski, 180 for 1-band, -250 for the gamma valley
                "CBX/_mobility": mobility_bowing("CBX/mobility", U("1e6 cm^2/V/s")),
                # No published bowing parameter for the L valley, so follow the
                # CBX treatment above: C appears in the *denominator* of
                # mobility_bowing, so a large C makes the bowing term negligible
                # and leaves a plain harmonic interpolation between the parents.
                # Substitute a real value here if one becomes available.
                "CBL/mobility": mobility_bowing("CBL/mobility", U("1e6 cm^2/V/s")),
                "VB/mobility": mobility_bowing("VB/mobility", U("125 cm^2/V/s")),

                #thermal velocity
                # "CB/vth": dolfin.conditional(dolfin.lt(X, 0.45), 3.07e5, 2.84e5)*U("m/s"),   #actual value, source: http://www.ioffe.ru/SVA/NSM/Semicond/AlGaAs/basic.html
                "CB/_vth": vegard("CB/vth", U("0")),
                "CBX/_vth": vegard("CBX/vth", U("0")),
                "CBL/vth": vegard("CBL/vth", U("0")),
                "VB/vth": vegard("VB/vth", U("0")),

                # SRH recombination lifetimes, Table 3.38
                #"SRH/CB/tau": U("8e-9 s"),  #actual value, from Palankovski

                # TODO: "SRH/CB/tau" & SRH/VB/tau" should depend on the Aluminium fraction: # https://www.iue.tuwien.ac.at/phd/quay/node42.html
                "SRH/CB/tau": U("0.14e-9 s"), # source (see note above): https://www.iue.tuwien.ac.at/phd/quay/node42.html
                "SRH/CBX/tau": U("0.3e-9 s"),
                "SRH/CBL/tau": U("0.3e-9 s"), #TODO - find appropriate lifetimes for these valleys
                #"SRH/VB/tau": U("8e-9 s"),  #actual value, from Palankovski
                "SRH/VB/tau": U("0.14e-9 s"), # source (see note above): https://www.iue.tuwien.ac.at/phd/quay/node42.html

                # TODO - absorpton coefficienty, update these values to reflect the actual material
                "opt_cv/alpha" : U("2e4 cm^-1"),  # source: http://www.matprop.ru/AlGaAs_optic
                # "opt_gv/alpha" : U("1e4 cm^-1"),
                # "opt_lg/alpha" : U("1e4 cm^-1"),
            }
        )

        T = self.temperature

        # Palankovski Table 3.20 gives only bowing parameters, mass values from: https://1aip-scitation-org.proxy.bib.uottawa.ca/doi/pdf/10.1063/1.336070
        # mn = dolfin.conditional(dolfin.lt(X, 0.45), U("0.1"), U("0.117"))
        mn = vegard("CB/mDOS", U("0"))
        mnX = vegard("CBX/mDOS", U("0"))
        mnL = vegard("CBL/mDOS", U("0"))
        mp = vegard("VB/mDOS", U("0"))
        m_e = U.electron_mass
        k_B = U.boltzmann_constant
        h = U.planck_constant
        DOS_term = lambda m : (2 * pi * m * m_e * k_B * T / h ** 2) ** (1.5)
        NC = 2 * d["CB/MC"] * DOS_term(mn)
        NCX = 2 * d["CBX/MC"] * DOS_term(mnX)
        NCL = 2 * d["CBL/MC"] * DOS_term(mnL)
        NV = 2 * DOS_term(mp)

        # Energy offset, Eqn (3.99)
        EgX_300K = vegard("CBX/Eg_300K", U("0 eV"))
        E_off = (
            AluminumArsenide["VB/E_off"] * (EgX_300K - GalliumArsenide["CB/Eg_300K"])
            - GalliumArsenide["VB/E_off"] * (EgX_300K - AluminumArsenide["CB/Eg_300K"])
        ) / (AluminumArsenide["CB/Eg_300K"] - GalliumArsenide["CB/Eg_300K"])

        EgX = d["CBX/Eg_300K"]
        Eg = d["CB/Eg_300K"]
        EgL = d["CBL/Eg_300K"]

        # Eg = d["CB/Eg_300K"]
        # EgX = d["CBX/Eg_300K"]
        # EgL = d["CBL/Eg_300K"]

        # _E_off = lambda bandgap : (
        #     AluminumArsenide["VB/E_off"] * (bandgap - GalliumArsenide["CB/Eg_300K"])
        #     - GalliumArsenide["VB/E_off"] * (bandgap - AluminumArsenide["CB/Eg_300K"])
        #     ) / (AluminumArsenide["CB/Eg_300K"] - GalliumArsenide["CB/Eg_300K"])

        # E_off = conditional(_E_off(Eg), _E_off(EgX))

        d.update(
            {
                "CB/vth": conditional(d["CB/_vth"], d["CBX/_vth"]),
                "CB/mobility" : conditional(d["CB/_mobility"], d["CBX/_mobility"]),
                "CB/mDOS" : conditional(mn, mnX),
                "CB/effective_density_of_states": conditional(NC, NCX),
                # This should be the correct energy level, accounting for alloy fraction, but the simulation does not work with this
                # "CB/energy_level": conditional(_E_off(Eg)+Eg, _E_off(EgX)+EgX),

                # Export the satellite-valley quantities that were previously
                # computed only as private intermediates for the CB conditional
                # above.  The X and L valleys are not needed for a single-CB
                # simulation, but a valley-resolved one asks for these keys and
                # the values already exist.
                "CBX/vth": d["CBX/_vth"],
                "CBX/mobility": d["CBX/_mobility"],

                "CBX/mDOS": mnX,
                "CBL/mDOS": mnL,
                "VB/mDOS": mp,
                "CB/energy_level": E_off + Eg,
                "CBX/energy_level": E_off + EgX,
                "CBL/energy_level": E_off + EgL,
                "VB/energy_level": E_off,
                "CBX/effective_density_of_states": NCX,
                "CBL/effective_density_of_states": NCL,
                "VB/effective_density_of_states": NV,
                "mole_fraction": X,
                "SRH/energy_level": E_off + Eg / 2, #TODO it has to be source- and destination-band dependent
            }
        )

        return d
