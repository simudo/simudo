from math import pi
from simudo.physics import Material


class GermaniumMaterial(Material):
    """Germanium material data based on Palankovski.

       V. Palankovski and R. Quay, "Analysis and Simulation of Heterostructure Devices",
       Springer-Verlag (2004).
    """

    name = "Germanium"

    def get_dict(self):
        d = super().get_dict()
        U = self.unit_registry

        d.update(
            {
                # Static dielectric constant, Palankovski Table 3.3
                "poisson/permittivity": U("16.0 vacuum_permittivity"),
                # Table 3.9
                # These X value are for strained Ge on Si, not bulk Ge. Used for interpolating alloys.
                "CB/EgX_0K": U("0.57 eV"),
                "CB/ChiX_0K": U("4.199 eV") - U("4 eV"),
                "CB/EgL_0K": U("0.7437 eV"),
                "CB/ChiL_0K": U("4.0253 eV"),
                "CB/varshni_alphaL": U("4.774e-4 eV/K"),
                "CB/varshni_betaL": U("636 K"),
                # Table 3.21
                "CB/MCL": U("4"),
                # Table 3.18
                "CB/mL": U("0.222"),
                "CB/m1n": U("0.0068"),
                # Table 3.19
                "VB/m0p": U("0.28"),
                "VB/m1p": U("0.1"),
                "VB/m2p": U("0.0"),
                "CB/SRV": U("4000 cm/s"),
                "VB/SRV": U("4000 cm/s"),
                "SRH/CB/tau": U("1e-9 s"),
                "SRH/VB/tau": U("1e-6 s"),
                # Properties for band-to-band tunneling
                "CB/eff_mass_tunnel": U("0.05"),
                "VB/eff_mass_tunnel": U("0.1"),
                "CB/tunnel_offset_energy": U("0 eV"),
                "VB/tunnel_offset_energy": U("0 eV"),
            }
        )

        T = self.temperature
        Eg = d["CB/EgL_0K"] - d["CB/varshni_alphaL"] * T ** 2 / (
            d["CB/varshni_betaL"] + T
        )
        Chi = d["CB/ChiL_0K"] + d["CB/varshni_alphaL"] * T ** 2 / (
            d["CB/varshni_betaL"] + T
        )

        mn = d["CB/mL"] + d["CB/m1n"] * (T / U("300 K"))
        mp = (
            d["VB/m0p"]
            + d["VB/m1p"] * (T / U("300 K"))
            + d["VB/m2p"] * (T / U("300 K")) ** 2
        )

        m_e = U.electron_mass
        k_B = U.boltzmann_constant
        h = U.planck_constant
        NC = 2 * d["CB/MCL"] * (2 * pi * mn * m_e * k_B * T / h ** 2) ** (3 / 2)
        NV = 2 * (2 * pi * mp * m_e * k_B * T / h ** 2) ** (3 / 2)

        d.update(
            {
                "CB/mDOS": mn,
                "VB/mDOS": mp,
                "CB/energy_level": -Chi,
                "VB/energy_level": -Chi - Eg,
                "CB/effective_density_of_states": NC,
                "VB/effective_density_of_states": NV,
                # TODO - update these from Palankovski
                "CB/mobility": U("1400 cm^2/V/s"),
                "VB/mobility": U(" 450 cm^2/V/s"),
                # SRH energy level position
                "SRH/energy_level": -Chi - Eg / 2,
            }
        )

        return d
