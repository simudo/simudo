from math import pi
from simudo.physics import Material
from .silicon import SiliconMaterial
from .germanium import GermaniumMaterial
from .helpers import Alloy

class SiliconGermaniumAlloy(Material):
    '''Silicon-Germanium material data based on Palankovski.

       This is for Si(1-x)Ge(x) strained to the Silicon lattice.
       mole fraction should be specified by the MoleFractionX spatial rule.

       V. Palankovski and R. Quay, "Analysis and Simulation of Heterostructure Devices",
       Springer-Verlag (2004).
    '''

    name = 'SiliconGermanium'

    def get_dict(self):
        d = super().get_dict()
        U = self.unit_registry
        MoleFractionX = self.problem_data.pdd.spatial.value_rules['MoleFractionX'][0].value
        Silicon = SiliconMaterial(problem_data=self.problem_data).get_dict()
        Germanium = GermaniumMaterial(problem_data=self.problem_data).get_dict()
        print(MoleFractionX)

        alloy = Alloy(U, MoleFractionX, at_x0=Silicon, at_x1=Germanium)
        vegard = alloy.vegard

        def linear_interp(param):
            return Silicon[param]*(U('1') - MoleFractionX) + Germanium[param]*MoleFractionX

        d.update({
            'poisson/permittivity': vegard('poisson/permittivity', U('0')),

            # Table 3.9
            'CB/EgX_0K': vegard('CB/EgX_0K', U('-0.4 eV')),
            'CB/ChiX_0K': vegard('CB/ChiX_0K', U('0.4 eV')),
            'CB/varshni_alphaX': Silicon['CB/varshni_alphaX'],
            'CB/varshni_betaX':  Silicon['CB/varshni_betaX'],

            # Table 3.21
            'CB/MCX': U('6'),

            # recombination velocity at interfaces with other materials
            'CB/SRV': U('500 cm/s'),
            'VB/SRV': U('500 cm/s'),

            'SRH/CB/tau':   U('1e-9 s'),
            'SRH/VB/tau':   U('1e-6 s'),

            # Properties for band-to-band tunneling
            'CB/eff_mass_tunnel': U('0.05'),
            'VB/eff_mass_tunnel': U('0.1'),
            'CB/tunnel_offset_energy': U('0 eV'),
            'VB/tunnel_offset_energy': U('0 eV')
        })

        T = self.temperature
        Eg  = d['CB/EgX_0K']  - d['CB/varshni_alphaX'] * T**2 / (d['CB/varshni_betaX'] + T)
        Chi = d['CB/ChiX_0K'] + d['CB/varshni_alphaX'] * T**2 / (d['CB/varshni_betaX'] + T)

        mn = vegard('CB/mDOS', U('-0.183'))
        mp = vegard('VB/mDOS', U('-0.096'))
        NC = 2 * d['CB/MCX'] * (2 * pi * mn * U.electron_mass * U.boltzmann_constant * T / U.planck_constant**2)**(3/2)
        NV = 2 * (2 * pi * mp * U.electron_mass * U.boltzmann_constant * T / U.planck_constant**2)**(3/2)

        d.update({
            'CB/mDOS': mn,
            'VB/mDOS': mp,
            'CB/energy_level': -Chi,
            'VB/energy_level': -Chi -Eg,
            'CB/effective_density_of_states': NC,
            'VB/effective_density_of_states': NV,

            # TODO - update these from Palankovski
            'CB/mobility': U('1400 cm^2/V/s'),
            'VB/mobility': U(' 450 cm^2/V/s'),

            'SRH/energy_level': -Chi -Eg/2
        })

        return d
