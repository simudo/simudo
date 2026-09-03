from math import pi
import numpy as np
from simudo.physics import Material

from .helpers import thermal_velocity


class GalliumNitrideMaterial(Material):
    """beta-GaN (FCC lattice) material data based on Springer 2017

        V. Palankovski and R. Quay, "Analysis and Simulation of Heterostructure Devices", Springer-Verlag (2004).
        Both gamma and L valley properties are included here for later use in the valleyPV GaNAs alloy.
        The X valley is deliberately absent -- its degeneracy is unknown, so none of it is exported.
        The CB energy level is still taken to be that of the lowest valley
    """

    ## NOTE: Note that, for values sourced from http://www.ioffe.ru/SVA/NSM/Semicond/GaN/index.html, we assume the zinc blende rather than wurzite configuration.
    ## The 2017 Springer handbook references both alpha-GaN and beta-GaN
    ## According to the book "Semiconductor Devices" by Amal Banerjee, this is a notation for group-III nitrides
    ## where alpha refers to a hexagonal close-packed configuration and beta refers to a face-centered cubic configuration.
    ## Palankovski page 5 (pdf 25) strongly implies the data from this book uses hexagonal GaN, which is supported by comparing the bandgap
    ## data listed here to that of the 2017 Springer handbook. Further confirmed with reference to a c-axis in Palankovski page 41

    ## Zincblende = FCC = beta
    ## Wurtzite = hexagonal cell = alpha

    name = "GalliumNitride"

    def get_dict(self):
        d = super().get_dict()
        U = self.unit_registry

        d.update(
            {
                # Static dielectric constant, Springer Table 30.23
                "poisson/permittivity": U("9.4 vacuum_permittivity"),
                # Springer handbook (2017) Table 30.12, Palankovski Table 3.10
                "CB/Eg_300K": U("3.42 eV"),
                "CBL/Eg_300K": U("5.5 eV"),
                # Table 3.15 band edge alignment
                "VB/E_off": U("-1.1 eV"),  # Palankovski
                # Table 3.21 Parameter for energy minima in the DOS model
                "CB/MC": U("1"),
                "CBL/MC": U("1"),
                # Springer handbook Table 30.17 Effective electron masses. Temperature dependence can be added later
                "CB/mn": U("0.15"),
                "CBL/mn": U("0.78"),
                # Springer Table 3.20 Effective hole masses
                # Hole mass calculated using eq. 3.107 in Palankovski
                "VB/mp": U("1.348"),  #actual value, source: http://www.ioffe.ru/SVA/NSM/Semicond/GaN/basic.html
                # Palankovski Table 3.22, lattice mobility temperature power
                "CB/y_0": U("-1.26"),
                "CBL/y_0": U("-1.26"), # NOTE: Palankovski
                "VB/y_0": U("-1.5"), # JB: Palankovski lists two VB mobilities, but this aligns with the one in this material file when I got it
                # conductivity mass, Springer Table 30.17
                "CBL/mc": U("0.36"),
                # Include other parameters for SRH etc. here.
                # SRH recombination lifetimes, Table 3.38
                "SRH/CB/tau": U("50e-9 s"),  #actual value not found, used value from GaAs in Palankovski
                "SRH/CBL/tau": U("0.3e-9 s"), #TODO - find appropriate lifetimes for these valleys
                "SRH/VB/tau": U("50e-9 s"),  #actual value not found, used value from GaAs in Palankovski

				# Springer 2017, Table 30.1
                "a": U("4.52e-10 m") # Lattice parameter
            }
        )

        T = self.temperature
        E_off = d["VB/E_off"] #add temperature dependence here, if desired
        # Effective electron and hole mass
        mn = d["CB/mn"]
        mnL = d["CBL/mn"]
        mp = d["VB/mp"]

        # Effective DOS. Palankovski eq. 3.111 & 3.112
        m_e = U.electron_mass
        k_B = U.boltzmann_constant
        h = U.planck_constant
        DOS_term = lambda m : (2 * pi * m * m_e * k_B * T / h ** 2) ** (3 / 2)
        NC = 2 * d["CB/MC"] * DOS_term(mn)
        NCL = 2 * d["CBL/MC"] * DOS_term(mnL)
        NV = 2 * DOS_term(mp)

        # Thermal velocity
        vth = lambda m : thermal_velocity(U, T, m)
        #Gamma valley
        mtc = d["CB/mn"]
        vth_c = vth(mtc)
        #L valley
        mtcL = d["CBL/mn"]
        vth_cL = vth(mtcL)
        # For valence band, just use the DOS effective mass
        mtv = d["VB/mp"]
        vth_v = vth(mtv)

        Eg = d["CB/Eg_300K"]
        EgL = d["CBL/Eg_300K"]

        # Palankovski eq. 3.115 incorporates temperature dependence with mobility
        mu_T = lambda mu, y_0 : mu * (T/U("300 K"))**y_0

        d.update(
            {
                # Table 3.22 Mobility
                "CB/mobility": mu_T(U("760 cm^2/V/s"), d["CB/y_0"]), # 760 in Springer, 1478 in Palankovski
                "CBL/mobility": mu_T(U("1478 cm^2/V/s"), d["CBL/y_0"]), #TODO - update L valley mobility
                "VB/mobility": mu_T(U("350 cm^2/V/s"), d["VB/y_0"]), # 350 in Springer, 30 in Palankovski

                "CB/mDOS": mn,
                "CBL/mDOS": mnL,
                "VB/mDOS": mp,
                "CB/energy_level": E_off + Eg,
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
                "CBL/effective_density_of_states": NCL,
                "CB/vth": vth_c,
                "CBL/vth": vth_cL,
                "VB/vth": vth_v,
            }
        )

        return d

	# Optical data sourced from:
    # T. Kawashima, H. Yoshikawa, S. Adachi. Optical properties of hexagonal GaN. J. Appl. Phys. 82, 3528-3535 (1997)
	# Data was collated into CSV files from:
    # https://refractiveindex.info/?shelf=main&book=GaN&page=Kawashima
    # TODO: This is simulated data for alhpa-GaN, though this site provides no good data for beta-GaN
    def optical_properties_table(self):
        '''Array of arrays, where each element is of the form [wl, alpha, n, k].'''
        return np.array([
			[123.98, 1.50972810e+06, 0.90117, 1.48950e+00],
			[124.64, 1.52421687e+06, 0.92162, 1.51180e+00],
			[125.30, 1.53805155e+06, 0.94340, 1.53360e+00],
			[125.97, 1.55101953e+06, 0.96654, 1.55480e+00],
			[126.65, 1.56293304e+06, 0.99104, 1.57520e+00],
			[127.33, 1.57383109e+06, 1.01690, 1.59470e+00],
			[128.03, 1.58328614e+06, 1.04410, 1.61310e+00],
			[128.72, 1.59149296e+06, 1.07260, 1.63020e+00],
			[129.43, 1.59790874e+06, 1.10220, 1.64580e+00],
			[130.15, 1.60248984e+06, 1.13290, 1.65970e+00],
			[130.87, 1.60529215e+06, 1.16450, 1.67180e+00],
			[131.60, 1.60584084e+06, 1.19680, 1.68170e+00],
			[132.34, 1.60417308e+06, 1.22940, 1.68940e+00],
			[133.09, 1.60023179e+06, 1.26210, 1.69480e+00],
			[133.84, 1.59408129e+06, 1.29450, 1.69780e+00],
			[134.61, 1.58542955e+06, 1.32620, 1.69830e+00],
			[135.38, 1.57483413e+06, 1.35690, 1.69660e+00],
			[136.16, 1.56221324e+06, 1.38600, 1.69270e+00],
			[136.95, 1.54787956e+06, 1.41330, 1.68690e+00],
			[137.75, 1.53223057e+06, 1.43830, 1.67960e+00],
			[138.56, 1.51565521e+06, 1.46070, 1.67120e+00],
			[139.38, 1.49853384e+06, 1.48040, 1.66210e+00],
			[140.21, 1.48159670e+06, 1.49720, 1.65310e+00],
			[141.05, 1.46520050e+06, 1.51120, 1.64460e+00],
			[141.90, 1.44995903e+06, 1.52230, 1.63730e+00],
			[142.76, 1.43629497e+06, 1.53090, 1.63170e+00],
			[143.63, 1.42488277e+06, 1.53730, 1.62860e+00],
			[144.51, 1.41603196e+06, 1.54200, 1.62840e+00],
			[145.40, 1.41021643e+06, 1.54550, 1.63170e+00],
			[146.30, 1.40763966e+06, 1.54880, 1.63880e+00],
			[147.22, 1.40840317e+06, 1.55260, 1.65000e+00],
			[148.14, 1.41280480e+06, 1.55810, 1.66550e+00],
			[149.08, 1.42041797e+06, 1.56650, 1.68510e+00],
			[150.03, 1.43085589e+06, 1.57920, 1.70830e+00],
			[150.99, 1.44331412e+06, 1.59770, 1.73420e+00],
			[151.96, 1.45626340e+06, 1.62320, 1.76100e+00],
			[152.95, 1.46754176e+06, 1.65650, 1.78620e+00],
			[153.95, 1.47457931e+06, 1.69750, 1.80650e+00],
			[154.96, 1.47486179e+06, 1.74390, 1.81870e+00],
			[155.99, 1.46641223e+06, 1.79190, 1.82030e+00],
			[157.03, 1.44981810e+06, 1.83620, 1.81170e+00],
			[158.08, 1.42770759e+06, 1.87220, 1.79600e+00],
			[159.15, 1.40413301e+06, 1.89790, 1.77830e+00],
			[160.23, 1.38337522e+06, 1.91440, 1.76390e+00],
			[161.32, 1.36810790e+06, 1.92470, 1.75630e+00],
			[162.44, 1.35921652e+06, 1.93290, 1.75700e+00],
			[163.56, 1.35636284e+06, 1.94250, 1.76540e+00],
			[164.71, 1.35833685e+06, 1.95600, 1.78040e+00],
			[165.87, 1.36376221e+06, 1.97510, 1.80010e+00],
			[167.04, 1.37106145e+06, 2.00090, 1.82250e+00],
			[168.23, 1.37891697e+06, 2.03390, 1.84600e+00],
			[169.44, 1.38590514e+06, 2.07410, 1.86870e+00],
			[170.67, 1.39093752e+06, 2.12130, 1.88910e+00],
			[171.92, 1.39295792e+06, 2.17510, 1.90570e+00],
			[173.18, 1.39124047e+06, 2.23480, 1.91730e+00],
			[174.46, 1.38485063e+06, 2.29940, 1.92260e+00],
			[175.76, 1.37310621e+06, 2.36770, 1.92050e+00],
			[177.08, 1.35570332e+06, 2.43850, 1.91040e+00],
			[178.42, 1.33228039e+06, 2.51020, 1.89160e+00],
			[179.78, 1.30297983e+06, 2.58120, 1.86410e+00],
			[181.17, 1.26794312e+06, 2.64980, 1.82800e+00],
			[182.57, 1.22786594e+06, 2.71440, 1.78390e+00],
			[183.99, 1.18342031e+06, 2.77370, 1.73270e+00],
			[185.44, 1.13554073e+06, 2.82660, 1.67570e+00],
			[186.91, 1.08526218e+06, 2.87230, 1.61420e+00],
			[188.41, 1.03360249e+06, 2.91050, 1.54970e+00],
			[189.93, 9.81530606e+05, 2.94100, 1.48350e+00],
			[191.47, 9.29991495e+05, 2.96420, 1.41700e+00],
			[193.04, 8.79658962e+05, 2.98050, 1.35130e+00],
			[194.63, 8.31150845e+05, 2.99050, 1.28730e+00],
			[196.25, 7.84845883e+05, 2.99500, 1.22570e+00],
			[197.90, 7.41028525e+05, 2.99470, 1.16700e+00],
			[199.57, 6.99817823e+05, 2.99030, 1.11140e+00],
			[201.28, 6.61345210e+05, 2.98270, 1.05930e+00],
			[203.01, 6.25502069e+05, 2.97230, 1.01050e+00],
			[204.78, 5.92266461e+05, 2.95980, 9.65150e-01],
			[206.57, 5.61541627e+05, 2.94560, 9.23080e-01],
			[208.40, 5.33136112e+05, 2.93030, 8.84150e-01],
			[210.26, 5.06934060e+05, 2.91410, 8.48200e-01],
			[212.15, 4.82776088e+05, 2.89740, 8.15040e-01],
			[214.08, 4.60473425e+05, 2.88030, 7.84460e-01],
			[216.04, 4.39910222e+05, 2.86320, 7.56290e-01],
			[218.04, 4.20913477e+05, 2.84610, 7.30330e-01],
			[220.07, 4.03366393e+05, 2.82920, 7.06400e-01],
			[222.15, 3.87110964e+05, 2.81260, 6.84340e-01],
			[224.26, 3.72060054e+05, 2.79640, 6.63980e-01],
			[226.42, 3.58071077e+05, 2.78070, 6.45170e-01],
			[228.61, 3.45087346e+05, 2.76550, 6.27790e-01],
			[230.85, 3.32974799e+05, 2.75080, 6.11690e-01],
			[233.13, 3.21681408e+05, 2.73660, 5.96780e-01],
			[235.46, 3.11101196e+05, 2.72310, 5.82920e-01],
			[237.84, 3.01182892e+05, 2.71020, 5.70040e-01],
			[240.26, 2.91867635e+05, 2.69790, 5.58030e-01],
			[242.74, 2.83077248e+05, 2.68620, 5.46810e-01],
			[245.27, 2.74777601e+05, 2.67510, 5.36310e-01],
			[247.84, 2.66923828e+05, 2.66460, 5.26440e-01],
			[250.48, 2.59449799e+05, 2.65480, 5.17150e-01],
			[253.17, 2.52330061e+05, 2.64550, 5.08360e-01],
			[255.92, 2.45518559e+05, 2.63690, 5.00010e-01],
			[258.73, 2.38990775e+05, 2.62890, 4.92060e-01],
			[261.60, 2.32708432e+05, 2.62140, 4.84440e-01],
			[264.53, 2.26648814e+05, 2.61460, 4.77110e-01],
			[267.54, 2.20763992e+05, 2.60830, 4.70010e-01],
			[270.61, 2.15046028e+05, 2.60260, 4.63090e-01],
			[273.75, 2.09462463e+05, 2.59740, 4.56300e-01],
			[276.97, 2.03987444e+05, 2.59280, 4.49600e-01],
			[280.26, 1.98602103e+05, 2.58870, 4.42930e-01],
			[283.64, 1.93271524e+05, 2.58520, 4.36240e-01],
			[287.09, 1.87994375e+05, 2.58220, 4.29490e-01],
			[290.63, 1.82734045e+05, 2.57960, 4.22620e-01],
			[294.26, 1.77464860e+05, 2.57760, 4.15560e-01],
			[297.98, 1.72175050e+05, 2.57600, 4.08270e-01],
			[301.80, 1.66831270e+05, 2.57480, 4.00670e-01],
			[305.71, 1.61417293e+05, 2.57400, 3.92690e-01],
			[309.73, 1.55902030e+05, 2.57360, 3.84260e-01],
			[313.85, 1.50263923e+05, 2.57350, 3.75290e-01],
			[318.09, 1.44464473e+05, 2.57350, 3.65680e-01],
			[322.44, 1.38489635e+05, 2.57360, 3.55350e-01],
			[326.91, 1.32302268e+05, 2.57360, 3.44180e-01],
			[331.51, 1.25883564e+05, 2.57320, 3.32090e-01],
			[336.24, 1.19231775e+05, 2.57200, 3.19030e-01],
			[341.11, 1.12394067e+05, 2.56920, 3.05090e-01],
			[346.12, 1.05571745e+05, 2.56370, 2.90780e-01],
			[351.27, 9.94412856e+04, 2.55350, 2.77970e-01],
			[356.59, 9.62661729e+04, 2.53570, 2.73170e-01],
			[362.07, 1.03735811e+05, 2.51670, 2.98890e-01],
			[367.72, 1.21815742e+05, 2.58220, 3.56460e-01],
			[373.54, 8.65321478e+04, 2.67120, 2.57220e-01],
			[379.56, 5.63460484e+04, 2.63420, 1.70190e-01],
			[385.77, 4.41062442e+04, 2.58910, 1.35400e-01],
			[392.19, 3.81134089e+04, 2.55390, 1.18950e-01],
			[398.83, 3.44761496e+04, 2.52610, 1.09420e-01],
			[405.69, 3.19138545e+04, 2.50310, 1.03030e-01],
			[412.80, 2.99117977e+04, 2.48350, 9.82590e-02],
			[420.16, 2.82465189e+04, 2.46620, 9.44430e-02],
			[427.78, 2.67989354e+04, 2.45080, 9.12280e-02],
			[435.69, 2.55036552e+04, 2.43670, 8.84240e-02],
			[443.90, 2.43214049e+04, 2.42380, 8.59140e-02],
			[452.42, 2.32278703e+04, 2.41180, 8.36260e-02],
			[461.27, 2.22063000e+04, 2.40060, 8.15120e-02],
			[470.48, 2.12446130e+04, 2.39000, 7.95390e-02],
			[480.06, 2.03345582e+04, 2.38010, 7.76820e-02],
			[490.04, 1.94698737e+04, 2.37060, 7.59250e-02],
			[500.45, 1.86452849e+04, 2.36160, 7.42540e-02],
			[511.31, 1.78572671e+04, 2.35300, 7.26590e-02],
			[522.65, 1.71026705e+04, 2.34470, 7.11320e-02],
			[534.50, 1.63790709e+04, 2.33680, 6.96670e-02],
			[546.90, 1.56837221e+04, 2.32920, 6.82570e-02],
			[559.89, 1.50148254e+04, 2.32190, 6.68980e-02],
			[573.52, 1.43707377e+04, 2.31480, 6.55870e-02],
			[587.82, 1.37502800e+04, 2.30800, 6.43200e-02],
			[602.86, 1.31516868e+04, 2.30140, 6.30940e-02],
			[618.68, 1.25742921e+04, 2.29510, 6.19070e-02],
			[635.36, 1.20167303e+04, 2.28890, 6.07570e-02],
			[652.96, 1.14780524e+04, 2.28290, 5.96410e-02],
			[671.57, 1.09573318e+04, 2.27710, 5.85580e-02],
			[691.26, 1.04537966e+04, 2.27150, 5.75050e-02],
			[712.15, 9.96663266e+03, 2.26600, 5.64820e-02],
			[734.34, 9.49519577e+03, 2.26070, 5.54870e-02],
			[757.96, 9.03881418e+03, 2.25560, 5.45190e-02],
			[783.14, 8.59703806e+03, 2.25050, 5.35770e-02],
			[810.06, 8.16893206e+03, 2.24570, 5.26590e-02],
			[838.89, 7.75427261e+03, 2.24090, 5.17650e-02],
			[869.85, 7.35230557e+03, 2.23630, 5.08930e-02],
			[903.18, 6.96271933e+03, 2.23180, 5.00430e-02],
			[939.17, 6.58497784e+03, 2.22740, 4.92140e-02],
			[978.15, 6.21862873e+03, 2.22310, 4.84050e-02],
		])
