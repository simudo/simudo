
from pint import UnitRegistry
import numpy as np

u = UnitRegistry()

T = 300*u.K
beta = (1/(u.k*T).to(u.eV))

E_c = 1.12 * u.eV
E_v = 0.0  * u.eV

# # smack in middle of bandgap
# # we take degeneracy g=1 because it won't move it that much
# # away from the middle of bandgap
# E_t = (E_c + E_v)/2 #*np.log(2)

# p1 u("1.87786e10 centimeter**-3")
# n1 u("4.69466e9  centimeter**-3")

N_v = (1.8e19)*u.cm**-3
N_c = (3.2e19)*u.cm**-3

n_i_eff = (N_v*N_c)**0.5 * np.exp((E_v - E_c)*beta/2)

E_trap = E_v - np.log(u('9389112518.281794 centimeter**-3')/N_v)/beta

SRH_p1 = N_v*np.exp((E_v-E_trap) * beta)
SRH_n1 = N_c*np.exp((E_trap-E_c) * beta)

N_A = 1e18*u.cm**-3
N_D = 1e16*u.cm**-3

# maybe want different one for holes?
D_n = D_p = 20 * u.cm**2/u.s

_mblty_units = u("cm^2/V/s")
mblty_n = (D_n*beta*u.e).to(_mblty_units)
mblty_p = (D_p*beta*u.e).to(_mblty_units)

# SRH tau
SRH_tau_p = SRH_tau_n = 1e-5 * u.s


# total device length
# left half p-type, right half n-type
dev_length = 200 * u.micrometer

# I've been using vacuum permittivity until just now, oops
# not sure what units you want this in
_vacuum_epsilon = 8.85418782e-12*u("C/m/V").to("elementary_charge/m/V")
epsilon = 11.7*_vacuum_epsilon

for k, v in sorted(globals().items()):
    if isinstance(v, u.Quantity) and not k.startswith("_"):
        print("{:>11s} = {}".format(k, v))

