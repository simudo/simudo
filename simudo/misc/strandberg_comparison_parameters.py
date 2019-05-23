
import sympy as S
from pint import UnitRegistry
import numpy as np
from math import pi

u = UnitRegistry()

_e = u.elementary_charge
T = 300*u.K
beta = (1/(u.k*T).to(u.eV))

E_c = 2.0 * u.eV
E_i = 1.2 * u.eV
E_v = 0.0 * u.eV

N_v = (2e19)*u.cm**-3
N_c = (2e19)*u.cm**-3

n_i_eff = (N_v*N_c)**0.5 * np.exp((E_v - E_c)*beta/2)

# maybe want different one for holes?
D_n = D_p = 500*0.02585 * u.cm**2/u.s

_mblty_units = u("cm^2/V/s")
mblty_n = (D_n*beta*u.e).to(_mblty_units)
mblty_p = (D_p*beta*u.e).to(_mblty_units)

# total device length
dev_length = 5 * u.um

# permittivity
_vacuum_epsilon = 8.85418782e-12*u("C/m/V").to("elementary_charge/m/V")
epsilon = 13*_vacuum_epsilon

# index of refraction
# index_of_refraction = (epsilon / _vacuum_epsilon)**0.5
index_of_refraction = 1.0 # mother of all inconsistencies
# yes I know this is wrong, see the paragraph just above the section
# 3.1 title

_strb_K = (8*pi*index_of_refraction**2
          / u.planck_constant**3 / u.speed_of_light**2)

def strb_compute_I(beta, E0, E1):
    E, beta_, E0_, E1_ = S.symbols('E beta E0 E1')
    I_ = S.integrate(E**2 * S.exp(-beta_*E), E)
    I_ = I_.subs({E: E1_}) - I_.subs({E: E0_})

    E_unit = u.eV
    val = I_.subs({E0_: E0.m_as(E_unit),
                   E1_: E1.m_as(E_unit),
                   beta_: beta.m_as(1/E_unit)})
    return val * E_unit**3

strb_I_cv = _strb_K * strb_compute_I(
    beta, E_c-E_v, 1e3*u.eV)         # because literal inf gives nan
strb_I_iv = _strb_K * strb_compute_I(
    beta, E_i-E_v, E_c-E_v)
strb_I_ci = _strb_K * strb_compute_I(
    beta, E_c-E_i, E_i-E_v)

_I_unit = u('cm**-2 * s**-1')
strb_I_cv = strb_I_cv.to(_I_unit)
strb_I_iv = strb_I_iv.to(_I_unit)
strb_I_ci = strb_I_ci.to(_I_unit)

strb_σ_iv = strb_σ_ci = 3e-14 * u.cm**2
strb_α_cv = 1e4 * u.cm**-1

strb_N_ib = 2.5e17 * u.cm**-3

# 1/c^2 * V_bi * mu * alpha^2 * tau

def tau_ci(n_ib):
    return 1/(strb_σ_ci * (strb_N_ib - n_ib) / N_c * np.exp(beta * (E_c - E_i)) * strb_I_ci)
def tau_iv(n_ib):
    return 1/(strb_σ_iv * n_ib / N_v * np.exp(beta * (E_i - E_v)) * strb_I_iv)

jk_fhalf_n_ib = strb_N_ib/2

jk_tau_ci_fhalf = tau_ci(jk_fhalf_n_ib)
jk_tau_iv_fhalf = tau_iv(jk_fhalf_n_ib)

jk_alpha_ci_fhalf = jk_fhalf_n_ib * strb_σ_ci
jk_alpha_iv_fhalf = jk_fhalf_n_ib * strb_σ_iv

# 1308237.46130036 / second
# 3750.0 / centimeter^2
# 1934.0864563095263 centimeter ** 2 / second / volt
# x volt

_Vth = 1/beta / _e
jk_nu_e = (jk_tau_ci_fhalf * jk_alpha_ci_fhalf**2 * mblty_n * _Vth).to('dimensionless')
jk_nu_h = (jk_tau_iv_fhalf * jk_alpha_iv_fhalf**2 * mblty_p * _Vth).to('dimensionless')

# def jk_nu(c, V, mu, N_I, σ):
#     alpha = N_I/2 * σ


# should be 3750 1/cm
zzdebug_strb_alpha_half_fill = (strb_N_ib/2 * strb_σ_iv).to('1/cm')

for k, v in sorted(globals().items()):
    if isinstance(v, u.Quantity) and not k.startswith("_"):
        print("{:>11s} = {}".format(k, v))

