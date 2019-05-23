from __future__ import division, absolute_import, print_function
from builtins import bytes, dict, int, range, str, super

import numpy as np

def depletion_approximation(u, N_D, N_A, N_i,
                            tau_n, tau_p,
                            beta, permittivity, diffusivity,
                            V_ext, x_cp, x_nc, x_pn):

    D = diffusivity
    epsilon = permittivity

    q_e = 1*u.elementary_charge

    n_n0 = 1/2 * (N_D + np.sqrt(N_D**2 + 4*N_i**2))
    p_p0 = 1/2 * (N_A + np.sqrt(N_A**2 + 4*N_i**2))

    V_bi = (1/beta/q_e)*np.log(n_n0*p_p0/N_i**2)
    DeltaV = V_bi - V_ext
    w = (2*(V_bi)*epsilon/q_e*(1/N_A+1/N_D))**0.5
    L_n = (tau_n*D)**0.5
    L_p = (tau_p*D)**0.5
    w_n = (2*DeltaV*epsilon/(q_e*(N_D/N_A+1)*N_D))**0.5
    w_p = (2*DeltaV*epsilon/(q_e*(N_A/N_D+1)*N_A))**0.5
    V_bi = V_bi.to(u.volt)
    L_p = L_p.to(u.m)
    L_n = L_n.to(u.m)
    w = w.to(u.m)

    # invoking law of the junction
    expbetaVextm1 = np.expm1(beta*q_e*V_ext)
    n_z_p_minus_n0 = N_i**2/N_A*expbetaVextm1
    p_z_n_minus_n0 = N_i**2/N_D*expbetaVextm1
    B_n = n_z_p_minus_n0
    B_p = p_z_n_minus_n0

    z_n = x_pn+w_n
    z_p = x_pn-w_p
    A_p = -B_p*np.tanh( (x_nc-z_n)/L_p)
    A_n = -B_n*np.tanh(-(x_cp-z_p)/L_n)

    # check convention whether it's min_maj or maj_min
    p_n0 = N_i**2/p_p0
    n_p0 = N_i**2/n_n0

    # for k,v in sorted(locals().items()):
    #     print(k, v)
    # exit(0)

    def u_(units, xs):
        return (x.m_as(units) for x in xs)

    charge_units = u.elementary_charge
    conc_units = 1 / u.mesh_unit**3
    length_units = u.mesh_unit
    energy_units = u.eV

    _A_p,_B_p,_n_p0,_A_n,_B_n,_p_n0,_N_A,_N_D = u_(conc_units, (
     A_p, B_p, n_p0, A_n, B_n, p_n0, N_A, N_D))
    _z_p,_z_n,_L_p,_L_n,_x_pn = u_(length_units, (
     z_p, z_n, L_p, L_n, x_pn))
    _epsilon, = u_(u.elementary_charge/u.mesh_unit/u.V, (epsilon,))
    _q_e, = u_(charge_units, (q_e,))
    _beta, = u_(1/energy_units, (beta,))

    def depl_V(x):
        if x < _z_p:
            return 0.0
        else:
            V_pn = _q_e/_epsilon*_N_A*0.5*(min(_x_pn, x) - _z_p)**2
            if x < _x_pn:
                return V_pn
            else:
                return V_pn + _q_e/_epsilon*_N_D*0.5*(
                    (_x_pn - _z_n)**2 - (_z_n - min(x, _z_n))**2)

    return dict(depl_V=depl_V)


