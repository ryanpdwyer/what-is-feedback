import numpy as np
import sympy as sm
from scipy import signal
from munch import Munch


# Lots of symbols that are useful for this material
f = sm.symbols('f', real=True)
omega = sm.symbols('omega', real=True)
tau = sm.symbols('tau', positive=True)
tau_d = sm.symbols('tau_d', positive=True)
s = sm.symbols('s')
k = sm.symbols('k', real=True)
f_c = sm.symbols('f_c', positive=True)
s_c = sm.symbols('s_c')
k_c = sm.symbols('k_c', positive=True)
V_0 = sm.symbols('V_0', positive=True)
alpha = sm.symbols('alpha', positive=True)
d = sm.symbols('d', positive=True)
K_R = sm.symbols('K_R', real=True)
omega_0 = sm.symbols('omega_0', real=True)
f_0 = sm.symbols('f_0', real=True)
omega_R = sm.symbols('omega_R', real=True)
omega_L = sm.symbols('omega_L', real=True)
K_i = sm.Symbol('K_i', real=True)
K_p = sm.Symbol('K_p', real=True)
K_d = sm.Symbol('K_d', real=True)
T_s = sm.Symbol('T_s', positive=True)
z = sm.symbols('z', real=True)
z_i = sm.Symbol('z_i', real=True)

# Continuous to discrete transformation
# pade = {s:2/T_s*(1-z**-1)/(1+z**-1)}


low_pass = (1/(1+s/omega_0)).subs(omega_0, 2*sm.pi*f_0)


# Function to take low_pass and discretize and lambdify it...
def discrete(args, laplace, s=s, z_i=z_i):
    """Output a function which returns a function that generates [b, a, T_s] in a form which can be used by scipy.signal.lfilter"""
    
    T_s = sm.Symbol('T_s', positive=True)
    pade = {s:2/T_s*(1-z_i)/(1+z_i)}

    numer_denom = laplace.subs(pade).simplify().as_numer_denom()
    numer_denom_coeffs = [sm.Poly(expr, z_i).all_coeffs() for expr in numer_denom]
    a0 = numer_denom_coeffs[1][-1]
    b = [c/a0 for c in numer_denom_coeffs[0][-1::-1]]
    a = [c/a0 for c in numer_denom_coeffs[1][-1::-1]]
    args.append(T_s)
    discrete_func = sm.lambdify(args, [b, a, T_s])
    return discrete_func


lp_discrete = discrete([f_0], low_pass)

closed_loop_lp = K_p*low_pass/(1+K_p*low_pass)
cl_discrete = discrete([f_0, K_p], closed_loop_lp)

