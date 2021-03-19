from ipywidgets import interact, FloatSlider, IntSlider, interact_manual
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sympy as sm
import pandas as pd
from scipy import signal
import inspect
from munch import Munch
import streamlit as st

# Print sympy quantities using nice LaTeX
sm.init_printing()

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
omega_R = sm.symbols('omega_R', real=True)
omega_L = sm.symbols('omega_L', real=True)
K_i = sm.Symbol('K_i', real=True)
K_p = sm.Symbol('K_p', real=True)
K_d = sm.Symbol('K_d', real=True)
T_s = sm.Symbol('T_s', positive=True)
z = sm.symbols('z', real=True)
z_i = sm.Symbol('z_i', real=True)

# Continuous to discrete transformation
pade = {s:2/T_s*(1-z**-1)/(1+z**-1)}

# Bode plot utility functions
def dB_sm(x):
    return 20*sm.log(sm.Abs(x), 10)

def dB(x):
    return 20*np.log10(np.abs(x))

def adjust_y_ticks(ax, delta):
    ylim = np.array(ax.get_ylim()) / delta
    ymin = ylim[0]//1  # Round towards - infinity
    ymax = -(-ylim[1]//1)  # Trick to round towards + infinity
    # Note: this rounds away from zero so we never make the axis limits smaller
    ax_new_lim = np.array([ymin, ymax]) * delta
    ax_new_ticks = np.arange(ax_new_lim[0], ax_new_lim[1]+1, delta)
    ax.set_ybound(*ax_new_lim)
    ax.set_yticks(ax_new_ticks)

def find_gain(mag, gain):
    if mag.max() > gain:
        return np.max(np.flatnonzero(mag > gain))
    else:
        return 0
    
def unity_gain(mag):
    return find_gain(mag, 0)
    
def phase_margin(mag, phase):
    return ((phase[unity_gain(mag)] % 360) - 180)  

def bode_f(func, xlim, N):
    f_min = xlim[0]
    f_max = xlim[1]
    f = np.logspace(np.log10(f_min), np.log10(f_max), N)
    
    transfer = func(f)
    mag = dB(transfer)
    phase = 180/np.pi*np.unwrap(np.angle(transfer))
    return f, mag, phase

def bode_f_kwargs(func, xlim, N, **kwargs):
    f_min = xlim[0]
    f_max = xlim[1]
    f = np.logspace(np.log10(f_min), np.log10(f_max), N)
    
    transfer = func(f, **kwargs)
    
    try:
        mag = dB(transfer)
        phase = 180/np.pi*np.unwrap(np.angle(transfer))
    except:
        print(transfer)
        raise
        
    return f, mag, phase

def get_bode_s(laplace, s, xlim, N=1000):
    f = sm.Symbol('f')
    laplace_numpy = sm.lambdify([f], laplace.subs(s, 2j*np.pi*f), modules='numpy')
    
    return bode_f(laplace_numpy, xlim, N)

def bode_quantities(loop_transfer, var, xlim, N):
    numerical_transfer = sm.lambdify(var, loop_transfer, modules='numpy')
    
    return bode_f(numerical_transfer, xlim, N)
    

def bode_plot(loop_transfer, var, xlim, N=1000, gain_point=0, mag_lim=None, phase_lim=None, figax=None):
    data = f_range, magnitude, phase = bode_quantities(loop_transfer, var, xlim, N)
    
    gain_index = find_gain(magnitude, gain_point)
    print(u"Freq at gain: {:.2g}".format(f_range[gain_index]))
    print(u"Phase at gain: {:.0f}°".format(phase[gain_index]))
    
    return bode(f_range, magnitude, phase, gain_point, xlim, mag_lim, phase_lim, figax)

def bode(freq, mag, phase, gain_point=0, xlim=None, mag_lim=None, phase_lim=None, figax=None):
    """Make a nice bode plot for the given frequency, magnitude, and phase data.
    
    If provided, xlim is the xlimits to set on the plot.
    If provided, mag_lim is a tuple with (mag_min, mag_max, mag_delta),
    and phase is also a tuple of the same type."""
    gain_index = find_gain(mag, gain_point)
    with mpl.rc_context({'figure.figsize': (8,6), 'lines.linewidth': 1.5, 'figure.dpi':300, 'savefig.dpi': 300, 'font.size': 16,}):
        if figax is None:
            fig, (ax1, ax2) = plt.subplots(nrows=2)
        else:
            fig, (ax1, ax2) = figax
        ax1.yaxis.grid(True, linestyle='-', color='.8')
        ax2.yaxis.grid(True, linestyle='-', color='.8')
        ax1.semilogx(freq, mag)

        if xlim is not None:
            ax1.set_xlim(*xlim)
        if mag_lim is not None:
            ax1.set_ylim(mag_lim[0], mag_lim[1])
            adjust_y_ticks(ax1, mag_lim[2])
        else:
            adjust_y_ticks(ax1, 20)
    
        ax1.axvline(x=freq[gain_index], color='k',  linestyle='--')

        ax2.semilogx(freq, phase, linewidth=1.5)
        
        if xlim is not None:
            ax2.set_xlim(*xlim)
        if phase_lim is not None:
            ax2.set_ylim(phase_lim[0], phase_lim[1])
            adjust_y_ticks(ax2, phase_lim[2])
        else:
            adjust_y_ticks(ax2, 30)

        ax2.axvline(x=freq[gain_index], color='k',  linestyle='--')
        ax1.set_ylabel('Magnitude [dB]')
        ax2.set_ylabel('Phase [degrees]')
        ax2.set_xlabel('Frequency')
        fig.tight_layout()
        fig.show()
        return fig, (ax1, ax2)
    
def bode_plot_s(loop_transfer, var, xlim, N=1000, gain_point=0, mag_lim=None, phase_lim=None, figax=None):
    """Create a bode plot for a sympy transfer function, expressed in terms of var s,
    plotted over a frequency range given in xlim.
    
    Other parameters described below.
    
    N is the number of points to calculated the transfer function at.
    gain_point determines where to draw a line on the plot
        this can be used to highlight the unity gain bandwidth, or the -3 dB point of a filter.
    
    mag_lim gives the magnitude axis limits as a tuple (low, high, tick_spacing)
    phase_lim gives the phase axis limits as a tuple (low, high, tick_spacing)
    figax allows the figure to be drawn on axes you provide, rather than a new set of axes.
        It should be a nested tuple, (fig, (ax1, ax2))
    """
    f = sm.symbols('f', real=True)
    return bode_plot(loop_transfer.subs(var, 2*sm.pi*sm.I*f), f, xlim, N, gain_point, mag_lim, phase_lim, figax)

def bode_plot_f(func, xlim, N, gain_point=0, mag_lim=None, phase_lim=None, **kwargs):
    freq, magnitude, phase =  bode_f_kwargs(func, xlim, N, **kwargs)
    gain_index = find_gain(magnitude, gain_point)
    print(u"Freq at gain: {:.2g}".format(freq[gain_index]))
    print(u"Phase at gain: {:.0f}°".format(phase[gain_index]))
    return bode(freq, magnitude, phase, gain_point=gain_point, mag_lim=mag_lim, phase_lim=phase_lim)

def bode_plot_interact(loop_transfer, var, xlim, N=1000, gain_point=0, mag_lim=None, phase_lim=None, subs={}, plot_kwargs={}):
    """A utility function for interacting with sympy transfer functions. Put variables to vary in subs, with the tuple you would
    pass to interact (min, max, delta), such as,
    
    {omega_0: (0, 1, 0.01)}"""
    f = sm.symbols('f', real=True)
    loop_transfer_f = loop_transfer.subs(var, 2j*sm.pi*f)
    
    variables = [f]
    variables.extend(subs.keys())
    func = sm.lambdify(variables, loop_transfer_f, dummify=False, modules='numpy')
    
    args = inspect.getargspec(func).args
    args.remove('f')
    new_kwargs = {key:val for key, val in zip(args, subs.values())}
    
    def _interact_f(**kwargs):
        print(kwargs)
        if 'gain_point' in plot_kwargs:
            return bode_plot_f(func, xlim, N, mag_lim=mag_lim, phase_lim=phase_lim, **kwargs)
        else:
            return bode_plot_f(func, xlim, N, gain_point=gain_point, mag_lim=mag_lim, phase_lim=phase_lim, **kwargs)
    
    new_kwargs.update(plot_kwargs)
    
    interact(_interact_f, **new_kwargs)

def difference_coeffs_exact(laplace, s):
    """Symbolic expression of numerator, denominator"""
    z = sm.symbols('z')
    z_i = sm.symbols('z_i')
    T_s = sm.symbols('T_s')
    pade = {s:2/T_s*(1-z**-1)/(1+z**-1)}
    numer_denom = laplace.subs(pade).subs({z:1/z_i}).simplify().as_numer_denom()
    numer_denom_coeffs = [sm.Poly(expr, z_i).all_coeffs() for expr in numer_denom]
    a0 = numer_denom_coeffs[1][-1]
    b = [x/a0 for x in numer_denom_coeffs[0][-1::-1]]
    a = [x/a0 for x in numer_denom_coeffs[1][-1::-1]]
    return (b, a)

def discrete(laplace, s, z_i, Ts):
    """Output b, a, dt in a form which can be used by scipy.signal.lfilter"""
    
    # z_i is shorthand for z**-1
    T_s = sm.symbols('T_s')
    pade = {s:2/T_s*(1-z_i)/(1+z_i)}
    numer_denom = laplace.subs(pade).simplify().subs(T_s, Ts).as_numer_denom()
    try:
        numer_denom_coeffs = [np.array(sm.Poly(expr, z_i).all_coeffs(), dtype=np.float64) for expr in numer_denom]
    except TypeError:
        print(sm.Poly(numer_denom[0], z_i).all_coeffs())
        raise
    
    a0 = numer_denom_coeffs[1][-1]
    b = numer_denom_coeffs[0][-1::-1]/a0
    a = numer_denom_coeffs[1][-1::-1]/a0
    return (b, a, Ts)

# Need to make all of this stuff a 1x function evaluation

@st.cache
def setup_pi_example():
    m = Munch()
    m.low_pass = 1/(1+s/omega_0)
    (b, a, dt) = signal.cont2discrete(signal.butter(1, 2*np.pi, analog=True), dt=0.001, method='bilinear')
    m.dt = dt
    m.t = np.arange(0, 10, dt)
    m.x_sq = np.where(m.t>2, 1,0)
    m.y = signal.lfilter(b[0], a, m.x_sq)
    m.transfer = K_p*m.low_pass/(1+K_p*m.low_pass)
    return m



@st.cache
def proportional_interact(m, Kp, f0):
    df = 1e-3
    subs = {K_p: Kp,
            omega_0: 2*np.pi*f0}
    b_lp, a_lp, _ = discrete(m.low_pass.subs(subs), s, z_i, m.dt)
    y_lp = signal.lfilter(b_lp, a_lp, m.x_sq)
    b_fb, a_fb, _ = discrete(m.transfer.subs(omega_0, 2*np.pi*f0)\
                                        .subs(K_p, Kp), s, z_i, 0.001)
    y_fb = signal.lfilter(b_fb, a_fb, m.x_sq)
    return y_lp, y_fb


def make_pi_plot(t, x_sq, y_lp, y_fb):
    fig, ax = plt.subplots()
    ax.set_xlim(1.9, 3)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Output")
    ax.plot(t, x_sq, 'b-', t, y_lp, 'g-', t, y_fb, 'r-', linewidth=1.5)
    return fig, ax

m1 = setup_pi_example()
m1.low_pass
T_s = sm.symbols('T_s')
pade = {s:2/T_s*(1-z_i)/(1+z_i)}
numer_denom = m1.low_pass.subs(pade).simplify().as_numer_denom()
numer_denom
numer_denom_coeffs = [sm.Poly(expr, z_i).all_coeffs() for expr in numer_denom]
a0 = numer_denom_coeffs[1][-1]
b = [c/a0 for c in numer_denom_coeffs[0][-1::-1]]
a = [c/a0 for c in numer_denom_coeffs[1][-1::-1]]
a0
"B coeffs:"
b[0]
b[1]
"A coeffs:"
a[0]
a[1]
ba_lp = sm.lambdify([omega_0, T_s], [b, a]) # This is what I need!

# To break this down, I need: 


out = ba_lp(2*np.pi*1, 0.001)
out

Kp = st.slider("Kp", 1, 100)
f0 = st.slider("f0", 0.5, 1.5, 0.1)
y_lp, y_fb = proportional_interact(m1, Kp, f0)
fig, ax = make_pi_plot(m1.t, m1.x_sq, y_lp, y_fb)
st.pyplot(fig)


