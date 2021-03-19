
import streamlit as st
import numpy as np
from scipy import signal
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
from munch import Munch
from feedback_helpers import lp_discrete, cl_discrete

st.title("Proportional Controller Example 1")

@st.cache
def setup_pi_example(dt):
    m = Munch()
    m.dt = dt
    m.t = np.arange(0, 10, dt)
    m.x_sq = np.where(m.t>2, 1,0)
    # m.transfer = K_p*m.low_pass/(1+K_p*m.low_pass)
    return m

Ts = 0.001
m = setup_pi_example(Ts)

f0 = st.slider("f0", 0.5, 1.5, 0.01)
Kp = st.slider("Kp", 1, 100)

lpba = lp_discrete(f0, Ts)
clba = cl_discrete(f0, Kp, Ts)
y_lp = signal.lfilter(lpba[0], lpba[1], m.x_sq)
y_fb = signal.lfilter(clba[0], clba[1], m.x_sq)


fig, ax = plt.subplots()
ax.plot(m.t, m.x_sq, label="Input")
ax.plot(m.t, y_lp, label="Output without feedback")
ax.plot(m.t, y_fb, label="Output with feedback")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Output")
ax.legend()
st.pyplot(fig)



st.markdown("""
- $K_p$ is the proportional gain
- $f_0$ is the low-pass filter frequency
- Feedback used to speed up response
- Steady state error $1/K_p$ (proportional droop)
- High gain makes response insensitive to variations in $f_0$ (compare changes in orange curve to changes in green curve)
""")
