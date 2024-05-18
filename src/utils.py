import qutip as qt
from numpy import exp, pi, sqrt, arange

# %%
def dial_states(k, S):
    N = int(2*S + 1)
    v = exp(1j*2*pi*k*S/N) * qt.spin_state(S, -S) 
    for m in arange(-S + 1, S+1):
        v = v + exp(-1j*2*pi*k*m/N) * qt.spin_state(S, m)
    return v/sqrt(N)

# %%
def Tc(τ, S):
    N = int(2*S + 1)
    return τ*sum([k*qt.ket2dm(dial_states(k, S)) for k in range(N)])