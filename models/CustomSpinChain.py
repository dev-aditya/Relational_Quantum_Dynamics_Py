# %%
import sys
sys.path.insert(0, '/home/adityadev/Relational_Time_MS_Thesis')
from src.FiniteQuantSystem import SpinQuantSystem
import qutip as qt
from qutip import tensor
from numpy import linspace, pi, abs
import random
import matplotlib.pyplot as plt
import scienceplots   
plt.style.use(['nature', 'grid'])
"""
THIS MDOEL GIVES FOR SOME STRAGNE REASON, ZERO INTERACTION POTENTIAL!!??

"""

# Define the parameters and other variables
N = 5 # number of spins
sigx = qt.sigmax()
sigz = qt.sigmaz()
sigy = qt.sigmay()
sigp = qt.sigmap()
sigm = qt.sigmam()
I = qt.identity(2)
spinup = qt.spin_state(1/2, 1/2)
spindw = qt.spin_state(1/2, -1/2)
superpos = (spinup + spindw).unit()

def sig(k, matrix, size = N): 
    """task_kwargs = {"norm": True}
    Returns the input matix at the kth position in the tensor product N spins-1/2 system
    """
    if k >= N:
        raise ValueError("k must be less than N")
    mat_lis = [I] * size
    mat_lis[k] = matrix
    return tensor(mat_lis)
# ----------------------------------------------
omega = 0.1 #transverse field
"""
Somehow the frequency of the emergent potential depends on g, the coupling constant
and the coefficient depend on the frequency of the transverse field
"""
g = 0.01 # coupling 
Hs = (sigz  + omega * sigx)
Hc = sum([sig(i, sigx, size = N-1) * sig(i+1, sigx, size = N-1) for i in range(1, N-2)])
V = g * sum([(sig(0, sigx) + sig(0, sigz)) * sig(i, sigz) for i in range(1, N)])
#-- Clock Spin System
def Xi(t, energy, size = N-1):
    state = []
    for i in range(size):
        random.seed(i)
        theta = random.uniform(0, 2*pi)
        phi =random.uniform(0, pi)
        state.append(qt.spin_coherent(1/2, theta, phi))
    state = tensor(state)
    eiden = energy*tensor([I] * size)
    return ((-1j*(Hc - eiden)*t).expm() * state).unit()
# %%
quant_sys = SpinQuantSystem(Hs, Hc, V = V, clock_state=Xi)
T = linspace(0, 2*pi/omega, 1000)
psi_lamb = qt.parallel.parallel_map(quant_sys.psi_lambda, T,progress_bar = True, num_cpus = 8,\
    task_kwargs = {"norm": True, "energy": quant_sys.glob_energy})

V_lis  = qt.parallel.parallel_map(quant_sys.Vs_lambda, T, progress_bar = True, num_cpus = 8, \
    task_kwargs = {"energy": quant_sys.glob_energy})
# -- Plotting
psi_c1 = [i.full()[0] for i in psi_lamb]
psi_c2 = [i.full()[1] for i in psi_lamb]

plt.figure(figsize=(12, 8))
plt.plot(T, abs(psi_c1), "-.", label="C1")
plt.plot(T, abs(psi_c2), "--", label="C2")
plt.plot(T, abs(psi_c1)**2 + abs(psi_c2)**2, "--", label="Norm")
plt.title("Coefficients of the clock state")
plt.xlabel("Time")
plt.ylabel("c's")
plt.grid(True)
plt.legend()
plt.show()

Vx = [0.5*(v*qt.sigmax()).tr() for v in V_lis]
Vy = [0.5*(v*qt.sigmay()).tr() for v in V_lis]
Vz = [0.5*(v*qt.sigmaz()).tr() for v in V_lis]
plt.figure(figsize=(12, 8))
plt.plot(T, Vy, "-.", label="Vy", )
plt.plot(T, Vx, "--", label="Vx")
plt.plot(T, Vz, "--", label="Vz")
plt.title("The effective interaction potential")
plt.xlabel("Time")
plt.ylabel("V's")
plt.grid(True)
plt.legend()
plt.show()
# %%
