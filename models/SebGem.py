# %%
import numpy as np
import qutip as qt
import sys
sys.path.insert(0, '/home/adityadev/Relational_Time_MS_Thesis')
from src.FiniteQuantSystem import SpinQuantSystem


Ec = V0 = 1
a = np.sqrt(3) + 1

print("""Setting the system and clock Hamiltonians""")
Hs = qt.Qobj(np.zeros([2, 2]), dims=qt.sigmaz().dims, )
Hc = qt.Qobj(Ec*qt.sigmaz())
Vs = V0 * (qt.sigmaz() + qt.sigmax())
Vc = qt.sigmax()
V = qt.tensor(Vs,  Vc)

def Xi(t, energy):
    phase = np.exp(1j*energy*t)/(2 * np.sqrt(1 + a * np.cos(t)**2))
    _state = phase * (np.exp(-1j*t)*qt.basis(2, 0) +
                        np.exp(1j*t)*qt.basis(2, 1))
    return _state
pos = 0
quant_sys = SpinQuantSystem(Hs, Hc, V, pos_of_glob_ket=pos, clock_state=Xi)
quant_sys.set_glob_state(ext_glob_state=qt.Qobj(np.array(
    [1, 0, -1, -a]), dims=quant_sys.glob_ket.dims, shape=quant_sys.glob_ket.shape))
energy = quant_sys.H.eigenenergies()[0]

def analy_state(t):
    phase = np.exp(1j*a*t) / (2 * np.sqrt(1 + a*np.cos(t)**2))
    state = phase * (qt.basis(2, 0) - (a*np.exp(-2j*t) + 1)*qt.basis(2, 1))
    return state.unit()

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['nature', 'grid'])

time = np.linspace(0, 2*np.pi, 100)
psi_lambda_list = [quant_sys.psi_lambda(t, energy=energy) for t in time]
psi_c1 = [i.full()[0] for i in psi_lambda_list]
psi_c2 = [i.full()[1] for i in psi_lambda_list]
analy_state_list = [analy_state(t) for t in time]
analy_c1 = [i.full()[0] for i in analy_state_list]
analy_c2 = [i.full()[1] for i in analy_state_list]

plt.figure(figsize=(12, 8))
plt.plot(time, np.abs(psi_c1), "-.", label="Numerical")
plt.plot(time, np.abs(analy_c1), "o", label="Analytical")
plt.title("C1")
plt.xlabel("Time")
plt.ylabel("C's")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(time, np.abs(psi_c2), "-.", label="Numerical")
plt.plot(time, np.abs(analy_c2), "o", label="Analytical")
plt.title("C1")
plt.xlabel("Time")
plt.ylabel("C's")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(time, np.abs(psi_c1), "-.", label="C1")
plt.plot(time, np.abs(psi_c2), "o", label="c2")
plt.title("Coefficients")
plt.xlabel("Time")
plt.ylabel("C's")
plt.grid(True)
plt.legend()
plt.show()
# ---------------------------------------
# Plots of the overlap of analytical and numerical solutionsSebGem_Sim
plt.figure(figsize=(12, 8))
plt.plot(time, [np.sqrt((quant_sys.psi_lambda(t, energy=energy).dag() * analy_state(t)).norm()) for t in time],
            "-.", color="#ff7f0e", label="Overlap")
plt.xlabel("Time")
plt.ylabel("Fidelity")
plt.grid(True)
plt.show()
# ---------------------------------------
Vx = [0.5*(quant_sys.Vs_lambda(t, energy=energy)*qt.sigmax()).tr() for t in time]
Vy = [0.5*(quant_sys.Vs_lambda(t, energy=energy)*qt.sigmay()).tr() for t in time]
Vz = [0.5*(quant_sys.Vs_lambda(t, energy=energy)*qt.sigmaz()).tr() for t in time]
plt.figure(figsize=(12, 8))
plt.plot(time, Vy, "-.", label="Vy", color="orange")
plt.plot(time, -((a/2)*np.sin(2*time))/(1 + a * (np.cos(time)**2)),
            color="green", label="Analytical")
plt.xlabel("Time")
plt.ylabel("Vy")
plt.grid(True)
plt.legend()
plt.show()
# ---------------------------------------
plt.figure(figsize=(12, 8))
plt.plot(time, Vx, "-.", label="Vx", color="blue")
plt.plot(time, Vz, "-.", label="Vz", color="green")
plt.plot(time, (np.cos(2*time) + a * np.cos(time)**2)/(1 + a *
                                                        (np.cos(time)**2)), "-.", color="green", label="Analytical")
plt.xlabel("Time")
plt.ylabel("V")
plt.grid(True)
plt.legend()
plt.show()
# ---------------------------------------
plt.figure(figsize=(12, 8))
plt.plot(time, np.imag([quant_sys.E(t, energy=energy) for t in time]), "-.",
            label="E(lambda).imag", color="blue")
plt.plot(time, np.real([quant_sys.E(t, energy=energy) for t in time]), "-.",
            label="E(lambda).real", color="green")
plt.plot(time, np.abs([quant_sys.E(t, energy) for t in time]), "-.",
            color="green", label="E(lambda).abs")
plt.xlabel("Time")
plt.ylabel("E(lambda)")
plt.grid(True)
plt.legend()
plt.show()

# %%
