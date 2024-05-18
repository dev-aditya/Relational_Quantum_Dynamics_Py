import subprocess
import sys
sys.path.insert(0, "/home/adityadev/Relational_Time_MS_Thesis")
from src.FiniteQuantSystem import SpinQuantSystem
from qutip import tensor
import qutip as qt
from numpy import linspace, pi, abs, sqrt, zeros, vdot, savez, real
import matplotlib.pyplot as plt
import numpy as np
plt.style.use(["ggplot"])

# --------------------------------------
print("Number of CPUs: {}".format(qt.settings.num_cpus))

# Define the parameters and other variables
N = 8  # number of spins
SIGX = qt.sigmax()
SIGZ = qt.sigmaz()
SIGY = qt.sigmay()
I = qt.identity(2)


def sig(k, matrix, size=N):
    """
    Returns the input matix at the kth position in the tensor product N spins-1/2 system
    """
    if k >= N:
        raise ValueError("k must be less than N")
    mat_lis = [I] * size
    mat_lis[k] = matrix
    return tensor(mat_lis)


# ----------------------------------------------
"""
Somehow the frequency of the emergent potential depends on g, the coupling constant
and the coefficient depend on the frequency of the transverse field
"""
g = 1 # coupling
Hs = SIGZ
Hc = sum([sig(i, SIGZ, size=N - 1) for i in range(N - 1)])
for i in range(N - 1):
    for j in range(i + 1, N - 1):
        Hc += sig(i, SIGX, size=N - 1) * sig(j, SIGX, size=N - 1)

V = g * sig(0, SIGX, size=N) * sum([sig(i, SIGX, size=N) for i in range(1, N)])
# -- Clock Spin System
#THETA, PHI = pi / 4 * sqrt(2), pi / sqrt(2) for plot1
THETA, PHI = pi / 4, pi / 4

HC_EIG_ENERGY, HC_EIG_STATES = Hc.eigenstates()
def Xi(t, energy, size=N - 1):
    state = []
    for i in range(size):
        state.append(qt.spin_coherent(1 / 2, THETA, PHI))
    state0 = qt.tensor(state)
    state = 0 * state0
    state0 = state0.full()
    for m in range(HC_EIG_ENERGY.__len__()):
        state += (
            np.vdot(HC_EIG_STATES[m].full(), state0)
            * np.exp(-1j * (HC_EIG_ENERGY[m] - energy) * t)
            * HC_EIG_STATES[m]
        )
    return state.unit()

def Xi(t, energy, size=N - 1):
    state = HC_EIG_STATES[0]*0
    for m in range(HC_EIG_ENERGY.__len__()):
        state += np.exp(-1j * (HC_EIG_ENERGY[m] - energy) * t) * HC_EIG_STATES[m]
    return state.unit()

T = linspace(0, 4 * pi, 1000)
#savez("time.npz", T)
QUANT_SYS = SpinQuantSystem(Hs, Hc, V=V, clock_state=Xi, pos_of_glob_ket=0)
GLOB_EIG_ENERGY, GLOB_EIG_STATE = QUANT_SYS.Henergies, QUANT_SYS.Hstates
#savez(f"{N}_GLOB_EIG_ENERGY.npz", GLOB_EIG_ENERGY)
#savez(f"{N}_GLOB_EIG_STATE.npz", GLOB_EIG_STATE)
for pos_eig in range(2**N):
    QUANT_SYS.set_glob_state(pos=pos_eig)
    GLOB_ENTROPY = QUANT_SYS.entanglement_entropy()
    #dir_name = "eigen_index_{}".format(pos_eig)
    #subprocess.run(["mkdir", dir_name])

    psi_lamb = qt.parallel.parallel_map(
        QUANT_SYS.psi_lambda,
        T,
        task_kwargs={"norm": 0, "energy": QUANT_SYS.glob_energy},
    )
    #savez(dir_name + f"/psi_lambda_for_eigenindex_{pos_eig}_spins_{N}.npz", psi_lamb)
    # -- Plotting
    psi_c1 = [i.full()[0] for i in psi_lamb]
    psi_c2 = [i.full()[1] for i in psi_lamb]
    norm_ = sqrt(np.abs(psi_c1)**2 + np.abs(psi_c2)**2)
    #savez(dir_name + f"/Vs_lambda_for_eigenindex_{pos_eig}_spins_{N}.npz", psi_lamb)
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(19.20, 15.80))
    if any(norm_) <= 1e-5:
        # Plot data on the subplots
        ax1.plot(T, np.abs(psi_c1), "-.", label="C1", linewidth=0.8)
        ax1.plot(T, np.abs(psi_c2), "--", label="C2", linewidth=0.8)
        norm = False
    else:
        ax1.plot(T, np.abs(psi_c1)/norm_, "-.", label="C1", linewidth=0.8)
        ax1.plot(T, np.abs(psi_c2)/norm_, "--", label="C2", linewidth=0.8)
        norm = True
    V_lis = qt.parallel.parallel_map(
        QUANT_SYS.Vs_lambda, T, task_kwargs={"energy": QUANT_SYS.glob_energy, "norm": norm}
    )
    ax1.set_title("Coefficients of the system state")
    ax1.set_ylabel("C's")
    ax1.grid(True)
    
    ax1.grid(True)
    Vx = [0.5 * (v * SIGX).tr() for v in V_lis]
    Vy = [0.5 * (v * SIGY).tr() for v in V_lis]
    Vz = [0.5 * (v * SIGZ).tr() for v in V_lis]
    ax2.plot(T, real(Vy), "-.", label="Vy", linewidth=0.8)
    ax2.plot(T, real(Vz), "--", label="Vz", linewidth=0.8)
    ax2.plot(T, real(Vx), "-.", label="Vx", linewidth=0.8)
    ax2.set_title("The effective potential")
    ax2.set_ylabel("V's")
    ax2.set_xlabel("Time")
    ax2.legend()
    # Specify the figure size
    ax2.grid(True)
    plt.suptitle(
        "The plot are for entanglement entropy {} and energy {}".format(
            GLOB_ENTROPY, QUANT_SYS.glob_energy
        )
    )
    # save the figure to file
    dir_name = "data"
    plt.savefig(dir_name + f"/Cs_Vs_for_eigenindex_{pos_eig}_spins_{N}.png",)
    plt.show()
    plt.close()
