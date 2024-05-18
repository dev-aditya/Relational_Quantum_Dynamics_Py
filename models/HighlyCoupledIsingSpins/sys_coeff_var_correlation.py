import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from qutip import tensor
import sys

sys.path.insert(0, "/home/adityadev/Relational_Time_MS_Thesis")
from src.FiniteQuantSystem import SpinQuantSystem

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
g = 1  # coupling
Hs = SIGZ
Hc = sum([sig(i, SIGZ, size=N - 1) for i in range(N - 1)])
for i in range(N - 1):
    for j in range(i + 1, N - 1):
        Hc += sig(i, SIGX, size=N - 1) * sig(j, SIGX, size=N - 1)

V = g * sig(0, SIGX, size=N) * sum([sig(i, SIGX, size=N) for i in range(1, N)])
# -- Clock Spin System
HC_EIG_ENERGY, HC_EIG_STATES = Hc.eigenstates()


def Xi(t, energy, size=N - 1):
    state = HC_EIG_STATES[0] * 0
    for m in range(HC_EIG_ENERGY.__len__()):
        state += np.exp(-1j * (HC_EIG_ENERGY[m] - energy) * t) * HC_EIG_STATES[m]
    return state.unit()


T = np.linspace(0, 4 * np.pi, 1000)
QUANT_SYS = SpinQuantSystem(Hs, Hc, V=V, clock_state=Xi, pos_of_glob_ket=0)
GLOB_EIG_ENERGY, GLOB_EIG_STATE = QUANT_SYS.Henergies, QUANT_SYS.Hstates
var_c1 = np.zeros(GLOB_EIG_ENERGY.__len__(), dtype=np.float64)
var_c2 = np.zeros(GLOB_EIG_ENERGY.__len__(), dtype=np.float64)
for pos_eig in range(2**N):
    QUANT_SYS.set_glob_state(pos=pos_eig)
    GLOB_ENTROPY = QUANT_SYS.entanglement_entropy()
    psi_lamb = qt.parallel.parallel_map(
        QUANT_SYS.psi_lambda,
        T,
        task_kwargs={"norm": 0, "energy": QUANT_SYS.glob_energy},
    )
    psi_c1 = [i.full()[0] for i in psi_lamb]
    psi_c2 = [i.full()[1] for i in psi_lamb]
    norm_ = np.sqrt(np.abs(psi_c1) ** 2 + np.abs(psi_c2) ** 2)
    if any(norm_) <= 1e-10:
        print("Norm is zero")
    else:
        psi_c1 = psi_c1 / norm_
        psi_c2 = psi_c2 / norm_
    c1_abs = np.abs(psi_c1)
    c2_abs = np.abs(psi_c2)
    var_c1[pos_eig] = np.var(c1_abs)
    var_c2[pos_eig] = np.var(c2_abs)

plt.figure(figsize=(1200, 600))
fig, axs = plt.subplots(1, 2, tight_layout=True)
hist = axs[0].hist2d(
    GLOB_EIG_ENERGY,
    var_c1,
    bins=(70, 70),
    cmap="plasma",
    norm=mpl.colors.LogNorm(),
)
axs[0].set_title(f"Variance of C1", )
axs[0].set_ylabel(r"$\sigma ^2 _{c_1}$")
axs[0].set_xlabel("Energy")
cbar = plt.colorbar(hist[3], ax=axs[0], orientation="horizontal")


hist = axs[1].hist2d(
    GLOB_EIG_ENERGY,
    var_c2,
    bins=(70, 70),
    cmap="plasma",
    norm=mpl.colors.LogNorm(),
)

# Set plot labels
axs[1].set_title(f"Variance of C2",)
axs[1].set_xlabel("Energy")
axs[1].set_ylabel(r"$\sigma ^2 _{c_2}$")
# +"theta" +f"= {theta/np.pi}" + "pi"+  " and " + "phi" +  f"= {phi/np.pi}"+ "pi"
fig.suptitle(f"Overlap for Homogeneous sum of clock States \n N = {N} and g = {g}", fontsize=11)
cbar = plt.colorbar(hist[3], ax=axs[1], orientation="horizontal")
plt.savefig("data/coeff_variance.png", dpi=300)
plt.show()
