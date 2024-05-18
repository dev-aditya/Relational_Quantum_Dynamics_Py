# %%
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["nature", "grid"])
from numpy import linspace, pi, abs, sqrt, zeros, vdot, savez, real
import matplotlib.pyplot as plt
import qutip as qt
from qutip import tensor
import sys, subprocess

sys.path.insert(0, "/home/dev/Relational_Time_MS_Thesis")
from src.FiniteQuantSystem import SpinQuantSystem

# --------------------------------------
print("Number of CPUs: {}".format(qt.settings.num_cpus))

# Define the parameters and other variables
N = 8  # number of spins
sigx = qt.sigmax()
sigz = qt.sigmaz()
sigy = qt.sigmay()
sigp = qt.sigmap()
sigm = qt.sigmam()
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
g = (sqrt(5) - 1) / 10  # coupling
Hs = sigz
Hc = sum([sig(i, sigz, size=N - 1) for i in range(N - 1)])
Hc = Hc + sum(
    [sig(i, sigx, size=N - 1) * sig(i + 1, sigx, size=N - 1) for i in range(N - 2)]
)
V = g * sum(
    [
        sig(0, sigx, size=N) * sig(1, sigx, size=N),
        sig(N - 1, sigx, size=N) * sig(0, sigx, size=N),
    ]
)
# -- Clock Spin System
theta, phi = pi / 4 * sqrt(2), pi / sqrt(2)


def Xi(t, energy, size=N - 1):
    state = []
    for i in range(size):
        state.append(qt.spin_coherent(1 / 2, theta, phi))
    state = tensor(state)
    eiden = energy * tensor([I] * size)
    return ((-1j * (Hc - eiden) * t).expm() * state).unit()

T = linspace(0, 4 * pi, 2000)
savez("data/time.npz", T)
quant_sys = SpinQuantSystem(Hs, Hc, V=V, clock_state=Xi, pos_of_glob_ket=0)
Henergy, Hstates = quant_sys.Henergies, quant_sys.Hstates
savez(f"data/{N}_Henergy.npz", Henergy)
savez(f"data/{N}_Hstates.npz", Hstates)
for pos_eig in range(2**N):
    quant_sys.set_glob_state(pos=pos_eig)
    dir_name = "data/eigen_index_{}".format(pos_eig)
    subprocess.run(["mkdir", dir_name])

    if (quant_sys.entanglement_entropy() <= 1e-5) or (
        abs(quant_sys.glob_energy) < 1e-12
    ):
        print(
            """
              For index {} the entanglement entropy is {} and energy is {}. 
              And I'v observed that the effective potential returns a division by zero error.
              Only the coefficients of the system state are plotted.
              
              """.format(
                pos_eig, quant_sys.entanglement_entropy(), quant_sys.glob_energy
            )
        )
        psi_lamb = qt.parallel.parallel_map(
            quant_sys.psi_lambda,
            T,
            task_kwargs={"norm": False, "energy": quant_sys.glob_energy},
        )
        savez(
            dir_name + f"/psi_lambda_for_eigenindex_{pos_eig}_spins_{N}.npz", psi_lamb
        )
        # -- Plotting
        psi_c1 = [i.full()[0] for i in psi_lamb]
        psi_c2 = [i.full()[1] for i in psi_lamb]

        plt.figure(figsize=(19.20, 10.80))
        plt.plot(T, abs(psi_c1), "-.", label="C1", linewidth=0.8)
        plt.plot(T, abs(psi_c2), "--", label="C2", linewidth=0.8)
        plt.title(
            "The plot are for entanglement entropy {} and energy {}".format(
                quant_sys.entanglement_entropy(), quant_sys.glob_energy
            )
        )
        plt.xlabel("Time")
        plt.ylabel("C's")
        plt.grid(True)
        plt.legend()
        plt.savefig(
            dir_name + f"/Cs_for_eigenindex_{pos_eig}_spins_{N}.pdf",
            format="pdf",
            dpi=1200,
        )
        continue

    psi_lamb = qt.parallel.parallel_map(
        quant_sys.psi_lambda,
        T,
        task_kwargs={"norm": False, "energy": quant_sys.glob_energy},
    )
    savez(dir_name + f"/psi_lambda_for_eigenindex_{pos_eig}_spins_{N}.npz", psi_lamb)
    # -- Plotting
    psi_c1 = [i.full()[0] for i in psi_lamb]
    psi_c2 = [i.full()[1] for i in psi_lamb]

    V_lis = qt.parallel.parallel_map(
        quant_sys.Vs_lambda, T, task_kwargs={"energy": quant_sys.glob_energy}
    )
    savez(dir_name + f"/Vs_lambda_for_eigenindex_{pos_eig}_spins_{N}.npz", psi_lamb)
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(19.20, 15.80))

    # Plot data on the subplots
    ax1.plot(T, abs(psi_c1), "-.", label="C1", linewidth=0.8)
    ax1.plot(T, abs(psi_c2), "--", label="C2", linewidth=0.8)
    ax1.plot(
        T,
        sqrt(abs(psi_c1) ** 2 + abs(psi_c2) ** 2),
        "-.",
        label="|C1|^2+|C2|^2",
        linewidth=0.8,
    )
    ax1.set_title("Coefficients of the system state")
    ax1.set_ylabel("C's")
    ax1.grid(True)
    ax1.legend()
    ax1.grid(True)
    Vx = [0.5 * (v * sigx).tr() for v in V_lis]
    Vy = [0.5 * (v * sigy).tr() for v in V_lis]
    Vz = [0.5 * (v * sigz).tr() for v in V_lis]
    ax2.plot(T, real(Vy), "-.", label="Vy", linewidth=0.8)
    ax2.plot(T, real(Vz), "--", label="Vz", linewidth=0.8)
    ax2.plot(T, real(Vx), "-.", label="Vx", linewidth=0.8)
    ax2.set_title("The effective potential")
    ax2.set_ylabel("V's")
    ax2.set_xlabel("Time")
    # Specify the figure size
    ax2.grid(True)
    plt.suptitle(
        "The plot are for entanglement entropy {} and energy {}".format(
            quant_sys.entanglement_entropy(), quant_sys.glob_energy
        )
    )
    # save the figure to file
    plt.savefig(dir_name + f"/Cs_Vs_for_eigenindex_{pos_eig}_spins_{N}.pdf", dpi=1200)
    plt.close()

# %%
