# %%
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

# version_table()
from scipy.constants import golden_ratio as alpha

# === Defining the class ===#
from qutip import Qobj, tensor, expect, entropy_vn, qeye
from numpy import vdot, zeros

class SpinBosonQuantSystem:
    def __init__(
        self,
        Hs,
        Hc,
        V=None,
        clock_state=None,
        pos_of_glob_ket=None,
        ext_glob_state=None,
    ) -> None:
        print("Initializing Boson-Boson Quantum System")
        # -- Set the Hamiltonians for the system and clock --#
        self.Hs = Hs
        self.Hc = Hc

        # -- Compute the eigenstates of the clock and system Hamiltonians --#
        self.clock_basis = self.Hc.eigenstates()[1]
        self.sys_basis = self.Hs.eigenstates()[1]
        # ---------------------------------------
        # Compute the dimensions of the system, environment, and global Hilbert spaces
        self.dimSys, self.dimEnv = self.Hs.shape[0], self.Hc.shape[0]
        self.dimGlob = self.dimSys * self.dimEnv

        # Compute the total Hamiltonian of the system and clock
        if V is None:
            self.V = tensor(
                Qobj(zeros(self.Hs.shape), dims=self.Hs.dims),
                Qobj(zeros(self.Hc.shape), dims=self.Hc.dims),
            )
            self.H = tensor(self.Hs, qeye(self.Hc.dims[0])) + tensor(
                qeye(self.Hs.dims[0]), self.Hc
            )
        else:
            self.V = V
            self.H = (
                tensor(self.Hs, qeye(self.Hc.dims[0]))
                + tensor(qeye(self.Hs.dims[0]), self.Hc)
                + self.V
            )
        # ---------------------------------------
        # Set the position of the global state
        if pos_of_glob_ket is None:
            self.pos = int(self.dimGlob / 2)
        else:
            self.pos = pos_of_glob_ket
        # ---------------------------------------
        # Compute the eigenstates of the total Hamiltonian and set the global state
        self.Henergies, self.Hstates = None, None
        self.set_glob_state(ext_glob_state=ext_glob_state)
        # Compute the entanglement entropy of the global state
        self.entang_entropy = self.entanglement_entropy()
        # Set the clock state
        self.clock_state = clock_state
        self._antiCommuPsiV = (
            self.V * self.glob_ket.proj() + self.glob_ket.proj() * self.V
        )
        # ---------------------------------------

    def set_glob_state(self, pos=None, ext_glob_state=None):
        if pos is None:
            pos = self.pos
        if ext_glob_state is None:
            if self.Henergies is None and self.Hstates is None:
                import time

                start = time.time()
                self.Henergies, self.Hstates = self.H.eigenstates()
                print(
                    "Took {} seconds to calculate the eigenstates of the Hamiltonian".format(
                        time.time() - start
                    )
                )
            self.glob_energy, self.glob_ket = self.Henergies[pos], self.Hstates[pos]
            self.pos = pos
            print(
                "Global eigenstate at pos {} from the list of eigestates with energy {} is used \n The position attribute is changed to {}".format(
                    pos, self.glob_energy, self.pos
                )
            )
            print(
                "The entanglement entropy of the global state with respect to the clock is {}".format(
                    self.entanglement_entropy()
                )
            )
        else:
            self.glob_ket = ext_glob_state.unit()
            self.glob_energy = expect(self.H, self.glob_ket)
            print(
                " An external global state is provided, the energy expectation of the state is {} ".format(
                    self.glob_energy
                )
            )

    def psi_lambda(self, t, norm=True, *args, **kwargs):
        psi_basis = self.sys_basis
        clock_state_t = self.clock_state(t, *args, **kwargs)
        clock_coeff_conj = [
            vdot(clock_state_t.full(), i.full()) for i in self.clock_basis
        ]
        state = psi_basis[0] * 0
        for a in range(psi_basis.__len__()):
            coeff = 0
            for b in range(self.clock_basis.__len__()):
                coeff += clock_coeff_conj[b] * vdot(
                    tensor(psi_basis[a], self.clock_basis[b]).full(),
                    self.glob_ket.full(),
                )
            state += coeff * psi_basis[a]
        if norm:
            return state.unit()
        return state

    def Vs_lambda(self, t, norm=True, *args, **kwargs):
        """
        energy: energy of clock state
        returns: Vs(t) in the eigenbasis of H(t)
        """
        clock_at_t = self.clock_state(t, *args, **kwargs).proj()
        if norm:
            denom_norm = expect(
                tensor(qeye(self.dimSys), clock_at_t),
                self.glob_ket,
            )
            oper_ = (
                tensor(qeye(self.dimSys), clock_at_t) * self._antiCommuPsiV
            ).ptrace(0) / denom_norm
            return oper_
        oper_ = (tensor(qeye(self.dimSys), clock_at_t) * self._antiCommuPsiV).ptrace(0)
        return oper_

    def entanglement_entropy(self, pos=None, base=2):
        if pos is None:
            pos = self.pos
        """Calculate entanglement entropy of the
        global state with respect to the clock hilbert space
        """
        rho = self.Hstates[pos]
        entang_ent = entropy_vn(rho.ptrace(0), base=base)
        return entang_ent

    def E(self, t, *args, **kwargs):
        N_lambda_ = expect(
            tensor(qeye(self.dimSys), self.clock_state(t, *args, **kwargs).proj()),
            self.glob_ket,
        )

        oper_ = self.V * tensor(
            qeye(self.dimSys), self.clock_state(t, *args, **kwargs).proj()
        )
        return expect(oper_, self.glob_ket) / N_lambda_


# ==========================#


# -- Parameters --#
size = 1  # no of harmonic oscillator bath modes
Ns = [100] * size # no of states in each harmonic oscillator
ws = [2 * np.pi] * size # frequencies of each harmonic oscillator
ms = [1] * size
As = [qt.destroy(N) for N in Ns]
As_dag = [i.dag() for i in As]
Xs = [(a + a_dag) / np.sqrt(2*m*w) for a, a_dag, m, w in zip(As, As_dag, ms, ws)]
Ps = [np.sqrt(m*w)*-1j*(a - a_dag) / np.sqrt(2) for a, a_dag, m, w in zip(As, As_dag, ms, ws)]

def H_SHO(k, size=size):
    """
    Returns the Hamiltonian of SHO at the kth position in the tensor product N spins-1/2 system
    """
    if k >= size:
        raise ValueError("k must be less than N")
    mat_lis = [qt.identity(i) for i in Ns]
    mat_lis[k] = ((ms[k]*ws[k])**2 * Xs[k] ** 2 + Ps[k] ** 2) / (2*ms[k])
    return tensor(mat_lis)


def OperAtk(k, oper, size=size):
    """
    Returns the input matix at the kth position in the tensor product N spins-1/2 system
    """
    if k >= size:
        raise ValueError("k must be less than N")
    mat_lis = [qt.identity(i) for i in Ns]
    mat_lis[k] = oper
    return tensor(mat_lis)

h = 1
K = 1
Hs = h*qt.sigmaz() + K*qt.sigmax()
Hc = sum([H_SHO(i) for i in range(size)])
g = [1] * size
V = qt.tensor(qt.sigmaz(), sum([g[i]*OperAtk(i, Xs[i]) for i in range(size)]))
def Xi(t):
    return qt.tensor(
        [qt.coherent(Ns[i], np.exp(-1j * ws[i] * t) * alpha) for i in range(size)]
    )
T = np.linspace(0.0, 1.0, 1000, dtype=np.float128)
X = Xs[0]
X_expect = [expect(X, Xi(t)) for t in T]
SpinBoson_sys = SpinBosonQuantSystem(Hs, Hc, V, clock_state=Xi)
# %%
for i in range(SpinBoson_sys.Hstates.__len__()):
    SpinBoson_sys.set_glob_state(i)
    psi_lis = qt.parallel.parallel_map(SpinBoson_sys.psi_lambda, T, progress_bar=True, task_kwargs={"norm": True})
    psi_c1 = [i.full()[0] for i in psi_lis]
    psi_c2 = [i.full()[1] for i in psi_lis]
    plt.figure(figsize=(12, 8))
    plt.plot(X_expect, np.abs(psi_c1), "-.", label="C1")
    plt.plot(X_expect, np.abs(psi_c2), "--", label="C2")
    plt.title(
        "Entropy {} and Energy {}".format(
            SpinBoson_sys.entanglement_entropy(), SpinBoson_sys.glob_energy
        )
    )
    plt.xlabel(r"\rangle X_{SHO} \langle")
    plt.ylabel("C's")
    plt.grid(True)
    plt.legend()
    plt.savefig("data/SpinBoson_size{}_psi_{}.png".format(size, i))
    plt.close()

# %%
