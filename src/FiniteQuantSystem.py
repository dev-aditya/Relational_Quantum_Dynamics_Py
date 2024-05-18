# %%
from matplotlib.pyplot import (
    figure,
    plot,
    show,
    title,
    xlabel,
    ylabel,
    legend,
    grid,
    style,
)
from qutip import Qobj, tensor, expect, ket2dm, entropy_vn, qeye
from numpy import vdot, zeros


class SpinQuantSystem:
    def __init__(
        self,
        Hs,
        Hc,
        V=None,
        clock_state=None,
        pos_of_glob_ket=None,
        ext_glob_state=None,
    ) -> None:
        print("Initializing SpinQuantSystem")
        # ---------------------------------------
        # Set the Hamiltonians for the system and clock
        self.Hs = Hs
        self.Hc = Hc
        # ---------------------------------------
        # Compute the eigenstates of the clock and system Hamiltonians
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
        # Set the clock state
        self.clock_state = clock_state
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
            self._antiCommuPsiV = (
                self.V * self.glob_ket.proj() + self.glob_ket.proj() * self.V
            )
            print(
                "Global eigenstate at pos {} from the list of eigestates with energy {} is used \n The position attribute is changed to {}".format(
                    pos, self.glob_energy, self.pos
                )
            )
        else:
            self.glob_ket = ext_glob_state.unit()
            self.glob_energy = expect(self.H, self.glob_ket)
            self._antiCommuPsiV = (
                self.V * self.glob_ket.proj() + self.glob_ket.proj() * self.V
            )
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
            # In general can be provided externally
            rho = self.glob_ket
        else: rho = self.Hstates[pos]
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

    def energy_spectrum(self, *args, **kwargs):
        N = len(self.eigenenergies)
        for n in range(N):
            rho_red = ket2dm(self.eigenstates[n]).ptrace(1)
            rho_red_diag = [expect(rho_red, m) for m in self.Hc.eigenstates()[1]]
            Xi_spectrum = [
                expect(self.clock_state(0, *args, **kwargs).proj(), m)
                for m in self.Hc.eigenstates()[1]
            ]
            Hc_eigen_val = self.Hc.eigenenergies()

            style.use(["nature", "grid"])
            figure(figsize=(12, 8))
            title("Energy Eigenstate: {}".format(n))
            plot(Hc_eigen_val, rho_red_diag, "-.", label="Rh_c")
            plot(Hc_eigen_val, Xi_spectrum, "--", label="Xi_c")
            legend()
            xlabel("Energy")
            ylabel("<m|rhp_c|m>")
            grid(True)
            show()
        return None


# %%
