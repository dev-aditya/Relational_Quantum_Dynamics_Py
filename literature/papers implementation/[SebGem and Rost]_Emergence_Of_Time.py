# %%
import qutip as qt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
# %%quant_sys.psi_lambda(t)
Ec = V0 = 1
a = np.sqrt(3) + 1

Hs = qt.Qobj(np.zeros([2, 2]), dims=[[2], [2]], )
Hc = qt.Qobj(Ec*qt.sigmaz())
Vs = V0 *(qt.sigmaz() + qt.sigmax())
Vc = qt.sigmax()

# %%
class QuantSys():
    def __init__(self, Hs, Hc, V = None, pos = 0) -> None:
        self.Hs = Hs ## Hamiltonian of system
        self.Hc = Hc ## Hamiltonian of clock
        if V is None: ## interaction Hamiltonian
            self.V = qt.tensor(qt.Qobj(np.zeros(Hs.shape)),\
                                qt.Qobj(np.zeros(Hc.shape)))
        else: 
            self.V = qt.tensor(V)
        self.dimSys, self.dimEnv = Hs.shape[0], Hc.shape[0] ## dimensions of system and environment
        self.dim = self.dimSys * self.dimEnv ## dimension of total Hilbert space
        self.H = qt.tensor(Hs, qt.qeye(self.dimEnv))\
            + qt.tensor(qt.qeye(self.dimSys), Hc) + self.V ## total Hamiltonian
        self.eigenenergies = self.H.eigenenergies() ## eigenenergies of total Hamiltonian
        self.eigenstates = self.H.eigenstates()     
        self.eigenstates = self.eigenstates[1]  ## eigenstates of total Hamiltonian
        self.eigenDict = {self.eigenenergies[i]: self.eigenstates[i]\
                           for i in range(len(self.eigenenergies))} ## eigenstates of total Hamiltonian as a dictionary
        self.pos = pos ## position of energy eigenvalue to be used
        self.glob_ket = self.glob_state(pos = self.pos) ## eigenstate of total Hamiltonian at position 'pos'

    def clock_state(self, t, pos = None):
        if pos is None:
            pos = self.pos
        energy = self.eigenenergies[pos]
        phase = np.exp(1j*energy*t)/(2 * np.sqrt(1 + a* np.cos(t)**2))
        _state =  phase *  (np.exp(-1j*t)*qt.basis(2, 0) + np.exp(1j*t)*qt.basis(2, 1))
        return _state
    
    def glob_state(self, pos = 0): ## returns: eigensate of total Hamiltonian at position 'pos'
        #return self.eigenstates[pos]
        return qt.Qobj(np.array([1, 0, -1, -a]), dims=self.eigenstates[pos].dims
                       , shape=self.eigenstates[pos].shape)

    
    def psi_lambda(self, t):
        """
        makes a density matrix of the form |psi(t)> <psi(t)|
        then traces out the clock state
        then finds the eigenstate corresponding to the maximum eigenvalue
        that's the state of the system
        """
        psi_dm = qt.tensor(qt.qeye(self.dimSys), self.clock_state(t).proj()) * self.glob_state(pos = self.pos).proj()
        denomi_ = qt.expect(qt.tensor(qt.qeye(self.dimSys), self.clock_state(t).proj()),\
                            self.glob_state(pos = self.pos))
        #print("Denomi", denomi_)
        #print("Psi_dm", psi_dm)
        psi_dm = (psi_dm/denomi_).ptrace(0) ## we need to keep system 0 after tracing
        valeig_, valstate_ = psi_dm.eigenstates()
        index_of_max_eigenvalue = valeig_.argmax()
        return valstate_[index_of_max_eigenvalue].unit()

    def Vs_lambda(self, t):
        """
        energy: energy of clock state
        returns: Vs(t) in the eigenbasis of H(t)
        """
        oper_ = self.V * self.glob_ket.proj() + self.glob_ket.proj() * self.V
        denom_norm = qt.expect(qt.tensor(qt.qeye(self.dimSys), self.clock_state(t).proj()),\
                            self.glob_state(pos = self.pos))
        oper_ = (qt.tensor(qt.qeye(self.dimSys), self.clock_state(t).proj()) * oper_).ptrace(0)/denom_norm
        return oper_

def analy_state(t):
    phase = np.exp(1j*a*t) / (2 * np.sqrt(1 + a*np.cos(t)**2))
    state = phase * (qt.basis(2, 0) -(a*np.exp(-2j*t) + 1)*qt.basis(2, 1))
    return state.unit()
        

        
quant_sys = QuantSys(Hs, Hc,V = [Vs, Vc], pos = 0)

# %%
# Plot of the expectation values of some operator for analytical and numerical solutions
time = np.linspace(0, 2*np.pi, 100)
oper = qt.sigmax() + qt.sigmay()
psi_lambda_list = [qt.expect(oper, quant_sys.psi_lambda(t)) for t in time]
analy_state_list = [qt.expect(oper, analy_state(t)) for t in time]

plt.figure(figsize=(12, 8))
plt.plot(time, psi_lambda_list, ":", label = "Numerical")
plt.plot(time, analy_state_list, "o", label = "Analytical")
plt.xlabel("Time")
plt.ylabel("<sigma_z>")
plt.grid(True)
plt.legend()
plt.show()
# %%
# Plots of the overlap of analytical and numerical solutions
plt.figure(figsize=(12, 8))
plt.plot(time, [np.sqrt((quant_sys.psi_lambda(t).dag() * analy_state(t)).norm()) for t in time])
plt.xlabel("Time")
plt.ylabel("Fidelity")
plt.grid(True)
plt.show()

# %%
Vx = [0.5*(quant_sys.Vs_lambda(t)*qt.sigmax()).tr() for t in time]
Vy = [0.5*(quant_sys.Vs_lambda(t)*qt.sigmay()).tr() for t in time]
Vz = [0.5*(quant_sys.Vs_lambda(t)*qt.sigmaz()).tr() for t in time]
plt.figure(figsize=(12, 8))
plt.plot(time, Vy, ":", label = "Vy", color = "orange")
plt.plot(time, -((a/2)*np.sin(2*time))/(1 + a* (np.cos(time)**2)), color = "green", label = "Analytical")
#plt.plot(time, Vz, ":", label = "Vz", color = "green")
plt.xlabel("Time")
plt.ylabel("V")
plt.grid(True)
plt.legend()
plt.show()
# %%
plt.figure(figsize=(12, 8))
plt.plot(time, Vx, ":", label = "Vx", color = "blue")
plt.plot(time, (np.cos(2*time) + a * np.cos(time)**2)/(1 + a* (np.cos(time)**2)), color = "green", label = "Analytical")
plt.xlabel("Time")
plt.ylabel("V")
plt.grid(True)
plt.legend()
plt.show()
# %%
