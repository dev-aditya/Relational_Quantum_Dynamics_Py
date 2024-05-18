# %%
import qutip as qt
import numpy as np
from scipy.special import comb

# %%
#energy, jx = qt.jmat(1/2, 'y').eigenstates()
#jx

# %%
j = 1 ## spin of particle
n = 3 ## number of particles

# %%
class QuantumSystem:
    def __init__(self, j, n, omega = 1):
        self.omega = omega ## frequency of rotation
        self.J = j ## total spin of particle
        self.N = n ## number of particles
        self.state = self.psi() ## state according to equation (3)
        self.mat_shape = qt.jmat(j, "x").shape[0] ## shape of hermitian matrix operators
    
    def psi(self): ## returns the state as defined in equation (3)
        """
        returns: creates the state as defined in equation (3) of Page and Wootters (1983)
        """
        psi_ = lambda J, m: np.sqrt(comb(2*J, J+m))*qt.spin_state(J, m)
        _ = (1/2**self.J)*sum([psi_(self.J, m) for m in np.arange(-self.J, self.J+1)])
        return qt.tensor([_ for j in range(self.N)])
    
    def J_operator(self, pos, direction): 
        """
        pos: index of particle in tensor product state
        direction: direction of J (angular momentum) operator
        returns: J operator for particle at index 'pos'
        """
        op_list = [qt.qeye(self.mat_shape)] * self.N
        op_list[pos] = qt.jmat(self.J, direction)
        return qt.tensor(op_list)
    
    def expectation(self, oper, state = None):
        """
        oper: operator
        returns: expectation value of operator 'oper' on the state
        """
        if state is None:
            state = self.state
        return qt.expect(oper, state)
    
    def J_state(self, direction, j = None):
        """
        direction: direction of J operator
        returns: dict eigenstates of J operator in given 'direction' with with keys as eigenvalues
        """
        if j is None:
            j = self.J
        jmat_ = qt.jmat(j, direction)
        energy_, js_ = jmat_.eigenstates()
        dict_ = {energy_[i]: js_[i] for i in range(len(energy_))}
        return dict_
    
    def J1x_(self, t):
        """
        pos: index of particle in tensor product state
        direction: direction of J operator
        returns: J operator for particle at index 'pos'
        """
        psi_ = lambda J, m, t: np.sqrt(comb(2*J, J+m)) * np.exp(-1j*m*self.omega*t) * qt.spin_state(J, m)
        jx_ = (1/2**self.J)*sum([psi_(self.J, m, t) for m in np.arange(-self.J, self.J+1)])
        return jx_
        
    def proj_operators(self, pos, direction):
        """
        Projection Oprators for particle at position 'pos'
        """
        proj_list = [qt.qeye(self.mat_shape)] * self.N
        jmat_ = qt.jmat(self.J, direction)
        energy_, js_ = jmat_.eigenstates()

qob = QuantumSystem(j, n)
print(qob.state)

# %%


# %%


# %%



