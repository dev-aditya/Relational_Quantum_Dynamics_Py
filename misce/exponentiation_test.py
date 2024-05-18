# %%
import time
import numpy as np
import qutip as qt

"""
This file is to test the exponentiation of the Hamiltonian using
1. Brute force method
2. Eigenstates method
"""
N = 13  # number of snp.pins
sigx = qt.sigmax()
sigz = qt.sigmaz()
sigy = qt.sigmay()
sigp = qt.sigmap()
sigm = qt.sigmam()
I = qt.identity(2)


def sig(k, matrix, size=N):
    """
    Returns the input matix at the kth position in the qt.tensor product N snp.pins-1/2 system
    """
    if k >= N:
        raise ValueError("k must be less than N")
    mat_lis = [I] * size
    mat_lis[k] = matrix
    return qt.tensor(mat_lis)


# ----------------------------------------------
"""
Somehow the frequency of the emergent potential depends on g, the coupling constant
and the coefficient depend on the frequency of the transverse field
"""
g = (np.sqrt(5) - 1) / 10  # coupling
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
# -- Clock Snp.pin System
theta, phi = np.pi / 4 * np.sqrt(2), np.pi / np.sqrt(2)


start = time.time()
hc_eigen, hc_eigstate = Hc.eigenstates()
print(f"Time elapsed for eigenstate calculation: {time.time()-start}")

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f"Time elapsed: {end-start}")
        return res

    return wrapper


@timeit
def Xi_brute(t, energy, size=N - 1):
    state = []
    for i in range(size):
        state.append(qt.spin_coherent(1 / 2, theta, phi))
    state = qt.tensor(state)
    eiden = energy * qt.tensor([I] * size)
    return ((-1j * (Hc - eiden) * t).expm() * state).unit()


@timeit
def Xi_eig(t, energy, size=N - 1):
    state = []
    for i in range(size):
        state.append(qt.spin_coherent(1 / 2, theta, phi))
    state0 = qt.tensor(state)
    state = 0 * state0
    state0 = state0.full()
    for m in range(hc_eigen.__len__()):
        state += (
            np.vdot(hc_eigstate[m].full(), state0)
            * np.exp(-1j * (hc_eigen[m] - energy) * t)
            * hc_eigstate[m]
        )
    return state.unit()


print("Brute force")
x1 = Xi_brute(np.sqrt(3), np.sqrt(2))
print("Eigenstates")
x2 = Xi_eig(np.sqrt(3), np.sqrt(2))
print("-----")
# %%
