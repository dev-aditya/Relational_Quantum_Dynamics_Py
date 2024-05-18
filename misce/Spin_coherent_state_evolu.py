# %% 
from numpy import exp, pi, sqrt, arange, linspace
from scipy.special import sph_harm
from scipy.special import factorial as fac
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# Define the spin_coherent function
def spin_coherent(z, theta, phi, t, J):
    norm = 1/(1+abs(z)**2)**J
    Y = 0
    for m in np.arange(-J, J+1):
        coeff = np.sqrt(fac(2*J)/(fac(J+m)*fac(J-m))) * z**(J- m)
        Y += coeff*exp(-1j*m*t)*sph_harm(m, J, theta, phi)
    return Y*norm

# Set up the figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set the parameters
J = 3/2
z = (1 + sqrt(5))/2
print(z)
theta = np.linspace(0, pi, 100)
phi = np.linspace(0, 2*pi, 100)
Theta, Phi = np.meshgrid(theta, phi)

# Define the animation function
def animate(t):
    # Compute the spin_coherent function for the given time t
    Y = spin_coherent(z, Theta, Phi, t, J)

    # Plot the real part of the function as a 3D surface
    ax.clear()
    ax.plot_surface(
        Theta, Phi, np.real(Y), cmap='viridis', alpha=0.8, linewidth=0.5, edgecolor='k'
    )
    ax.set_xlabel('Theta')
    ax.set_ylabel('Phi')
    ax.set_zlabel('Real(Y)')

    # Plot the imaginary part of the function as a separate 3D surface
    ax.plot_surface(
        Theta, Phi, np.imag(Y), cmap='magma', alpha=0.8, linewidth=0.5, edgecolor='k'
    )
    ax.set_xlabel('Theta')
    ax.set_ylabel('Phi')
    ax.set_zlabel('Imag(Y)')

    # Set the title to show the current time
    ax.set_title(f't = {t:.2f}')

# Create the animation
anim = FuncAnimation(fig, animate, frames=np.linspace(0, 2*pi, 100), interval=50)

# Save the animation as an MP4 file
#anim.save('spin_coherent_animation.mp4')

# Show the animation
plt.show()
# %%
"""
This  is implementation of the Asher Peres Quantum Claock System
"""
import numpy as np
from scipy.special import factorial as fac
from numpy import exp, pi, log
import qutip as qt
import matplotlib.pyplot as plt

j = 3/2
N = int(2*j+1)

θ = np.linspace(0, pi, 100)
ϕ = np.linspace(0, 2*pi, 100)

np.random.seed(0)

θ = 0*np.random.choice(θ)
ϕ = np.random.choice(ϕ)

print("""
      
      Theta: {}
      Phi: {}
      
      """.format(θ, ϕ))

#init_state = qt.spin_coherent(j, θ, ϕ)

def clock_state(k):
    state = 0*exp(-1j*2*pi*k*(-j)/N)*qt.spin_state(j, -j)
    for m in np.arange(-j+1, j+1):
        state += exp(-1j*2*pi*k*m/N)*qt.spin_state(j, m)
    return state.unit()

τ = 0.01
def Time():
    oper = 0*qt.ket2dm(clock_state(0))
    for k in np.arange(1, N+1):
        oper += k*qt.ket2dm(clock_state(k))
    return τ*oper

init_state = clock_state(0)

ω = 2*pi/(N*τ)
H = ω*qt.jmat(j, 'z')
T = np.linspace(0, 2*N*τ, 1000)
result = qt.sesolve(H, init_state, T, e_ops =  {"J2": Time()}, progress_bar=True)

plt.plot(T, result.expect["J2"])
# %%
