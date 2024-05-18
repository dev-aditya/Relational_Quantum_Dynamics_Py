# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def comb(n, k):
    return gamma(n + 1) / (gamma(k + 1) * gamma(n - k + 1))

N = 10

def f(p, k):
    return comb(N, k) * p**k * (1 - p)**(N - k)

for N in [10, 20, 100, 200]:
    y = np.linspace(0, N, 1000)
    x = np.linspace(0, 1, 1000)
    X, Y = np.meshgrid(x, y)
    Z = np.round(f(X, Y), 5)
    fig, ax = plt.subplots()
    contour = ax.contour(X, Y, Z, levels=50)
    contourf = ax.contourf(X, Y, Z, levels=50, cmap='YlGnBu')
    quiver = ax.quiver(X, Y, 2*X, 2*Y, scale=30, alpha=0.5)
    ax.set_xlabel('Probability of success')
    ax.set_ylabel('Number of successes')
    ax.set_title(f'N = {N}')
    # Add a colorbar to the plot
    cbar = fig.colorbar(contourf)
    cbar.ax.set_ylabel('Probability density')
    plt.show()
    contourf = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    quiver = ax.quiver(X, Y, 2*X, 2*Y, scale=30, alpha=0.5)
    ax.set_xlabel('Probability of success')
    ax.set_ylabel('Number of successes')
    ax.set_title(f'N = {N}')
    # Add a colorbar to the plot
    cbar = fig.colorbar(contourf)
    cbar.ax.set_ylabel('Probability density')
    plt.show()

# %%
import numpy as np

def summ(x, N):
    return np.sum([np.cos(2*np.pi*i*x/N) for i in range(0, N)])
for x in np.arange(1, 100):
    for N  in np.arange(1, 100):
        print(summ(x, N))

# %%
