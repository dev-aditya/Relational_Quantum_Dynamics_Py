# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scienceplots
plt.style.use(["nature", "grid", "no-latex"])

def gaussian_wave(x, t):
    w = np.sqrt(1 + 2j * t)
    return np.abs(np.exp(-x**2 / w**2) / np.sqrt(2/np.pi) / w)

x = np.linspace(-100, 100, 1000)
t = np.linspace(0, 10, 1000)

# Create the figure and axis
fig, ax = plt.subplots()
line, = ax.plot(x, gaussian_wave(x, 0),)

# Animation function
def animate(t_val):
    y = gaussian_wave(x, t_val)
    print(f"Animating t = {t_val}") 
    line.set_ydata(y)
    ax.set_title(f'Gaussian Wave Animation (t = {t_val})')
    return line,

# Set plot labels and title
ax.set_xlabel('x')
ax.set_ylabel('Amplitude')
ax.set_title('Gaussian Wave Animation')

# Create the animation
animation = FuncAnimation(fig, animate, frames=t, interval=50, blit=True)


# To save the animation as a GIF file
animation.save('gaussian_wave_animation.gif', writer='imagemagick')

print("Done!")
# %%
