import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.constants import h, hbar, m_e, e
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Display settings
skip_frame = 6  # number of frames to skip before updating display

# Constants and simulation parameters
Lx, Ly = 5.0e-9, 5.0e-9  # dimensions of the quantum box in meters
Nx, Ny = 260, 260  # grid resolution
x_min, x_max = 0, Lx
y_min, y_max = 0, Ly
dx = (x_max - x_min) / Nx
dy = (y_max - y_min) / Ny
x = np.arange(x_min, x_max, dx)
y = np.arange(y_min, y_max, dy)
Nt = 1
d2 = dx * dy / np.sqrt(dx**2 + dy**2)
A1 = 0.1
dt = A1 * 2 * m_e * d2**2 / hbar
A2 = e * dt / hbar

Y, X = np.meshgrid(x, y)

# Initial wave packet
x0, y0 = Lx / 4, Ly / 2  # initial position of the packet
sigma_x, sigma_y = 2.0e-10, 2.0e-10  # width of the packet
Lambda = 1.5e-10  # de Broglie wavelength
Psi_Real = np.exp(-0.5 * ((X - x0)**2 / sigma_x**2 + (Y - y0)**2 / sigma_y**2)) * np.cos(2 * np.pi * (X - x0) / Lambda)
Psi_Imag = np.exp(-0.5 * ((X - x0)**2 / sigma_x**2 + (Y - y0)**2 / sigma_y**2)) * np.sin(2 * np.pi * (X - x0) / Lambda)
Psi_Prob = Psi_Real**2 + Psi_Imag**2

Ec = (h / Lambda)**2 / (2 * m_e * e)

# Define the potential barrier
U0 = 70  # potential in joules (convert from eV)
a = 5e-11  # barrier width
center_x, center_y = Lx / 2, Ly / 2  # barrier center
U = np.zeros((Ny, Nx))
U[(X >= center_x - a / 2) & (X <= center_x + a / 2)] = U0  # 2D barrier

# Optim
alpha = hbar * dt / (2 * m_e)
dx2 = dx**2
dy2 = dy**2

# Plot preparation
fig, (ax, ax_prob) = plt.subplots(1, 2, figsize=(16, 8), dpi=80, gridspec_kw={'width_ratios': [1, 1]})
cmap = plt.cm.get_cmap("inferno").copy()
cmap.set_bad('gray')

# Main 2D plot
im = ax.imshow(Psi_Prob.T, extent=(0, Lx * 1e9, 0, Ly * 1e9), origin='lower', cmap=cmap, vmin=0, vmax=Psi_Prob.max())
ax.set_xlabel('x [nm]')
ax.set_ylabel('y [nm]')
ax.set_title('2D Quantum Tunneling')

divider = make_axes_locatable(ax)
cax = divider.append_axes("left", size="5%", pad=0.7)
cbar = plt.colorbar(im, cax=cax, orientation='vertical')
cbar.set_label('Probability Density')
cax.yaxis.set_label_position('left')
cax.yaxis.tick_left()

# Subplot for 1D density plot at x=D
D = (Lx / 2 + a) * 10**9  # in nm
line, = ax.plot([D, D], [0, Ly * 1e9], color='white', linestyle='--', linewidth=1)
ax_prob.set_ylim(0, Ly * 1e9)
ax_prob.set_xlim(0, 0.25)  # Adjust y-axis dynamically
ax_prob.set_xlabel('$|\\psi|^2$')
ax_prob.set_aspect(0.2)  # Aspect ratio (width:height = 2:1)
ax_prob.set_yticks([])  # This removes the y-axis ticks
line_prob, = ax_prob.plot([], [], color='blue', linewidth=2)  # Placeholder line

# Set the title along the right y-axis
ax_prob.set_ylabel(f'Probability Density at $x = {np.round(D,2)}$ nm', rotation=-90, labelpad=20, fontsize=12)

# Move the label to the right side
ax_prob.yaxis.set_label_position("right")

plt.subplots_adjust(wspace=-0.34, left=-0.165)  # Reduce space between subplots

# Define the theoretical function with Gaussian envelope
def theoretical_function(y):
    # Calculate the wave vector K
    K = np.sqrt(2 * m_e * (U0 - Ec) * e) / hbar
    
    # Calculate T
    T = 1 / ((U0**2 / (4 * Ec * (U0 - Ec)) * np.sinh(K * a)**2) + 1)
    y_scaled = (y - y0*10**9)
    s_scaled = sigma_y*1e9
    
    
    return (1.09*T)**2* np.exp(-0.5 * ((y_scaled)**2 / s_scaled**2))/(2*np.pi*s_scaled)**0.25


# Calculate the theoretical curve using the modified function
y_values_theoretical = np.linspace(0, Ly * 1e9, Ny)  # y values in nanometers
f_values = theoretical_function(y_values_theoretical)

# Add the theoretical line to the 1D plot
line_theoretical, = ax_prob.plot(f_values, y_values_theoretical, color='red', linestyle='--', label='Theoretical')

# Update the legend
ax_prob.legend(loc='upper left')

# Update function to ensure the theoretical curve remains visible
def update(frame):
    global Psi_Real, Psi_Imag, Psi_Prob
    for i in range(skip_frame):
        # Calculate the finite difference for the wave function evolution
        Psi_Real[1:-1, 1:-1] = Psi_Real[1:-1, 1:-1] - alpha * (
            (Psi_Imag[2:, 1:-1] - 2 * Psi_Imag[1:-1, 1:-1] + Psi_Imag[:-2, 1:-1]) / dx2 +
            (Psi_Imag[1:-1, 2:] - 2 * Psi_Imag[1:-1, 1:-1] + Psi_Imag[1:-1, :-2]) / dy2
        ) + A2 * U[1:-1, 1:-1] * Psi_Imag[1:-1, 1:-1]

        Psi_Imag[1:-1, 1:-1] = Psi_Imag[1:-1, 1:-1] + alpha * (
            (Psi_Real[2:, 1:-1] - 2 * Psi_Real[1:-1, 1:-1] + Psi_Real[:-2, 1:-1]) / dx2 +
            (Psi_Real[1:-1, 2:] - 2 * Psi_Real[1:-1, 1:-1] + Psi_Real[1:-1, :-2]) / dy2
        ) - A2 * U[1:-1, 1:-1] * Psi_Real[1:-1, 1:-1]

    # Update probability density
    Psi_Prob = (Psi_Real**2 + Psi_Imag**2)

    # Update main 2D plot
    Uplot = Psi_Prob
    Uplot[U > 0] = np.nan  # Hide the potential barrier area for visualization
    im.set_array(Uplot.T)

    # Ensure the vertical dashed line remains visible
    line.set_data([D, D], [0, Ly * 1e9])

    # Extract and update the 1D plot for |psi|^2 at x = D nm
    x_index = int(D * 1e-9 / dx)  # Convert D nm to grid index
    prob_density_at_x = Psi_Prob[x_index, :]  # |psi|^2 at x = D nm

    line_prob.set_data(prob_density_at_x, y_values_theoretical)

    # Update the theoretical line (remains constant)
    line_theoretical.set_data(f_values, y_values_theoretical)

    return [im, line, line_prob, line_theoretical]

# Run the animation
ani = FuncAnimation(fig, update, frames=Nt, blit=True, interval=1)
plt.tight_layout()

# # Définir le chemin de sauvegarde
# output_path = r"P:\\Cours Physique - Universités\\Cours fac UBFC\\M2\\S9\\Free Numerical Project\\beta version\\QT_2D.mp4"

# # Création du writer pour sauvegarder en MP4
# writer = FFMpegWriter(fps=30, metadata=dict(artist='Hugo Alexandre'), bitrate=1800)

# # Enregistrement de l'animation
# with writer.saving(fig, output_path, dpi=200):
#     for frame in range(0, int(Nt/dt)):
#         update(frame)  # Appel de la fonction update pour chaque frame
#         writer.grab_frame()

plt.show()

