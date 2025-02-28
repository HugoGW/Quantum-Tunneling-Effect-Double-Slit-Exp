#################################### Import Libraries ####################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.constants import h, hbar, m_e, e
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

##################################### Display Settings ####################################

# Define the number of frames to skip before updating the display
skip_frame = 6  # number of frames to skip before updating display

################################### Constants and Simulation Parameters ##################

# Dimensions of the quantum box (in meters)
Lx, Ly = 3.0e-9, 3.0e-9  
# Grid resolution (number of points along x and y axes)
Nx, Ny = 300, 300  
# Define the minimum and maximum values for x and y
x_min, x_max = 0, Lx
y_min, y_max = 0, Ly
# Calculate grid spacing
dx = (x_max - x_min) / Nx
dy = (y_max - y_min) / Ny
# Generate grid arrays for x and y coordinates
x = np.arange(x_min, x_max, dx)
y = np.arange(y_min, y_max, dy)
# Set number of time steps (Nt = 1 for simplicity in this example)
Nt = 1
# Pre-calculated value for numerical stability
d2 = dx * dy / np.sqrt(dx**2 + dy**2)
# Parameter A1 used to define dt (time step)
A1 = 0.1
# Define time step dt based on A1 and other physical constants
dt = A1 * 2 * m_e * d2**2 / hbar
# Another parameter A2 used in the calculation of wave function evolution
A2 = e * dt / hbar

# Create a mesh grid for X and Y values
Y, X = np.meshgrid(x, y)

################################### Initial Wave Packet ###################################

# Initial position of the wave packet
x0, y0 = Lx / 4, Ly / 2  
# Width of the packet (in meters)
sigma_x, sigma_y = 2.0e-10, 2.0e-10  
# De Broglie wavelength
Lambda = 8e-11  
# Define the real part of the wave function (Gaussian envelope with cosine function)
Psi_Real = np.exp(-0.5 * ((X - x0)**2 / sigma_x**2 + (Y - y0)**2 / sigma_y**2)) * np.cos(2 * np.pi * (X - x0) / Lambda)
# Define the imaginary part of the wave function (Gaussian envelope with sine function)
Psi_Imag = np.exp(-0.5 * ((X - x0)**2 / sigma_x**2 + (Y - y0)**2 / sigma_y**2)) * np.sin(2 * np.pi * (X - x0) / Lambda)
# Compute the probability density as the sum of squares of the real and imaginary parts
Psi_Prob = Psi_Real**2 + Psi_Imag**2

# Calculate the energy of the wave packet based on de Broglie wavelength
Ec = (h / Lambda)**2 / (2 * m_e * e)

################################### Define the Potential Barrier ###########################

# Define the potential height in joules (converted from eV)
U0 = 1e3  
# Width of the potential barrier (in meters)
a = 2.5e-11  
# Width of the slits in the barrier
l = 5e-11  
# Distance between the slits
d = 2.3e-10  
# Intensity of the wave (related to the amplitude of Psi)
I0 = 0.21   
# Center coordinates of the barrier (in the middle of the quantum box)
center_x, center_y = Lx / 2, Ly / 2  

# Calculate the position of the upper and lower slits based on the center and slit separation
upper_slit_bottom = center_y + d / 2 + l / 2  # Bottom of the upper slit
lower_slit_top = center_y - d / 2 - l / 2  # Top of the lower slit

################################### Define the Potential U ################################

# Define the potential U using np.where to set the barrier and slit positions
U = np.where(
    (X >= center_x - a / 2) & (X <= center_x + a / 2)  # Define the barrier
    & ~(((Y >= upper_slit_bottom - l) & (Y <= upper_slit_bottom))  # Upper slit
        | ((Y >= lower_slit_top) & (Y <= lower_slit_top + l))),  # Lower slit
    U0, 0  # Barrier potential U0, otherwise 0
)

##################################### Optimized Constants ####################################

# Optimized parameters for numerical stability and calculation of the wave function
alpha = hbar * dt / (2 * m_e)  # Factor for time evolution, dependent on Planck's constant and electron mass
dx2 = dx**2  # Squared grid spacing in x-direction
dy2 = dy**2  # Squared grid spacing in y-direction

##################################### Plot Preparation ####################################

# Create a figure with a specific size and resolution
fig = plt.figure(figsize=(21, 8), dpi=80)

# 2D Plot Setup: Probability Density
ax_2d = fig.add_subplot(1, 3, 1)  # Create subplot for 2D plot (1 row, 3 columns, first subplot)
# Define a colormap for the plot, using 'inferno' colormap with gray for bad values
cmap = plt.cm.get_cmap("inferno").copy()
cmap.set_bad('gray')  # Set 'gray' for values that are considered invalid or out of range
# Display the probability density (Psi_Prob) using imshow (2D heatmap)
im = ax_2d.imshow(Psi_Prob.T, extent=(0, Lx * 1e9, 0, Ly * 1e9), origin='lower', cmap=cmap, vmin=0, vmax=Psi_Prob.max())
# Label the axes
ax_2d.set_xlabel('x [nm]')
ax_2d.set_ylabel('y [nm]')
ax_2d.set_title('2D Quantum Tunneling')

# Add a color bar to the left of the 2D plot for better understanding of the values
divider = make_axes_locatable(ax_2d)
cax = divider.append_axes("left", size="5%", pad=0.7)  # Position color bar to the left
cbar = plt.colorbar(im, cax=cax, orientation='vertical')  # Create colorbar for the plot
cbar.set_label('Probability Density')  # Label for the colorbar
cax.yaxis.set_ticks_position('left')  # Move the ticks to the left side
cax.yaxis.set_label_position('left')  # Move the label to the left side of the colorbar

##################################### 1D Density Plot Setup ####################################

# Create another subplot for a 1D probability density plot at a specific position (D)
ax_prob = fig.add_subplot(1, 3, 2)  # Create subplot for 1D probability density
D = (7 * Lx / 10) * 1e9  # Convert the position D to nanometers
# Add a vertical line at position D in the 2D plot
line, = ax_2d.plot([D, D], [0, Ly * 1e9], color='white', linestyle='--', linewidth=1)
# Set limits for the y-axis and x-axis of the 1D plot
ax_prob.set_ylim(0, Ly * 1e9)
ax_prob.set_xlim(0, 0.25)  # Adjust the x-axis scale if necessary
ax_prob.set_xlabel('$|\\psi|^2$')  # Label the x-axis as probability density
ax_prob.set_aspect(0.2)  # Set the aspect ratio to stretch the plot vertically
ax_prob.set_yticks([])  # Remove the y-axis ticks for a cleaner display
# Define the 1D plot line for the probability density (empty initially)
line_prob, = ax_prob.plot([], [], color='blue', linewidth=2)

# Set the label for the y-axis of the 1D plot and position it on the right side
ax_prob.set_ylabel(f'Probability Density at $x = {np.round(D,2)}$ nm', rotation=-90, labelpad=20, fontsize=12)
ax_prob.yaxis.set_label_position("right")

# Adjust the spacing between subplots to make the layout cleaner
plt.subplots_adjust(wspace=-0.38, left=0.05)  # Reduce space between subplots

##################################### 3D Plot Setup #####################################

# Create the third subplot as a 3D plot for visualizing the wave function's 3D evolution
ax_3d = fig.add_subplot(1, 3, 3, projection='3d')  # 3D plot setup
surf = None  # Placeholder for the 3D surface plot (we will add it later)
# Label the axes for the 3D plot
ax_3d.set_xlabel('x [nm]')
ax_3d.set_ylabel('y [nm]')
ax_3d.set_zlabel('$|\\psi|^2$')
ax_3d.set_title('3D Quantum Tunneling')

##################################### Theoretical Function #####################################

# Define a function to calculate the theoretical probability density for comparison
def theoretical_function(y, I0, l, d, lambda_, D):
    # Adjust the y values by centering them around y = Ly / 2
    y_scaled = (y - Ly * 1e9 / 2)
    # Theoretical formula for probability density using sinc and cosine functions
    return I0 * (np.sinc(1.3 * np.pi * l * y_scaled / (lambda_ * D)))**2 * (np.cos(2.9 * np.pi * d * y_scaled / (lambda_ * D)))**2

# Calculate the theoretical values for the probability density along y
y_values_theoretical = np.linspace(0, Ly * 1e9, Ny)  # Generate y values in nanometers
f_values = theoretical_function(y_values_theoretical, I0, l, d, Lambda, D)  # Calculate the theoretical function values

# Plot the theoretical curve on the 1D plot for comparison
line_theoretical, = ax_prob.plot(f_values, y_values_theoretical, color='red', linestyle='--', label='Theoretical')

# Update the legend to include the theoretical line
ax_prob.legend(loc='upper left')

# Update function for the animation
def update(frame):
    """
    Function to update the wave function and its visualizations at each frame in the animation.
    This includes updating the real and imaginary parts of the wave function,
    the probability density, the 2D and 1D plots, and the 3D surface plot.
    """
    global Psi_Real, Psi_Imag, Psi_Prob

    # Loop over a defined number of frames to evolve the wave function
    for _ in range(skip_frame):
        # Finite difference method for updating the real part of the wave function
        # The equation calculates the real part (Psi_Real) based on the imaginary part (Psi_Imag)
        Psi_Real[1:-1, 1:-1] = Psi_Real[1:-1, 1:-1] - alpha * (
            # Second derivative with respect to x (finite difference method)
            (Psi_Imag[2:, 1:-1] - 2 * Psi_Imag[1:-1, 1:-1] + Psi_Imag[:-2, 1:-1]) / dx2 +
            # Second derivative with respect to y (finite difference method)
            (Psi_Imag[1:-1, 2:] - 2 * Psi_Imag[1:-1, 1:-1] + Psi_Imag[1:-1, :-2]) / dy2
        ) + A2 * U[1:-1, 1:-1] * Psi_Imag[1:-1, 1:-1]
        
        # Finite difference method for updating the imaginary part of the wave function
        # The equation calculates the imaginary part (Psi_Imag) based on the real part (Psi_Real)
        Psi_Imag[1:-1, 1:-1] = Psi_Imag[1:-1, 1:-1] + alpha * (
            # Second derivative with respect to x (finite difference method)
            (Psi_Real[2:, 1:-1] - 2 * Psi_Real[1:-1, 1:-1] + Psi_Real[:-2, 1:-1]) / dx2 +
            # Second derivative with respect to y (finite difference method)
            (Psi_Real[1:-1, 2:] - 2 * Psi_Real[1:-1, 1:-1] + Psi_Real[1:-1, :-2]) / dy2
        ) - A2 * U[1:-1, 1:-1] * Psi_Real[1:-1, 1:-1]

    # Update the probability density (|Psi|^2) which is the sum of the squares of the real and imaginary parts
    Psi_Prob[:] = Psi_Real**2 + Psi_Imag**2

    # Update the 2D plot of the probability density
    Uplot = Psi_Prob.copy()
    Uplot[U > 0] = np.nan  # Set potential barriers to NaN so they don't show up in the plot
    im.set_array(Uplot.T)  # Update the image for the 2D plot with the new probability density

    # Ensure the vertical dashed line (at x = D nm) remains visible in the 2D plot
    line.set_data([D, D], [0, Ly * 1e9])  # Update the line position at x = D

    # Extract the 1D probability density at x = D nm and update the corresponding plot
    x_index = int(D * 1e-9 / dx)  # Convert D from nm to grid index (for x-axis)
    prob_density_at_x = Psi_Prob[x_index, :]  # Extract |Psi|^2 at x = D nm

    # Update the data for the 1D probability density plot
    line_prob.set_data(prob_density_at_x, y_values_theoretical)

    # Update the theoretical curve, which remains constant in time
    line_theoretical.set_data(f_values, y_values_theoretical)

    # Update the 3D surface plot of the probability density
    global surf
    if surf:  # If a surface exists, remove the previous one
        surf.remove()
    X_3d, Y_3d = np.meshgrid(np.linspace(0, Lx * 1e9, Nx), np.linspace(0, Ly * 1e9, Ny))  # Create meshgrid for 3D plot
    surf = ax_3d.plot_surface(X_3d, Y_3d, Psi_Prob.T, cmap="inferno", edgecolor='none')  # Plot the new surface

    # Return updated objects for the animation
    return [im, surf]

# Run the animation using FuncAnimation
ani = FuncAnimation(fig, update, frames=Nt, blit=False, interval=1)  # Update the figure every frame

# Ensure the layout is tight (no overlapping elements)
plt.tight_layout()

# Define the output path for saving the animation as an MP4 file
# output_path = r"OUTPUT_PATH_FOR_SAVING_ANIMATION\DSE2D+3D.mp4"

# Create a writer object to save the animation in MP4 format with specified properties (fps, bitrate)
# writer = FFMpegWriter(fps=30, metadata=dict(artist='Hugo Alexandre'), bitrate=1800)

# Save the animation to the output file
# with writer.saving(fig, output_path, dpi=200):
#     for frame in range(0, int(Nt/dt)):  # Loop through all frames
#         update(frame)  # Call the update function to update the frame data
#         writer.grab_frame()  # Grab and save the current frame

# Show the plot (this step is for interactive use)
plt.show()
