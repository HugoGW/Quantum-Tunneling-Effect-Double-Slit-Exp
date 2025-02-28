######################### Importing Libraries #########################

import numpy as np  # For mathematical operations and array manipulations
import matplotlib.pyplot as plt  # For creating 2D plots
from matplotlib.animation import FuncAnimation, FFMpegWriter  # For creating animations and exporting them to video
from scipy.constants import h, hbar, m_e, e  # Useful constants like Planck's constant, reduced Planck's constant, electron mass, and charge
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting (though not used here explicitly)
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For enhancing axis labeling and layout

###################### Display and Performance Settings ######################

skip_frame = 6  # Number of frames to skip before updating the display to optimize animation performance

################## Constants and Simulation Parameters ##################

Lx, Ly = 5.0e-9, 5.0e-9  # Dimensions of the quantum box in meters
Nx, Ny = 260, 260  # Grid resolution along x and y axes
x_min, x_max = 0, Lx  # Range of x-coordinates
y_min, y_max = 0, Ly  # Range of y-coordinates
dx = (x_max - x_min) / Nx  # Spatial step size along the x-axis
dy = (y_max - y_min) / Ny  # Spatial step size along the y-axis

x = np.arange(x_min, x_max, dx)  # Array of x-values spanning the grid
y = np.arange(y_min, y_max, dy)  # Array of y-values spanning the grid
Nt = 1000  # Number of time steps for the simulation
d2 = dx * dy / np.sqrt(dx**2 + dy**2)  # Stability factor for the time step calculation
A1 = 0.1  # Constant ensuring stability of the wave function update
dt = A1 * 2 * m_e * d2**2 / hbar  # Time step derived from stability conditions
A2 = e * dt / hbar  # A constant used in the FDTD method

# Create 2D grid arrays for x and y coordinates
Y, X = np.meshgrid(x, y)

################### Initial Wave Packet Parameters ###################

x0, y0 = Lx / 4, Ly / 2  # Initial position of the wave packet (quarter along x, center along y)
sigma_x, sigma_y = 2.0e-10, 2.0e-10  # Width of the wave packet along x and y directions
Lambda = 1.5e-10  # de Broglie wavelength of the wave packet

# Real part of the wave packet (initial condition)
Psi_Real = np.exp(-0.5 * ((X - x0)**2 / sigma_x**2 + (Y - y0)**2 / sigma_y**2)) * \
           np.cos(2 * np.pi * (X - x0) / Lambda)

# Imaginary part of the wave packet (initial condition)
Psi_Imag = np.exp(-0.5 * ((X - x0)**2 / sigma_x**2 + (Y - y0)**2 / sigma_y**2)) * \
           np.sin(2 * np.pi * (X - x0) / Lambda)

# Probability density derived from the wave function
Psi_Prob = Psi_Real**2 + Psi_Imag**2

# Kinetic energy of the wave packet in electronvolts
Ec = (h / Lambda)**2 / (2 * m_e * e)

####################### Defining the Potential Barrier #######################

U0 = 70  # Barrier height in joules (converted from eV)
a = 5e-11  # Barrier width in meters
center_x, center_y = Lx / 2, Ly / 2  # Center of the potential barrier

# Initialize the potential to zero everywhere
U = np.zeros((Ny, Nx))

# Define a rectangular potential barrier at the center of the domain
U[(X >= center_x - a / 2) & (X <= center_x + a / 2)] = U0

# optim
alpha = hbar * dt / (2 * m_e)  # constant for the FDTD calculation method
dx2, dy2 = dx**2, dy**2  # constants for the FDTD calculation method

#################################### Plot Preparation ####################################

# Create a figure with custom size and resolution
fig = plt.figure(figsize=(18, 8), dpi=80)

############################# 2D Plot for Probability Density #############################

# Add a subplot for the 2D plot
ax_2d = fig.add_subplot(1, 3, 1)

# Define the colormap for the 2D plot and set undefined values to gray
cmap = plt.cm.get_cmap("inferno").copy()
cmap.set_bad('gray')

# Plot the probability density Psi_Prob in 2D, with proper scaling and colormap
im = ax_2d.imshow(Psi_Prob.T, extent=(0, Lx * 1e9, 0, Ly * 1e9), origin='lower', 
                  cmap=cmap, vmin=0, vmax=Psi_Prob.max())

# Add labels and a title to the plot
ax_2d.set_xlabel('x [nm]')
ax_2d.set_ylabel('y [nm]')
ax_2d.set_title('2D Quantum Tunneling')

# Add a color bar to the left of the 2D plot
divider = make_axes_locatable(ax_2d)
cax = divider.append_axes("left", size="5%", pad=0.5)  # Position to the left
cbar = plt.colorbar(im, cax=cax, orientation='vertical')  # Add the color bar
cbar.set_label('Probability Density')  # Label for the color bar

# Adjust ticks and label positions for the color bar
cax.yaxis.set_ticks_position('left')  # Move ticks to the left
cax.yaxis.set_label_position('left')  # Move label to the left

######################## Subplot for 1D Probability Density ########################

# Add a subplot for the 1D density plot at a specific x-position (D)
ax_prob = fig.add_subplot(1, 3, 2)

# Define the position D in nanometers where the 1D density is computed
D = (Lx / 2 + a) * 1e9  # Center position plus the width of the barrier

# Add a vertical white dashed line in the 2D plot to indicate the x-position (D)
line, = ax_2d.plot([D, D], [0, Ly * 1e9], color='white', linestyle='--', linewidth=1)

# Configure the 1D density plot
ax_prob.set_ylim(0, Ly * 1e9)  # Set y-axis range
ax_prob.set_xlim(0, 0.25)  # Adjust x-axis range (modify as needed)
ax_prob.set_xlabel('$|\\psi|^2$')  # Label for the x-axis

# Set the aspect ratio to create a tall and narrow plot
ax_prob.set_aspect(0.2)

# Suppress ticks on the y-axis for clarity
ax_prob.set_yticks([])

# Placeholder for the 1D probability density line
line_prob, = ax_prob.plot([], [], color='blue', linewidth=2)

# Add a rotated y-axis label for the plot
ax_prob.set_ylabel(f'Probability Density at $x = {np.round(D, 2)}$ nm', 
                   rotation=-90, labelpad=20, fontsize=12)

# Position the y-axis label on the right side of the plot
ax_prob.yaxis.set_label_position("right")

# Adjust spacing between subplots and set the left margin
plt.subplots_adjust(wspace=-0.38, left=0.05)

######################## 3D Plot for Quantum Tunneling ########################

ax_3d = fig.add_subplot(1, 3, 3, projection='3d')  # Add a subplot for the 3D visualization of the probability density
surf = None  # Placeholder for the 3D surface plot

# Add labels and a title to the 3D plot
ax_3d.set_xlabel('x [nm]')
ax_3d.set_ylabel('y [nm]')
ax_3d.set_zlabel('$|\\psi|^2$')
ax_3d.set_title('3D Quantum Tunneling')

######################## Theoretical Function for 1D Density ########################

# Define a theoretical function with a Gaussian envelope for comparison
def theoretical_function(y):
    # Calculate the wave vector K
    K = np.sqrt(2 * m_e * (U0 - Ec) * e) / hbar  # Wave vector in the barrier region

    # Calculate the transmission coefficient T
    T = 1.09 / ((U0**2 / (4 * Ec * (U0 - Ec)) * np.sinh(K * a)**2) + 1)

    # Scale the y-coordinates and the Gaussian width
    y_scaled = (y - y0 * 10**9)
    s_scaled = sigma_y * 1e9

    # Return the Gaussian-modulated transmission function
    return T**2 * np.exp(-0.5 * (y_scaled**2 / s_scaled**2)) / (2 * np.pi * s_scaled)**0.25

################################### Calculate Theoretical Curve ###################################

# Generate a range of y values in nanometers for the theoretical curve
y_values_theoretical = np.linspace(0, Ly * 1e9, Ny)  # y values in nanometers

# Calculate the theoretical probability density function for these y values
f_values = theoretical_function(y_values_theoretical)

# Add the theoretical line to the 1D plot
line_theoretical, = ax_prob.plot(f_values, y_values_theoretical, color='red', linestyle='--', label='Theoretical')

# Update the legend in the 1D plot
ax_prob.legend(loc='upper left')

######################################## Update Function ########################################

# Define the update function for the animation
def update(frame):
    global Psi_Real, Psi_Imag, Psi_Prob
    
    # Loop over the number of skipped frames to apply finite difference
    for _ in range(skip_frame):
        # Finite difference for wave function evolution in the real part (Psi_Real)
        Psi_Real[1:-1, 1:-1] = Psi_Real[1:-1, 1:-1] - alpha * (
            (Psi_Imag[2:, 1:-1] - 2 * Psi_Imag[1:-1, 1:-1] + Psi_Imag[:-2, 1:-1]) / dx2 +
            (Psi_Imag[1:-1, 2:] - 2 * Psi_Imag[1:-1, 1:-1] + Psi_Imag[1:-1, :-2]) / dy2
        ) + A2 * U[1:-1, 1:-1] * Psi_Imag[1:-1, 1:-1]
        
        # Finite difference for wave function evolution in the imaginary part (Psi_Imag)
        Psi_Imag[1:-1, 1:-1] = Psi_Imag[1:-1, 1:-1] + alpha * (
            (Psi_Real[2:, 1:-1] - 2 * Psi_Real[1:-1, 1:-1] + Psi_Real[:-2, 1:-1]) / dx2 +
            (Psi_Real[1:-1, 2:] - 2 * Psi_Real[1:-1, 1:-1] + Psi_Real[1:-1, :-2]) / dy2
        ) - A2 * U[1:-1, 1:-1] * Psi_Real[1:-1, 1:-1]

    # Update the probability density by adding the squared real and imaginary parts
    Psi_Prob[:] = Psi_Real**2 + Psi_Imag**2

    # Update the 2D plot for the probability density
    Uplot = Psi_Prob.copy()
    Uplot[U > 0] = np.nan  # Hide potential barrier in the plot
    im.set_array(Uplot.T)  # Update the image with the new probability density

    # Ensure the vertical dashed line (at x = D) remains visible in the 2D plot
    line.set_data([D, D], [0, Ly * 1e9])

    # Extract and update the 1D plot for probability density at x = D nm
    x_index = int(D * 1e-9 / dx)  # Convert D nm to grid index
    prob_density_at_x = Psi_Prob[x_index, :]  # Get the probability density at x = D nm

    line_prob.set_data(prob_density_at_x, y_values_theoretical)  # Update the 1D plot

    # Update the theoretical line (this remains constant in the plot)
    line_theoretical.set_data(f_values, y_values_theoretical)

    ######################### Update 3D Plot for Quantum Tunneling #########################

    global surf
    if surf:
        surf.remove()  # Remove the previous surface plot if it exists
    
    # Create meshgrid for the 3D plot based on x and y ranges
    X_3d, Y_3d = np.meshgrid(np.linspace(0, Lx * 1e9, Nx), np.linspace(0, Ly * 1e9, Ny))
    
    # Plot the surface of the probability density in 3D
    surf = ax_3d.plot_surface(X_3d, Y_3d, Psi_Prob.T, cmap="inferno", edgecolor='none')

    return [im, surf]  # Return the updated plot objects

####################################### Animation Setup #######################################

# Set up the FFMpeg writer (commented out as this section is not active)
# metadata = {
#     'title': '2DQT_+3D',
#     'artist': 'Hugo.A',
#     'comment': 'quantum tunneling simulation 2D and 3D'
# }
# writer = FFMpegWriter(fps=30, metadata=metadata)

# Define the output file path for saving the animation (commented out)
# output_file = r"ADRESSE_POUR_ENREGISTRER_LANIMATION\QT_2D_and_3D.mp4"

ani = FuncAnimation(fig, update, frames=Nt, blit=False, interval=1) # Create the animation using FuncAnimation, calling the update function for each frame
plt.show()  # Display the animation

# Uncomment the following line to save the animation (if needed)
# ani.save(output_file, writer=writer)

#print(f"Animation saved as {output_file}.")

