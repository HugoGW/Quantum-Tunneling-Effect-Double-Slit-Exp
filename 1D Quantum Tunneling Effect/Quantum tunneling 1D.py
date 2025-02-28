#################################### Importing Libraries ####################################

import numpy as np  # Provides mathematical functions and array handling
import matplotlib.pyplot as plt  # Enables plotting graphs
from matplotlib.animation import FuncAnimation, FFMpegWriter  # For creating and saving animations
from scipy.constants import h, hbar, e, m_e  # Imports useful constants like Planck's constant and electron mass

############################### Constants and Parameters ###############################

DeuxPi = 2.0 * np.pi  # A constant for 2*pi
L = 5.0e-9  # Length of the quantum box in meters
Nx = 2000  # Number of spatial points forming the grid
xmin, xmax = 0.0, L  # Minimum and maximum spatial positions
dx = (xmax - xmin) / Nx  # Spatial step size determined by the grid points
x = np.arange(0.0, L, dx)  # Array of points from 0 to L with spacing dx
Nt = 1e-12  # Total duration of the animation in seconds
a2 = 0.1  # Constant ensuring stability of the animation
dt = a2 * 2 * m_e * dx**2 / hbar  # Time step calculated using spatial step size
a3 = e * dt / hbar  # Constant used in the FDTD method

######################### Initial Parameters of the Wave Packet #########################

x0 = x[int(Nx / 4)]  # Initial position of the wave packet, set to L/4
sigma = 2.0e-10  # Width of the wave packet in meters
Lambda = 1.5e-10  # de Broglie wavelength of the electron in meters
Ec = (h / Lambda)**2 / (2 * m_e * e)  # Kinetic energy in electronvolts (eV)

############################### Potential Definition ################################

U0 = 80  # Barrier height in eV
a = 7e-11  # Barrier width in meters
center = L / 2  # Center position of the barrier
barrier_start = center - a / 2  # Start position of the barrier
barrier_end = center + a / 2  # End position of the barrier

################# Constructing the Potential Based on Barrier Position ################

U = np.zeros(Nx)  # Initialize potential to zero everywhere
U[(x >= barrier_start) & (x <= barrier_end)] = U0  # Set potential to U0 within the barrier

# Determine indices corresponding to the barrier start and end
barrier_start_index = int(barrier_start / dx)
barrier_end_index = int(barrier_end / dx)

######################## Initial Gaussian Wave Packet #########################

Psi_Real = np.exp(-0.5 * ((x - x0) / sigma)**2) * np.cos(DeuxPi * (x - x0) / Lambda)  # Real part of the Gaussian wave packet
Psi_Imag = np.exp(-0.5 * ((x - x0) / sigma)**2) * np.sin(DeuxPi * (x - x0) / Lambda)  # Imaginary part of the Gaussian wave packet
Psi_Prob = Psi_Real**2 + Psi_Imag**2  # Probability density

####################### Setting up the Plot and Axes ###########################

fig, ax1 = plt.subplots(figsize=(10, 8))  # Create a figure with a specified size
line_prob, = ax1.plot([], [], 'darkred')  # Initialize the probability density curve
ax1.set_xlim(0, L * 1.e9)  # Set the x-axis limits from 0 to L (converted to nm)
ax1.set_ylim(0.00001, 1.1 * max(Psi_Prob / np.sum(Psi_Prob)))  # Set y-axis limits based on Psi_Prob
ax1.set_xlabel('x [nm]')  # Label for x-axis
ax1.set_ylabel(r'Detection probability density [$m^{-1}$]')  # Label for y-axis
ax1.grid(True)  # Enable grid lines

ax2 = ax1.twinx()  # Create a secondary y-axis for the potential
ax2.plot(x * 1.e9, U, 'royalblue')  # Plot the potential in blue
ax2.set_ylabel('U [eV]')  # Label for the potential axis
ax2.set_ylim(0.2, 90)  # Set limits for the potential axis

################### Adding Text Labels for Probabilities ###################

left_prob_text = ax1.text(0.1, 0.91, '', transform=ax1.transAxes, color='black', fontsize=14)  # Text for probability on the left
right_prob_text = ax1.text(0.55, 0.91, '', transform=ax1.transAxes, color='black', fontsize=14)  # Text for probability on the right
T_text = ax1.text(0.77, 0.91, '', transform=ax1.transAxes, color='black', fontsize=14)  # Text for transmission coefficient
time_text = ax1.text(0.77, 0.81, '', transform=ax1.transAxes, color='blue', fontsize=14)  # Text for elapsed time

# Precompute constants for transmission calculation
K = np.sqrt(2 * m_e * (U0 - Ec) * e) / hbar  # Intermediate calculation for K
T = 1 / ((U0**2 / (4 * Ec * (U0 - Ec)) * np.sinh(K * a)**2) + 1)  # Transmission coefficient

# Initialize time variable
current_time = 0.0  # Start time at 0

###################### Update Function for Animation ######################

def update(frame):
    global Psi_Real, Psi_Imag, Psi_Prob, current_time

    # Stop animation if elapsed time exceeds total duration
    if current_time > Nt / 1000:
        ani.event_source.stop()
        plt.close(fig)
        return

    # Update the wave function using the FDTD method
    for _ in range(60):  # Update multiple steps per frame
        Psi_Real[1:-1] -= a2 * (Psi_Imag[2:] - 2 * Psi_Imag[1:-1] + Psi_Imag[:-2]) + a3 * U[1:-1] * Psi_Imag[1:-1]  # updating the real part
        Psi_Imag[1:-1] += a2 * (Psi_Real[2:] - 2 * Psi_Real[1:-1] + Psi_Real[:-2]) - a3 * U[1:-1] * Psi_Real[1:-1]  # updating the imag part
        Psi_Prob[1:-1] = Psi_Real[1:-1]**2 + Psi_Imag[1:-1]**2  # updating the probability
        current_time += dt  # updating the current time as current_time = current_time + dt

    # Update probability density curve
    line_prob.set_data(x * 1.e9, Psi_Prob / np.sum(Psi_Prob))

    # Calculate probabilities on the left and right sides of the barrier
    left_side_prob = np.sum(Psi_Prob[:barrier_start_index]) / np.sum(Psi_Prob)
    right_side_prob = np.sum(Psi_Prob[barrier_end_index:]) / np.sum(Psi_Prob)

    # Update text annotations
    left_prob_text.set_text(f'$\Psi^2_L = $ {left_side_prob:.2e}')
    right_prob_text.set_text(f'$\Psi^2_R = $ {right_side_prob:.2e}')
    T_text.set_text(f'$T =$ {T:.2e}')
    time_text.set_text(f'Time: {current_time * 1e15:.2f} fs')  # Convert time to femtoseconds

    return line_prob, left_prob_text, right_prob_text, T_text, time_text

####################### Creating the Animation #########################

ani = FuncAnimation(fig, update, frames=range(0, int(Nt / dt)), interval=1, blit=True)  # Set up the animation
plt.tight_layout()  # Adjust layout to fit all elements

# Uncomment the following section to save the animation
# output_path = r"ADRESSE_POUR_ENREGISTRER_LANIMATION\QT_1D.mp4"
# writer = FFMpegWriter(fps=30, metadata=dict(artist='Hugo Alexandre'), bitrate=1800)
# with writer.saving(fig, output_path, dpi=200):
#     for frame in range(0, int(Nt / dt)):
#         update(frame)
#         writer.grab_frame()

plt.show()  # Display the animation
