# Gold Version - 2D Quantum Double-Slit Experiment Simulation

This Python code simulates the quantum double-slit experiment using the Finite Difference Time Domain (FDTD) method. The simulation models the behavior of a quantum wave packet as it encounters a potential barrier with two slits. It visualizes the resulting probability density, wave interference patterns, and other key aspects of the experiment. The simulation produces real-time animations to illustrate quantum mechanical phenomena.

## Prerequisites
To run the code, ensure the following Python libraries are installed:

- **numpy**: For numerical computations.
- **matplotlib**: For plotting and animation.
- **scipy**: For physical constants.

Install the libraries using pip if needed:

```bash
pip install numpy matplotlib scipy
```

## File Structure
- **Main Simulation Code**: The script contains the entire quantum double-slit simulation.
- **Animation**: The simulation generates real-time visualizations of the probability density evolution.

## How to Run the Code

### Step 1: Set Up Your Python Environment
Ensure you have Python 3.7 or newer installed on your system. After installing the required libraries, run the script using the following command:

```bash
python quantum_double_slit_simulation.py
```

### Step 2: Script Configuration
#### Constants and Parameters
- **Lx, Ly**: Dimensions of the quantum simulation box (meters).
- **Nx, Ny**: Number of spatial points in the x and y directions (grid resolution).
- **x0, y0**: Initial position of the wave packet.
- **sigma_x, sigma_y**: Width of the wave packet in x and y directions (meters).
- **Lambda**: de Broglie wavelength of the electron (meters).
- **U0**: Height of the potential barrier (joules).
- **a**: Width of the potential barrier (meters).
- **l**: Width of the slits in the barrier (meters).
- **d**: Distance between the slits (meters).

These parameters are pre-configured for the simulation but can be modified in the script to experiment with different setups.

### Step 3: Animation of the Double-Slit Experiment
The simulation will generate a real-time animation showcasing:

- **2D Probability Density**: The evolution of the quantum wave packet.
- **Potential Barrier**: The double-slit barrier is visualized, including the slit positions.
- **Interference Patterns**: The resulting patterns of wave interference beyond the barrier.
- **Dynamical Fit**: The wavefunction shape is dynamically fitted at a chosen distance to highlight the interference patterns more clearly.

### Plot Features
- The animation includes a color-coded representation of the probability density.
- Axes are labeled in nanometers for clarity.
- A color bar helps interpret the probability density values.

### Step 4: (Optional) Saving the Animation
To save the animation as a video, uncomment the relevant code section:

```python
output_path = r"PATH_TO_SAVE_ANIMATION/DSE2D+3D.mp4"
writer = FFMpegWriter(fps=30, metadata=dict(artist='Hugo Alexandre'), bitrate=1800)
with writer.saving(fig, output_path, dpi=200):
    for frame in range(0, int(Nt / dt)):
        update(frame)
        writer.grab_frame()
```

Replace `PATH_TO_SAVE_ANIMATION` with your desired save directory. The video will be saved as `QDSE_2D.mp4`.

## Key Output
### Visualizations:
- **2D Heatmap**: Displays the real-time evolution of the wave packet’s probability density.
- **Potential Barrier**: Highlights the double-slit barrier geometry.
- **Interference Patterns**: Shows the distinctive fringes created by quantum interference.
- **Dynamical Fit of Wavefunction**: Enhances the visualization of interference fringes at a chosen distance.

### Physical Insights:
- Observe how the quantum wave packet splits, diffracts, and interferes after passing through the slits.
- Explore the impact of varying the slit width, separation, or wave packet properties.

## Understanding the Code
### Main Components:
- **Psi_Real and Psi_Imag**: Real and imaginary parts of the wave function.
- **Psi_Prob**: Probability density, calculated as the sum of the squares of the real and imaginary parts.
- **U**: Potential profile defining the double-slit barrier.
- **FDTD Method**: Used to update the wave function's time evolution.
- **Dynamical Fit**: Applies a fitting function to extract interference fringes at a given distance.

### Numerical Methods:
- Time evolution of the wave packet uses the FDTD algorithm for stability and accuracy.
- Potential barriers are defined using numpy arrays with logical conditions for slit positions.

## Troubleshooting
- **Missing Libraries**: Ensure all dependencies are installed via `pip install`.
- **Animation Issues**: If the animation doesn't display, check your `matplotlib` installation or run the code in a different environment (e.g., Jupyter Notebook).


