# β Version - 2D Quantum Tunneling Simulation

## Description
This Python code simulates a 2D quantum wave packet encountering a potential barrier using the Finite Difference Time Domain (FDTD) method. The simulation computes the time evolution of the wave packet and visualizes the resulting probability density, potential, and other key quantities. The simulation runs in real-time and generates an animated plot of the wave packet's evolution.

## Prerequisites
To run the code, you need to have the following Python libraries installed:

- `numpy` for numerical computations and array manipulations.
- `matplotlib` for 2D and 3D plotting and animation.
- `scipy` for physical constants.
- `mpl_toolkits` for advanced plotting (e.g., 3D visualization and color bar adjustments).

You can install these libraries using pip if you don't have them installed already:

```bash
pip install numpy matplotlib scipy
```

## File Structure
- **Main Simulation Code**: The entire quantum tunneling simulation is contained in a single Python script.
- **Animation**: The animation is generated in real-time and can optionally be saved to a video file.

## How to Run the Code

### Step 1: Set Up Your Python Environment
Ensure you have Python installed on your system. It's recommended to use Python 3.7 or newer. Once the necessary libraries are installed, you can run the script using the following command:

```bash
python quantum_tunneling_simulation.py
```

### Step 2: Script Configuration
#### Constants and Parameters
- `Lx`, `Ly`: Dimensions of the quantum box in meters.
- `Nx`, `Ny`: Number of grid points along the x and y axes (resolution).
- `x_min`, `x_max`, `y_min`, `y_max`: Range of coordinates for the quantum box.
- `sigma_x`, `sigma_y`: Width of the wave packet in the x and y directions.
- `Lambda`: de Broglie wavelength of the wave packet.
- `U0`: Height of the potential barrier in joules (converted from eV).
- `a`: Width of the potential barrier in meters.
- `dt`: Time step calculated based on the stability factor for the simulation.

You can modify these values within the script to experiment with different scenarios.

#### Plot Configuration
The code will generate 2D and 3D plots displaying:
- The probability density of the wave packet in the 2D grid.
- The potential barrier.
- A dynamic fit of the wavefunction shape at a chosen distance.
- A separate 1D plot showing the probability density at a specific x-position.

### Step 3: Generate Animation
The simulation will generate an animation of the wave packet's interaction with the potential barrier. The plot will show:

- The real-time evolution of the probability density.
- The potential barrier as a blue curve.
- The dynamically fitted wavefunction at a chosen distance.

The animation will run for the specified total duration (`Nt`). You can stop the animation early if necessary.

### Step 4: (Optional) Saving the Animation
If you'd like to save the animation as a video, you can uncomment the lines under Creating the Animation:

```python
# Uncomment the following section to save the animation
output_path = r"ADRESSE_POUR_ENREGISTRER_LANIMATION/QT_2D.mp4"
writer = FFMpegWriter(fps=30, metadata=dict(artist='Hugo Alexandre'), bitrate=1800)
with writer.saving(fig, output_path, dpi=200):
    for frame in range(0, int(Nt / dt)):
        update(frame)
        writer.grab_frame()
```

Replace `ADRESSE_POUR_ENREGISTRER_LANIMATION` with the directory where you'd like to save the video file. The file will be saved as `QT_2D.mp4`.

### Step 5: View the Animation
The plot will be displayed using `matplotlib`. You can interact with the animation, zoom in/out, or pause the simulation. Once the simulation finishes, the animation will stop, and the plot window will close.

## Key Output
- **2D Probability Density Plot**: Shows the wave packet's evolution over time in the quantum box.
- **1D Probability Density Plot**: Displays the wave packet's probability density at a specific position (`x`).
- **3D Probability Density Plot**: Provides a 3D visualization of the probability density across the entire grid.
- **Dynamical Fit of the Wavefunction**: A fit of the wave packet shape at a chosen distance for analysis.

## Understanding the Code

### Main Parameters:
- `Psi_Real` and `Psi_Imag` represent the real and imaginary parts of the wave function.
- `U` is the potential profile with a defined barrier.
- `Psi_Prob` is the probability density, which is updated at each time step.

### Numerical Methods:
- The FDTD method is used to update the wave function at each time step.
- The simulation runs multiple steps per frame for smoother animation.
- The code uses physical constants like Planck's constant, electron mass, and elementary charge.

### Theoretical Comparison:
A theoretical function with a Gaussian envelope is compared to the numerical results for validation of the simulation's accuracy.

## Troubleshooting
- If you encounter an error related to missing libraries, ensure that all dependencies are installed.
- If the animation doesn't display correctly, check the `matplotlib` installation or try running the code in a different environment (e.g., Jupyter Notebook).

## License
This code is provided for educational purposes and can be modified and redistributed freely. Please give credit to the original author, **Hugo Alexandre**, if you use or adapt this code in your projects.

## Author
Hugo Alexandre

