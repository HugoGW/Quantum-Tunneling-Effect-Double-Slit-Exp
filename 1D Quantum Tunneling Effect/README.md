# 1D Quantum Tunneling Simulation (α Version)

## Description
This Python code simulates a 1D quantum wave packet encountering a potential barrier using the Finite Difference Time Domain (FDTD) method. The simulation computes the time evolution of the wave packet and visualizes the resulting probability density, potential, and key quantities like the transmission coefficient. It runs in real-time and generates an animated plot of the wave packet's evolution.

## Prerequisites
Ensure you have the following Python libraries installed:
```bash
pip install numpy matplotlib scipy
```

## File Structure
- **Main Simulation Code**: The entire quantum tunneling simulation is contained in a single Python script.
- **Animation**: The animation is generated in real-time and can optionally be saved as a video file.

## How to Run the Code

### Step 1: Set Up Your Python Environment
Ensure you have Python installed (recommended: Python 3.7 or newer). Run the script using:
```bash
python quantum_tunneling_simulation.py
```

### Step 2: Script Configuration
Modify the following parameters within the script to experiment with different scenarios:
- `L`: Length of the quantum box (m)
- `Nx`: Number of spatial points (grid resolution)
- `x0`: Initial position of the wave packet
- `sigma`: Width of the wave packet
- `Lambda`: de Broglie wavelength of the electron
- `U0`: Height of the potential barrier (eV)
- `a`: Width of the potential barrier (m)

### Step 3: Generate Animation
The simulation visualizes:
- The **probability density** evolution over time
- The **potential barrier** (blue curve)
- The **transmission coefficient (T)** and wave packet probabilities on both sides of the barrier

### Step 4: (Optional) Save the Animation
To save the animation as a video file, uncomment the following section in the script:
```python
output_path = r"ADRESSE_POUR_ENREGISTRER_LANIMATION/QT_1D.mp4"
writer = FFMpegWriter(fps=30, metadata=dict(artist='Hugo Alexandre'), bitrate=1800)
with writer.saving(fig, output_path, dpi=200):
    for frame in range(0, int(Nt / dt)):
        update(frame)
        writer.grab_frame()
```
Replace `ADRESSE_POUR_ENREGISTRER_LANIMATION` with your desired save location.

### Step 5: View the Animation
The animation is displayed using `matplotlib`. You can interact with it, zoom in/out, or pause the simulation.

## Key Output
- **Probability Density Curve**: Evolution of the wave packet over time
- **Transmission Coefficient (T)**: Probability of the wave packet passing through the barrier
- **Left & Right Probabilities**: Likelihood of finding the particle on either side of the barrier

## Understanding the Code
### Main Components
- `Psi_Real` and `Psi_Imag`: Real and imaginary parts of the wave function
- `U`: Potential profile with a defined barrier
- `Psi_Prob`: Probability density, updated at each time step
- `T`: Transmission coefficient, calculated based on barrier parameters

### Numerical Methods
- FDTD method updates the wave function at each time step
- Multiple steps per frame for smoother animation
- Uses physical constants (Planck's constant, electron mass, elementary charge)

## Troubleshooting
- If libraries are missing, ensure dependencies are installed.
- If animation issues arise, check `matplotlib` installation or try a different environment (e.g., Jupyter Notebook).

## License
This code is provided for educational purposes and can be modified or redistributed freely. Please credit the original author, **Hugo Alexandre**, if used or adapted in projects.

## Author
Hugo Alexandre

