# Noise Control (EAC1051)
Undergraduate course at UFSM lectured by Dr. Paulo Mareze.

## Sound absorption and 1 degree of freedom vibration modeling
- Adds `helmholtz_layer` to the [TMM](https://github.com/rinaldipp/tmm) package;
- Transmissibility plots for vibration control analysis.

## Piston on an infinite baffle simulator
- Simulates a radial piston sound source using a discrete approximation of the Rayleigh integral.
- SPL per frequency, SPL along axis, directivity and contour plots.
- Comparisons with the analytical models (exact and far field approximation) seen on Kinsler's Fundamentals of Acoustics (4th ed), chapter 7.

## Transformer noise calculator
- Calculates the sound power level and pressule level for the noise of an industrial transformer humming indoors.
- Equations taken from Barron's Industrial Noise Control and Acoustics (1st ed).

## Installation
The [uv](https://docs.astral.sh/uv/) package manager is the recommended way to use this project. Clone the repository, `cd` into the project folder and run `uv sync`. This will install all dependencies. Afterwards, you can run `uv run --with jupyter jupyter lab` to start the JupyterLab environment and explore the notebooks.