# Solar system simulation

This program uses information collected from NASA's Horizons servers to calculate the 2D planar positions of the Sun and all planets in the solar system.

The data collected from NASA is stored in the bodies.inp file as a CSV, and contains the data for the positions and velocities of the Sun and planets on the 16th June, 2019. From this point, the Euler method is then applied to find desired phase data up to a specified date.

The output is formatted Pandas DataFrames, then stored to CSV in data/traj.out.

## To do
- [ ] Automate data collection from Horizon servers
- [ ] Write visualiser module
- [ ] Implement periodic file writing to speed up array indexing during recording of data
- [ ] Use a higher order numerical integration scheme, like RK4
- [ ] Extend to 3D
