# to do:
# get the real astronomical data at
# https://ssd.jpl.nasa.gov/horizons.cgi?CGISESSID=0be6725975911d33c2020b132ee1a912&s_disp=1#top

import numpy as np
import pandas as pd

au_length, au_mass, au_time = 1.495978707E+11, 1.98892E+30, 8.64E+4  # metres, kilograms, seconds
grav_constant = 6.67408E-11 * (au_mass * au_time ** 2) / (au_length ** 3)


# -------------------------------------------- READING SOLAR SYSTEM DATA --------------------------------------------- #
data = pd.read_csv('bodies.inp', index_col=0)
masses = data['Mass'].values[:, np.newaxis]
positions = data.loc[:, 'X':'Y'].values
velocities = data.loc[:, 'VX':'VY'].values


# --------------------------- CALCULATING THE ACCELERATIONS OF THESE BODIES DUE TO GRAVITY --------------------------- #
def accelerate(position, mass):
    displacement = position[:, np.newaxis] - position
    distance = np.linalg.norm(displacement, axis=2)
    inv_cube_dist = np.power(distance, -3, where=(distance != 0.0))[:, :, np.newaxis]

    return - grav_constant * np.sum(inv_cube_dist * np.swapaxes(mass * displacement, 0, 1), axis=0)


# --------------------------------------- PERFORMING THE NUMERICAL INTEGRATION --------------------------------------- #
num_steps = 30200
timespan = 30200.0
time_change = timespan / float(num_steps)

velocities = velocities[:, :, np.newaxis]
positions = positions[:, :, np.newaxis]

counter = 0
percnt = 0

for step in np.arange(1, num_steps, dtype=int):
    velocity_change = accelerate(positions[:, :, step - 1], masses) * time_change
    velocities = np.dstack([velocities, velocities[:, :, step - 1] + velocity_change])
    position_change = velocities[:, :, step] * time_change
    positions = np.dstack([positions, positions[:, :, step - 1] + position_change])
    counter = counter + 1
    if (counter >= 3020):
        counter = 0
        percnt = percnt + 10
        print('{}% complete'.format(percnt))
