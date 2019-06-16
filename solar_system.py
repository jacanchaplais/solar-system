# to do:
# get the real astronomical data at
# https://ssd.jpl.nasa.gov/horizons.cgi?CGISESSID=0be6725975911d33c2020b132ee1a912&s_disp=1#top

import numpy as np

au_length, au_mass, au_time = 1.495978707E+11, 1.98892E+30, 8.64E+4 # metres, kilograms, seconds
grav_constant = 6.67408E-11 * (au_mass * au_time ** 2) / (au_length ** 3)
#-------------------------------------- GENERATING RANDOM ASTRONOMICAL BODY DATA --------------------------------------#
np.random.seed(0)

N = 4 # number of bodies in the solar system

masses = np.random.random(N) # uniformly random astronomical bodies
masses[0] = 1.0 # sets the star to have a mass ~ 3 - 4 orders of magnitude higher than the planets
masses[1:] *= 1.0E-6
masses = masses[:, np.newaxis]

system_radius = 1.0 # distance to neptune in AU
orbital_radii = np.linspace(0.0, system_radius, N)[:, np.newaxis]
angles = np.random.random(N) * 2.0 * np.pi
positions = orbital_radii * np.array([np.cos(angles), np.sin(angles)]).T

# set up all planets to rotate counterclockwise about the star
directions = np.cross( np.insert(positions, 2, 0.0, axis=1), np.array([0.0, 0.0, 1.0]), axisa=1)
directions = directions[:, :2] # recasting back to 2 dimensions
directions[1:] = directions[1:] / orbital_radii[1:]
# apply roughly the correct speed given the orbital radius to keep in stable orbit
velocities = np.zeros((N, 2))
velocities[1:,:] = np.sqrt( grav_constant *  masses[0] / orbital_radii[1:]) * directions[1:, :]

#---------------------------- CALCULATING THE ACCELERATIONS OF THESE BODIES DUE TO GRAVITY ----------------------------#
def accelerate(position, mass):
    displacement = position[:, np.newaxis] - position
    distance = np.linalg.norm(displacement, axis=2)
    inv_cube_dist = np.power(distance, -3, where=(distance!=0.0))[:, :, np.newaxis]

    return - grav_constant * np.sum(inv_cube_dist * np.swapaxes(mass * displacement, 0, 1), axis=0)

#---------------------------------------- PERFORMING THE NUMERICAL INTEGRATION ----------------------------------------#
num_steps = 370
timespan = 365.25
time_change = timespan / float(num_steps)

velocities = velocities[:, :, np.newaxis]
positions = positions[:, :, np.newaxis]

for step in np.arange(1, num_steps, dtype=int):
    velocity_change = accelerate(positions[:, :, step - 1], masses) * time_change
    velocities = np.dstack([velocities, velocities[:, :, step - 1] + velocity_change])
    position_change = velocities[:, :, step] * time_change
    positions = np.dstack([positions, positions[:, :, step - 1] + position_change])
