# to do:
# get the real astronomical data at
# https://ssd.jpl.nasa.gov/horizons.cgi?CGISESSID=0be6725975911d33c2020b132ee1a912&s_disp=1#top

import numpy as np
import pandas as pd

au_length, au_mass, au_time = 1.495978707E+11, 1.98892E+30, 8.64E+4  # metres, kilograms, seconds
grav_constant = 6.67408E-11 * (au_mass * au_time ** 2) / (au_length ** 3)


# -------------------------------------------- READING SOLAR SYSTEM DATA --------------------------------------------- #
data = pd.read_csv('bodies.inp', index_col=[0,1], parse_dates=True)

data.index.set_levels(data.index.levels[0][data.index.codes[0]], level='Name', inplace=True)
data.index.set_codes(np.sort(data.index.codes[0]), level='Name', inplace=True)

trajectory = data.drop(columns=['Mass','Z', 'VZ'])


# --------------------------- CALCULATING THE ACCELERATIONS OF THESE BODIES DUE TO GRAVITY --------------------------- #
def accelerate(position, mass):
    """Calculates the cumulative Newtonian gravitational acceleration exerted on every body within the system.
    
    Keyword arguments:
    position -- 
    mass --
    """
    mass = mass[:, np.newaxis]
    displacement = position[:, np.newaxis] - position
    distance = np.linalg.norm(displacement, axis=2)
    inv_cube_dist = np.power(distance, -3, where=(distance != 0.0))[:, :, np.newaxis]

    return - grav_constant * np.sum(inv_cube_dist * np.swapaxes(mass * displacement, 0, 1), axis=0)


# --------------------------------------- PERFORMING THE NUMERICAL INTEGRATION --------------------------------------- #
num_steps = 30200
timespan = 30200.0
time_change = timespan / float(num_steps)

counter = 0
percnt = 0

idx = pd.IndexSlice
start_date = trajectory.index.levels[1]
date_delta = pd.to_timedelta(time_change, 'D')

cur_traj = trajectory.copy()
cur_date = start_date

for step in np.arange(1, num_steps, dtype=int):
    cur_date = cur_date + date_delta
    
    # update the entry to the current date in this iteration:
    cur_traj.index.set_levels(cur_date, level=1, inplace=True)
    
    velocity = cur_traj.loc[idx[:, cur_date], ['VX','VY']]
    position = cur_traj.loc[idx[:, cur_date], ['X','Y']]
    
    new_velocity = velocity + accelerate(position.values, data['Mass'].values) * time_change
    new_position = position + new_velocity.values * time_change
    
    cur_traj.loc[idx[:, cur_date], ['VX','VY']] = new_velocity
    cur_traj.loc[idx[:, cur_date], ['X','Y']] = new_position
    
    trajectory = trajectory.append(cur_traj)
    
    counter = counter + 1
    if (counter >= 302):
        counter = 0
        percnt = percnt + 1
        print('{}% complete'.format(percnt))
        
trajectory.sort_index()
