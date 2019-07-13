# astronomical data obtained from NASA Horizons server:
# https://ssd.jpl.nasa.gov/horizons.cgi?CGISESSID=0be6725975911d33c2020b132ee1a912&s_disp=1#top

import numpy as np
import pandas as pd
import sys, os


# ------------------------------------------- DEFINING ASTRONOMICAL UNITS -------------------------------------------- #
au_length, au_mass, au_time = 1.495978707E+11, 1.98892E+30, 8.64E+4  # metres, kilograms, seconds
grav_constant = 6.67408E-11 * (au_mass * au_time ** 2) / (au_length ** 3)


# -------------------------------------------- READING SOLAR SYSTEM DATA --------------------------------------------- #
data = pd.read_csv('bodies.inp', index_col=[0,1], parse_dates=True)
body_order = data.index.levels[0][data.index.codes[0]] # preserve the order of the bodies, for later sorting
start_date = data.index.levels[1][0] # the first date from the input file
traj = data.drop(columns=['Mass','Z', 'VZ']) # create a DataFrame in which the calculations will be stored


# --------------------------- CALCULATING THE ACCELERATIONS OF THESE BODIES DUE TO GRAVITY --------------------------- #
def accelerate(position, mass):
    """Calculates the cumulative Newtonian gravitational acceleration exerted on all n bodies within the system.
    
    Keyword arguments:
    position -- (n x 2 array) n rows of 2d position for each body
    mass -- (1-d array of length n) containing the masses of each body
    
    Returns:
    (n x 2 array) n rows of 2-dimensional acceleration for each body
    """
    mass = mass[:, np.newaxis]
    displacement = position[:, np.newaxis] - position
    distance = np.linalg.norm(displacement, axis=2)
    inv_cube_dist = np.power(distance, -3, where=(distance != 0.0))[:, :, np.newaxis]

    return - grav_constant * np.sum(inv_cube_dist * np.swapaxes(mass * displacement, 0, 1), axis=0)


# ------------------------------------------ SETTING UP THE LOOP VARIABLES ------------------------------------------- #
# define the length of the calculation (in days), and the number of calculation steps to perform:
timespan = 30200.0
num_steps = 30200
time_change = timespan / float(num_steps) # the resulting time difference between each step

# set up the variables to give progress readouts during execution:
cntr = 0
pcnt = 0
pcnt_change = 10 # (int) change this value to alter after how many percent of the calculation the readout is shown
cntr_change = int((pcnt_change / 100.0) * num_steps)


# ---------------------------- PREPARING THE PHYSICAL DATA STRUCTURES FROM THE INPUT FILE ---------------------------- #
# the calculation will be performed using Numpy, and formatted using Pandas afterwards

idx_slc = pd.IndexSlice # a pandas object to make index slicing easier

pos = traj.loc[idx_slc[:, start_date], ['X','Y']].values # get the initial position as 2D array
vel = traj.loc[idx_slc[:, start_date], ['VX','VY']].values

cur_pos = pos # store 2D array of positions at first timestep, for calculations
cur_vel = vel

pos = pos[:, :, np.newaxis] # add a time axis, for recording
vel = vel[:, :, np.newaxis]

mass = data['Mass'].values # get the mass as a 1D array


# --------------------------------------- PERFORMING THE NUMERICAL INTEGRATION --------------------------------------- #
for step in range(1, num_steps):
    # apply the euler method of integration to calculate new velocities and positions:
    cur_vel = cur_vel + accelerate(cur_pos, mass) * time_change
    cur_pos = cur_pos + cur_vel * time_change

    # record the next 2D arrays of positions and velocities along the time axes of their respective arrays:
    vel = np.dstack((vel, cur_vel))
    pos = np.dstack((pos, cur_pos))

    # update the progress readout:
    cntr = cntr + 1
    if (cntr >= cntr_change):
        cntr = 0
        pcnt = pcnt + pcnt_change
        sys.stdout.write('\r{}% complete'.format(pcnt))
        sys.stdout.flush()


# ------------------------------------------------ FORMATTING THE DATA ----------------------------------------------- #
# first, set up the multi-level index for this newly calculated data:
index = traj.index.copy() # copy the index for one timestep
date_range = start_date + pd.to_timedelta(np.arange(num_steps), 'D') # create a datetime array from start to finish

index.set_levels(date_range, level=1, inplace=True) # set the dates for our new index
index.set_codes([
    np.tile(index.codes[0], num_steps), # copies mapping of planet name to number for every timestep
    np.repeat( np.arange(num_steps), len(index.codes[1]) ) # applies the date to the set of planets at each timestep
], verify_integrity=False, inplace=True)

column = traj.columns.copy()

# reshapes the output data from n x s x t, to (n * t) x s, glueing the time slices as rows, forming long columns
pos = pos.transpose(2,0,1).reshape(pos.shape[0] * pos.shape[2], pos.shape[1])
vel = vel.transpose(2,0,1).reshape(vel.shape[0] * vel.shape[2], vel.shape[1])

traj_data = np.hstack((pos, vel)) # sticks the position and velocity data in adjacent columns, for the dataframe

# puts it all together to actually create the dataframe:
traj = pd.DataFrame(data=traj_data, index=index, columns=column)
traj.sort_index(inplace=True)
traj = traj.reindex(body_order, level='Name')

del traj_data, pos, vel, cur_pos, cur_vel, date_range # cleaning up some data

# creates a data/ storage directory, if it does not already exist:
data_dir = 'data/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

with open(data_dir + 'traj.out', 'w') as f: # save as csv
    traj.to_csv(f) # writes out to data file
