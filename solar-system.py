###############################################################################
#                          Solar System Simulation                            #
#                      Written by Jacan Chaplais, 2019                        #
#                         jacan.chaplais@gmail.com                            #
###############################################################################

# This code takes input of the positions and velocities of all bodies
# within a gravitationally interacting system and evolves them through
# time.
#
# The program reads and writes in CSV format using Pandas. Input data
# is provided in bodies.inp, output data is written to data/traj.out

import os
import sys

import numpy as np
import pandas as pd

from solar import io, config, algo


# ------------------------ READING SOLAR SYSTEM DATA ------------------------ #
data = io.read('bodies.inp')
traj = data.drop(columns=['Mass','Z', 'VZ'])  # DataFrame to stores calcs

index = traj.index.copy()
columns = traj.columns.copy()
start_date = index.levels[1][0]  # first date from input file


# ------- CALCULATING THE ACCELERATIONS OF THESE BODIES DUE TO GRAVITY ------ #
def accelerate(position, mass):
    """Calculates the cumulative Newtonian gravitational acceleration
    exerted on all n bodies within the system.
    
    Keyword arguments:
    position -- (n x 2 array) n rows of 2d position for each body
    mass -- (1-d array of length n) containing the masses of each body
    
    Returns:
    (n x 2 array) n rows of 2-dimensional acceleration for each body
    """
    mass = mass[:, np.newaxis]  # formats as column vector
    
    # Subtracts position row vector from col vector, to form square
    # antisymm. matrix of displacements D_ij, from body i to j.
    displacement = position[:, np.newaxis] - position
    
    # Calc matrix of distance d_ij from displacement vectors in D_ij.
    distance = np.linalg.norm(displacement, axis=2)
    
    # Calc matrix of (1 / d_ij) ^ 3, except where d_ij = 0.
    inv_cube_dist = np.power(
        distance, -3, where=(distance != 0.0))
    
    inv_cube_dist = inv_cube_dist[:, :, np.newaxis]

    acc = - config.GRAV_CONST * np.sum(
        inv_cube_dist * np.swapaxes(mass * displacement, 0, 1),
        axis=0)
    
    return acc


# ---------------------- SETTING UP THE LOOP VARIABLES ---------------------- #
# Define length of the calc (days), & the num of calc steps to perform.
timespan = 60400.0
num_steps = 60400
time_change = timespan / float(num_steps)  # resulting time diff between steps

# Set up the variables to give progress readouts during execution.
cntr = 0
pcnt = 0
pcnt_change = 10  # (int) set how much change in % of calc for readout update
cntr_change = int((pcnt_change / 100.0) * num_steps)


# -------- PREPARING THE PHYSICAL DATA STRUCTURES FROM THE INPUT FILE ------- #
# The calculation is in Numpy, formatted in Pandas after.

idx_slc = pd.IndexSlice  # a pandas object to make index slicing easier

# Get initial 2D arrays for position and velocity:
pos = traj.loc[idx_slc[:, start_date], ['X','Y']].values
vel = traj.loc[idx_slc[:, start_date], ['VX','VY']].values

# store 2D array of pos at first timestep, for calcs
cur_pos = pos
cur_vel = vel

# add a time axis, for recording
pos = pos[:, :, np.newaxis]
vel = vel[:, :, np.newaxis]

# get the mass as a 1D array
mass = data['Mass'].values


# ------------------- PERFORMING THE NUMERICAL INTEGRATION ------------------ #
# Create a data/ storage directory, if it does not already exist.
data_dir = 'data/'
fpath = data_dir + 'traj.out'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
elif (os.path.isfile(fpath)):
    os.remove(fpath)


for step in range(1, num_steps):
    # Apply euler method to calculate new velocities & positions.
    cur_vel = cur_vel + accelerate(cur_pos, mass) * time_change
    cur_pos = cur_pos + cur_vel * time_change

    # Record next 2D arrays of pos & vels along t axes of arrays.
    vel = np.dstack((vel, cur_vel))
    pos = np.dstack((pos, cur_pos))


# ----------------- WRITE TO OUTPUT FILE AND UPDATE PROGRESS ---------------- #
    cntr = cntr + 1
    last_step = step == num_steps - 1
    
    io.display_progress(last_step, cntr, cntr_change, pcnt)
    
    if (cntr == cntr_change or last_step):
        pcnt = pcnt + pcnt_change
        # Record all but the last timestep of data.
        cur_date = start_date + pd.to_timedelta(step - cntr, 'D')
        
        if not last_step:
            cur_traj = io.format_data(
                pos[:, :, :-1], vel[:, :, :-1],
                index.copy(), columns,
                cntr_change, cur_date)
            
            pos = pos[:, :, -1]
            vel = vel[:, :, -1]
        else:
            cur_traj = io.format_data(
                pos, vel,
                index.copy(), columns,
                cntr_change, cur_date)
        
        io.write(fpath, cur_traj)

        cntr = 0
