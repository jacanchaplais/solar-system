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
traj = data.drop(columns=['Mass','Z', 'VZ']) # create a DataFrame in which the calculations will be stored

index = traj.index.copy()
columns = traj.columns.copy()
start_date = index.levels[1][0] # the first date from the input file


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


# ------------------------------------------------ FORMATTING THE DATA ----------------------------------------------- #
def format_data(pos_data, vel_data, index, column, num_iter, cur_date):
    """Takes position and velocity arrays of shape nbod x ndim x nt (bodies, spatial dimensions, time) and returns a
    DataFrame of the data, sorted column-wise.
    
    Keyword arguments:
    pos_data -- (nbod x ndim x nt array) position data
    vel_data -- (nbod x ndim x nt array) velocity data
    index -- Pandas.MultiIndex object for a single date
    column -- (list / array-like) column headers for the formatted DataFrame
    num_iter -- (int) number of dates over which the data represents
    cur_date -- (Pandas.DateTime object) the date at the start of the data
    
    Returns:
    cur_traj -- the formatted DataFrame 
    """
    date_range = cur_date + pd.to_timedelta(np.arange(num_iter), 'D') # create a datetime array from start to finish

    body_order = index.levels[0][index.codes[0]]

    index.set_levels(date_range, level=1, inplace=True) # set the dates for our new index
    index.set_codes([
        np.tile(index.codes[0], num_iter), # copies mapping of planet name to number for every timestep
        np.repeat( np.arange(num_iter), len(index.codes[1]) ) # applies the date to the set of planets at each timestep
    ], verify_integrity=False, inplace=True)

    # reshapes the output data from n x s x t, to (n * t) x s, glueing the time slices as rows, forming long columns
    pos_data = pos_data.transpose(2,0,1).reshape(pos_data.shape[0] * pos_data.shape[2], pos_data.shape[1])
    vel_data = vel_data.transpose(2,0,1).reshape(vel_data.shape[0] * vel_data.shape[2], vel_data.shape[1])

    traj_data = np.hstack((pos_data, vel_data)) # position and velocity data in adjacent columns, for the dataframe

    # puts it all together to actually create the dataframe:
    cur_traj = pd.DataFrame(data=traj_data, index=index, columns=column)
    cur_traj.sort_index(inplace=True)
    cur_traj = cur_traj.reindex(body_order, level='Name')
    
    return cur_traj

# ------------------------------------------ SETTING UP THE LOOP VARIABLES ------------------------------------------- #
# define the length of the calculation (in days), and the number of calculation steps to perform:
timespan = 60400.0
num_steps = 60400
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
# creates a data/ storage directory, if it does not already exist:
data_dir = 'data/'
fpath = data_dir + 'traj.out'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
elif (os.path.isfile(fpath)):
    os.remove(fpath)


for step in range(1, num_steps):
    # apply the euler method of integration to calculate new velocities and positions:
    cur_vel = cur_vel + accelerate(cur_pos, mass) * time_change
    cur_pos = cur_pos + cur_vel * time_change

    # record the next 2D arrays of positions and velocities along the time axes of their respective arrays:
    vel = np.dstack((vel, cur_vel))
    pos = np.dstack((pos, cur_pos))

    # update the progress readout:
    cntr = cntr + 1
    last_step = step == num_steps - 1
    if (cntr == cntr_change or last_step):
        pcnt = pcnt + pcnt_change
        sys.stdout.write('\r{}% complete'.format(pcnt))
        sys.stdout.flush()
        
        # record all but the last data
        cur_date = start_date + pd.to_timedelta(step - cntr, 'D')
        
        if not last_step:
            cur_traj = format_data(pos[:, :, :-1], vel[:, :, :-1], index.copy(), columns, cntr_change, cur_date)
            pos = pos[:, :, -1]
            vel = vel[:, :, -1]
        else:
            cur_traj = format_data(pos, vel, index.copy(), columns, cntr_change, cur_date)
        
        file_exists = os.path.isfile(fpath)
        write_mode = 'a' if file_exists else 'w' # if the storage file exists, append to it

        with open(fpath, write_mode) as f: # save as csv
            cur_traj.to_csv(f, header=(not file_exists)) # writes out to data file

        cntr = 0
