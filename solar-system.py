# astronomical data obtained from NASA Horizons server:
# https://ssd.jpl.nasa.gov/horizons.cgi?CGISESSID=0be6725975911d33c2020b132ee1a912&s_disp=1#top

import numpy as np
import pandas as pd
import sys, os


# ----------------------- DEFINING ASTRONOMICAL UNITS ------------------------ #
# metres, kilograms, seconds:
au_len, au_mass, au_time = 1.495978707E+11, 1.98892E+30, 8.64E+4
grav_const = 6.67408E-11 * (au_mass * au_time ** 2) / (au_len ** 3)


# ------------------------ READING SOLAR SYSTEM DATA ------------------------- #
data = pd.read_csv('bodies.inp', index_col=[0,1], parse_dates=True)
traj = data.drop(columns=['Mass','Z', 'VZ']) # DataFrame to stores calcs

index = traj.index.copy()
columns = traj.columns.copy()
start_date = index.levels[1][0] # first date from input file


# ------- CALCULATING THE ACCELERATIONS OF THESE BODIES DUE TO GRAVITY ------- #
def accelerate(position, mass):
    """Calculates the cumulative Newtonian gravitational acceleration
    exerted on all n bodies within the system.
    
    Keyword arguments:
    position -- (n x 2 array) n rows of 2d position for each body
    mass -- (1-d array of length n) containing the masses of each body
    
    Returns:
    (n x 2 array) n rows of 2-dimensional acceleration for each body
    """
    mass = mass[:, np.newaxis] # formats as column vector
    
    # subtracts position row vector from col vector, to form square
    # antisymm. matrix of displacements D_ij, from body i to j
    displacement = position[:, np.newaxis] - position
    
    # calc matrix of distance d_ij from displacement vectors in D_ij
    distance = np.linalg.norm(displacement, axis=2)
    
    # calc matrix of (1 / d_ij) ^ 3, except where d_ij = 0
    inv_cube_dist = np.power(
        distance, -3, where=(distance != 0.0))
    
    inv_cube_dist = inv_cube_dist[:, :, np.newaxis]

    acc = - grav_const * np.sum(
        inv_cube_dist * np.swapaxes(
            mass * displacement, 0, 1),
        axis=0)
    
    return acc


# ---------------------------- FORMATTING THE DATA --------------------------- #
def format_data(pos_data, vel_data, index, column, num_iter, cur_date):
    """Takes position and velocity arrays of shape nbod x ndim x nt
    (bodies, spatial dimensions, time) and returns a DataFrame of the
    data, sorted column-wise.
    
    Keyword arguments:
    pos_data -- (nbod x ndim x nt array) position data
    vel_data -- (nbod x ndim x nt array) velocity data
    index -- Pandas.MultiIndex for a single date
    column -- (array-like) column headers for the formatted DataFrame
    num_iter -- (int) number of dates over which the data represents
    cur_date -- (Pandas.DateTime) the date at the start of the data
    
    Returns:
    cur_traj -- the formatted DataFrame 
    """
    
    # create a datetime array from start to finish:
    date_range = cur_date + pd.to_timedelta(np.arange(num_iter), 'D') 

    body_order = index.levels[0][index.codes[0]]

    index.set_levels(date_range, level=1, inplace=True) # set dates of new index
    index.set_codes(
        [
            np.tile( # map planet name to number
                index.codes[0],
                num_iter),
                
            np.repeat( # apply date to planets at each timestep
                np.arange(num_iter),
                len(index.codes[1]))
        ],
        verify_integrity=False, # allow index to be erroneous for now
        inplace=True # change index in place
    )

    # reshapes 3d arrays of to 2d, making time axis add repeating rows:
    # position
    pos_data = pos_data.transpose(2,0,1).reshape( # moves t axis to front
            # rows are each body repeated for every timestep:
            pos_data.shape[0] * pos_data.shape[2],
            # columns give x & y pos for each body:
            pos_data.shape[1])
    
    # same for velocity:
    vel_data = vel_data.transpose(2,0,1).reshape(
            vel_data.shape[0] * vel_data.shape[2],
            vel_data.shape[1])

    traj_data = np.hstack((pos_data, vel_data)) # pos & vel data in adj cols

    # puts it all together to actually create the dataframe:
    cur_traj = pd.DataFrame(data=traj_data, index=index, columns=column)
    cur_traj.sort_index(inplace=True)
    cur_traj = cur_traj.reindex(body_order, level='Name')
    
    return cur_traj

# ---------------------- SETTING UP THE LOOP VARIABLES ----------------------- #
# define length of the calc (in days), & the num of calc steps to perform:
timespan = 60400.0
num_steps = 60400
time_change = timespan / float(num_steps) # resulting time diff between steps

# set up the variables to give progress readouts during execution:
cntr = 0
pcnt = 0
pcnt_change = 10 # (int) set how much change in % of calc for readout update
cntr_change = int((pcnt_change / 100.0) * num_steps)


# -------- PREPARING THE PHYSICAL DATA STRUCTURES FROM THE INPUT FILE -------- #
# the calc is in Numpy, formatted in Pandas after

idx_slc = pd.IndexSlice # a pandas object to make index slicing easier

pos = traj.loc[idx_slc[:, start_date], ['X','Y']].values # get init pos 2D array
vel = traj.loc[idx_slc[:, start_date], ['VX','VY']].values

cur_pos = pos # store 2D array of positions at first timestep, for calculations
cur_vel = vel

pos = pos[:, :, np.newaxis] # add a time axis, for recording
vel = vel[:, :, np.newaxis]

mass = data['Mass'].values # get the mass as a 1D array


# ------------------- PERFORMING THE NUMERICAL INTEGRATION ------------------- #
# creates a data/ storage directory, if it does not already exist:
data_dir = 'data/'
fpath = data_dir + 'traj.out'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
elif (os.path.isfile(fpath)):
    os.remove(fpath)


for step in range(1, num_steps):
    # apply euler method to calc new vels & pos:
    cur_vel = cur_vel + accelerate(cur_pos, mass) * time_change
    cur_pos = cur_pos + cur_vel * time_change

    # record next 2D arrays of pos & vels along t axes of arrays:
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
            cur_traj = format_data(
                pos[:, :, :-1], vel[:, :, :-1],
                index.copy(), columns,
                cntr_change, cur_date)
            
            pos = pos[:, :, -1]
            vel = vel[:, :, -1]
        else:
            cur_traj = format_data(
                pos, vel,
                index.copy(), columns,
                cntr_change, cur_date)
        
        file_exists = os.path.isfile(fpath)
        write_mode = 'a' if file_exists else 'w' # if store file exists, append

        with open(fpath, write_mode) as f: # save as csv
            cur_traj.to_csv(f, header=(not file_exists)) # write to data file

        cntr = 0
