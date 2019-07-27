"""
"""

import os
import sys

import numpy as np
import pandas as pd


def read(fname):
    return pd.read_csv(fname, index_col=[0,1], parse_dates=True)


def write(fpath, data):
    file_exists = os.path.isfile(fpath)
    write_mode = 'a' if file_exists else 'w'  # if store file exist, append

    with open(fpath, write_mode) as f: # save as csv
        data.to_csv(f, header=(not file_exists))


def display_progress(last_step, cntr, cntr_change, pcnt):
    if (cntr == cntr_change or last_step):
        sys.stdout.write('\r{}% complete'.format(pcnt))
        sys.stdout.flush()

# --------------------------- FORMATTING THE DATA --------------------------- #
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
    
    # Create a datetime array from start to finish.
    date_range = cur_date + pd.to_timedelta(np.arange(num_iter), 'D') 

    body_order = index.levels[0][index.codes[0]]

    index.set_levels(date_range, level=1, inplace=True)  # set dates new index
    index.set_codes(
        [
            np.tile(  # map planet name to number
                index.codes[0],
                num_iter),
                
            np.repeat(  # apply date to planets at each timestep
                np.arange(num_iter),
                len(index.codes[1]))
        ],
        verify_integrity=False,  # allow index to be erroneous for now
        inplace=True
    )

    # Reshapes 3d arrays of to 2d, making time axis add repeating rows.
    # position:
    pos_data = pos_data.transpose(2,0,1).reshape(  # moves t axis to front
            # rows are each body repeated for every timestep:
            pos_data.shape[0] * pos_data.shape[2],
            # columns give x & y pos for each body:
            pos_data.shape[1])
    
    # same for velocity:
    vel_data = vel_data.transpose(2,0,1).reshape(
            vel_data.shape[0] * vel_data.shape[2],
            vel_data.shape[1])

    traj_data = np.hstack((pos_data, vel_data))  # pos & vel data in adj cols

    # Puts it all together to create the DataFrame.
    cur_traj = pd.DataFrame(data=traj_data, index=index, columns=column)
    cur_traj.sort_index(inplace=True)
    cur_traj = cur_traj.reindex(body_order, level='Name')
    
    return cur_traj