import numpy as np

from solar import config


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

