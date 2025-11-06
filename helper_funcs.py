#!/usr/bin/env python3

import numpy as np

def tendon_length_kinematics( config_d, n_motors, r_t, ten_angles, L ):

    phi_d = np.deg2rad(config_d[0])
    theta_d = np.deg2rad(config_d[1])

    n_ten = n_motors
    l = np.zeros(n_ten)

    for i in range(n_ten):
        l[i] = L - r_t*theta_d*(np.cos(ten_angles[i] - phi_d))
  
    return l

def length_to_step_conversion( l, n_motors, metres_per_step, l_curr ):

    ds = np.zeros(n_motors)
    for i in range(n_motors):
        ds[i] = round(l_curr[i] - (l[i] / metres_per_step))

    return ds

