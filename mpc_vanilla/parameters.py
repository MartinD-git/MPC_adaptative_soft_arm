import numpy as np

def arm_parameters():
    L = 0.3
    d =0.1/L
    return {
        'L_segs': [L, L, L],  # Length of each segment m
        'm': 0.5,# Mass per unit length kg/m
        'g': [0, 0, -9.81],# Gravity vector m/s^2
        'd_eq': [d, d, d],
        'K': np.diag([0, 0.2, 0, 0.6, 0, 0.2])
    }

def mpc_parameters():
    q=15
    r=1
    return {
        'N': 15, #Horizon length
        'Q': np.diag([
            q,
            q,
            q,
            q,
            q,
            q
        ]),
        'R': np.diag([r,r,r,r,r,r])
    }

def get_simulation_params():
    return {
        'dt': 0.01, # s 0.05
        'T': 10, # Total simu time s
        'x0': np.array([
            # q0:
            np.deg2rad(0), np.deg2rad(90),
            np.deg2rad(0),  np.deg2rad(-120),
            np.deg2rad(0),  np.deg2rad(120),
            # q_dot0: 
            0, 0, 0, 0, 0, 0
        ])
    }