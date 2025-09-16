import numpy as np

def get_robot_params():
    L = 0.5
    d =2/L
    return {
        'L_segs': [L, L, L],  # Length of each segment m
        'm': 0.5,# Mass per unit length kg/m
        'g': [0, 0, -9.81],# Gravity vector m/s^2
        'd_eq': [d, d, d],
        'K': np.diag([0, 0.2, 0, 0.6, 0, 0.6])
    }

def get_simulation_params():
    return {
        'dt': 0.1, # s 0.05
        'T': 5, # Total simu time s
        'x0': np.array([
            # q0:
            np.deg2rad(0), np.deg2rad(90),
            np.deg2rad(0),  np.deg2rad(-90),
            np.deg2rad(0),  np.deg2rad(90),
            # q_dot0: 
            0, 0, 0, 0, 0, 0
        ])
    }