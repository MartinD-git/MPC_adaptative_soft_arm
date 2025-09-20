import numpy as np

L = 0.5
d = 2 / L

ARM_PARAMETERS = {
    "L_segs": [L, L, L],
    "m": 0.5,
    "d_eq": [d, d, d],
    "K": np.diag([0, 0.2, 0, 0.6, 0, 0.2]),
}

MPC_PARAMETERS = {
    "N": 20,
    "Q":  np.diag([15]*6 + [1]*6),
    "Qf": np.diag([30]*6 + [2]*6),  # stronger terminal weight helps convergence
    "R": np.diag([1, 1, 1, 1, 1, 1]),
    "u_bound": 5,
}

SIM_PARAMETERS = {
    "dt": 0.1,
    "T": 4,
    "x0": np.array([
        0, np.deg2rad(90), 0, np.deg2rad(-120), 0, np.deg2rad(120),
        0, 0, 0, 0, 0, 0
    ]),
}