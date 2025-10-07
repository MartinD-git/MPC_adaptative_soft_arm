import numpy as np

L = 0.5
d = 2 / L
num_segments = 2

ARM_PARAMETERS = {
    "L_segs": [L, L, L],
    "m": 0.5,
    "d_eq": [d, d, d],
    "K": np.diag([0, 0.2, 0, 0.6, 0, 0.2]),
    "num_segments": num_segments,
}

MPC_PARAMETERS = {
    "N": 20,
    "Q":  np.diag([15]*2*num_segments + [1]*2*num_segments),
    "Qf": np.diag([30]*2*num_segments + [2]*2*num_segments),  # stronger terminal weight helps convergence
    "R": np.eye(2*num_segments)*0.5,
    "u_bound": 5,
}
if num_segments==3:
    ARM_PARAMETERS = {
        "L_segs": [L, L, L],
        "m": 0.5,
        "d_eq": [d, d, d],
        "K": np.diag([0, 0.2, 0, 0.2, 0, 0.2]),
        "num_segments": num_segments,
    }

    SIM_PARAMETERS = {
        "dt": 0.1,
        "T": 4,
        "x0": np.array([
            0, np.deg2rad(90), 0, np.deg2rad(-120), 0, np.deg2rad(120),
            0, 0, 0, 0, 0, 0
        ]),
    }
elif num_segments==2:
    SIM_PARAMETERS = {
        "dt": 0.1,
        "T": 4,
        "x0": np.array([
            0, np.deg2rad(90), 0, np.deg2rad(-120),
            0, 0, 0, 0
        ]),
    }

    ARM_PARAMETERS = {
        "L_segs": [L, L],
        "m": 0.5,
        "d_eq": [d, d],
        "K": np.diag([0, 0.2, 0, 0.2]),
        "num_segments": num_segments,
    }