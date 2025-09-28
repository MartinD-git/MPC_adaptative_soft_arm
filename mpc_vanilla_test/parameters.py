import numpy as np

# we consider a segment as a cylinder:
#material: TPU
E=23e6 #Young modulus
rho=1220 #density

#arm
r_o     = 0.025   # outer radius [m]
t_wall  = 0.005   # wall thickness [m]
r_i     = r_o - t_wall
L = 0.5 #length of each segment
A= np.pi*(r_o**2 - r_i**2)  #cross section area
I = np.pi*(r_o**4 - r_i**4)/4 #second moment of area
m = rho * A * L #mass of each segment
k_phi = 0
k_theta = (E*I)/L 
xi=0.05
d = 2*xi*1.875**2 * np.sqrt((rho*A*E*I)/(L**2))
u_bound = 100

horizon_time = 2  #seconds
dt = 0.05  #seconds

num_segments = 2

MPC_PARAMETERS = {
    "N": int(np.ceil(horizon_time/dt)),
    "Q":  np.diag([15]*2*num_segments + [1]*2*num_segments),
    "Qf": np.diag([30]*2*num_segments + [2]*2*num_segments),  # stronger terminal weight helps convergence
    "R": np.eye(2*num_segments),
    "u_bound": u_bound,
}
if num_segments==3:
    ARM_PARAMETERS = {
        "L_segs": [L, L, L],
        "m": m,
        "d_eq": [d, d, d],
        "K": np.diag([k_phi, k_theta, k_phi, k_theta, k_phi, k_theta]),
        "num_segments": num_segments,
    }

    SIM_PARAMETERS = {
        "dt": dt,
        "T": 4,
        "x0": np.array([
            0, np.deg2rad(90), 0, np.deg2rad(-120), 0, np.deg2rad(120),
            0, 0, 0, 0, 0, 0
        ]),
    }
elif num_segments==2:
    SIM_PARAMETERS = {
        "dt": dt,
        "T": 4,
        "x0": np.array([
            0, np.deg2rad(45), 0, np.deg2rad(60),
            0, 0, 0, 0
        ]),
    }

    ARM_PARAMETERS = {
        "L_segs": [L, L],
        "m": m,
        "d_eq": [d, d],
        "K": np.diag([k_phi, k_theta, k_phi, k_theta]),
        "num_segments": num_segments,
    }


'''L = 0.5
d = 2 / L
num_segments = 2

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
        "dt": 0.05,
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
    }'''



