import numpy as np

# we consider a segment as a cylinder:
#material: TPU
E= 23e6 #Young modulus23e6
rho=1220 #density

#arm
r_o     = 0.04 # outer radius [m]
t_wall  = 0.004 # wall thickness [m]
r_i     = r_o - t_wall
r_d = 0.036  # radius at which tendons are located
L = 0.315 #length of each segment

A= np.pi*(r_o**2 - r_i**2)  # area
I = np.pi*(r_o**4 - r_i**4)/4 #second moment of area
m = rho * A * L #mass of each segment
k_phi = 0
k_theta = 0.015634 #gotten from static simulation or (E*I)/L
xi=0.15
d = 2*xi*1.875**2 * np.sqrt((rho*A*E*I)/(L**2))

rho_water = 1000 #density of water
rho_air = 1.225 #density of air

rho_liquid = rho_water  # density of the surrounding fluid


horizon_time = 2  #seconds
dt = 0.1  #seconds

num_segments = 2

MPC_PARAMETERS = {
    "N": int(np.ceil(horizon_time/dt)),
    "Q":  np.diag([1e3]*3 + [1]*2*num_segments),
    "Qf": np.diag([1e3]*3 + [1]*2*num_segments),  # stronger terminal weight helps convergence
    "R": 1e-4*np.eye(3*num_segments),
    "u_bound": [2,30],#[0,tension_bound],
}

SIM_PARAMETERS = {
    "dt": dt,
    "T": 10,
    "x0": np.array([ # phi, theta
        np.deg2rad(0), np.deg2rad(1e-6), np.deg2rad(0), np.deg2rad(0), # phi is angle at base, theta is curvature
        0, 0, 0, 0
    ]),
    "T_loop": 10,  # seconds
    "radius_trajectory": 0.5*L,
    "center_trajectory": np.array([1, 0, 1])*L,
    "rotation_angles_trajectory": np.array([np.deg2rad(0), np.deg2rad(60), np.deg2rad(0)]),
}

"""
motor id list:
id 1: segment 1, at pi/3
id 5: segment 1, at pi
id 3: segment 1, at 5*pi/3

id 2: segment 2, at 0
id 0: segment 2, at 2pi/3
id 4: segment 2, at 4pi/3

"""

ARM_PARAMETERS = {
    "L_segs": [L, L],
    "r_o": r_o,
    "r_i": r_i,
    "sigma_k": [np.pi/3, np.pi, 5*np.pi/3,0, 2*np.pi/3, 4*np.pi/3],  # tendon routing angles
    "rho_arm": rho,
    "d_eq": [d, d],
    "K": np.diag([k_phi, k_theta, k_phi, k_theta]),
    "num_segments": num_segments,
    "rho_liquid": rho_liquid,
    "r_d": r_d,
}
print("MPC_PARAMETERS:", MPC_PARAMETERS)
print("ARM_PARAMETERS:", ARM_PARAMETERS)
print("SIM_PARAMETERS:", SIM_PARAMETERS)
