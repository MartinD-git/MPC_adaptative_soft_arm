import numpy as np

# we consider a segment as a cylinder:
#material: TPU
E= 5e6 #Young modulus23e6
rho=1220 #density

#arm
r_o     = 0.025 # outer radius [m]
t_wall  = 0.005 # wall thickness [m]
r_i     = r_o - t_wall
r_i = 0
r_d = r_o  # radius at which tendons are located
L = 0.2 #length of each segment
A= np.pi*(r_o**2 - r_i**2)  # area
I = np.pi*(r_o**4 - r_i**4)/4 #second moment of area
m = rho * A * L #mass of each segment
k_phi = 0
k_theta = (E*I)/L
xi=0.2
d = 2*xi*1.875**2 * np.sqrt((rho*A*E*I)/(L**2))
tension_bound = 2/0.01 *100#2 is motor limit, shaft is at 0.01m
u_bound = tension_bound*r_d

rho_water = 1000 #density of water
rho_air = 1.225 #density of air

rho_liquid = rho_water  # density of the surrounding fluid


horizon_time = 2  #seconds
dt = 0.1  #seconds

num_segments = 2

MPC_PARAMETERS = {
    "N": int(np.ceil(horizon_time/dt)),
    "Q":  np.diag([15]*2*num_segments + [1]*2*num_segments),
    "Qf": np.diag([15]*2*num_segments + [2]*2*num_segments),  # stronger terminal weight helps convergence
    "R": np.eye(2*num_segments),
    "u_bound": u_bound,
}

SIM_PARAMETERS = {
    "dt": dt,
    "T": 10,
    "x0": np.array([
        np.deg2rad(45), np.deg2rad(45), np.deg2rad(45), np.deg2rad(45),
        0, 0, 0, 0
    ]),
}

ARM_PARAMETERS = {
    "L_segs": [L, L],
    "r_o": r_o,
    "r_i": r_i,
    "sigma_k": [0, 2*np.pi/3,2*2*np.pi/3],
    "rho_arm": rho,
    "d_eq": [d, d],
    "K": np.diag([k_phi, k_theta, k_phi, k_theta]),
    "num_segments": num_segments,
    "rho_liquid": rho_liquid,
    "r_d": r_d,
    "maximum_tension": tension_bound,
}
'''print("MPC_PARAMETERS:", MPC_PARAMETERS)
print("ARM_PARAMETERS:", ARM_PARAMETERS)
print("SIM_PARAMETERS:", SIM_PARAMETERS)'''
