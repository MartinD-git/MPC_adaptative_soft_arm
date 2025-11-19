import numpy as np

# we consider a segment as a cylinder:
#material: TPU
E= 1e6 #Young modulus23e6
rho=1220 #density

#arm
r_o     = 0.04 # outer radius [m]
t_wall  = 0.005 # wall thickness [m]
r_i     = r_o - t_wall
r_d = r_o  # radius at which tendons are located
L = 0.2 #length of each segment
A= np.pi*(r_o**2 - r_i**2)  # area
I = np.pi*(r_o**4 - r_i**4)/4 #second moment of area
m = rho * A * L #mass of each segment
k_phi = 0
k_theta = 0.015634 #gotten from static simulation or (E*I)/L
xi=0.2
d = 2*xi*1.875**2 * np.sqrt((rho*A*E*I)/(L**2))
tension_bound = 2/0.05 #2 is motor limit, shaft is at 0.01m
u_bound = tension_bound*r_d

rho_water = 1000 #density of water
rho_air = 1.225 #density of air

rho_fluid_initial = rho_air  # initial density of the surrounding fluid
true_rho_fluid = rho_water  # density of the surrounding fluid


horizon_time = 2  #seconds
dt = 0.1  #seconds

num_segments = 2

MPC_PARAMETERS = {
    "N": int(np.ceil(horizon_time/dt)),
    "Q":  np.diag([5]*2*num_segments + [1]*2*num_segments),
    "Qf": np.diag([5]*2*num_segments + [1]*2*num_segments),  # stronger terminal weight helps convergence
    "R": np.eye(2*num_segments),
    "u_bound": u_bound,
    "N_rho": 60, #number of previous steps to consider for rho fluid estimation
}

SIM_PARAMETERS = {
    "dt": dt,
    "T": 20,
    "x0": np.array([
        np.deg2rad(-45), np.deg2rad(45), np.deg2rad(45), np.deg2rad(45),
        0, 0, 0, 0
    ]),
    "T_loop": 10,  # seconds
}

ARM_PARAMETERS = {
    "m": m,
    "L_segs": [L, L],
    "r_o": r_o,
    "r_i": r_i,
    "sigma_k": [0, 2*np.pi/3,2*2*np.pi/3],
    "rho_arm": rho,
    "d_eq": [d, d],
    "K": np.diag([k_phi, k_theta, k_phi, k_theta]),
    "num_segments": num_segments,
    "rho_fluid_initial": rho_fluid_initial,
    "true_rho_fluid": true_rho_fluid,
    "r_d": r_d,
    "maximum_tension": tension_bound,
}

'''print("MPC_PARAMETERS:", MPC_PARAMETERS)
print("ARM_PARAMETERS:", ARM_PARAMETERS)
print("SIM_PARAMETERS:", SIM_PARAMETERS)'''
