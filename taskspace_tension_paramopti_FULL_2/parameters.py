import numpy as np

# we consider a segment as a cylinder:
#material: TPU
E= 23e6 #Young modulus23e6
rho=1220 #density

#arm
r_o     = 0.04/2 # outer radius [m]
t_wall  = 0.002 # wall thickness [m]
r_i     = r_o - t_wall
r_d = 0.036/2  # radius at which tendons are located
L = 0.315 #length of each segment

A= np.pi*(r_o**2 - r_i**2)  # area
I = np.pi*(r_o**4 - r_i**4)/4 #second moment of area
m = rho * np.pi*(r_o**2) * L*0.5 #mass of each segment estimated 0.5 infill

print("Mass of each segment:", m)
k_theta = 0.356*1.75 #0.356*1.75 gotten from static simulation
k_phi = 0
beta = 0.03

rho_water = 1000 #density of water
rho_air = 1.225 #density of air

rho_liquid = rho_water  # density of the surrounding fluid


horizon_time = 3  #seconds
dt = 0.1  #seconds

num_segments = 2

MPC_PARAMETERS = {
    "N": int(np.ceil(horizon_time/dt)),
    "Q":  np.diag([5e2]*3 + [1e-2]*2*num_segments),
    "Qf": np.diag([5e2]*3 + [1e-2]*2*num_segments),  # stronger terminal weight helps convergence
    "R": 1e-5*np.eye(3*num_segments),
    "u_bound": [2,100],#[0,tension_bound],
    "N_p_adaptative": 20, #number of previous steps to consider for parameter estimation
}
if num_segments ==3:
    SIM_PARAMETERS = {
        "dt": dt,
        "T": 50,#180
        "x0": np.array([ # phi, theta
            np.deg2rad(1e1), np.deg2rad(1e1), np.deg2rad(1e1), np.deg2rad(1e1), np.deg2rad(1e1), np.deg2rad(1e1), # phi is angle at base, theta is curvature
            0, 0, 0, 0, 0, 0
        ]),
        "T_loop": 15,  # 5 seconds
        "radius_trajectory": 1*L, #0.5*L,
        "center_trajectory": np.array([0, 0, 1.4+0.8])*L,#np.array([1+0.2, 0, 1.4+0.8])*L,
        "rotation_angles_trajectory": np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]),#np.array([np.deg2rad(0), np.deg2rad(60), np.deg2rad(0)]),
    }
elif num_segments ==2:
    SIM_PARAMETERS = {
        "dt": dt,
        "T": 50,#180
        "x0": np.array([ # phi, theta
            np.deg2rad(1e1), np.deg2rad(1e1), # segment 1
            np.deg2rad(1e1), np.deg2rad(1e1), # segment 2
            0, 0, 0, 0
        ]),
        "T_loop": 10,  # seconds
        "shape": 'circle', # "rectangle", "circle", "lemniscate"
        "radius_trajectory": 0.4*L, #0.4*L
        "center_trajectory": np.array([1, 0, 1.4])*L,#np.array([1, 0, 1.4])*L,
        "rotation_angles_trajectory": np.array([np.deg2rad(0), np.deg2rad(60), np.deg2rad(0)]),#np.array([np.deg2rad(0), np.deg2rad(60), np.deg2rad(0)]),
    }

# # round loop:
#         "radius_trajectory": 1*L, 
#         "center_trajectory": np.array([0, 0, 1.4])*L,
#         "rotation_angles_trajectory": np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]),

if num_segments ==3:
    sigma_k = [np.pi/3, np.pi, 5*np.pi/3,0, 2*np.pi/3, 4*np.pi/3, np.pi/3, np.pi, 5*np.pi/3]  # tendon routing angles
elif num_segments ==2:
    sigma_k = [np.pi/3, np.pi, 5*np.pi/3,0, 2*np.pi/3, 4*np.pi/3]  # tendon routing angles


ARM_PARAMETERS = {
    "L_segs": [L]*num_segments,
    "r_o": r_o,
    "r_i": r_i,
    "sigma_k": sigma_k,  # tendon routing angles
    "rho_arm": rho,
    "beta": [beta]*num_segments,
    "K": np.diag([k_phi, k_theta]*num_segments),
    "num_segments": num_segments,
    "rho_liquid": rho_liquid,
    "r_d": r_d,
    "m": m,
}

