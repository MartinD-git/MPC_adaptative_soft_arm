import casadi as ca
import numpy as np

from utils import pcc_forward_kinematics, pcc_dynamics, shape_function, dynamics2integrator

class PCCSoftArm:
    def __init__(self, arm_param_dict, dt):
        print("Initializing PCC Soft Arm Model...")
        #define the robot arm
        self.L_segs = arm_param_dict['L_segs']
        self.d_eq = arm_param_dict['d_eq']
        self.K = arm_param_dict['K']
        self.r_o = arm_param_dict['r_o']
        self.r_i = arm_param_dict['r_i']
        self.rho = arm_param_dict['rho_arm']
        self.rho_liquid = arm_param_dict['rho_liquid']
        self.r_d = arm_param_dict['r_d']
        self.sigma_k = arm_param_dict['sigma_k']
        self.rho_air = 1.225 # kg/m^3
        self.C_d = 1.17  # drag coefficient, approx for cylinder
        self.max_tension = arm_param_dict['maximum_tension']  # max tension for each tendon
        self.dt = dt
        self.current_state = None
        self.true_current_state = None  # for simulation with noise
        self.history = []
        self.history_d = []
        self.history_u = []
        self.history_u_tendon = []
        self.num_segments = arm_param_dict['num_segments']

        self.s = ca.SX.sym('s')
        q = ca.SX.sym('q', 2*self.num_segments)
        q_dot = ca.SX.sym('q_dot', 2*self.num_segments)

        # compute the kinematics
        tips, jacobians = pcc_forward_kinematics(self.s, q, self.L_segs,self.num_segments)
        print("Kinematics done")

        # shape function for visualization
        self.shape_func = shape_function(q, tips,self.s)

        # compute the dynamics
        self.dynamics_func = pcc_dynamics(self,q, q_dot, tips, jacobians,sim=False)
        dynamics_func_sim = pcc_dynamics(self,q, q_dot, tips, jacobians,sim=True)
        print("Dynamics done")
        # create integrators
        self.integrator = dynamics2integrator(self,self.dynamics_func)
        self.integrator_sim = dynamics2integrator(self,dynamics_func_sim)

    def next_step(self, u):
        # simulate one step
        error =  self.meas_error()
        self.true_current_state = self.integrator_sim(x0=self.true_current_state, u=u, q0=self.true_current_state)['xf'].full().flatten()

        self.current_state = self.true_current_state + error

    
    def log_history(self,u,q_d,u_tendon):
        self.history.append(self.true_current_state)
        self.history_u.append(u)
        self.history_d.append(q_d)
        self.history_u_tendon.append(u_tendon)

    def meas_error(self):
        std_angle = np.deg2rad(1)
        std_velocity = np.deg2rad(1)/self.dt
        return np.random.normal(0, [std_angle]*2*self.num_segments+[std_velocity]*2*self.num_segments, size=4*self.num_segments)




