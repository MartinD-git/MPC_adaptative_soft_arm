import casadi as ca
import numpy as np

from utils import pcc_forward_kinematics, pcc_dynamics, shape_function, dynamics2integrator

class PCCSoftArm:
    def __init__(self, arm_param_dict, dt,history_size):
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
        self.dt = dt
        self.num_segments = arm_param_dict['num_segments']
        self.current_state = None
        self.true_current_state = None  # for simulation with noise
        self.history = np.zeros((4*self.num_segments, history_size))
        self.history_d = np.zeros((4*self.num_segments, history_size))
        self.history_u = np.zeros((2*self.num_segments, history_size))
        self.history_u_tendon = np.zeros((3*self.num_segments, history_size))
        self.history_index = 0
        self.history_pred = np.zeros((4*self.num_segments, history_size))
        self.num_adaptive_params =  2*self.num_segments + 1 # damping per segment + bending stiffness per segment
        self.history_adaptive_param = np.zeros((self.num_adaptive_params, history_size+1))
        self.m = arm_param_dict['m']
        

        self.s = ca.SX.sym('s')
        q = ca.SX.sym('q', 2*self.num_segments)
        q_dot = ca.SX.sym('q_dot', 2*self.num_segments)

        # compute the kinematics
        tips, jacobians = pcc_forward_kinematics(self.s, q, self.L_segs,self.num_segments)
        self.end_effector = ca.Function('end_effector', [q], [ca.substitute(tips[self.num_segments-1], self.s, 1)])
        print("Kinematics done")

        # shape function for visualization
        self.shape_func = shape_function(q, tips,self.s)

        # compute the dynamics
        self.dynamics_func = pcc_dynamics(self,q, q_dot, tips, jacobians,water=False)
        dynamics_func_sim = pcc_dynamics(self,q, q_dot, tips, jacobians,water=False)
        print("Dynamics done")
        # create integrators
        self.integrator = dynamics2integrator(self,self.dynamics_func)
        self.integrator_sim = dynamics2integrator(self,dynamics_func_sim)

    def next_step(self, u):
        # simulate one step
        error =  self.meas_error()
        self.true_current_state = self.integrator_sim(x0=self.true_current_state, u=u, p_global=np.hstack([self.true_current_state, np.zeros(self.num_adaptive_params)]))['xf'].full().flatten()

        self.current_state = self.true_current_state + error

    
    def log_history(self,u,q_d,u_tendon,x1):
        self.history[:, self.history_index] = self.true_current_state
        #self.history_d[:, self.history_index] = q_d
        self.history_u[:, self.history_index] = u
        self.history_u_tendon[:, self.history_index] = u_tendon
        self.history_pred[:, self.history_index] = x1
        self.history_index += 1

    def meas_error(self):
        std_angle = 0*np.deg2rad(5)
        std_velocity = 0*np.deg2rad(5)/self.dt
        return np.random.normal(0, [std_angle]*2*self.num_segments+[std_velocity]*2*self.num_segments, size=4*self.num_segments)




