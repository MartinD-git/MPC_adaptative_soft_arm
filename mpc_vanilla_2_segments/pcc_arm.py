import casadi as ca

from utils import pcc_forward_kinematics, pcc_dynamics, shape_function, dynamics2integrator

class PCCSoftArm:
    def __init__(self, arm_param_dict, dt):
        print("Initializing PCC Soft Arm Model...")
        #define the robot arm
        self.L_segs = arm_param_dict['L_segs']
        self.m = arm_param_dict['m']
        self.d_eq = arm_param_dict['d_eq']
        self.K = arm_param_dict['K']
        self.m_buoy = arm_param_dict['m_buoy']
        self.r_o = arm_param_dict['r_o']
        self.rho_water = 1000  # density of water
        self.C_d = 1.17  # drag coefficient, approx for cylinder
        self.dt = dt
        self.current_state = None
        self.history = []
        self.history_d = []
        self.history_u = []
        if arm_param_dict['num_segments'] not in [2,3]:
            raise ValueError("num_segments must be 2 or 3")
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
        dynamics_func = pcc_dynamics(self,q, q_dot, tips, jacobians,sim=True)
        dynamics_func_sim = pcc_dynamics(self,q, q_dot, tips, jacobians,sim=True)
        print("Dynamics done")
        # create integrators
        self.integrator = dynamics2integrator(self,dynamics_func)
        self.integrator_sim = dynamics2integrator(self,dynamics_func_sim)

    def next_step(self, u):
        # simulate one step
        self.current_state = self.integrator_sim(x0=self.current_state, u=u, q0=self.current_state)['xf'].full().flatten()

        return self.current_state
    
    def log_history(self,u,q_d):
        self.history.append(self.current_state)
        self.history_u.append(u)
        self.history_d.append(q_d)




