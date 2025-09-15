import casadi as ca

from .utils import pcc_forward_kinematics, pcc_dynamics, shape_function, dynamics2integrator

class PCCSoftArm:
    def __init__(self, L_segs, m, d_eq, K):
        print("Initializing PCC Soft Arm Model...")
        #define the robot arm
        self.L_segs = L_segs
        self.m = m
        self.d_eq = d_eq
        self.K = K
        self.current_state = None
        self.dt = None
        self.history = []
        self.history_d = []
        self.history_u = []


        s = ca.SX.sym('s')
        q = ca.SX.sym('q', 6)
        q_dot = ca.SX.sym('q_dot', 6)

        # compute the kinematics
        tips, jacobians = pcc_forward_kinematics(s, q, self.L_segs)
        print("Kinematics done")

        # shape function for visualization
        self.shape_func = shape_function(q, tips, s)

        # compute the dynamics
        self.dynamics_func = pcc_dynamics(q, q_dot, tips, jacobians, s)
        print("Dynamics done")

    def create_integrator(self, dt):
        # create inegrator
        self.integrator = dynamics2integrator(self,dt)
        self.dt=dt
        print("Integrator done")

    def next_step(self, u):
        p=ca.vertcat(u, self.m, self.d_eq, ca.reshape(self.K, 36, 1))
        # simulate one step
        self.current_state = self.integrator(x0=self.current_state, p=p)['xf'].full().flatten()

        self.history_u
        self.history.append(self.current_state)

        return self.current_state
    
    def log_history(self,u,q_d):
        self.history.append(self.current_state)
        self.history_u.append(u)
        self.history_d.append(q_d)



