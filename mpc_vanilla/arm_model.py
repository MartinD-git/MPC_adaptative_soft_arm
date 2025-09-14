import casadi as ca
import numpy as np

from utils import pcc_forward_kinematics, pcc_dynamics, shape_function

class PCCSoftArm:
    def __init__(self, L_segs, m, g_vec, d_eq, K):
        print("Initializing PCC Soft Arm Model...")
        #define the robot arm
        self.L_segs = L_segs
        self.m = m
        self.g_vec = g_vec
        self.d_eq = d_eq
        self.K = K

        s = ca.SX.sym('s')
        q = ca.SX.sym('q', 6)
        q_dot = ca.SX.sym('q_dot', 6)

        # compute the kinematics
        tips, jacobians = pcc_forward_kinematics(s, q, self.L_segs)
        print("Kinematics done")

        # shape function for visualization
        self.shape_func = shape_function(q, self.L_segs, tips, s)

        # compute the dynamics
        self.dynamics_func = pcc_dynamics(q, q_dot, self.L_segs, jacobians)
        print("Dynamics done")



