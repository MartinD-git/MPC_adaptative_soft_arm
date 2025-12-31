'''
This script defines basic functions used across the project.
'''

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def sinc_cosc(u, eps=1e-4):
    '''
    Compute sinc(u) and cosc(u) with series expansion around 0 for numerical stability.
    '''
    # Precompute powers
    u2 = u*u
    u4 = u2*u2
    u6 = u4*u2
    small = ca.fabs(u) <= eps

    # sinc(u)  = 1 - u^2/6 + u^4/120
    sinc_ser = 1 - (u2*(1/6.0)) + (u4*(1/120.0))
    # cosc(u)  = u/2 - u^3/24 + u^5/720
    cosc_ser = (u*(1/2.0)) - (u*u2*(1/24.0)) + (u4*u*(1/720.0))

    # Exact away from 0
    sinc_ex  = ca.sin(u) / u
    cosc_ex  = (1 - ca.cos(u)) / u

    sinc_val = ca.if_else(small, sinc_ser, sinc_ex)
    cosc_val = ca.if_else(small, cosc_ser, cosc_ex)
    return sinc_val, cosc_val

def pcc_segment_transform(s_var, phi, theta, L, eps=1e-4):
    '''
    Compute the homogeneous transformation matrix of a PCC segment at normalized length s_var (0 to 1),
    '''
    u  = s_var*theta
    cp = ca.cos(phi); sp = ca.sin(phi)
    cu = ca.cos(u);   su = ca.sin(u)

    sinc_u, cosc_u = sinc_cosc(u, eps)

    # Translation: no /theta anywhere
    tx = L * cp * s_var * cosc_u
    ty = L * sp * s_var * cosc_u
    tz = L *  s_var * sinc_u
    t  = ca.vertcat(tx, ty, tz)

    # Rotation (already division-free)
    R = ca.vertcat(
        ca.horzcat(cp*cp*(cu-1)+1,  sp*cp*(cu-1),        cp*su),
        ca.horzcat(sp*cp*(cu-1),    cp*cp*(1-cu)+cu,     sp*su),
        ca.horzcat(-cp*su,          -sp*su,              cu)
    )

    T = ca.vertcat(ca.horzcat(R, t), ca.horzcat(0,0,0,1))
    return T

def pcc_forward_kinematics(s, q, L_segs,num_segments):
    '''
    Compute the forward kinematics of the 3-segment PCC robot arm.
    '''
    if num_segments == 2:
        L1, L2 = L_segs[0], L_segs[1]
        phi1, th1, phi2, th2 = q[0], q[1], q[2], q[3]
    elif num_segments == 3:
        L1, L2, L3 = L_segs[0], L_segs[1], L_segs[2]
        phi1, th1, phi2, th2, phi3, th3 = q[0], q[1], q[2], q[3], q[4], q[5]
    else:
        raise ValueError("num_segments must be 2 or 3")

    # Chain the transformations
    T_tip1 = pcc_segment_transform(1, phi1, th1, L1)
    T_tip2 = pcc_segment_transform(1, phi2, th2, L2)

    T_global1 = pcc_segment_transform(s, phi1, th1, L1)
    T_global2 = T_tip1 @ pcc_segment_transform(s, phi2, th2, L2)

    p1 = T_global1[:3, 3]
    p2 = T_global2[:3, 3]

    J1 = ca.jacobian(p1, q)
    J2 = ca.jacobian(p2, q)
    
    if num_segments==3:
        T_global3 = T_tip1 @ T_tip2 @ pcc_segment_transform(s, phi3, th3, L3)
        p3 = T_global3[:3, 3]
        J3 = ca.jacobian(p3, q)

    if num_segments==2:
        return [p1, p2], [J1,J2]
    elif num_segments==3:
        return [p1, p2, p3], [J1,J2,J3]
    else:
        raise ValueError("num_segments must be 2 or 3")

    

def gauss_legendre(result,integrand, s):
    '''
    Perform Gauss-Legendre quadrature on the provided integrands to go faster than symbolic integration.
    '''
    num_points = 6 # Arbitrary number
    gl_points, gl_weights = np.polynomial.legendre.leggauss(num_points)
    
    # Our integral is over [0, 1], so we need to scale the points and weights
    # Change of variables: s = (t+1)/2, where t is in [-1, 1]. Then ds = dt/2.
    s_eval_points = 0.5 * (gl_points + 1)
    s_weights = 0.5 * gl_weights
    

    for k in range(num_points):
        s_k = s_eval_points[k]
        w_k = s_weights[k]

        result += ca.substitute(integrand, s, s_k) * w_k

    return result

def shape_function(q, tips,s):
    '''
    Compute a list of points along the robot shape for visualization.
    '''

    s_points = np.linspace(0, 1, 20) # 20 points per segment
    points1 = []
    points2 = []
    points3 = []

    for s_val in s_points: points1.append(ca.substitute(tips[0], s, s_val))
    for s_val in s_points: points2.append(ca.substitute(tips[1], s, s_val))
    if len(tips)>2:
        for s_val in s_points: points3.append(ca.substitute(tips[2], s, s_val))
        

    P1 = ca.horzcat(*points1)
    P2 = ca.horzcat(*points2)
    if len(tips)>2:
        P3 = ca.horzcat(*points3)

    if len(tips)<=2:
        return ca.Function('arm_shape_func', [q], [P1, P2])
    else:
        return ca.Function('arm_shape_func', [q], [P1, P2, P3])


def pcc_dynamics(arm,q, q_dot, tips, jacobians,water=False):
    '''
    Compute the dynamics of the PCC robot arm.
    '''
    m = arm.m

    num_segments = arm.num_segments
    s = arm.s
    M = ca.SX.zeros(2*num_segments, 2*num_segments)
    D = ca.SX.zeros(2*num_segments, 2*num_segments)
    G_pot = ca.SX(0)
    g_vec = ca.vertcat(0.0, 0.0, -9.81)

    p_adaptative = ca.SX.sym('p_adaptative', arm.num_adaptive_params, 1)
    m = arm.m + p_adaptative[0]  # mass adaptative
    if arm.num_segments ==2:
        K = arm.K + ca.diag(ca.vertcat(0, p_adaptative[3], 0, p_adaptative[4]))  # stiffness adaptative
    elif arm.num_segments ==3:
        K = arm.K + ca.diag(ca.vertcat(0, p_adaptative[4], 0, p_adaptative[5], 0, p_adaptative[6]))  # stiffness adaptative

    J = ca.vertcat(*jacobians)

    D_fluid = 0
    m_buoy = 0
    m_displaced = 0
    if water:
        rho_fluid = arm.rho_liquid   
        m_buoy = rho_fluid * np.pi*(arm.r_o**2) * arm.L_segs[0]*0.5 #buoyancy mass of each segment
        m_displaced = rho_fluid * np.pi*arm.r_o**2 * arm.L_segs[0]*0.5 #displaced mass of each segment
        
        for i, Ji in enumerate(jacobians):
            v_i = Ji @ q_dot
            vmag = ca.norm_2(v_i)+ 1e-6 # to be smooth
            Aproj = (2*arm.r_o)*arm.L_segs[i] # projected area per segment
            D_fluid += 0.5 * rho_fluid * arm.C_d * Aproj * vmag * (Ji.T @ Ji)
        D_fluid_gauss = gauss_legendre(ca.SX.zeros(2*num_segments, 2*num_segments), D_fluid, s)
    else:
        D_fluid_gauss = ca.SX.zeros(2*num_segments, 2*num_segments)
            
    # for i in range(num_segments):
    #     D_integrand += (jacobians[i].T @ jacobians[i]) * d_eq[i]

    G_integrand = (m_buoy-m) * sum(ca.dot(g_vec, tip) for tip in tips)
    M_integrand = (m+m_displaced) * (J.T @ J)
    I_phi = 1e-3  # regularization to avoid singularities
    M_reg = ca.DM.zeros(2*num_segments, 2*num_segments)
    for i in range(num_segments):
        M_reg[2*i, 2*i] = I_phi

    M = gauss_legendre(ca.SX.zeros(2*num_segments, 2*num_segments), M_integrand, s) + M_reg

    G_pot = gauss_legendre(ca.SX(0), G_integrand, s)
    G = ca.gradient(G_pot, q)

    beta = np.ones(arm.num_segments)*0.03

    beta_adaptive = p_adaptative[1:1+arm.num_segments]
    D_blocks = []
    for i in range(arm.num_segments):
        beta = arm.beta[i] + beta_adaptive[i]
        
        K_block = K[2*i : 2*i+2, 2*i : 2*i+2]
        
        # Add to list
        D_blocks.append(beta * K_block)
    D_stiffness = ca.diagcat(*D_blocks)
    D = D_fluid_gauss + D_stiffness #+1e-5* ca.DM.eye(2*num_segments) # could add 0.5*M to have rayleigh damping but this works well for now

    M_func = ca.Function('M_func', [q, p_adaptative[0]], [M])
    G_func = ca.Function('G_func', [q, p_adaptative[0]], [G])
    D_func = ca.Function('D_func', [q,q_dot, p_adaptative[1:1+arm.num_segments+arm.num_segments]], [D])
    arm.M_func = M_func
    arm.D_func = D_func

    # Coriolis C
    M_q = M_func(q, p_adaptative[0])
    M_dot_q_dot = ca.jtimes(M_q, q, q_dot) # This calculates (dM/dq)*q_dot

    KE = 0.5 * q_dot.T @ M_q @ q_dot
    KE_grad = ca.gradient(KE, q)

    c_vec = M_dot_q_dot @ q_dot - KE_grad
    C_vec_func = ca.Function('C_vec_func', [q, q_dot, p_adaptative[0]], [c_vec])

    x = ca.SX.sym('x', 4*num_segments)
    q0 = ca.SX.sym('q0', 4*num_segments)

    q_from_x = x[:2*num_segments]
    q_dot_from_x = x[2*num_segments:]

    # Calculate q_ddot

    M_term= M_func(q0[:2*num_segments],p_adaptative[0])+1e-3* ca.DM.eye(2*num_segments)
    C_term= C_vec_func(q0[:2*num_segments], q0[2*num_segments:], p_adaptative[0])
    G_term= G_func(q0[:2*num_segments], p_adaptative[0])
    D_term= D_func(q0[:2*num_segments], q0[2*num_segments:], p_adaptative[1:arm.num_segments+arm.num_segments+1]) @ q_dot_from_x
    K_term= K @ q_from_x
  
    J_tendon = ca.SX.zeros((3*num_segments, 2*num_segments))
    u_tendon = ca.SX.sym('u', 3*num_segments)

    for i in range(num_segments):
        for k in range(3): #number of tendons
            phi =q0[2*i]
            theta = q0[1+2*i]
            J_tendon[k+3*i,2*i] = -theta*arm.r_d*ca.sin(arm.sigma_k[k+3*i]-phi)
            J_tendon[k+3*i,2*i+1] = arm.r_d*ca.cos(arm.sigma_k[k+3*i]-phi)

    #added for speed
    rhs = J_tendon.T @ u_tendon - C_term - G_term - D_term - K_term

    # Robust SPD solve via Cholesky
    q_ddot = ca.solve(M_term , rhs)
    x_dot = ca.vertcat(q_dot_from_x, q_ddot)
    p_global = ca.vertcat(q0, p_adaptative)

    return ca.Function('pcc_f', [x, u_tendon, p_global], [x_dot])

def dynamics2integrator(pcc_arm,f,n_substeps=1):
    x0 = ca.MX.sym('x0', 4*pcc_arm.num_segments)
    u  = ca.MX.sym('u', 3*pcc_arm.num_segments)
    q0 = ca.MX.sym('q0', 4*pcc_arm.num_segments) 
    p_adaptative = ca.MX.sym('p_adaptative', pcc_arm.num_adaptive_params)
    p_global = ca.vertcat(q0, p_adaptative)

    dt=pcc_arm.dt

    h  = dt / n_substeps # internal RK4 step for better stability
    x = x0
    for _ in range(n_substeps):
        k1 = f(x,             u, ca.vertcat(q0, p_adaptative))
        k2 = f(x + 0.5*h*k1,  u, ca.vertcat(q0, p_adaptative))
        k3 = f(x + 0.5*h*k2,  u, ca.vertcat(q0, p_adaptative))
        k4 = f(x + h*k3,      u, ca.vertcat(q0, p_adaptative))
        x  = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    xf = x
    #xf[:2*pcc_arm.num_segments] += ca.DM.ones(2*pcc_arm.num_segments,1) *1e-5 # to avoid singularities

    F = ca.Function('pcc_F_map', [x0, u, p_global], [xf], ['x0', 'u','p_global'], ['xf'])

    return F

###############################################################################
#
#-----------------------------Trajectory generation----------------------------
#
###############################################################################

# Circular trajectory generation

def circle_trajectory(radius, center, rotation_angles, num_points):
    '''
    Generate a circular trajectory for the end-effector of radius float+, center [x,y,z] and rotations angles [roll, pitch, yaw]
    '''
    angles = np.linspace(0, 2*np.pi, num_points,endpoint=False)
    trajectory = np.empty((num_points, 3))

    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros_like(x)
    trajectory = np.vstack((x, y, z)).T

    # Apply rotation if needed
    # precalculate cos and sin for each angle
    cy, sy = np.cos(rotation_angles[2]),   np.sin(rotation_angles[2])     # yaw (psi)
    cp, sp = np.cos(rotation_angles[1]), np.sin(rotation_angles[1])   # pitch
    cr, sr = np.cos(rotation_angles[0]),  np.sin(rotation_angles[0])    # roll

    # Rotation matrices for yaw, pitch, roll
    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [ 0,   0, 1]
    ])
    Ry = np.array([
        [cp, 0, sp],
        [ 0, 1,  0],
        [-sp, 0, cp]
    ])
    Rx = np.array([
        [1,  0,   0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Rotate each trajectory point
    trajectory = trajectory @ R.T

    # Translate to center
    trajectory += center

    return trajectory

# Recatangle trajectory generation
def rectangle_trajectory(width, height, center, rotation_angles, num_points):
    '''
    Generate a rectangular trajectory for the end-effector of width float+, height float+, center [x,y,z] and rotations angles [roll, pitch, yaw]
    '''
    points_per_side = num_points // 4
    points = np.arange(0,points_per_side)

    # Bottom side
    x_b = -width / 2 + (width / points_per_side) * points
    y_b = (-height / 2) * np.ones_like(points)

    # Right side
    x_r = (width / 2) * np.ones_like(points)
    y_r = -height / 2 + (height / points_per_side) * points

    # Top side
    x_t = width / 2 - (width / points_per_side) * points
    y_t = (height / 2) * np.ones_like(points)

    # Left side
    x_l = (-width / 2) * np.ones_like(points)
    y_l = height / 2 - (height / points_per_side) * points

    x=np.concatenate([x_b,x_r,x_t,x_l])
    y=np.concatenate([y_b,y_r,y_t,y_l])
    z=np.zeros_like(x)

    trajectory=np.vstack([x,y,z]).T
   

    # Apply rotation if needed
    # precalculate cos and sin for each angle
    cy, sy = np.cos(rotation_angles[2]),   np.sin(rotation_angles[2])     # yaw (psi)
    cp, sp = np.cos(rotation_angles[1]), np.sin(rotation_angles[1])   # pitch
    cr, sr = np.cos(rotation_angles[0]),  np.sin(rotation_angles[0])    # roll

    # Rotation matrices for yaw, pitch, roll
    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [ 0,   0, 1]
    ])
    Ry = np.array([
        [cp, 0, sp],
        [ 0, 1,  0],
        [-sp, 0, cp]
    ])
    Rx = np.array([
        [1,  0,   0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Rotate each trajectory point
    trajectory = trajectory @ R.T

    # Translate to center
    trajectory += center

    return trajectory

# eight trajectory generation

def eight_trajectory(alpha, center, rotation_angles, num_points):
    '''
    Generate an eight trajectory for the end-effector of alpha float+, center [x,y,z] and rotations angles [roll, pitch, yaw]
    '''
    angles = np.linspace(0, 2*np.pi, num_points,endpoint=False)
    trajectory = np.empty((num_points, 3))

    y = alpha * np.sqrt(2) * np.cos(angles) / (np.sin(angles)**2 + 1)
    x = alpha * np.sqrt(2) * np.cos(angles) * np.sin(angles) / (np.sin(angles)**2 + 1)
    z = np.zeros_like(x)
    trajectory = np.vstack((x, y, z)).T

    # Apply rotation if needed
    # precalculate cos and sin for each angle
    cy, sy = np.cos(rotation_angles[2]),   np.sin(rotation_angles[2])     # yaw (psi)
    cp, sp = np.cos(rotation_angles[1]), np.sin(rotation_angles[1])   # pitch
    cr, sr = np.cos(rotation_angles[0]),  np.sin(rotation_angles[0])    # roll

    # Rotation matrices for yaw, pitch, roll
    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [ 0,   0, 1]
    ])
    Ry = np.array([
        [cp, 0, sp],
        [ 0, 1,  0],
        [-sp, 0, cp]
    ])
    Rx = np.array([
        [1,  0,   0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Rotate each trajectory point
    trajectory = trajectory @ R.T

    # Translate to center
    trajectory += center

    return trajectory

def generate_total_trajectory(arm,SIM_PARAMETERS,N,stabilizing_time=0):

    T = SIM_PARAMETERS['T']
    dt = SIM_PARAMETERS['dt']
    x0 = arm.end_effector(SIM_PARAMETERS['x0'][:2*arm.num_segments])
    x0 = np.array(x0).flatten()
    radius = SIM_PARAMETERS['radius_trajectory']
    center = SIM_PARAMETERS['center_trajectory']
    rotation_angles = SIM_PARAMETERS['rotation_angles_trajectory']
    loop_time = SIM_PARAMETERS['T_loop']
    shape = SIM_PARAMETERS['shape']

    if T < stabilizing_time + loop_time:
        raise ValueError("Total sim time must be greater than stabilizing_time + loop_time")

    number_of_loops = int(np.ceil((T-stabilizing_time+N*dt)/loop_time))+10 #+1 to be sure

    # Create trajectory

    # stabilize first at initial position
    num_stabilize_points = int(stabilizing_time//dt)
    
    # follow the circular trajectory
    if shape == 'circle':
        xyz_circular_traj = circle_trajectory(radius=radius, center=center, rotation_angles=rotation_angles, num_points=int(loop_time//dt))
    elif shape == 'rectangle':
        xyz_circular_traj = rectangle_trajectory(width=radius, height=2*radius, center=center, rotation_angles=rotation_angles, num_points=int(loop_time//dt))
    elif shape == 'lemniscate':
        xyz_circular_traj = eight_trajectory(alpha=radius, center=center, rotation_angles=rotation_angles, num_points=int(loop_time//dt))
    else:
        raise ValueError("Shape must be 'circle', 'rectangle' or 'lemniscate'")
    dottet_plotting_traj = xyz_circular_traj.copy()

    # get to closest x0 point
    idx0 = np.argmin(np.linalg.norm(xyz_circular_traj - x0, axis=1))
    xyz_circular_traj = np.unwrap(np.roll(xyz_circular_traj, -idx0, axis=0),axis=0)
    
    xyz_circular_traj = np.tile(xyz_circular_traj, (int(number_of_loops), 1))

    xyz_stabilize_traj = np.tile(xyz_circular_traj[0], (num_stabilize_points, 1))
    xyz_tot_traj = np.vstack((xyz_stabilize_traj, xyz_circular_traj))

    return xyz_tot_traj, dottet_plotting_traj
