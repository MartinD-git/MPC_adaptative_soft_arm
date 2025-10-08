'''
This script defines basic functions used across the project.
'''

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

'''def pcc_segment_transform(s_var, phi, theta, L): 

    eps=1e-8

    theta = ca.if_else(ca.fabs(theta)>eps, theta, ca.sign(theta)*eps) #avoid division by 0 by clamping

    #Translation
    tx = L * ca.cos(phi) * (1 - ca.cos(s_var*theta)) / theta
    ty = L * ca.sin(phi) * (1 - ca.cos(s_var*theta)) / theta
    tz = L * ca.sin(s_var*theta) / theta

    t_c = ca.vertcat(tx, ty, tz)

    # Rotation Matrix
    R_c = ca.vertcat(
        ca.horzcat(ca.cos(phi)**2 * (ca.cos(s_var*theta) - 1) + 1,    ca.sin(phi)*ca.cos(phi)*(ca.cos(s_var*theta) - 1),                ca.cos(phi)*ca.sin(s_var*theta)),
        ca.horzcat(ca.sin(phi)*ca.cos(phi)*(ca.cos(s_var*theta) - 1), ca.cos(phi)**2 * (1 - ca.cos(s_var*theta)) + ca.cos(s_var*theta), ca.sin(phi)*ca.sin(s_var*theta)),
        ca.horzcat(-ca.cos(phi)*ca.sin(s_var*theta),                 -ca.sin(phi)*ca.sin(s_var*theta),                                  ca.cos(s_var*theta))
    )

    T = ca.vertcat(ca.horzcat(R_c, t_c), ca.horzcat(0,0,0,1))

    return T'''
def sinc_cosc(u, eps=1e-4):
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

def pcc_forward_kinematics(s, q, L_segs,num_segments=3):
    '''
    Compute the forward kinematics of the 3-segment PCC robot arm.
    '''

    L1, L2 = L_segs[0], L_segs[1]
    phi1, th1, phi2, th2 = q[0], q[1], q[2], q[3]


    # Chain the transformations
    T_tip1 = pcc_segment_transform(1, phi1, th1, L1)
    T_tip2 = pcc_segment_transform(1, phi2, th2, L2)

    T_global1 = pcc_segment_transform(s, phi1, th1, L1)
    T_global2 = T_tip1 @ pcc_segment_transform(s, phi2, th2, L2)

    p1 = T_global1[:3, 3]
    p2 = T_global2[:3, 3]

    J1 = ca.jacobian(p1, q)
    J2 = ca.jacobian(p2, q)
    
    return [p1, p2], [J1,J2]

    

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

    for s_val in s_points: points1.append(ca.substitute(tips[0], s, s_val))
    for s_val in s_points: points2.append(ca.substitute(tips[1], s, s_val))

    P1 = ca.horzcat(*points1)
    P2 = ca.horzcat(*points2)

    return ca.Function('arm_shape_func', [q], [P1, P2])


def pcc_dynamics(arm,q, q_dot, tips, jacobians,sim=False):

    m = arm.rho * np.pi*(arm.r_o**2 - arm.r_i**2) * arm.L_segs[0] # mass of each segment


    num_segments = arm.num_segments
    s = arm.s
    M = ca.SX.zeros(2*num_segments, 2*num_segments)
    D = ca.SX.zeros(2*num_segments, 2*num_segments)
    G_pot = ca.SX(0)
    g_vec = ca.vertcat(0.0, 0.0, -9.81)
    d_eq = arm.d_eq

    J = ca.vertcat(*jacobians)

    if not sim:
        rho_fluid = arm.rho_air
    else:
        rho_fluid = arm.rho_liquid

    m_buoy = rho_fluid * np.pi*(arm.r_o**2 - arm.r_i**2) * arm.L_segs[0] #buoyancy mass of each segment
    m_displaced = rho_fluid * np.pi*arm.r_o**2 * arm.L_segs[0] #displaced mass of each segment

    G_integrand = (m_buoy-m) * sum(ca.dot(g_vec, tip) for tip in tips)
    M_integrand = (m+m_displaced) * (J.T @ J)
    D_fluid = 0
    for i, Ji in enumerate(jacobians):
        v_i = Ji @ q_dot
        vmag = ca.norm_2(v_i)+ 1e-8 # + 1e-6 to be smooth
        Aproj = (2*arm.r_o)*arm.L_segs[i] # projected area per segment
        D_fluid += 0.5 * rho_fluid * arm.C_d * Aproj * vmag * (Ji.T @ Ji)
        
    D_integrand = (jacobians[0].T @ jacobians[0]) * d_eq[0]+(jacobians[1].T @ jacobians[1]) * d_eq[1] + D_fluid

    M = gauss_legendre(M, M_integrand, s)

    G_pot = gauss_legendre(G_pot, G_integrand, s)
    G = ca.gradient(G_pot, q)

    D = gauss_legendre(D, D_integrand, s)

    M_func = ca.Function('M_func', [q], [M])
    G_func = ca.Function('G_func', [q], [G])
    D_func = ca.Function('D_func', [q,q_dot], [D])

    # Coriolis C
    M_q = M_func(q)
    M_dot_q_dot = ca.jtimes(M_q, q, q_dot) # This calculates (dM/dq)*q_dot

    KE = 0.5 * q_dot.T @ M_q @ q_dot
    KE_grad = ca.gradient(KE, q)

    c_vec = M_dot_q_dot @ q_dot - KE_grad
    C_vec_func = ca.Function('C_vec_func', [q, q_dot], [c_vec])

    x = ca.SX.sym('x', 4*num_segments)
    u = ca.SX.sym('u', 2*num_segments)
    q0 = ca.SX.sym('x', 4*num_segments)
    K = arm.K

    q_from_x = x[:2*num_segments]
    q_dot_from_x = x[2*num_segments:]

    # Calculate q_ddot
    M_term= M_func(q0[:2*num_segments])+1e-8* np.eye(2*num_segments)
    C_term= C_vec_func(q0[:2*num_segments], q0[2*num_segments:])
    G_term= G_func(q0[:2*num_segments])
    D_term= D_func(q0[:2*num_segments], q0[2*num_segments:]) @ q0[2*num_segments:]
    K_term= K @ q0[:2*num_segments]

    #q_ddot = ca.solve(M_term , u - C_term - G_term - D_term - K_term) #Ax=b

    #added for speed
    rhs = u - C_term - G_term - D_term - K_term

    # Robust SPD solve via Cholesky
    R = ca.chol(M_term) # M_term = R.T @ R
    y = ca.solve(R.T, rhs)  # R^T y = rhs
    q_ddot = ca.solve(R, y) # R q_ddot = y
    x_dot = ca.vertcat(q_dot_from_x, q_ddot)

    return ca.Function('pcc_f', [x, u,q0], [x_dot])

def dynamics2integrator(pcc_arm,f):
    x0 = ca.MX.sym('x0', 4*pcc_arm.num_segments)
    u  = ca.MX.sym('u', 2*pcc_arm.num_segments) 
    q0 = ca.MX.sym('q0', 4*pcc_arm.num_segments)

    dt=pcc_arm.dt

    k1 = f(x0,u,q0)
    k2 = f(x0 + 0.5*dt*k1,u,q0)
    k3 = f(x0 + 0.5*dt*k2,u,q0)
    k4 = f(x0 + dt*k3,u,q0)
    xf = x0 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    F = ca.Function('pcc_F_map', [x0, u,q0], [xf], ['x0', 'u','q0'], ['xf'])

    return F

def circle_trajectory(radius, height, angle, num_points):
    '''
    Generate a circular trajectory for the end-effector, rotated around the x axis by 'angle' radians.
    '''
    angles = np.linspace(0, 2*np.pi, num_points,endpoint=False)
    trajectory = np.empty((num_points, 3))

    # Rotation matrix around x axis
    ca_ = np.cos(angle)
    sa_ = np.sin(angle)
    R_x = np.array([
        [1,    0,     0],
        [0,  ca_, -sa_],
        [0,  sa_,  ca_]
    ])

    for i, ang in enumerate(angles):
        x = radius * np.cos(ang)
        y = radius * np.sin(ang)
        z = height
        point = np.array([x, y, z])
        rotated_point = R_x @ point
        trajectory[i, :] = rotated_point

    return trajectory

def taskspace_to_jointspace(arm, traj_xyz, w_reg=1e-4):
    """
    Convert a sequence of Cartesian points (Nx3) into joint vectors (Nx, 2*num_segments).
    """
    Nseg = arm.num_segments
    dof = 2 * Nseg
    tip_index = (Nseg - 1)
    q0 = np.array([0, np.deg2rad(10), 0, np.deg2rad(10)])

    # Variables
    q = ca.SX.sym('q', dof)
    p_des = ca.SX.sym('p', 3)

    # Build tip position at s=1 using your FK
    tips, _ = pcc_forward_kinematics(arm.s, q, arm.L_segs, num_segments=Nseg)
    p_tip = ca.substitute(tips[tip_index], arm.s, 1.0)  

    # Small least-squares
    cost = ca.sumsqr(p_tip - p_des) + w_reg * ca.sumsqr(q)

    nlp = {'x': q, 'p': p_des, 'f': cost}
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    solver = ca.nlpsol('ik', 'ipopt', nlp, opts)

    # Prepare bounds and initial guess
    delta = np.array([np.deg2rad(99999)]*dof)

    Q = np.zeros((traj_xyz.shape[0], dof))
    q_prev = q0

    for i, p in enumerate(traj_xyz):
        lbx = q_prev - delta
        ubx = q_prev + delta
        sol = solver(x0=q_prev, p=p, lbx=lbx, ubx=ubx)
        q_sol = np.array(sol['x']).flatten()
        Q[i, :] = q_sol
        q_prev = q_sol  # warm-start next point

    debug_trajectory_generation_plot(arm, traj_xyz, Q)

    return Q

'''def taskspace_to_jointspace(arm, traj_xyz, w_smooth=1e-6):
    """
    Convert a sequence of Cartesian points (N x 3) into joint vectors (N x 2*num_segments).
    """
    N_seg = arm.num_segments
    dof = 2 * N_seg
    tip_index = N_seg - 1

    traj_xyz = np.asarray(traj_xyz)
    num_points = traj_xyz.shape[0]

    q0 = np.array([np.deg2rad(0), np.deg2rad(10), np.deg2rad(10), np.deg2rad(10)])
    delta = np.deg2rad(20) * np.ones(dof)

    q = ca.SX.sym('q', num_points, dof)

    cost = 0
    constr_vars, constr_lbx, constr_ubx = [], [], []

    for i in range(num_points):
        i_prev = (i - 1) % num_points

        # FK at step i
        tips, _ = pcc_forward_kinematics(arm.s, q[i, :], arm.L_segs, num_segments=N_seg)
        p_tip = ca.substitute(tips[tip_index], arm.s, 1.0)

        # tracking + smoothness
        dq = anglediff(q[i, :], q[i_prev, :])
        cost += ca.sumsqr(p_tip - traj_xyz[i, :]) + w_smooth * ca.sumsqr(dq)

        constr_vars.append(ca.transpose(dq))
        constr_lbx.append(-delta)
        constr_ubx.append(+delta)

    g   = ca.vertcat(*constr_vars)
    glb = np.concatenate(constr_lbx)
    gub = np.concatenate(constr_ubx)

    nlp  = {'x': ca.vec(q), 'f': cost, 'g': g}
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    solver = ca.nlpsol('ik', 'ipopt', nlp, opts)

    X0_mat = ca.repmat(ca.DM(q0).T, num_points, 1)
    x0     = ca.vec(X0_mat)

    nx  = num_points * dof
    lbx = -np.inf * np.ones(nx)
    ubx = +np.inf * np.ones(nx)

    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=glb, ubg=gub)

    Qsol_mat = ca.reshape(sol['x'], num_points, dof) 
    Qsol = np.array(Qsol_mat)

    debug_trajectory_generation_plot(arm, traj_xyz, Qsol)

    return Qsol'''

def generate_total_trajectory(arm,T,dt,q0,num_mpc_steps,stabilizing_time=0, loop_time=6.0):

    if T < stabilizing_time + loop_time:
        raise ValueError("Total sim time must be greater than stabilizing_time + loop_time")

    number_of_loops = int(np.ceil((T-stabilizing_time)/loop_time)+1) #+1 to be sure

    # Create trajectory

    # stabilize first at initial position
    num_stabilize_points = int(stabilizing_time//dt)
    q_stabilize_traj = np.tile(q0, (num_stabilize_points, 1))

    # follow the circular trajectory
    xyz_circular_traj = circle_trajectory(radius=0.6*arm.L_segs[0], height=0.8*arm.L_segs[0], angle=-np.deg2rad(75), num_points=int(loop_time//dt))

    q_traj = taskspace_to_jointspace(arm, xyz_circular_traj)
    
    idx0 = np.argmin(np.linalg.norm(q_traj - q0[:2*arm.num_segments], axis=1))
    q_traj = np.unwrap(np.roll(q_traj, -idx0, axis=0),axis=0) #start at the closest point to q0

    q_dot_traj = np.diff(np.vstack((q_traj,q_traj[0,:])),axis=0)/dt
    q_circ_traj = np.hstack((q_traj, q_dot_traj))

    q_circ_traj = np.tile(q_circ_traj, (int(number_of_loops), 1)) #more than necessary

    q_tot_traj = np.vstack((q_stabilize_traj, q_circ_traj))


    return q_tot_traj[:int(T//dt+num_mpc_steps+2),:], xyz_circular_traj #+1 for the last point, +1 for the diff to be sure

def anglediff(a, b):
    # CasADi-safe angle difference in [-pi, pi] # used when trying to solve ik smoothness problems
    return ca.atan2(ca.sin(a - b), ca.cos(a - b))

def debug_trajectory_generation_plot(arm, traj_xyz, Qsol):

    # NB: This function has been AI generated and may require adjustments.
    # Used for debugging purposes only.

    N_seg = arm.num_segments
    num_points = traj_xyz.shape[0]
    dof = 2 * N_seg
    q_row = ca.SX.sym('q_row', dof)
    tips_row, _ = pcc_forward_kinematics(arm.s, q_row, arm.L_segs, num_segments=N_seg)
    p_tip_row = ca.substitute(tips_row[N_seg - 1], arm.s, 1.0)
    fk_tip = ca.Function('fk_tip', [q_row], [p_tip_row])

    # Evaluate tip positions for the solved joint path
    P = np.array([np.array(fk_tip(Qsol[i, :])).squeeze() for i in range(num_points)])

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj_xyz[:,0], traj_xyz[:,1], traj_xyz[:,2], 'o', label='desired')
    ax.plot(P[:,0], P[:,1], P[:,2], '-', label='IK tip')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('IK vs desired circle'); ax.legend()

    # Make axes roughly equal
    xs, ys, zs = P[:,0], P[:,1], P[:,2]
    xm, ym, zm = xs.mean(), ys.mean(), zs.mean()
    r = 0.5 * max(xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min())
    ax.set_xlim(xm - r, xm + r); ax.set_ylim(ym - r, ym + r); ax.set_zlim(zm - r, zm + r)
    phi   = Qsol[:, 0::2]                 # shape: (N, N_seg)
    theta = Qsol[:, 1::2]                 # shape: (N, N_seg)

    # unwrap φ to avoid ±2π jumps for readability
    phi_un = np.unwrap(phi, axis=0)
    phi_un=phi

    # to degrees (nicer to read)
    rad2deg = 180.0/np.pi
    phi_deg   = phi_un * rad2deg
    theta_deg = theta   * rad2deg

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    for k in range(N_seg):
        ax1.plot(phi_deg[:, k], label=f'φ{k+1}')
        ax2.plot(theta_deg[:, k], label=f'θ{k+1}')

    ax1.set_ylabel('φ (deg)')
    ax2.set_ylabel('θ (deg)')
    ax2.set_xlabel('waypoint index')
    ax1.set_title('Joint angles along trajectory')
    ax1.legend(ncol=max(1, N_seg//2), fontsize=8)
    ax2.legend(ncol=max(1, N_seg//2), fontsize=8)
    plt.tight_layout()
    plt.show()

