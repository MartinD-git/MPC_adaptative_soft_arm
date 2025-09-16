'''
This script defines basic functions used across the project.
'''

import casadi as ca
import numpy as np

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
    # Precompute powers (integers => fast)
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

def pcc_forward_kinematics(s, q, L_segs):
    '''
    Compute the forward kinematics of the 3-segment PCC robot arm.
    '''

    L1, L2, L3 = L_segs[0], L_segs[1], L_segs[2]
    phi1, th1, phi2, th2, phi3, th3 = q[0], q[1], q[2], q[3], q[4], q[5]

    # Chain the transformations
    T_tip1 = pcc_segment_transform(1, phi1, th1, L1)
    T_tip2 = pcc_segment_transform(1, phi2, th2, L2)

    T_global1 = pcc_segment_transform(s, phi1, th1, L1)
    T_global2 = T_tip1 @ pcc_segment_transform(s, phi2, th2, L2)
    T_global3 = T_tip1 @ T_tip2 @ pcc_segment_transform(s, phi3, th3, L3)

    p1 = T_global1[:3, 3]
    p2 = T_global2[:3, 3]
    p3 = T_global3[:3, 3]

    J1 = ca.jacobian(p1, q)
    J2 = ca.jacobian(p2, q)
    J3 = ca.jacobian(p3, q)

    return [p1, p2, p3], [J1,J2,J3]

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

def shape_function(q, tips, s):
    '''
    Compute a list of points along the robot shape for visualization.
    '''
    s_points = np.linspace(0, 1, 20) # 20 points per segment
    points1 = []
    points2 = []
    points3 = []

    for s_val in s_points: points1.append(ca.substitute(tips[0], s, s_val))
    for s_val in s_points: points2.append(ca.substitute(tips[1], s, s_val))
    for s_val in s_points: points3.append(ca.substitute(tips[2], s, s_val))

    P1 = ca.horzcat(*points1)
    P2 = ca.horzcat(*points2) 
    P3 = ca.horzcat(*points3)

    return ca.Function('arm_shape_func', [q], [P1, P2, P3])

def pcc_dynamics(q, q_dot, tips, jacobians, s):
    m = ca.SX.sym('m')
    M = ca.SX.zeros(6, 6)
    g_vec = np.array([0.0, 0.0, -9.81])
    G_pot = 0
    d_eq = ca.SX.sym('d_eq', 3)
    D = ca.SX.zeros(6, 6)

    J = ca.vertcat(*jacobians)

    M_integrand = m * (J.T @ J)

    G_integrand = -m * sum(ca.dot(g_vec, tip) for tip in tips)

    D_integrand = (jacobians[0].T @ jacobians[0]) * d_eq[0]+(jacobians[1].T @ jacobians[1]) * d_eq[1]+(jacobians[2].T @ jacobians[2]) * d_eq[2]

    M = gauss_legendre(M, M_integrand, s)

    G_pot = gauss_legendre(G_pot, G_integrand, s)
    G = ca.gradient(G_pot, q)

    D = gauss_legendre(D, D_integrand, s)

    M_func = ca.Function('M_func', [q, m], [M])
    G_func = ca.Function('G_func', [q, m], [G])
    D_func = ca.Function('D_func', [q, d_eq], [D])

    # Coriolis C
    M_q = M_func(q, m)
    M_dot_q_dot = ca.jtimes(M_q, q, q_dot) # This calculates (dM/dq)*q_dot

    KE = 0.5 * q_dot.T @ M_q @ q_dot
    KE_grad = ca.gradient(KE, q)

    c_vec = M_dot_q_dot @ q_dot - KE_grad
    C_vec_func = ca.Function('C_vec_func', [q, q_dot, m], [c_vec])

    x = ca.SX.sym('x', 12)
    u = ca.SX.sym('u', 6)
    K = ca.SX.sym('K', 6, 6)

    q_from_x = x[:6]
    q_dot_from_x = x[6:]

    # Calculate q_ddot
    M_term= M_func(q_from_x, m)+1e-8* ca.SX.eye(6)
    C_term= C_vec_func(q_from_x, q_dot_from_x, m)
    G_term= G_func(q_from_x, m)
    D_term= D_func(q_from_x, d_eq) @ q_dot_from_x
    K_term= K @ q_from_x

    #q_ddot = ca.solve(M_term , u - C_term - G_term - D_term - K_term) #Ax=b

    #added for speed
    rhs = u - C_term - G_term - D_term - K_term

    # Robust SPD solve via Cholesky (SX-safe)
    R = ca.chol(M_term)                    # M_term = R.T @ R
    y = ca.solve(R.T, rhs)                 # R^T y = rhs
    q_ddot = ca.solve(R, y)                # R q_ddot = y
    x_dot = ca.vertcat(q_dot_from_x, q_ddot)

    return ca.Function('f', [x, u, m,d_eq,K], [x_dot])

'''def dynamics2integrator(pcc_arm,dt):
    # Create integrator
    x_int  = ca.MX.sym('x', 12)
    u_int  = ca.MX.sym('u', 6)
    m_int  = ca.MX.sym('m')
    d_int  = ca.MX.sym('d_eq', 3)
    K_int  = ca.MX.sym('K', 6, 6)

    xdot = pcc_arm.dynamics_func(x_int, u_int, m_int, d_int, K_int)

    ode  = {'x': x_int, 'p': ca.vertcat(u_int, m_int, d_int, ca.reshape(K_int, 36, 1)), 'ode': xdot}

    return ca.integrator('F', 'cvodes', ode, 0.0, dt)'''

def dynamics2integrator(pcc_arm):
    # Pure-CasADi RK4 map: xf = F(x0, p)
    x0 = ca.MX.sym('x0', 12)
    p  = ca.MX.sym('p', 46)  # [u(6), m(1), d(3), vec(K)(36)] = 46
    dt=pcc_arm.dt

    u  = p[0:6]
    m  = p[6]
    d  = p[7:10]
    K  = ca.reshape(p[10:], 6, 6)

    f = pcc_arm.dynamics_func  # f(x, u, m, d, K) -> xdot

    k1 = f(x0,                 u, m, d, K)
    k2 = f(x0 + 0.5*dt*k1,     u, m, d, K)
    k3 = f(x0 + 0.5*dt*k2,     u, m, d, K)
    k4 = f(x0 + dt*k3,         u, m, d, K)
    xf = x0 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # Make it look like a CasADi "integrator" call-site: F(x0=..., p=...)['xf']
    return ca.Function('F', [x0, p], [xf], ['x0', 'p'], ['xf'])


