'''
This script defines basic functions used across the project.
'''

import casadi as ca
import numpy as np

def pcc_segment_transform(s_var, phi, theta, L): 

    eps=1e-8

    if ca.fabs(theta)<eps:
        theta=eps
    
    #Translation
    tx = L * ca.cos(phi) * (1 - ca.cos(s_var*theta)) / theta
    ty = L * ca.sin(phi) * (1 - ca.cos(s_var*theta)) / theta
    tz = L * ca.sin(s_var*theta) / theta, L * s_var

    t_c = ca.vertcat(tx, ty, tz)

    # Rotation Matrix
    R_c = ca.vertcat(
        ca.horzcat(ca.cos(phi)**2 * (ca.cos(s_var*theta) - 1) + 1,    ca.sin(phi)*ca.cos(phi)*(ca.cos(s_var*theta) - 1),                ca.cos(phi)*ca.sin(s_var*theta)),
        ca.horzcat(ca.sin(phi)*ca.cos(phi)*(ca.cos(s_var*theta) - 1), ca.cos(phi)**2 * (1 - ca.cos(s_var*theta)) + ca.cos(s_var*theta), ca.sin(phi)*ca.sin(s_var*theta)),
        ca.horzcat(-ca.cos(phi)*ca.sin(s_var*theta),                 -ca.sin(phi)*ca.sin(s_var*theta),                                  ca.cos(s_var*theta))
    )

    T = ca.vertcat(ca.horzcat(R_c, t_c), ca.horzcat(0,0,0,1))

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

def gauss_legendre(result,integrands):
    '''
    Perform Gauss-Legendre quadrature on the provided integrands to go faster than symbolic integration.
    '''
    num_points = 10 # Arbitrary number
    gl_points, gl_weights = np.polynomial.legendre.leggauss(num_points)

    s_sub = lambda expr: ca.substitute(expr, s, s_k) #helper function to substitute s by the gl point
    
    # Our integral is over [0, 1], so we need to scale the points and weights
    # Change of variables: s = (t+1)/2, where t is in [-1, 1]. Then ds = dt/2.
    s_eval_points = 0.5 * (gl_points + 1)
    s_weights = 0.5 * gl_weights

    for k in range(num_points):
        s_k = s_eval_points[k]
        w_k = s_weights[k]
        
        s_sub = lambda expr: ca.substitute(expr, s, s_k)
        
        for integrand in integrands:
            result += s_sub(integrand) * w_k

    return result

def shape_function(q, L_segs, tips, s):
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

    return ca.Function('arm_shape_func', [q, L_segs], [P1, P2, P3])

def pcc_dynamics(q, q_dot,  L_segs, jacobians):
    m = ca.SX.sym('m')
    M = ca.SX.zeros(6, 6)
    g_vec = ca.SX.sym('g', 3)
    G_pot = 0
    d_eq = ca.SX.sym('d_eq', 3)
    D = ca.SX.zeros(6, 6)

    J = ca.vertcat(*jacobians)

    M_integrand = m * (J.T @ J)

    G_integrand = -m * sum(ca.dot(g_vec, tip) for tip in tips)

    D_integrand = sum((J.T @ J) * d for (J,d) in zip(J,d_eq))

    M = gauss_legendre(M, M_integrand)

    G_pot = gauss_legendre(G_pot, G_integrand)
    G = ca.gradient(G_pot, q)

    D = gauss_legendre(D, D_integrand)

    M_func = ca.Function('M_func', [q, L_segs, m], [M])
    G_func = ca.Function('G_func', [q, L_segs, m, g_vec], [G])
    D_func = ca.Function('D_func', [q, L_segs, d_eq], [D])

    # Coriolis C
    M_q = M_func(q, L_segs, m)
    M_dot_q_dot = ca.jtimes(M_q, q, q_dot) # This calculates (dM/dq)*q_dot

    KE = 0.5 * q_dot.T @ M_q @ q_dot
    KE_grad = ca.gradient(KE, q)

    c_vec = M_dot_q_dot @ q_dot - KE_grad
    C_vec_func = ca.Function('C_vec_func', [q, q_dot, L_segs, m], [c_vec])

    x = ca.SX.sym('x', 12)
    u = ca.SX.sym('u', 6)
    K = ca.SX.sym('K', 6, 6)

    q_from_x = x[:6]
    q_dot_from_x = x[6:]

    # Calculate q_ddot
    M_term= M_func(q_from_x, L_segs, m)+1e-8* ca.SX.eye(6)
    C_term= C_vec_func(q_from_x, q_dot_from_x, L_segs, m)
    G_term= G_func(q_from_x, L_segs, m, g_vec)
    D_term= D_func(q_from_x, L_segs, d_eq) @ q_dot_from_x
    K_term= K @ q_from_x

    q_ddot = ca.solve(M_term , u - C_term - G_term - D_term - K_term) #Ax=b
    x_dot = ca.vertcat(q_dot_from_x, q_ddot)

    return ca.Function('f', [x, u, L_segs, m, g_vec,d_eq,K], [x_dot])
