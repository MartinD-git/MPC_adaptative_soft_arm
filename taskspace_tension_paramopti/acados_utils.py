
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, AcadosSim, AcadosSimSolver
import casadi as ca
import numpy as np

def export_pcc_acados_model(pcc_arm, name="pcc_arm_ocp"):
    nx = 4 * pcc_arm.num_segments
    nu = 3 * pcc_arm.num_segments

    x = ca.SX.sym('x', nx)
    u = ca.SX.sym('u', nu)
    p_global = ca.SX.sym('p', nx+pcc_arm.num_adaptive_params)  # q0

    model = AcadosModel()
    model.name = name
    model.x = x
    model.u = u
    model.p_global = p_global
    model.f_expl_expr = pcc_arm.dynamics_func(x, u, p_global)

    return model


def setup_ocp_solver(pcc_arm, MPC_PARAMETERS, N, Tf):

    ocp = AcadosOcp()
    model = export_pcc_acados_model(pcc_arm)
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = 3 + 2*pcc_arm.num_segments + nu
    ny_e = 3 + 2*pcc_arm.num_segments
    u_bound = MPC_PARAMETERS['u_bound']
    Q = MPC_PARAMETERS['Q']
    R = MPC_PARAMETERS['R']
    Qf = MPC_PARAMETERS['Qf']
    
    # Horizon
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
    ocp.solver_options.nlp_solver_max_iter = 200

    # ?? works better with these globalization settings if its breaking
    '''ocp.solver_options.globalization_fixed_step_length = 0.5 
    ocp.solver_options.globalization_full_step_dual = 1        # keep duals stable when primals take smaller steps'''

    # ease the NLP stopping a bit around where you plateau
    ocp.solver_options.nlp_solver_tol_stat  = 1e-4
    ocp.solver_options.nlp_solver_tol_eq    = 1e-8
    ocp.solver_options.nlp_solver_tol_ineq  = 1e-8
    ocp.solver_options.nlp_solver_tol_comp  = 1e-7

    # Cost as NONLINEAR_LS on y = [x; u]
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    ocp.model.cost_y_expr = ca.vertcat(pcc_arm.end_effector(model.x[:2*pcc_arm.num_segments]),model.x[2*pcc_arm.num_segments:], model.u)
    ocp.model.cost_y_expr_e = ca.vertcat(pcc_arm.end_effector(model.x[:2*pcc_arm.num_segments]),model.x[2*pcc_arm.num_segments:])

    W = np.block([
        [Q,                np.zeros((3+2*pcc_arm.num_segments, nu))],
        [np.zeros((nu, 3+2*pcc_arm.num_segments)),      R        ],
    ])
    ocp.cost.W = W
    ocp.cost.W_e = Qf
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # bounds
    lbu = u_bound[0] * np.ones(nu)
    ubu = u_bound[1] * np.ones(nu)
    #lbu[5]=0 #simulate broken tendon
    #ubu[5]=0
    ocp.constraints.lbu = lbu
    ocp.constraints.ubu = ubu

    ocp.constraints.idxbu = np.arange(nu, dtype=int)

    ocp.constraints.x0 = np.zeros(nx)
    ocp.p_global_values = np.zeros(nx+pcc_arm.num_adaptive_params)

    # Generate & compile
    ocp.code_export_directory = 'c_generated_code_pcc_ocp'

    acados_ocp_solver = AcadosOcpSolver(ocp)

    return acados_ocp_solver


def setup_acados_integrator(pcc_arm, dt):
    """
    Optional external simulator (you can keep using your own integrator_sim if you prefer).
    """
    sim = AcadosSim()
    sim.model = export_pcc_acados_model(pcc_arm, name="pcc_arm_sim")
    sim.solver_options.T = dt
    sim.solver_options.num_steps = 2
    sim.code_export_directory = 'c_generated_code_pcc_sim'
    return AcadosSimSolver(sim)


def mpc_step_acados(ocp_solver, x0, q_goal, p_adaptive,N):

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu

    # x0
    ocp_solver.set(0, 'lbx', x0)
    ocp_solver.set(0, 'ubx', x0)

    # Parameters p_global = q0 (same at every stage)
    ocp_solver.set_p_global_and_precompute_dependencies(np.hstack([x0, p_adaptive]))

    # yref for each stage/terminal
    for i in range(N):
        yref_i = np.hstack([q_goal[:, i], np.zeros(nu)])
        ocp_solver.set(i, 'yref', yref_i)
    ocp_solver.set(N, 'yref', q_goal[:, N])  # terminal

    # solve
    u0 = ocp_solver.solve_for_x0(x0)
    x1 = ocp_solver.get(1, 'x')

    return u0, x1
