
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, AcadosSim, AcadosSimSolver
import casadi as ca
import numpy as np

def export_pcc_acados_model(pcc_arm, name="pcc_arm_ocp"):
    nx = 4 * pcc_arm.num_segments
    nu = 2 * pcc_arm.num_segments

    x = ca.SX.sym('x', nx)
    u = ca.SX.sym('u', nu)
    p = ca.SX.sym('p', nx)  # q0

    model = AcadosModel()
    model.name = name
    model.x = x
    model.u = u
    model.p = p
    model.f_expl_expr = pcc_arm.dynamics_func(x, u, p)

    return model


def setup_ocp_solver(pcc_arm, MPC_PARAMETERS, N, Tf):

    ocp = AcadosOcp()
    model = export_pcc_acados_model(pcc_arm)
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx
    u_bound = MPC_PARAMETERS['u_bound']
    Q = MPC_PARAMETERS['Q']
    R = MPC_PARAMETERS['R']
    Qf = MPC_PARAMETERS['Qf']
    
    # Horizon
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf

    # Cost as NONLINEAR_LS on y = [x; u]
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x

    W = np.block([
        [Q,                np.zeros((nx, nu))],
        [np.zeros((nu, nx)),      R        ],
    ])
    ocp.cost.W = W
    ocp.cost.W_e = Qf
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # bounds
    ocp.constraints.lbu = -u_bound * np.ones(nu)
    ocp.constraints.ubu = +u_bound * np.ones(nu)
    ocp.constraints.idxbu = np.arange(nu, dtype=int)

    ocp.constraints.x0 = np.zeros(nx)
    ocp.parameter_values = np.zeros(nx)

    # Generate & compile
    ocp.code_export_directory = 'c_generated_code_pcc_ocp'

    ocp.solver_options.nlp_solver_type = 'SQP'              # more than 1 Gauss-Newton step per solve
    ocp.solver_options.nlp_solver_max_iter = 50             # allow iterations when needed
    ocp.solver_options.qp_warm_start = 1

    # Use a slightly finer simulation inside multiple shooting
    ocp.solver_options.integrator_type = 'ERK'              # (implicit requires f_impl)
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 5             # default is often too low for stiff-ish mech. dyn.

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


def mpc_step_acados(ocp_solver, x0, q_goal,N):

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu

    # x0
    ocp_solver.set(0, 'lbx', x0)
    ocp_solver.set(0, 'ubx', x0)

    # Parameters p = q0 (same at every stage)
    for i in range(N+1):
        ocp_solver.set(i, 'p', x0)

    # yref for each stage/terminal
    for i in range(N):
        yref_i = np.hstack([q_goal[:, i], np.zeros(nu)])
        ocp_solver.set(i, 'yref', yref_i)
    ocp_solver.set(N, 'yref', q_goal[:, N])  # terminal

    if not hasattr(ocp_solver, "_ws_init"):
        for i in range(N+1):
            ocp_solver.set(i, 'x', x0)     # flat initial guess
        for i in range(N):
            ocp_solver.set(i, 'u', np.zeros(nu))
        ocp_solver._ws_init = True

    # solve
    u0 = ocp_solver.solve_for_x0(x0)

    return u0
