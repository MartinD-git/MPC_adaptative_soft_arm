

'''
Main script to run acados MPC with full parameter adaptation on a PCC soft arm following a circular trajectory.

'''


import traceback
import casadi as ca
import numpy as np
from tqdm import tqdm
import time

from acados_utils import setup_ocp_solver, mpc_step_acados
from pcc_arm import PCCSoftArm
from parameters import ARM_PARAMETERS, MPC_PARAMETERS, SIM_PARAMETERS
from visualisation import history_plot
from utils import generate_total_trajectory

def main():
    start_time = time.time()
    num_iter = int(SIM_PARAMETERS['T']/SIM_PARAMETERS['dt'])
    N=MPC_PARAMETERS['N']

    pcc_arm = PCCSoftArm(ARM_PARAMETERS,SIM_PARAMETERS['dt'],num_iter)
    pcc_arm.true_current_state=SIM_PARAMETERS['x0']
    pcc_arm.current_state=pcc_arm.true_current_state + pcc_arm.meas_error()

    #generate circular trajectory (N,4*num_segments)
    print("Generating trajectory")
    q_tot_traj, xyz_circular_traj = generate_total_trajectory(pcc_arm,SIM_PARAMETERS,N,stabilizing_time=0, loop_time=SIM_PARAMETERS['T_loop'])
    print("Trajectory is generated")
    
    # Create Acados OCP solver
    Tf = N * SIM_PARAMETERS['dt']
    ocp_solver = setup_ocp_solver(pcc_arm, MPC_PARAMETERS, N, Tf)

    # create generalized force to tendon tension solver
    force2tendon_solver = create_force2tendon_function(pcc_arm)
    initial_tendon_guess = 0.1*np.ones(3*pcc_arm.num_segments)
    lb_tendon = np.zeros(3*pcc_arm.num_segments)
    ub_tendon = pcc_arm.max_tension*np.ones(3*pcc_arm.num_segments)

    #create adaptive solver
    param_solver, error_func = create_adaptative_parameters_solver(pcc_arm, MPC_PARAMETERS['N_rho'])
    #bounds:
    lb_adaptive = ca.vertcat(-1*np.array(ARM_PARAMETERS['d_eq']), -np.diag(ARM_PARAMETERS['K']))
    ub_adaptive = [1e6]*pcc_arm.num_adaptive_params

    # Simu loop
    best_error = 1e10
    error_history = []
    best_error_history = []
    error_history_time = []
    best_error_history_time = []
    with tqdm(total=num_iter*SIM_PARAMETERS['dt'], desc="MPC loop", bar_format='{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {postfix}]') as pbar:
        for t in range(num_iter):
            try:
                loop_time_00 = time.time()


                # MPC
                loop_time_0 = time.time()

                q_goal_value = q_tot_traj[t:t+N+1,:].T
                if t == 0:
                    adapt_param = pcc_arm.history_adaptive_param[:,0]
                else:
                    adapt_param = pcc_arm.history_adaptive_param[:,pcc_arm.history_index-1]
                u0, x1 = mpc_step_acados(ocp_solver, pcc_arm.current_state, q_goal_value, adapt_param, N)

                loop_time_1 = time.time()

                #convert generalized forces to tensions for the motors
                p= np.concatenate([pcc_arm.current_state, u0])
                tendon_solution = force2tendon_solver(x0=initial_tendon_guess, p=p, lbx=lb_tendon, ubx=ub_tendon)
                u_tendon = np.array(tendon_solution['x']).flatten()
                initial_tendon_guess = u_tendon
                loop_time_2 = time.time()

                pcc_arm.log_history(u0, q_goal_value[:,0],u_tendon, x1)
                # apply the first control input to the real system
                pcc_arm.next_step(u0)

                pbar.update(SIM_PARAMETERS['dt'])

#######################################
# Update rho fluid based on history
                if pcc_arm.history_index > (MPC_PARAMETERS['N_rho'] + 5) and False:
                    start_idx = pcc_arm.history_index - MPC_PARAMETERS['N_rho']
                    end_idx = pcc_arm.history_index
                    #To do: use meas states not true states thus creates a new history array
                    states = np.hstack((pcc_arm.history[:,start_idx:end_idx],pcc_arm.true_current_state.reshape(-1,1))) # add current state because it has not been logged yet
                    inputs = np.hstack((pcc_arm.history_u[:,start_idx:end_idx],np.zeros((2*pcc_arm.num_segments,1)))) #add zeros that will never be accessed, just for the vstack
                    adaptative_solver_parameters = np.vstack((states, inputs)) 
                    error = np.sum(np.round(np.square(np.linalg.norm(pcc_arm.history[:, t-10:t-1] - pcc_arm.history_pred[:, t-11:t-2], axis=0)), decimals=4))
                    error_history.append(error)
                    error_history_time.append(t*SIM_PARAMETERS['dt'])

                    if error > 0.3:
                        best_error = error
                        solution = param_solver(x0=pcc_arm.history_adaptive_param[:,pcc_arm.history_index-1],p=adaptative_solver_parameters,lbx=lb_adaptive,ubx=ub_adaptive)
                        param_sol = np.array(solution['x']).flatten()
                        objective_val = solution['f']
                        print(-lb_adaptive)
                        print(param_sol)
                    else:
                        param_sol = pcc_arm.history_adaptive_param[:,pcc_arm.history_index-1]
                    #print("\n Adapting parameters, error:", error, "objective value:", objective_val)
                    '''if error > 0.1 and error < best_error: #solve only if significant error
                        best_error = error
                        best_error_history.append(best_error)
                        best_error_history_time.append(t*SIM_PARAMETERS['dt'])
                        #solution = param_solver(x0=pcc_arm.history_adaptive_param[:,pcc_arm.history_index-1],p=adaptative_solver_parameters)
                        solution = param_solver(x0=pcc_arm.history_adaptive_param[:,pcc_arm.history_index-1],p=adaptative_solver_parameters,lbx=lb_adaptive,ubx=ub_adaptive)
                        param_sol = np.array(solution['x']).flatten()
                        objective_val = solution['f']
                        print("\n Adapting parameters, error:", error, "objective value:", objective_val)
                        print("\n A:", param_sol[:4])
                        print("\n B:", param_sol[4:8])
                        print("\n C:", param_sol[8:12])
                        print("\n D:", param_sol[12:16])
                    else: 
                        param_sol = pcc_arm.history_adaptive_param[:,pcc_arm.history_index-1]'''
                else:
                    param_sol = np.zeros(pcc_arm.num_adaptive_params)
                pcc_arm.history_adaptive_param[:, pcc_arm.history_index] = param_sol
#######################################
                loop_time_3 = time.time()
                # Timing
                adapt_time = (loop_time_0 - loop_time_00) * 1000
                mpc_time = (loop_time_1 - loop_time_0) * 1000
                qp_time = (loop_time_2 - loop_time_1) * 1000
                fk_time = (loop_time_3 - loop_time_2) * 1000
                total_time = (loop_time_3 - loop_time_00) * 1000

                # Set the postfix with the calculated times
                pbar.set_postfix(adapt=f'{adapt_time:.2f}ms', MPC=f'{mpc_time:.2f}ms', QP=f'{qp_time:.2f}ms', FK=f'{fk_time:.2f}ms', Total=f'{total_time:.2f}ms', refresh=True)


            except:
                traceback.print_exc()
                break
    import matplotlib.pyplot as plt
    '''plt.figure()
    plt.plot(error_history_time, error_history, label='Error')
    plt.plot(best_error_history_time, best_error_history, label='Best error')
    plt.xlabel('Time [s]')
    plt.ylabel('Error')
    plt.title('Adaptative parameters error over time')
    plt.show()'''
    print("--- %s seconds ---" % (time.time() - start_time))
    history_plot(pcc_arm,MPC_PARAMETERS['u_bound'],xyz_circular_traj)

def create_force2tendon_function(arm):
    num_segments = arm.num_segments
    J_tendon = ca.SX.zeros((3*num_segments, 2*num_segments))
    u_tendon = ca.SX.sym('u', 3*num_segments)
    u_generalized = ca.SX.sym('u_gen', 2*num_segments)
    q = ca.SX.sym('x', 4*num_segments)
    
    for i in range(num_segments):
        for k in range(3): #number of tendons
            phi =q[2*i]
            theta = q[1+2*i]
            J_tendon[k+3*i,2*i] = -theta*arm.r_d*ca.sin(arm.sigma_k[k]-phi)
            J_tendon[k+3*i,2*i+1] = -arm.r_d*ca.cos(arm.sigma_k[k]-phi)

    # Formulate the QP:
    objective = ca.sumsqr(J_tendon.T @ u_tendon - u_generalized)

    qp   = {'x': u_tendon, 'p': ca.vertcat(q, u_generalized), 'f': objective}
    opts = {'print_time': False, 'printLevel': 'none'}
    solver = ca.qpsol('force2tendon', 'qpoases', qp, opts)
    return solver

def create_adaptative_parameters_solver(arm,N):
    '''
    add SDP constraint because M+A must be pos def as it represents mass, first tries without it
    '''
    p_adaptative = ca.MX.sym('p_adaptative', arm.num_adaptive_params)
    p= ca.MX.sym('p', 4*arm.num_segments + 2*arm.num_segments,N+1) #state, control
    state_history = p[0:4*arm.num_segments,:]
    u_history = p[4*arm.num_segments:,:]
    cost=0
    for i in range(N):
        p_global = ca.vertcat(state_history[:,-(i+2)], p_adaptative)
        q_pred = arm.integrator(x0=state_history[:,-(i+2)], u=u_history[:,-(i+2)], p_global=p_global)['xf']
        cost += ca.sumsqr(q_pred - state_history[:,-(i+1)])
    cost+= ca.sumsqr(p_adaptative)  #regularization term to avoid too large parameters
    nlp = {'x': p_adaptative, 'p': p, 'f': cost}
    '''opts = {
        'ipopt.print_level': 0, 'print_time': 0,
        # warm-start & early-exit
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.max_cpu_time': 0.2,
        'ipopt.tol': 1e-2,
        'ipopt.acceptable_tol': 5e-2,
        'ipopt.acceptable_iter': 1,
    }'''
    opts = {
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.acceptable_iter': 1,
        'ipopt.max_cpu_time': 0.2,
        'ipopt.max_iter': 10,
        'ipopt.print_level': 0, 'print_time': 0,
    }
    
    solver = ca.nlpsol('adaptative_solver', 'ipopt', nlp, opts)

    return solver, ca.Function('error_func', [p_adaptative, p], [cost])

if __name__ == "__main__":
    main()
