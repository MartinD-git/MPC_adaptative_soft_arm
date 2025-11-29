

'''

torque(N) to current (A):
a = (1.68-0.18)/(2.8-0.07) = 0.54945054945
b = 0.14153846153

I = a*tau + b = a*r_d*tension + b

control with position first and print the actual current to see if it matches simulation
id 5 is zero

radisu pulley = 6mm
measure length should be 3*105
d = 40 mm
tendon diameter is 36mm

'''


import traceback
import numpy as np
from tqdm import tqdm
import time
from acados_utils import setup_ocp_solver, mpc_step_acados
from pcc_arm import PCCSoftArm
from parameters import ARM_PARAMETERS, MPC_PARAMETERS, SIM_PARAMETERS
from visualisation import history_plot
from utils import generate_total_trajectory
import casadi as ca
import matplotlib.pyplot as plt


def main():
    start_time = time.perf_counter()
    num_iter = int(SIM_PARAMETERS['T']/SIM_PARAMETERS['dt'])
    N=MPC_PARAMETERS['N']

    pcc_arm = PCCSoftArm(ARM_PARAMETERS,SIM_PARAMETERS['dt'],num_iter)
    pcc_arm.true_current_state=SIM_PARAMETERS['x0']
    pcc_arm.current_state=pcc_arm.true_current_state + pcc_arm.meas_error()

    #generate circular trajectory (N,4*num_segments)
    print("Generating trajectory")
    xyz_circular_traj, dottet_plotting_traj = generate_total_trajectory(pcc_arm,SIM_PARAMETERS,N,stabilizing_time=0, loop_time=SIM_PARAMETERS['T_loop'])
    print("Trajectory is generated")    
    
    # Create Acados OCP solver
    Tf = N * SIM_PARAMETERS['dt']
    ocp_solver = setup_ocp_solver(pcc_arm, MPC_PARAMETERS, N, Tf)

    param_solver, _ = create_adaptative_parameters_solver_SQP(pcc_arm, MPC_PARAMETERS['N_p_adaptative'])
    #bounds:
    if pcc_arm.num_segments ==2:
        lb_adaptive_abs = ca.vertcat(-0*pcc_arm.m,-0*pcc_arm.d_eq[0], -0*pcc_arm.d_eq[1], -0.9*pcc_arm.K[1,1], -0.9*pcc_arm.K[3,3])
    elif pcc_arm.num_segments ==3:
        lb_adaptive_abs = ca.vertcat(-0.9*pcc_arm.m,-0.9*pcc_arm.d_eq[0], -0.9*pcc_arm.d_eq[1], -0.9*pcc_arm.d_eq[2], -0.8*pcc_arm.K[1,1], -0.8*pcc_arm.K[3,3], -0.8*pcc_arm.K[5,5])
    ub_adaptive = [1e6]*pcc_arm.num_adaptive_params

    error_list = []
    instant_error_list = []
    opti_index = [0]
    loop_time=np.zeros(num_iter)

    # Simu loop
    with tqdm(total=num_iter*SIM_PARAMETERS['dt'], desc="MPC loop", bar_format='{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {postfix}]') as pbar:
        for t in range(num_iter):
            try:
                # MPC
                loop_time_0 = time.perf_counter()

                q_goal_value = np.vstack((xyz_circular_traj[t:t+N+1,:].T,np.zeros((2*pcc_arm.num_segments,N+1))))  # zero velocities
                if t == 0:
                    adapt_param = pcc_arm.history_adaptive_param[:,0]
                    u_prev = np.zeros((3*pcc_arm.num_segments))
                else:
                    adapt_param = pcc_arm.history_adaptive_param[:,pcc_arm.history_index-1]
                    u_prev = pcc_arm.history_u_tendon[:,pcc_arm.history_index-1]

                u_prev = None
                u0, x1 = mpc_step_acados(ocp_solver, pcc_arm.current_state, q_goal_value, adapt_param, N, MPC_PARAMETERS['u_bound'], u_prev)

                loop_time_1 = time.perf_counter()
                pcc_arm.log_history(np.zeros(2*pcc_arm.num_segments), q_goal_value[:,0],u0, x1)

                # apply the first control input to the real system
                pcc_arm.next_step(u0)

                pbar.update(SIM_PARAMETERS['dt'])

                loop_time_2 = time.perf_counter()
#######################################

# Update params
                if (pcc_arm.history_index > (MPC_PARAMETERS['N_p_adaptative']+100)):
                    start_idx = pcc_arm.history_index - MPC_PARAMETERS['N_p_adaptative']-1
                    end_idx = pcc_arm.history_index-1

                    states = np.hstack((pcc_arm.history[:,start_idx:end_idx],pcc_arm.true_current_state.reshape(-1,1))) # add current state because it has not been logged yet
                    inputs = np.hstack((pcc_arm.history_u_tendon[:,start_idx:end_idx],np.zeros((3*pcc_arm.num_segments,1)))) #add zeros that will never be accessed, just for the vstack
                    p_states = states.flatten(order='F')
                    p_inputs = inputs.flatten(order='F')
                    prev_params = pcc_arm.history_adaptive_param[:,pcc_arm.history_index-1]
                    adaptative_solver_parameters = np.concatenate((p_states, p_inputs, prev_params))
                    error = np.mean(np.round(np.square(np.linalg.norm(pcc_arm.history[:, t-MPC_PARAMETERS['N_p_adaptative']:t-1] - pcc_arm.history_pred[:, t-MPC_PARAMETERS['N_p_adaptative']-1:t-2], axis=0)), decimals=4))
                    # *0.8 creates an error because 0 at the beginning thus both bounds are at 0
                    #bound_coef = [1]*pcc_arm.num_adaptive_params  # how much of the abs value of the param to allow to change
                    #lb_adaptive = np.maximum(np.array(pcc_arm.history_adaptive_param[:,pcc_arm.history_index-1]+bound_coef*lb_adaptive_abs), np.array(lb_adaptive_abs))
                    #ub_adaptive = pcc_arm.history_adaptive_param[:,pcc_arm.history_index-1]-bound_coef*lb_adaptive_abs
                    error_list.append(error)
                    instant_error_list.append(np.linalg.norm(pcc_arm.history[:, pcc_arm.history_index-1] - pcc_arm.history_pred[:, pcc_arm.history_index-2]))
                    
                    if (error > 0.005) and (pcc_arm.history_index > opti_index[-1]+40) :  # only optimizeif error is significant
                        opti_index.append(pcc_arm.history_index)
                        solution = param_solver(x0=pcc_arm.history_adaptive_param[:,pcc_arm.history_index-1],p=adaptative_solver_parameters,lbx=lb_adaptive_abs,ubx=ub_adaptive)
                        param_sol = np.array(solution['x']).flatten()
                        '''prev_mean = 1
                        prev_mean = min(prev_mean, len(opti_index)-1)
                        gamma = np.power(0.7, np.arange(prev_mean))
                        gamma = gamma / np.sum(gamma)
                        param_sol = np.sum(gamma * pcc_arm.history_adaptive_param[:, opti_index[-prev_mean:]], axis=1)'''
                    else:
                        param_sol = pcc_arm.history_adaptive_param[:,pcc_arm.history_index-1]

                else:
                    param_sol = np.zeros(pcc_arm.num_adaptive_params)
                    error_list.append(0)
                    instant_error_list.append(0)
                pcc_arm.history_adaptive_param[:, pcc_arm.history_index] = param_sol
                loop_time_3 = time.perf_counter()

                # Timing
                mpc_time = (loop_time_1 - loop_time_0) * 1000
                fk_time = (loop_time_2 - loop_time_1) * 1000
                adapt_time = (loop_time_3 - loop_time_2) * 1000
                total_time = (loop_time_3 - loop_time_0) * 1000
                loop_time[t] = total_time

                # Set the postfix with the calculated times
                pbar.set_postfix(MPC=f'{mpc_time:.2f}ms', FK=f'{fk_time:.2f}ms', Adapt=f'{adapt_time:.2f}ms', Total=f'{total_time:.2f}ms', refresh=True)
            except:
                print("status:", ocp_solver.get_status())
                print("alpha:", ocp_solver.get_stats('alpha'))
                print("qp_iter:", ocp_solver.get_stats('qp_iter'))
                print("residuals:", ocp_solver.get_residuals())  
                traceback.print_exc()
                break

    print("--- %s seconds ---" % (time.perf_counter() - start_time))
    save = False
    out_dir = "csv_and_plots_adapt/"

    plt.figure()
    plt.plot(np.arange(len(loop_time))*pcc_arm.dt,loop_time)
    plt.title("Computation time per MPC step")
    plt.xlabel("Time [s]")
    plt.ylabel("Time [ms]")
    if save:
        plt.savefig(out_dir + "computation_time_per_MPC_step.png", dpi=200)
    
    print("Mean computation time per MPC step: ", np.mean(loop_time), "ms")
    print("Max computation time per MPC step: ", np.max(loop_time), "ms")
    print("Min computation time per MPC step: ", np.min(loop_time), "ms")


    history_plot(pcc_arm,MPC_PARAMETERS['u_bound'],dottet_plotting_traj, save=save,opti_index=opti_index)


def create_adaptative_parameters_solver_SQP(arm,N):

    p_adaptative = ca.MX.sym('p_adaptative', arm.num_adaptive_params)
    p= ca.MX.sym('p', (4*arm.num_segments + 3*arm.num_segments)*(N+1)+arm.num_adaptive_params) #state, control, prev adaptative params
    
    state_history = p[:4*arm.num_segments*(N+1)].reshape((4*arm.num_segments,N+1))
    u_history = p[4*arm.num_segments*(N+1):-arm.num_adaptive_params,:].reshape((3*arm.num_segments,N+1))
    p_adaptative_prev = p[-arm.num_adaptive_params:]

    cost=0
    #weights = ca.diag([1]*4*arm.num_segments)  #weight more the curvature states
    for i in range(N):
        p_global = ca.vertcat(state_history[:,-(i+2)], p_adaptative)
        q_pred = arm.integrator(x0=state_history[:,-(i+2)], u=u_history[:,-(i+2)], p_global=p_global)['xf']
        cost += ca.sumsqr(q_pred - state_history[:,-(i+1)])  #prediction errorweights @ 

    weights_difference = ca.diag([10]*1 + [1]*arm.num_segments + [10]*arm.num_segments)*5  #weight more the mass and stiffness changes
    cost +=  ca.sumsqr(weights_difference @ (p_adaptative - p_adaptative_prev))

    nlp = {'x': p_adaptative, 'p': p, 'f': cost}

    # SQP
    opts = {
        #'jit': True,
        #'compiler': 'shell',
        #'jit_options': {'flags': ['-O2']}, 
        'qpsol': 'qrqp',          #QP solverqrqp, osqp, qpoases
        'qpsol_options': {'print_iter': False, 'print_header': False},
        'max_iter': 2,
        'print_time': 0,
        'print_header': False,
        'print_iteration': False
    }

    solver = ca.nlpsol('adaptative_solver', 'sqpmethod', nlp, opts)

    return solver, ca.Function('error_func', [p_adaptative, p], [cost])

if __name__ == "__main__":
    main()