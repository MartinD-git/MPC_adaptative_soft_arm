

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
tendon radius is 36mm

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

def main():
    start_time = time.time()
    num_iter = int(SIM_PARAMETERS['T']/SIM_PARAMETERS['dt'])
    N=MPC_PARAMETERS['N']

    pcc_arm = PCCSoftArm(ARM_PARAMETERS,SIM_PARAMETERS['dt'],num_iter)
    pcc_arm.true_current_state=SIM_PARAMETERS['x0']
    pcc_arm.current_state=pcc_arm.true_current_state + pcc_arm.meas_error()

    #generate circular trajectory (N,4*num_segments)
    print("Generating trajectory")
    q_tot_traj, xyz_circular_traj, dottet_plotting_traj = generate_total_trajectory(pcc_arm,SIM_PARAMETERS,N,stabilizing_time=0, loop_time=SIM_PARAMETERS['T_loop'])
    print("Trajectory is generated")
    
    # Create Acados OCP solver
    Tf = N * SIM_PARAMETERS['dt']
    ocp_solver = setup_ocp_solver(pcc_arm, MPC_PARAMETERS, N, Tf)

    param_solver, error_func = create_adaptative_parameters_solver(pcc_arm, MPC_PARAMETERS['N_p_adaptative'])
    #bounds:
    lb_adaptive = ca.vertcat(-1*np.array(ARM_PARAMETERS['d_eq']), -np.diag(ARM_PARAMETERS['K']))
    ub_adaptive = [1e6]*pcc_arm.num_adaptive_params
    best_error = 1e10

    # Simu loop
    with tqdm(total=num_iter*SIM_PARAMETERS['dt'], desc="MPC loop", bar_format='{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {postfix}]') as pbar:
        for t in range(num_iter):
            try:
                # MPC
                loop_time_0 = time.time()

                #q_goal_value = q_tot_traj[t:t+N+1,:].T
                #q_goal_value = np.vstack((xyz_circular_traj[t:t+N+1,:].T,q_tot_traj[t:t+N+1,2*pcc_arm.num_segments:].T))  # shifted by one time step
                q_goal_value = np.vstack((xyz_circular_traj[t:t+N+1,:].T,np.zeros((2*pcc_arm.num_segments,N+1))))  # zero velocities
                if t == 0:
                    adapt_param = pcc_arm.history_adaptive_param[:,0]
                else:
                    adapt_param = pcc_arm.history_adaptive_param[:,pcc_arm.history_index-1]
                u0, x1 = mpc_step_acados(ocp_solver, pcc_arm.current_state, q_goal_value, adapt_param, N)

                loop_time_1 = time.time()
                pcc_arm.log_history(np.zeros(2*pcc_arm.num_segments), q_goal_value[:,0],u0, x1)

                # apply the first control input to the real system
                pcc_arm.next_step(u0)

                pbar.update(SIM_PARAMETERS['dt'])

                loop_time_2 = time.time()
#######################################
# Update params
                if pcc_arm.history_index > (MPC_PARAMETERS['N_p_adaptative'] + 5):
                    start_idx = pcc_arm.history_index - MPC_PARAMETERS['N_p_adaptative']
                    end_idx = pcc_arm.history_index

                    states = np.hstack((pcc_arm.history[:,start_idx:end_idx],pcc_arm.true_current_state.reshape(-1,1))) # add current state because it has not been logged yet
                    inputs = np.hstack((pcc_arm.history_u_tendon[:,start_idx:end_idx],np.zeros((3*pcc_arm.num_segments,1)))) #add zeros that will never be accessed, just for the vstack
                    adaptative_solver_parameters = np.vstack((states, inputs)) 
                    error = np.sum(np.round(np.square(np.linalg.norm(pcc_arm.history[:, t-MPC_PARAMETERS['N_p_adaptative']:t-1] - pcc_arm.history_pred[:, t-MPC_PARAMETERS['N_p_adaptative']-1:t-2], axis=0)), decimals=4))

                    if error > 0.3 and True:
                        solution = param_solver(x0=pcc_arm.history_adaptive_param[:,pcc_arm.history_index-1],p=[adaptative_solver_parameters,pcc_arm.history_adaptive_param[:,pcc_arm.history_index-1]],lbx=lb_adaptive,ubx=ub_adaptive)
                        param_sol = np.array(solution['x']).flatten()
                        #objective_val = solution['f']
                        print(-lb_adaptive)
                        print(param_sol)
                    else:
                        param_sol = pcc_arm.history_adaptive_param[:,pcc_arm.history_index-1]

                else:
                    param_sol = np.zeros(pcc_arm.num_adaptive_params)
                pcc_arm.history_adaptive_param[:, pcc_arm.history_index] = param_sol
                loop_time_3 = time.time()

                # Timing
                mpc_time = (loop_time_1 - loop_time_0) * 1000
                fk_time = (loop_time_2 - loop_time_1) * 1000
                adapt_time = (loop_time_3 - loop_time_2) * 1000
                total_time = (loop_time_3 - loop_time_0) * 1000

                # Set the postfix with the calculated times
                pbar.set_postfix(MPC=f'{mpc_time:.2f}ms', FK=f'{fk_time:.2f}ms', Adapt=f'{adapt_time:.2f}ms', Total=f'{total_time:.2f}ms', refresh=True)
            except:
                print("status:", ocp_solver.get_status())
                print("alpha:", ocp_solver.get_stats('alpha'))
                print("qp_iter:", ocp_solver.get_stats('qp_iter'))
                print("residuals:", ocp_solver.get_residuals())  
                traceback.print_exc()
                break

    print("--- %s seconds ---" % (time.time() - start_time))
    history_plot(pcc_arm,MPC_PARAMETERS['u_bound'],dottet_plotting_traj)

def create_adaptative_parameters_solver(arm,N):
    '''
    add SDP constraint because M+A must be pos def as it represents mass, first tries without it
    '''
    p_adaptative = ca.MX.sym('p_adaptative', arm.num_adaptive_params)
    p_adaptative_prev = ca.MX.sym('p_adaptative_prev', arm.num_adaptive_params)
    p= ca.MX.sym('p', 4*arm.num_segments + 3*arm.num_segments,N+1) #state, control

    state_history = p[0:4*arm.num_segments,:]
    u_history = p[4*arm.num_segments:,:]
    cost=0
    for i in range(N):
        p_global = ca.vertcat(state_history[:,-(i+2)], p_adaptative)
        q_pred = arm.integrator(x0=state_history[:,-(i+2)], u=u_history[:,-(i+2)], p_global=p_global)['xf']
        cost += ca.sumsqr(q_pred - state_history[:,-(i+1)])
    cost+= ca.sumsqr(p_adaptative)  #regularization term to avoid too large parameters
    cost+= 10*ca.sumsqr(p_adaptative - p_adaptative_prev)  #regularization term to avoid too large changes
    nlp = {'x': p_adaptative, 'p': [p,p_adaptative_prev], 'f': cost}
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
        'ipopt.max_cpu_time': 0.1,
        'ipopt.print_level': 0, 'print_time': 0,
    }
    
    solver = ca.nlpsol('adaptative_solver', 'ipopt', nlp, opts)

    return solver, ca.Function('error_func', [p_adaptative, p], [cost])

if __name__ == "__main__":
    main()
