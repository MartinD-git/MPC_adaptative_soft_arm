

'''

Check why it comes back to zero when simu is different, this did not happeen with casadi, is it the new parameters??

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
    N=MPC_PARAMETERS['N']

    pcc_arm = PCCSoftArm(ARM_PARAMETERS,SIM_PARAMETERS['dt'])
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

    # Simu loop
    num_iter = int(SIM_PARAMETERS['T']/SIM_PARAMETERS['dt'])
    with tqdm(total=num_iter*SIM_PARAMETERS['dt'], desc="MPC loop", bar_format='{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {postfix}]') as pbar:
        for t in range(num_iter):
            try:
                # MPC
                loop_time_0 = time.time()

                q_goal_value = q_tot_traj[t:t+N+1,:].T

                u0 = mpc_step_acados(ocp_solver, pcc_arm.current_state, q_goal_value, N)
                
                loop_time_1 = time.time()


                #convert generalized forces to tensions for the motors
                p= np.concatenate([pcc_arm.current_state, u0])
                tendon_solution = force2tendon_solver(x0=initial_tendon_guess, p=p, lbx=lb_tendon, ubx=ub_tendon)
                u_tendon = np.array(tendon_solution['x']).flatten()
                initial_tendon_guess = u_tendon
                loop_time_2 = time.time()

                # apply the first control input to the real system
                pcc_arm.next_step(u0)

                pcc_arm.log_history(u0, q_goal_value[:,0],u_tendon)
                pbar.update(SIM_PARAMETERS['dt'])

                loop_time_3 = time.time()
                # Timing
                mpc_time = (loop_time_1 - loop_time_0) * 1000
                qp_time = (loop_time_2 - loop_time_1) * 1000
                fk_time = (loop_time_3 - loop_time_2) * 1000
                total_time = (loop_time_3 - loop_time_0) * 1000

                # Set the postfix with the calculated times
                pbar.set_postfix(MPC=f'{mpc_time:.2f}ms', QP=f'{qp_time:.2f}ms', FK=f'{fk_time:.2f}ms', Total=f'{total_time:.2f}ms', refresh=True)


            except:
                traceback.print_exc()
                break

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

if __name__ == "__main__":
    main()
