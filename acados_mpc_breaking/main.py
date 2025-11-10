

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

    # Simu loop
    with tqdm(total=num_iter*SIM_PARAMETERS['dt'], desc="MPC loop", bar_format='{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {postfix}]') as pbar:
        for t in range(num_iter):
            try:
                # MPC
                loop_time_0 = time.time()

                #q_goal_value = q_tot_traj[t:t+N+1,:].T
                q_goal_value = np.vstack((xyz_circular_traj[t:t+N+1,:].T,q_tot_traj[t:t+N+1,2*pcc_arm.num_segments:].T))  # shifted by one time step
                #q_goal_value = np.vstack((xyz_circular_traj[t:t+N+1,:].T,np.zeros((2*pcc_arm.num_segments,N+1))))  # zero velocities

                u0 = mpc_step_acados(ocp_solver, pcc_arm.current_state, q_goal_value, N)+ np.ones_like(u0)*

                loop_time_1 = time.time()
                pcc_arm.log_history(np.zeros(2*pcc_arm.num_segments), q_goal_value[:,0],u0)

                # apply the first control input to the real system
                pcc_arm.next_step(u0)

                pbar.update(SIM_PARAMETERS['dt'])

                loop_time_2 = time.time()
                # Timing
                mpc_time = (loop_time_1 - loop_time_0) * 1000
                fk_time = (loop_time_2 - loop_time_1) * 1000
                total_time = (loop_time_2 - loop_time_0) * 1000

                # Set the postfix with the calculated times
                pbar.set_postfix(MPC=f'{mpc_time:.2f}ms', FK=f'{fk_time:.2f}ms', Total=f'{total_time:.2f}ms', refresh=True)

            except:
                print("status:", ocp_solver.get_status())
                print("alpha:", ocp_solver.get_stats('alpha'))
                print("qp_iter:", ocp_solver.get_stats('qp_iter'))
                print("residuals:", ocp_solver.get_residuals())  
                traceback.print_exc()
                break

    print("--- %s seconds ---" % (time.time() - start_time))
    history_plot(pcc_arm,MPC_PARAMETERS['u_bound'],dottet_plotting_traj)

if __name__ == "__main__":
    main()
