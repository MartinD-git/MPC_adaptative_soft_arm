

'''

torque(N) to current (A):
a = (1.68-0.18)/(2.8-0.07) = 0.54945054945
b = 0.14153846153

I = a*tau + b = a*r_d*tension + b

id 5 is zero

pulley radius = 6mm

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
    N_adapt_list = [2, 10, 20, 100, 150]
    #N_maxiter_list = [1, 2, 4, 8, 100]
    labels = [f'N_previous={n}' for n in N_adapt_list]
    #labels = [f'N_iter={n}' for n in N_maxiter_list]
    cmap = plt.get_cmap('plasma')
    colors = cmap(np.linspace(0, 1, 5))
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True) # adapt param plot
    axes = axes.ravel()
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True) # error plot
    fig, ax = plt.subplots() #time plot
    axes_dict = {'time': ax, 'adapt_param': axes, 'error': axs}

    for run_idx in range(5):
        def create_adaptative_parameters_solver(arm,N):

            p_adaptative = ca.MX.sym('p_adaptative', arm.num_adaptive_params)
            p = ca.MX.sym('p', (4*arm.num_segments + 3*arm.num_segments)*(N+1)+arm.num_adaptive_params) #state, control, prev adaptative params
            
            state_history = p[:4*arm.num_segments*(N+1)].reshape((4*arm.num_segments,N+1))
            u_history = p[4*arm.num_segments*(N+1):(4*arm.num_segments + 3*arm.num_segments)*(N+1),:].reshape((3*arm.num_segments,N+1))
            p_adaptative_prev = p[-arm.num_adaptive_params:]

            cost=0
            weights_regul = ca.diag([1]*arm.num_adaptive_params)  #weight more some states
            for i in range(N):
                p_global = ca.vertcat(state_history[:,-(i+2)], p_adaptative)
                q_pred = arm.integrator(x0=state_history[:,-(i+2)], u=u_history[:,-(i+2)], p_global=p_global)['xf']
                cost += ca.sumsqr(q_pred - state_history[:,-(i+1)])  #prediction errorweights 

            weights_difference = ca.diag([1e-3]*1 + [1e-3]*arm.num_segments + [1e-3]*arm.num_segments)  #weight more the mass and stiffness changes
            cost +=  ca.sumsqr(weights_regul @ p_adaptative)
            cost +=  ca.sumsqr(weights_difference @ (p_adaptative - p_adaptative_prev))

            nlp = {'x': p_adaptative, 'p': p, 'f': cost}

            opts = {
                'ipopt.max_iter': N_adapt_list[run_idx],
                "jit": True,
                "compiler": "shell",
                "jit_options": {"flags": ["-O2"]},
                'print_time': 0,
                'ipopt.print_level': 0,
            }

            solver = ca.nlpsol('adaptative_solver', 'ipopt', nlp, opts)
            return solver

        #MPC_PARAMETERS['N_p_adaptative'] = N_adapt_list[run_idx]
        start_time = time.perf_counter()
        num_iter = int(SIM_PARAMETERS['T']/SIM_PARAMETERS['dt'])
        N=MPC_PARAMETERS['N']

        pcc_arm = PCCSoftArm(ARM_PARAMETERS,SIM_PARAMETERS['dt'],num_iter)
        pcc_arm.true_current_state=SIM_PARAMETERS['x0']
        pcc_arm.current_state=pcc_arm.true_current_state + pcc_arm.meas_error()

        #generate circular trajectory (N,4*num_segments)
        print("Generating trajectory")
        xyz_circular_traj, dottet_plotting_traj = generate_total_trajectory(pcc_arm,SIM_PARAMETERS,N,stabilizing_time=0)
        print("Trajectory is generated")    
        
        # Create Acados OCP solver
        Tf = N * SIM_PARAMETERS['dt']
        ocp_solver = setup_ocp_solver(pcc_arm, MPC_PARAMETERS, N, Tf)

        param_solver = create_adaptative_parameters_solver(pcc_arm, MPC_PARAMETERS['N_p_adaptative'])
        #bounds:
        if pcc_arm.num_segments ==2:
            lb_adaptive = ca.vertcat(-0.9*pcc_arm.m,-0.9*pcc_arm.beta[0], -0.9*pcc_arm.beta[1], -0.9*pcc_arm.K[1,1], -0.9*pcc_arm.K[3,3])
        elif pcc_arm.num_segments ==3:
            lb_adaptive = ca.vertcat(-0.9*pcc_arm.m,-0.9*pcc_arm.beta[0], -0.9*pcc_arm.beta[1], -0.9*pcc_arm.beta[2], -0.8*pcc_arm.K[1,1], -0.8*pcc_arm.K[3,3], -0.8*pcc_arm.K[5,5])
        ub_adaptive = [2e1]*pcc_arm.num_adaptive_params

        opti_index = [0]
        loop_time=np.zeros(num_iter)
        # Simu loop
        with tqdm(total=num_iter*SIM_PARAMETERS['dt'], desc="MPC loop", bar_format='{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {postfix}]') as pbar:
            for t in range(num_iter):
                try:
                    # MPC
                    loop_time_0 = time.perf_counter()
                    q_goal_value = np.vstack((xyz_circular_traj[t:t+N+1,:].T,np.zeros((2*pcc_arm.num_segments,N+1))))  # zero velocities

                    adapt_param = pcc_arm.history_adaptive_param[:,pcc_arm.history_index]
                    u0, x1 = mpc_step_acados(ocp_solver, pcc_arm.current_state, q_goal_value, adapt_param, N, MPC_PARAMETERS['u_bound'])
                    #no control:
                    #u0 = np.zeros((3*pcc_arm.num_segments))
                    #x1 = pcc_arm.current_state
                    loop_time_1 = time.perf_counter()
                    pcc_arm.log_history(u0, x1)

                    # apply the first control input to the real system
                    pcc_arm.next_step(u0)

                    pbar.update(SIM_PARAMETERS['dt'])

                    loop_time_2 = time.perf_counter()
    #######################################

    # Update params
                    prev_params = pcc_arm.history_adaptive_param[:,pcc_arm.history_index-1]
                    if (pcc_arm.history_index > (MPC_PARAMETERS['N_p_adaptative']+100)):
                        start_idx = pcc_arm.history_index-1 - MPC_PARAMETERS['N_p_adaptative']
                        end_idx = pcc_arm.history_index-1

                        states = np.hstack((pcc_arm.history_meas[:,start_idx:end_idx],pcc_arm.current_state.reshape(-1,1))) # add current state because it has not been logged yet
                        inputs = np.hstack((pcc_arm.history_u_tendon[:,start_idx:end_idx],np.zeros((3*pcc_arm.num_segments,1)))) #add zeros that will never be accessed, just for the vstack
                        p_states = states.flatten(order='F')
                        p_inputs = inputs.flatten(order='F')
                        adaptative_solver_parameters = np.concatenate((p_states, p_inputs, prev_params))

                        if (pcc_arm.history_index > opti_index[-1]+100) :  # only optimize if error is significant
                            opti_index.append(pcc_arm.history_index)
                            print(prev_params)
                            solution = param_solver(x0=prev_params,p=adaptative_solver_parameters, lbx=lb_adaptive, ubx=ub_adaptive)
                            param_sol = np.array(solution['x']).flatten()
                        else:
                            param_sol = prev_params

                    else:
                        param_sol = prev_params
                    pcc_arm.history_adaptive_param[:, pcc_arm.history_index] = param_sol
                    loop_time_3 = time.perf_counter()

                    # Timing
                    mpc_time = (loop_time_1 - loop_time_0) * 1000
                    fk_time = (loop_time_2 - loop_time_1) * 1000
                    adapt_time = (loop_time_3 - loop_time_2) * 1000
                    total_time = (loop_time_3 - loop_time_0) * 1000
                    loop_time[t] = mpc_time + adapt_time

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
        print(prev_params)
        save = False
        out_dir = "csv_and_plots_adapt/"

        ax = axes_dict['time']
        #ax.scatter(np.arange(len(loop_time))*pcc_arm.dt,loop_time, color = colors[run_idx], label=labels[run_idx], zorder = len(colors)-run_idx, s=2)
        #ax.plot(np.arange(len(loop_time))*pcc_arm.dt,loop_time, color = colors[run_idx], label=labels[run_idx], zorder = len(colors)-run_idx, alpha = 0.6, linewidth=1)
        N_mean = 20
        mean_error = np.convolve(loop_time, np.ones(N_mean)/N_mean, mode='valid')
        time_axis = np.arange(len(loop_time))*pcc_arm.dt
        ax.plot(time_axis[-len(mean_error):], mean_error, color = colors[run_idx], zorder = len(colors)-run_idx, linewidth=1, label=labels[run_idx])
        ax.set_title("Computation time per MPC step")
        ax.set_yscale('log')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Time [ms] (averaged over "+str(N_mean*pcc_arm.dt)+"s)")
        ax.legend()

        if save:
            plt.savefig(out_dir + "computation_time_per_MPC_step.png", dpi=200)

            history = pcc_arm.history[:, :pcc_arm.history_index].T
        history_param = pcc_arm.history_adaptive_param[:, :pcc_arm.history_index].T
        history_pred = pcc_arm.history_pred[:, :pcc_arm.history_index].T
        history_meas = pcc_arm.history_meas[:, :pcc_arm.history_index].T


        time_x = np.arange(history_meas.shape[0]) * pcc_arm.dt


        # adaptative parameters plot
        if pcc_arm.num_segments ==3:
            titles = ['Mass', 'Damping Segment 1', 'Damping Segment 2', 'Damping Segment 3', 'Stiffness Segment 1', 'Stiffness Segment 2', 'Stiffness Segment 3']
        elif pcc_arm.num_segments ==2:
            titles = ['Mass', 'Damping Segment 1', 'Damping Segment 2', 'Stiffness Segment 1', 'Stiffness Segment 2']
        initial_param = np.concatenate(([pcc_arm.m], pcc_arm.beta, np.diag(pcc_arm.K)[1::2]))

        axes = axes_dict['adapt_param']
        for i in range(pcc_arm.num_adaptive_params):
            ax = axes[i]
            ax.plot(time_x, history_param[:, i], label=labels[run_idx], color = colors[run_idx])
            if run_idx ==0:
                ax.axhline(initial_param[i], linestyle='--', label='Initial Value')  # horizontal reference line
            ax.set_title(f'Adaptive {titles[i]} over Time')
            if i >= 2:  # bottom row
                ax.set_xlabel('Time [s]')
            if i % 2 == 0:  # left column
                ax.set_ylabel('Parameter Value')
            ax.legend()

        fig.tight_layout()

        if save:
            plt.savefig(out_dir + "adaptive_parameters.png", dpi=200)

        # Error plot over time
        try:
            q_error = np.linalg.norm(history_meas[1:,:] - history_pred[:-1,:], axis=1)
            # Generate XYZ coordinates
            #history_xyz = np.array([pcc_arm.end_effector(x[:2*pcc_arm.num_segments]).full().flatten() for x in history[:-1,:]])
            history_xyz_meas = np.array([pcc_arm.end_effector(x[:2*pcc_arm.num_segments]).full().flatten() for x in history_meas[1:,:]])
            history_pred_xyz = np.array([pcc_arm.end_effector(x[:2*pcc_arm.num_segments]).full().flatten() for x in history_pred[:-1,:]])
            xyz_error = np.linalg.norm(history_xyz_meas - history_pred_xyz, axis=1)

            N_mean = int(SIM_PARAMETERS['T_loop'] // pcc_arm.dt)
            time_axis = np.arange(1,int(q_error.shape[0])+1) * pcc_arm.dt
            axs = axes_dict['error']
            mean_error = np.convolve(q_error, np.ones(N_mean)/N_mean, mode='valid')
            axs[0].plot(time_axis[-len(mean_error):], mean_error, label=labels[run_idx], color = colors[run_idx])
            axs[0].set_ylabel('Loop-Mean Error')
            axs[0].legend()

            mean_error = np.convolve(xyz_error, np.ones(N_mean)/N_mean, mode='valid')
            axs[1].plot(time_axis[-len(mean_error):], mean_error, label=labels[run_idx], color = colors[run_idx])
            axs[1].set_xlabel('Time [s]')
            axs[1].set_ylabel('Loop-Mean Error [m]')
            axs[1].legend()

            # Add optimization vertical lines
            if run_idx ==0:
                for i in range(1, len(opti_index)):
                    time_loc = opti_index[i] * pcc_arm.dt
                    axs[0].axvline(x=time_loc, color='r', linestyle='--', alpha=0.5)
                    axs[1].axvline(x=time_loc, color='r', linestyle='--', alpha=0.5)
            plt.suptitle("Loop averaged error over time")
            plt.tight_layout()
        
            if save:
                plt.savefig(out_dir + "error_over_time.png", dpi=200)
        except:
            print("Could not plot error over time")
        
    plt.show()


if __name__ == "__main__":
    main()