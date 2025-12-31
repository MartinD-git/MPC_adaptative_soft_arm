import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

out_dir = "csv_and_plots_adapt/"
colored_line = True

def history_plot(pcc_arm,u_bound,xyz_traj=None, save=False, opti_index=None, sim_parameters=None):
    history = pcc_arm.history[:, :pcc_arm.history_index].T
    history_u_tendon = pcc_arm.history_u_tendon[:, :pcc_arm.history_index].T
    history_param = pcc_arm.history_adaptive_param[:, :pcc_arm.history_index].T
    history_pred = pcc_arm.history_pred[:, :pcc_arm.history_index].T
    history_meas = pcc_arm.history_meas[:, :pcc_arm.history_index].T
    if save:
        np.savetxt(out_dir + "history_angles.csv", history, delimiter=",")
        np.savetxt(out_dir + "history_u_tendon.csv", history_u_tendon, delimiter=",")

    r_pulley = 0.006  # radius pulley in m
    history_u_tendon_current = 0.54945054945 * (history_u_tendon*r_pulley) + 0.14153846153  # I = a*tau + b
    history_u_tendon_current = np.clip(history_u_tendon_current, 0, 1.5)  # limit current to 1.5A for safety
    history_u_tendon_current = history_u_tendon_current * 1000  # convert to mA * -1 because of motor orientation

    time = np.arange(history.shape[0]) * pcc_arm.dt


    # adaptative parameters plot
    if pcc_arm.num_segments ==3:
        titles = ['Mass', 'Damping Segment 1', 'Damping Segment 2', 'Damping Segment 3', 'Stiffness Segment 1', 'Stiffness Segment 2', 'Stiffness Segment 3']
    elif pcc_arm.num_segments ==2:
        titles = ['Mass', 'Damping Segment 1', 'Damping Segment 2', 'Stiffness Segment 1', 'Stiffness Segment 2']
    initial_param = np.concatenate(([pcc_arm.m], pcc_arm.beta, np.diag(pcc_arm.K)[1::2]))


    for i in range(pcc_arm.num_adaptive_params):
        ax = axes[i]
        ax.plot(time, history_param[:, i], label=f'Param {i+1}')
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

        N_mean = int(sim_parameters['T_loop'] // pcc_arm.dt)
        time_axis = np.arange(1,int(q_error.shape[0])+1) * pcc_arm.dt

        

        axs[0].plot(time_axis, q_error, label='RMSE jointspace')
        mean_error = np.convolve(q_error, np.ones(N_mean)/N_mean, mode='valid')
        axs[0].plot(time_axis[-len(mean_error):], mean_error, label='Loop average')
        axs[0].set_ylabel('Error')
        axs[0].legend()

        axs[1].plot(time_axis, xyz_error, label='RMSE taskspace')
        mean_error = np.convolve(xyz_error, np.ones(N_mean)/N_mean, mode='valid')
        axs[1].plot(time_axis[-len(mean_error):], mean_error, label='Loop average')
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Error [m]')
        axs[1].legend()

        # Add optimization vertical lines
        for i in range(1, len(opti_index)):
            time_loc = opti_index[i] * pcc_arm.dt
            axs[0].axvline(x=time_loc, color='r', linestyle='--', alpha=0.5)
            axs[1].axvline(x=time_loc, color='r', linestyle='--', alpha=0.5)

        plt.suptitle("Error over time")
        plt.tight_layout()
    
        if save:
            plt.savefig(out_dir + "error_over_time.png", dpi=200)
    except:
        print("Could not plot error over time")

    plt.show()