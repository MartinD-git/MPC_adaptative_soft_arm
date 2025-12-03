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

    fig, axs = plt.subplots(4, pcc_arm.num_segments, figsize=(14, 8), sharex=True, constrained_layout=True)
    for i in range(pcc_arm.num_segments):

        fig.suptitle(f"States and Torques for Segment")

        axs[0, i].set_title(f"Segment {i+1} Theta")
        # theta
        axs[0, i].plot(time, history[:,0+2*i], label=r'$\theta$', linestyle='-', color='b')
        axs[0, i].set_ylabel('rad')
        axs[0, i].tick_params(axis='y', labelcolor='b')
        axs[0, i].set_ylim(-1.2*np.pi, 1.2*np.pi)
        #axs[0, i].legend()
        # theta dot
        ax2 = axs[0, i].twinx() 
        ax2.plot(time, history[:,0+2*pcc_arm.num_segments+2*i], label=r'$\dot{\theta}$', linestyle='-', color='c')
        ax2.set_ylabel('rad/s')
        ax2.tick_params(axis='y', labelcolor='c')
        #ax2.legend()
    

        axs[1, i].set_title(f"Segment {i+1} Phi")
        # phi
        axs[1, i].plot(time, history[:,1+2*i], label=r'$\phi$', linestyle='-', color='b')
        axs[1, i].set_ylabel('rad')
        axs[1, i].tick_params(axis='y', labelcolor='b')
        axs[1, i].set_ylim(-1.2*np.pi, 1.2*np.pi)
        #axs[1, i].legend()
        # phi dot
        ax2 = axs[1, i].twinx() 
        ax2.plot(time, history[:,1+2*pcc_arm.num_segments+2*i], label=r'$\dot{\phi}$', linestyle='-', color='c')
        ax2.set_ylabel('rad/s')
        ax2.tick_params(axis='y', labelcolor='c')
        #ax2.legend()
    
        # tendon tensions
        labels = [r'$T_1$', r'$T_2$', r'$T_3$']
        color = ['b', 'r', 'm']
        for k in range(3):
            axs[2, i].plot(time, history_u_tendon[:,3*i+k], label=labels[k], linestyle='-', color=color[k])
        axs[2,i].set_title(f'Segment {i+1}')
        axs[2,i].set_xlabel('Time [s]')
        axs[2,i].set_ylabel('N')
        axs[2,i].set_ylim(0, u_bound[1]*1.1)
        axs[2,i].legend()

        # motor currents
        labels = [r'$I_1$', r'$I_2$', r'$I_3$']
        color = ['b', 'r', 'm']

        for k in range(3):
            axs[3, i].plot(time, history_u_tendon_current[:,3*i+k], label=labels[k], linestyle='-', color=color[k])
        axs[3,i].set_title(f'Segment {i+1}')
        axs[3,i].set_xlabel('Time [s]')
        axs[3,i].set_ylabel('N')
        axs[3,i].set_ylim(0, 1.6*1000)
        axs[3,i].legend()

        if save:
            plt.savefig(out_dir + f"States_Torques.png", dpi=200)

    # adaptative parameters plot
    if pcc_arm.num_segments ==3:
        titles = ['Mass', 'Damping Segment 1', 'Damping Segment 2', 'Damping Segment 3', 'Stiffness Segment 1', 'Stiffness Segment 2', 'Stiffness Segment 3']
    elif pcc_arm.num_segments ==2:
        titles = ['Mass', 'Damping Segment 1', 'Damping Segment 2', 'Stiffness Segment 1', 'Stiffness Segment 2']
    initial_param = np.concatenate(([pcc_arm.m], pcc_arm.beta, np.diag(pcc_arm.K)[1::2]))

    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()

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

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

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


    # 3d animation
    #get posture from shape function
    if pcc_arm.num_segments ==3:
        ani = animate_3d_3seg(pcc_arm, history, xyz_traj, save, opti_index)
    elif pcc_arm.num_segments ==2:
        ani = animate_3d_2seg(pcc_arm, history, xyz_traj, save, opti_index)
    
    plt.show()

def animate_3d_3seg(pcc_arm, history, xyz_traj=None, save=False, opti_index=None):
    points1 = []
    points2 = []
    points3 = []

    for x in history:
        q = x[:2*pcc_arm.num_segments]
        segment1, segment2, segment3 = pcc_arm.shape_func(q)
        points1.append(segment1.full())
        points2.append(segment2.full())
        points3.append(segment3.full())

    tip_trajectory = np.array(points3)[:,:,-1] # (N, 3)
    if colored_line:

        p_start = tip_trajectory[:-1] # Points 0 to N-1
        p_end = tip_trajectory[1:]    # Points 1 to N
        # stack along axis 1 to get shape (N-1, 2, 3)
        tip_segments = np.stack((p_start, p_end), axis=1)

        segment_colors = np.zeros((len(tip_segments), 4))
        opti_index.append(len(tip_trajectory))
        
        # Discrete colormap
        cmap = plt.cm.jet(np.linspace(0, 1, len(opti_index) - 1))
        
        # Fill colors per interval
        for i in range(len(opti_index) - 1):
            start_idx = opti_index[i]
            end_idx = opti_index[i+1]
            
            color = cmap[i]
            safe_end = min(end_idx, len(tip_segments))
            if start_idx < len(tip_segments):
                segment_colors[start_idx:safe_end] = color

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title("PCC Arm Simulation")

    # Create lines initially without data
    line1=ax.plot(points1[0][0,:], points1[0][1,:], points1[0][2,:],'b-', label='Segment 1')
    line2=ax.plot(points2[0][0,:], points2[0][1,:], points2[0][2,:],'r-', label='Segment 2')
    line3=ax.plot(points3[0][0,:], points3[0][1,:], points3[0][2,:],'g-', label='Segment 3')
    
    # Create an *empty* 3D collection at the start
    if colored_line:
        # Initialize Collection with first segment
        tip_line_collection = Line3DCollection(
            tip_segments[:1],
            colors=segment_colors[:1],
            linewidth=2
        )
        ax.add_collection3d(tip_line_collection)
    else:
        tip_line = ax.plot(tip_trajectory[0,0], tip_trajectory[0,1], tip_trajectory[0,2],'g-', label='Tip trajectory')

    lines = [line1[0], line2[0], line3[0]]

    # Setting the Axes properties
    max_length = np.sum(pcc_arm.L_segs)
    ax.set(xlim3d=(-1.1 * max_length, 1.1 * max_length), xlabel='X')
    ax.set(ylim3d=(-1.1 * max_length, 1.1 * max_length), ylabel='Y')
    ax.set(zlim3d=(-1.1 * max_length, 1.1 * max_length), zlabel='Z')

    # add target trajectory if provided
    if xyz_traj is not None:
        xyz_traj = np.vstack((xyz_traj, xyz_traj[0])) #loop back to start
        ax.plot(xyz_traj[:,0], xyz_traj[:,1], xyz_traj[:,2],'k--', label='Target trajectory')
        ax.legend()

    # Creating the Animation object
    if colored_line:
        fargs = (points1, points2, points3, lines, None, None, tip_line_collection, tip_segments, segment_colors, ax, pcc_arm.dt)
    else:
        fargs = (points1, points2, points3, lines, tip_line[0], tip_trajectory,None, None, None, ax,pcc_arm.dt)
    ani = animation.FuncAnimation(
        fig, 
        func=update_line_3seg, 
        frames=len(history),
        interval=pcc_arm.dt*1000,
        fargs=fargs,
    )
    if save:
        print("Saving animation")
        mpl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        ani.save(out_dir + 'broken_base_tendon.mp4', writer='ffmpeg', fps=int(round(1.0 / pcc_arm.dt)), dpi=200)
        print("Animation saved")
    
    return ani


def update_line_3seg(num, points1, points2, points3, lines,
                tip_line=None, tip_trajectory=None,
                tip_line_collection=None, tip_segments=None, tip_colors=None,
                ax=None, dt=0.1):

    pts1 = points1[num]
    pts2 = points2[num]
    pts3 = points3[num]
    
    # Update arm segments
    lines[0].set_data(pts1[0, :], pts1[1, :])
    lines[0].set_3d_properties(pts1[2, :])
    lines[1].set_data(pts2[0, :], pts2[1, :])
    lines[1].set_3d_properties(pts2[2, :])
    lines[2].set_data(pts3[0, :], pts3[1, :])
    lines[2].set_3d_properties(pts3[2, :])

    # Update title
    ax.set_title(f'Kite 3D Trajectory Animation - Time: {num*dt:.2f} s')

    # Update tip trajectory with colormap along time
    if colored_line:

        current_idx = max(1, min(num, len(tip_segments)))
        
        tip_line_collection.set_segments(tip_segments[:current_idx])
        tip_line_collection.set_color(tip_colors[:current_idx])
        tip_line_collection.set_alpha(0.7)
        
        # Return all artists that change
        return lines + ([tip_line_collection] if tip_line_collection is not None else [])
    else:
        if tip_line is not None:
            tip_line.set_data(tip_trajectory[:num+1,0], tip_trajectory[:num+1,1])
            tip_line.set_3d_properties(tip_trajectory[:num+1,2])
        return lines

def animate_3d_2seg(pcc_arm, history, xyz_traj=None, save=False, opti_index=None):
    points1 = []
    points2 = []

    for x in history:
        q = x[:2*pcc_arm.num_segments]
        segment1, segment2 = pcc_arm.shape_func(q)
        points1.append(segment1.full())
        points2.append(segment2.full())

    tip_trajectory = np.array(points2)[:,:,-1] # (N, 3)
    if colored_line:

        p_start = tip_trajectory[:-1] # Points 0 to N-1
        p_end = tip_trajectory[1:]    # Points 1 to N
        # stack along axis 1 to get shape (N-1, 2, 3)
        tip_segments = np.stack((p_start, p_end), axis=1)

        segment_colors = np.zeros((len(tip_segments), 4))
        opti_index.append(len(tip_trajectory))
        
        # Discrete colormap
        cmap = plt.cm.jet(np.linspace(0, 1, len(opti_index) - 1))
        
        # Fill colors per interval
        for i in range(len(opti_index) - 1):
            start_idx = opti_index[i]
            end_idx = opti_index[i+1]
            
            color = cmap[i]
            safe_end = min(end_idx, len(tip_segments))
            if start_idx < len(tip_segments):
                segment_colors[start_idx:safe_end] = color

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title("PCC Arm Simulation")

    # Create lines initially without data
    line1=ax.plot(points1[0][0,:], points1[0][1,:], points1[0][2,:],'b-', label='Segment 1')
    line2=ax.plot(points2[0][0,:], points2[0][1,:], points2[0][2,:],'r-', label='Segment 2')

    # Create an *empty* 3D collection at the start
    if colored_line:
        # Initialize Collection with first segment
        tip_line_collection = Line3DCollection(
            tip_segments[:1],
            colors=segment_colors[:1],
            linewidth=2
        )
        ax.add_collection3d(tip_line_collection)
    else:
        tip_line = ax.plot(tip_trajectory[0,0], tip_trajectory[0,1], tip_trajectory[0,2],'g-', label='Tip trajectory')

    lines = [line1[0], line2[0]]

    # Setting the Axes properties
    max_length = np.sum(pcc_arm.L_segs)
    ax.set(xlim3d=(-1.1 * max_length, 1.1 * max_length), xlabel='X')
    ax.set(ylim3d=(-1.1 * max_length, 1.1 * max_length), ylabel='Y')
    ax.set(zlim3d=(-1.1 * max_length, 1.1 * max_length), zlabel='Z')

    # add target trajectory if provided
    if xyz_traj is not None:
        xyz_traj = np.vstack((xyz_traj, xyz_traj[0])) #loop back to start
        ax.plot(xyz_traj[:,0], xyz_traj[:,1], xyz_traj[:,2],'k--', label='Target trajectory')
        ax.legend()

    # Creating the Animation object
    if colored_line:
        fargs = (points1, points2, lines, None, None, tip_line_collection, tip_segments, segment_colors, ax, pcc_arm.dt)
    else:
        fargs = (points1, points2, lines, tip_line[0], tip_trajectory,None, None, None, ax,pcc_arm.dt)
    ani = animation.FuncAnimation(
        fig, 
        func=update_line_2seg, 
        frames=len(history),
        interval=pcc_arm.dt*1000,
        fargs=fargs,
    )
    if save:
        print("Saving animation")
        mpl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        ani.save(out_dir + 'broken_base_tendon.mp4', writer='ffmpeg', fps=int(round(1.0 / pcc_arm.dt)), dpi=200)
        print("Animation saved")
    return ani


def update_line_2seg(num, points1, points2, lines,
                tip_line=None, tip_trajectory=None,
                tip_line_collection=None, tip_segments=None, tip_colors=None,
                ax=None, dt=0.1):

    pts1 = points1[num]
    pts2 = points2[num]
    
    # Update arm segments
    lines[0].set_data(pts1[0, :], pts1[1, :])
    lines[0].set_3d_properties(pts1[2, :])
    lines[1].set_data(pts2[0, :], pts2[1, :])
    lines[1].set_3d_properties(pts2[2, :])

    # Update title
    ax.set_title(f'Kite 3D Trajectory Animation - Time: {num*dt:.2f} s')

    # Update tip trajectory with colormap along time
    if colored_line:

        current_idx = max(1, min(num, len(tip_segments)))
        
        tip_line_collection.set_segments(tip_segments[:current_idx])
        tip_line_collection.set_color(tip_colors[:current_idx])
        tip_line_collection.set_alpha(0.7)
        
        # Return all artists that change
        return lines + ([tip_line_collection] if tip_line_collection is not None else [])
    else:
        if tip_line is not None:
            tip_line.set_data(tip_trajectory[:num+1,0], tip_trajectory[:num+1,1])
            tip_line.set_3d_properties(tip_trajectory[:num+1,2])
        return lines
