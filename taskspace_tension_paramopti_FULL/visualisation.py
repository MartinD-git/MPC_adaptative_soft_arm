import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

out_dir = "csv_and_plots_adapt/"
colored_line = True

def history_plot(pcc_arm,u_bound,xyz_traj=None, save=False, opti_index=None):
    history = pcc_arm.history[:, :pcc_arm.history_index].T
    history_d = pcc_arm.history_d[:, :pcc_arm.history_index].T
    history_u = pcc_arm.history_u[:, :pcc_arm.history_index].T
    history_u_tendon = pcc_arm.history_u_tendon[:, :pcc_arm.history_index].T
    history_param = pcc_arm.history_adaptive_param[:, :pcc_arm.history_index].T
    if save:
        np.savetxt(out_dir + "history_u.csv", history_u, delimiter=",")
        np.savetxt(out_dir + "history_d.csv", history_d, delimiter=",")
        np.savetxt(out_dir + "history_angles.csv", history, delimiter=",")
        np.savetxt(out_dir + "history_u_tendon.csv", history_u_tendon, delimiter=",")
    M_raw = np.hstack((history, history_d, history_u)).T  # (30, T)
    M = normalize(M_raw,u_bound,pcc_arm.num_segments)
    time = np.arange(history.shape[0]) * pcc_arm.dt

    for i in range(pcc_arm.num_segments):
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        fig.suptitle(f"Segment {i+1}")

        labels = [r'$\phi$', r'$\phi_d$', r'$\dot{\phi}$', r'$\dot{\phi}_d$', r'$\tau$']
        linestyle = ['-', '--', '-', '--', '-']
        color = ['b', 'b', 'c', 'c', 'r']
        idx = np.array([2*i, 4*pcc_arm.num_segments+2*i, 2*pcc_arm.num_segments+2*i, 4*pcc_arm.num_segments+2*pcc_arm.num_segments+2*i, 8*pcc_arm.num_segments+2*i]) 
        for j in range(5):
            axs[0].plot(time, M[idx[j], :], label=labels[j], linestyle=linestyle[j], color=color[j])
        axs[0].set_title('Phi and Torque')
        axs[0].set_xlabel('Time [s]'); axs[0].set_ylabel('Normalized'); axs[0].legend()

        labels = [r'$\theta$', r'$\theta_d$', r'$\dot{\theta}$', r'$\dot{\theta}_d$', r'$\tau$']
        linestyle = ['-', '--', '-', '--', '-']
        color = ['b', 'b', 'c', 'c', 'r']
        idx = np.array([2*i+1, 4*pcc_arm.num_segments+2*i+1, 2*pcc_arm.num_segments+2*i+1, 4*pcc_arm.num_segments+2*pcc_arm.num_segments+2*i+1, 8*pcc_arm.num_segments+2*i+1])
        for j in range(5):
            axs[1].plot(time, M[idx[j], :], label=labels[j], linestyle=linestyle[j], color=color[j])
        axs[1].set_title('Theta and Torque')
        axs[1].set_xlabel('Time [s]'); axs[1].set_ylabel('Normalized'); axs[1].legend()
        plt.tight_layout()
        if save:
            plt.savefig(out_dir + f"segment_{i+1}_states_and_torques.png", dpi=200)


    # tendon plot 
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    fig.suptitle("Control Inputs (Tendon Tensions)")

    labels = [r'$T_1$', r'$T_2$', r'$T_3$']
    color = ['b', 'r', 'm']
    for i in range(pcc_arm.num_segments):
        for k in range(3):
            axs[i].plot(time, history_u_tendon[:,2*i+k], label=labels[k], linestyle='-', color=color[k])
        axs[i].set_title(f'Segment {i+1}')
        axs[i].set_xlabel('Time [s]')
        axs[i].set_ylabel('N')
        axs[i].set_ylim(0, u_bound[1]*1.1)
        axs[i].legend()
        
    plt.tight_layout()
    if save:
        plt.savefig(out_dir + "tendon_tensions.png", dpi=200)

    #error plot
    #initial_param = np.concatenate([np.array(pcc_arm.d_eq), [pcc_arm.K[1,1]], [pcc_arm.K[3,3]]])  # damping per segment + bending stiffness per segment
    initial_param = np.array([pcc_arm.m, pcc_arm.d_eq[0], pcc_arm.d_eq[1], pcc_arm.K[1,1], pcc_arm.K[3,3]])

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()

    for i in range(5):
        ax = axes[i]
        ax.plot(time, history_param[:, i], label=f'Param {i+1}')
        ax.axhline(initial_param[i], linestyle='--', label='Initial Value')  # horizontal reference line
        ax.set_title(f'Adaptive Parameter {i+1} over Time')
        if i >= 2:  # bottom row
            ax.set_xlabel('Time [s]')
        if i % 2 == 0:  # left column
            ax.set_ylabel('Parameter Value')
        ax.legend()

    fig.tight_layout()

    if save:
        plt.savefig(out_dir + "adaptive_parameters.png", dpi=200)



    # 3d animation
    #get posture from shape function

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
        func=update_line, 
        frames=len(history), 
        fargs=fargs,
        interval=pcc_arm.dt * 1000 /5
    )
    if save:
        print("Saving animation")
        mpl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        ani.save(out_dir + 'broken_base_tendon.mp4', writer='ffmpeg', fps=int(round(1.0 / pcc_arm.dt)), dpi=200)
        print("Animation saved")
    plt.show()

def normalize(M,u_bound,num_segments, eps=1e-8):
    max_abs = np.max(np.abs(M), axis=1, keepdims=True)  # (n_rows, 1)

    angle_limit = np.pi
    torque_limit = u_bound[1]
    angle_divider = np.full(2*num_segments, angle_limit)
    velocity_divider = max_abs[2*num_segments:4*num_segments].flatten()
    torque_divider = np.full(2*num_segments, torque_limit)
    divider = np.hstack([angle_divider, velocity_divider, angle_divider, velocity_divider, torque_divider])

    out = np.zeros_like(M, dtype=float)
    np.divide(M, divider.reshape(-1,1), out=out, where=divider.reshape(-1,1) >= eps)
    return out

def update_line(num, points1, points2, lines,
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

    