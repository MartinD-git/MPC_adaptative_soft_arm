import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def history_plot(pcc_arm,u_bound,xyz_traj=None):
    history = np.array(pcc_arm.history)
    history_d = np.array(pcc_arm.history_d)
    history_u = np.array(pcc_arm.history_u)
    np.savetxt("history_u.csv", history_u, delimiter=",")
    np.savetxt("history_d.csv", history_d, delimiter=",")
    np.savetxt("history_angles.csv", history, delimiter=",")

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

    # 3d animation
    #get posture from shape function

    points1 = []
    points2 = []
    if pcc_arm.num_segments ==3:
        points3 = []

    for x in history:
        q = x[:2*pcc_arm.num_segments]
        if pcc_arm.num_segments==2:
            segment1, segment2 = pcc_arm.shape_func(q)
        elif pcc_arm.num_segments==3:
            segment1, segment2, segment3 = pcc_arm.shape_func(q)
        points1.append(segment1.full())
        points2.append(segment2.full())
        if pcc_arm.num_segments ==3:
            points3.append(segment3.full())

    tip_trajectory = np.array(points2)[:,:,-1] # (N, 3)

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Create lines initially without data
    line1=ax.plot(points1[0][0,:], points1[0][1,:], points1[0][2,:],'b-', label='Segment 1')
    line2=ax.plot(points2[0][0,:], points2[0][1,:], points2[0][2,:],'r-', label='Segment 2')
    tip_line = ax.plot(tip_trajectory[0,0], tip_trajectory[0,1], tip_trajectory[0,2],'g-', label='Tip trajectory')

    if pcc_arm.num_segments ==3:
        line3=ax.plot(points3[0][0,:], points3[0][1,:], points3[0][2,:],'m-', label='Segment 3')
        lines = [line1[0], line2[0], line3[0]]
    else:
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
    if pcc_arm.num_segments==2:
        ani = animation.FuncAnimation(
            fig, 
            func=update_line, 
            frames=len(history), 
            fargs=(points1, points2, None, lines,tip_line[0], tip_trajectory),
            interval=pcc_arm.dt * 1000
        )
    elif pcc_arm.num_segments==3:
        ani = animation.FuncAnimation(
            fig, 
            func=update_line, 
            frames=len(history), 
            fargs=(points1, points2, points3, lines),
        interval=pcc_arm.dt * 1000
    )

    plt.show()

def normalize(M,u_bound,num_segments, eps=1e-8):
    max_abs = np.max(np.abs(M), axis=1, keepdims=True)  # (n_rows, 1)

    angle_limit = np.pi
    torque_limit = u_bound
    angle_divider = np.full(2*num_segments, angle_limit)
    velocity_divider = max_abs[2*num_segments:4*num_segments].flatten()
    torque_divider = np.full(2*num_segments, torque_limit)
    divider = np.hstack([angle_divider, velocity_divider, angle_divider, velocity_divider, torque_divider])

    out = np.zeros_like(M, dtype=float)
    np.divide(M, divider.reshape(-1,1), out=out, where=divider.reshape(-1,1) >= eps)
    return out

def update_line(num, points1, points2, points3, lines, tip_line=None, tip_trajectory=None):
    pts1 = points1[num]
    pts2 = points2[num]
    if points3 is not None:
        pts3 = points3[num]
    lines[0].set_data(pts1[0, :], pts1[1, :])
    lines[0].set_3d_properties(pts1[2, :])
    lines[1].set_data(pts2[0, :], pts2[1, :])
    lines[1].set_3d_properties(pts2[2, :])
    if points3 is not None:
        lines[2].set_data(pts3[0, :], pts3[1, :])
        lines[2].set_3d_properties(pts3[2, :])
    if tip_line is not None:
        tip_line.set_data(tip_trajectory[:num+1,0], tip_trajectory[:num+1,1])
        tip_line.set_3d_properties(tip_trajectory[:num+1,2])
    return lines