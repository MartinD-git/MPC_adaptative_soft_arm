import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def history_plot(pcc_arm):
    history = np.array(pcc_arm.history)
    history_d = np.array(pcc_arm.history_d)
    history_u = np.array(pcc_arm.history_u)
    history = np.hstack((history, history_d, history_u))

    history_normalized = normalize(history)

    time = np.arange(history.shape[0]) * pcc_arm.dt

    segment_list=["Segment 1", "Segment 2", "Segment 3"]

    for i in range(3):
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        fig.suptitle(segment_list[i])  # Add global title

        labels = [r'\phi', r'\phi_d', r'\dot{\phi}', r'\dot{\phi}_d', r'\tau']
        colors = ['b-','b--','m-','m--','r-']
        indices = np.array([3*i, 3*i+12,3*i+1,3*i+12+1,3*i+12+12])
        for j in range(5):
            axs[0].plot(time, history_normalized[indices[j], :], label=labels[j], color=colors[j])
        axs[0].set_title(f'Phi and Torque History')
        axs[0].set_xlabel('Time [s]')
        axs[0].set_ylabel('Normalized Value')

        labels = [r'\theta', r'\theta_d', r'\dot{\theta}', r'\dot{\theta}_d', r'\tau']
        colors = ['b-','b--','m-','m--','r-']
        indices = np.array([3*i, 3*i+12,3*i+1,3*i+12+1,3*i+12+12])+1
        for j in range(5):
            axs[1].plot(time, history_normalized[indices[j], :], label=labels[j], color=colors[j])
        axs[1].set_title(f'Theta and Torque History')
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Normalized Value')

    plt.show()

def normalize(M, eps=1e-8):
    max_abs = np.max(np.abs(M), axis=1, keepdims=True)  # (n_rows, 1)

    out = np.zeros_like(M, dtype=float)
    np.divide(M, max_abs, out=out, where=max_abs >= eps)
    return out