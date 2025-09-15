import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def history_plot(pcc_arm):
    history = np.array(pcc_arm.history)
    history_d = np.array(pcc_arm.history_d)
    history_u = np.array(pcc_arm.history_u)

    M = np.hstack((history, history_d, history_u)).T  # (30, T)
    M = normalize(M)
    time = np.arange(history.shape[0]) * pcc_arm.dt

    for i in range(3):
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        fig.suptitle(f"Segment {i+1}")

        labels = [r'$\phi$', r'$\phi_d$', r'$\dot{\phi}$', r'$\dot{\phi}_d$', r'$\tau$']
        idx = np.array([2*i, 12+2*i, 6+2*i, 12+6+2*i, 24+2*i]) 
        for j in range(5):
            axs[0].plot(time, M[idx[j], :], label=labels[j])
        axs[0].set_title('Phi and Torque')
        axs[0].set_xlabel('Time [s]'); axs[0].set_ylabel('Normalized'); axs[0].legend()

        labels = [r'$\theta$', r'$\theta_d$', r'$\dot{\theta}$', r'$\dot{\theta}_d$', r'$\tau$']
        idx = np.array([2*i+1, 12+2*i+1, 6+2*i+1, 12+6+2*i+1, 24+2*i+1])
        for j in range(5):
            axs[1].plot(time, M[idx[j], :], label=labels[j])
        axs[1].set_title('Theta and Torque')
        axs[1].set_xlabel('Time [s]'); axs[1].set_ylabel('Normalized'); axs[1].legend()

    plt.tight_layout()
    plt.show()

def normalize(M, eps=1e-8):
    max_abs = np.max(np.abs(M), axis=1, keepdims=True)  # (n_rows, 1)

    out = np.zeros_like(M, dtype=float)
    np.divide(M, max_abs, out=out, where=max_abs >= eps)
    return out