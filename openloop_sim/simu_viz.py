import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# parameters and functions from other files
from sim_param import get_robot_params, get_simulation_params
from sim_arm import get_functions

robot_params = get_robot_params()
sim_params = get_simulation_params()

# Load the pre-compiled CasADi functions
dynamics_func, shape_func = get_functions()

##################
# integrator
##################

#define param
x = ca.SX.sym('x', 12)

p = ca.vertcat(
    ca.SX.sym('u', 6),
    ca.SX.sym('L', 3),
    ca.SX.sym('m'),
    ca.SX.sym('g', 3),
    ca.SX.sym('d_eq', 3),
    ca.SX.sym('K', 36)
)

ode = {'x': x, 'p': p, 'ode': dynamics_func(x, p[0:6], p[6:9], p[9], p[10:13], p[13:16], ca.reshape(p[16:], 6, 6))}

integrator = ca.integrator('F', 'cvodes', ode, 0.0, sim_params['dt'])

##################
# Loop
##################
# Run open loop simu
print("Starting open-loop simulation")

# arrays for plotting
x_current = sim_params['x0']
x_history = [x_current]

u_openloop = np.zeros(6)

p_params = ca.vertcat(
    u_openloop,
    robot_params['L_segs'],
    robot_params['m'],
    robot_params['g'],
    robot_params['d_eq'],
    ca.reshape(robot_params['K'], -1, 1)
)

# The simulation loop
num_steps = int(sim_params['T'] / sim_params['dt'])
for step in range(num_steps):
    res = integrator(x0=x_current, p=p_params)
    x_current = res['xf'].full().flatten() # .full().flatten() converts CasADi DM to numpy 1d array
    
    # Store the result
    x_history.append(x_current)
    
x_history = np.array(x_history)
print("Simulation finished.")

#Plot the Results
time_vec = np.linspace(0, sim_params['T'], num_steps + 1)

plt.figure(figsize=(12, 8))
plt.plot(time_vec, np.rad2deg(x_history[:, :6]))
plt.title('PCC Robot Joint Angles vs. Time (Open Loop)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (degrees)')
plt.legend(['φ₁', 'θ₁', 'φ₂', 'θ₂', 'φ₃', 'θ₃'])
plt.grid(True)
plt.show()

# 3d animation
#get posture from shape function

points1 = []
points2 = []
points3 = []

for x in x_history:
    q = x[:6]
    segment1, segment2, segment3 = shape_func(q, robot_params['L_segs'])
    points1.append(segment1.full())
    points2.append(segment2.full())
    points3.append(segment3.full())

def update_line(num, points1, points2, points3, lines):
    pts1 = points1[num]
    pts2 = points2[num]
    pts3 = points3[num]
    lines[0].set_data(pts1[0, :], pts1[1, :])
    lines[0].set_3d_properties(pts1[2, :])
    lines[1].set_data(pts2[0, :], pts2[1, :])
    lines[1].set_3d_properties(pts2[2, :])
    lines[2].set_data(pts3[0, :], pts3[1, :])
    lines[2].set_3d_properties(pts3[2, :])
    return lines

# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# Create lines initially without data
line1=ax.plot(points1[0][0,:], points1[0][1,:], points1[0][2,:],'b-', label='Segment 1')
line2=ax.plot(points2[0][0,:], points2[0][1,:], points2[0][2,:],'r-', label='Segment 2')
line3=ax.plot(points3[0][0,:], points3[0][1,:], points3[0][2,:],'m-', label='Segment 3')
lines = [line1[0], line2[0], line3[0]]

# Setting the Axes properties
max_length = np.sum(robot_params['L_segs'])
ax.set(xlim3d=(-0.5 * max_length, 1.1 * max_length), xlabel='X')
ax.set(ylim3d=(-1.1 * max_length, 1.1 * max_length), ylabel='Y')
ax.set(zlim3d=(-1.1 * max_length, 1.1 * max_length), zlabel='Z')

# Creating the Animation object
ani = animation.FuncAnimation(
    fig, 
    func=update_line, 
    frames=len(x_history), 
    fargs=(points1, points2, points3, lines),
    interval=sim_params['dt'] * 1000
)

plt.show()
#print("Saving animation to MP4")
#ani.save('pcc_robot_simulation_open_loop.mp4', writer='ffmpeg')
#print("Animation saved successfully")