#!/usr/bin/env python

from dynamixel_controller import DynamixelController, BaseModel
import os
import time

import numpy as np
from helper_funcs import *
from getch import getch
from tqdm import tqdm


"""
motor id list:
id 5: segment 1, at pi
id 3: segment 1, at -pi/3
id 1: segment 1, at pi/3

id 2: segment 2, at 0
id 4: segment 2, at -2pi/3
id 0: segment 2, at 2pi/3
"""


if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
    

print("Attempting to connect to dynamixels...")

n_motors = 6  # Number of motors to control
motor_list = []
for i in range(n_motors):
    motor_list.append(BaseModel(i))

print("Motor list: " + str(motor_list))
PORT = '/dev/ttyUSB0' 
controller = DynamixelController(PORT, motor_list, baudrate=57600, latency_time=10)
controller.activate_controller()
controller.torque_off()
controller.set_operating_mode_all("current_control")
controller.torque_on()

result_info = controller.read_info_with_unit(pwm_unit="percent", angle_unit="deg", current_unit="mA", retry=False, fast_read=False)
pos = result_info[0]
goal_pos = np.zeros(n_motors)
for i in range(n_motors):
    print("Dynamixel" + str(i) + ":  " + str(pos[i]))

index = 0
r_pulley = 0.006  # radius pulley in m
openloop_tension_trajectory = np.loadtxt("csv_and_plots/history_u_tendon.csv", delimiter=',')
openloop_current_trajectory = 0.54945054945 * (openloop_tension_trajectory*r_pulley) + 0.14153846153  # I = a*tau + b
openloop_current_trajectory = np.clip(openloop_current_trajectory, 0, 1.5)  # limit current to 1.5A for safety
openloop_current_trajectory = openloop_current_trajectory * 1000  # convert to mA

#to test only the base:
#openloop_current_trajectory[:, 3:] = 0
#openloop_current_trajectory += 40 #add base current

# use the actual motor order
motor_permutation = [1,5,3,2,0,4]
idx = np.empty_like(motor_permutation)
idx[motor_permutation] = np.arange(len(motor_permutation))
openloop_current_trajectory = openloop_current_trajectory[:, idx]


index = 0
print("Press any key to continue! (or press ESC to quit!)")
if getch() == chr(0x1b):
    exit()
traj_tension = np.ones((1,n_motors))*300
print(f"Desired mA: {traj_tension[index, :]}")
controller.set_goal_current_mA(traj_tension[0, :])

for i in range(10):
    time.sleep(0.1)
    result_info = controller.read_info_with_unit(retry=False, fast_read=False)
    pos = result_info[0]
    current = result_info[2]
    print(f"Measured mA: {current}")

print("Press any key to continue! (or press ESC to quit!)")
if getch() == chr(0x1b):
    exit()

for index in tqdm(range(openloop_current_trajectory.shape[0])):
        
    controller.set_goal_current_mA(openloop_current_trajectory[index,:])
    
    # wait 3s to let the motors move
    time.sleep(0.06)
    if index % 20 == 0:
        result_info = controller.read_info_with_unit(retry=False, fast_read=False)
        pos = result_info[0]
        current = result_info[2]
        print(f"Desired mA: {openloop_current_trajectory[index, :]}")
        print(f"Measured mA: {current}")
    # Change goal position
    if index < openloop_current_trajectory.shape[0] - 1:
        index += 1
    else:
        break




