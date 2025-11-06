#!/usr/bin/env python

from dynamixel_controller import DynamixelController, BaseModel
import os
import time
import numpy as np
from helper_funcs import *

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
    
n_motors = 3

print("Attempting to connect to dynamixels...")

num_motors = 3  # Number of motors to control
motor_list = []
for i in range(num_motors):
    motor_list.append(BaseModel(2*i))

print("Motor list: " + str(motor_list))

controller = DynamixelController("COM9", motor_list, baudrate=57600, latency_time=10)
controller.activate_controller()
controller.torque_on()
controller.set_operating_mode_all("extended_position_control")
controller.set_profile_velocity([100,100,100])

result_info = controller.read_info_with_unit(pwm_unit="percent", angle_unit="deg", current_unit="mA", retry=False, fast_read=False)
pos = result_info[0]
goal_pos = np.zeros(n_motors)
for i in range(num_motors):
    print("Dynamixel" + str(i) + ":  " + str(pos[i]))

index = 0
goal_theta = [0, np.pi/3]
goal_phi = [0, 0] 
l_b = 0.3
l_curr = [l_b, l_b, l_b]  # Initial lengths of the tendons
r_t = 0.0175  # Radius of the tendon routing
ten_angles = np.zeros(n_motors)

for i in range(n_motors):
    ten_angles[i] = ((2*(i+1)-1) * np.pi / n_motors)

r_pulley = 0.005
enc_steps_per_rev = 4096
metres_per_step = 2*np.pi*r_pulley / enc_steps_per_rev

    
while 1:
    print("Press any key to continue! (or press ESC to quit!)")
    if getch() == chr(0x1b):
        break

    config_d = [goal_phi[index], goal_theta[index]]
    l = tendon_length_kinematics( config_d, num_motors, r_t, ten_angles, l_b )
    dl = l - l_curr
    ds = length_to_step_conversion( dl, n_motors, metres_per_step, l_curr )

    for i in range(n_motors):
        print(dl[i])
        print(ds[i])
        ds[i] = int(ds[i])
    
    l_curr = l

    result_info = controller.read_info(retry=False, fast_read=False)
    pos = result_info[0]

    for i in range(n_motors):
        goal_pos[i] = pos[i] + ds[i]
        
    #controller.set_goal_position([int(goal_pos[0]), int(goal_pos[1]), int(goal_pos[2])])
    
    # wait 3s to let the motors move
    time.sleep(3)

    # Change goal position
    if index < len(goal_theta) - 1:
        index += 1
    else:
        index = 0



