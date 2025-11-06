#!/usr/bin/env python

from dynamixel_controller import DynamixelController, BaseModel
import os
import time
import numpy as np

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
    
########### MOTOR CONFIG ############

print("Attempting to connect to dynamixels...")

num_motors = 3  # Number of motors to control
motor_list = []
for i in range(num_motors):
    motor_list.append(BaseModel(i))

print("Motor list: " + str(motor_list))

controller = DynamixelController("COM23", motor_list, baudrate=57600, latency_time=10)
controller.activate_controller()
controller.torque_on()
controller.set_operating_mode_all("extended_position_control")
controller.set_profile_velocity([100,100,100])

result_info = controller.read_info_with_unit(pwm_unit="percent", angle_unit="deg", current_unit="mA", retry=False, fast_read=False)
pos = result_info[0]

for i in range(num_motors):
    print("Dynamixel" + str(i) + ":  " + str(pos[i]))


########### MAIN LOOP ############

index = 0


goal_theta = [0, np.pi/2, np.pi/2, np.pi/2, 0]
goal_phi = [0, np.pi/2, np.pi/4, np.pi, np.pi] 

def get_step_change(theta, phi, l):

    n_ten = 3
    r_t = 0.1
    r_pulley = 0.05
    enc_steps = 4096
    metres_per_step = 2*np.pi*r_pulley/enc_steps

    steps = np.zeros(n_ten, dtype=int)

    for i in range(n_ten):
        ten_angle = (2*(i+1)-1) * np.pi / (n_ten)
        print(ten_angle)
        l[i] = (-r_t*theta*(np.cos(ten_angle - phi)))
        steps[i] = round(l[i] / metres_per_step)

    return steps

l_b = 0.5
l = [l_b, l_b, l_b]  # Initial lengths of the tendons
prev_steps = np.zeros(4, dtype=int)  # Previous steps for each tendon
# start_enc_pos = controller.read_info(fast_read=False, retry=False)
# print(start_enc_pos)

while 1:
    print("Press any key to continue! (or press ESC to quit!)")
    if getch() == chr(0x1b):
        break

    print(goal_theta[index])
    print(goal_phi[index])

    # Calculate tendon lengths based on the goal angles
    
    enc_steps = get_step_change(goal_theta[index], goal_phi[index], l)
    print(prev_steps - enc_steps)

    prev_steps = enc_steps
 
    # step_change = start_enc_pos[0] + enc_steps[0]
    # print(step_change)
    
    #controller.set_goal_position([enc_steps[0], enc_steps[1]])
    # current_pos = controller.read_info(fast_read=False, retry=False)
    # print(current_pos)
    
    # wait 3s to let the motors move
    time.sleep(3)

    # Change goal position
    if index < len(goal_theta) - 1:
        index += 1
    else:
        index = 0



