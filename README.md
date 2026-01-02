# Data-Enabled Model Predictive Control for Compliant Manipulators

This repository contains all the code of my Autumn 2025 semester project.

The code has been run only on a windows machine using WSL (required for acados or JIT C compilation)

The repository contains multiple folders having the same architecture:
- main.py: script to be run for the adaptive MPC
- parameters.py: script containing all parameters for the simulation and MPC
- pcc_arm.py: the arm class
- utils.py: a lot of utils function for the kinematics, dynamics or trajectory generation
- visualization.py: visualization functions
- acados_utils.py: all functions related to the MPC setup in acados

The folders are:
- vertical\: Adaptive MPC where the arm is pointing up
- vertical_plot_with_without_adapt\: same as previous but the simulation is runs twice (with/without adaptation) to have the error plot containing both errors
- upside_down\: Adaptive MPC where the arm is pointing down.
- convergence analysis\: vertical adaptive MPC run multiple time with a varying parameter such as the number of adaptive solver iteration to plot and see the differences

other folders and scripts:
- depreciated\: folder containing old code, pictures and videos
- csv_and_plots_adapt\: folder where the figures and csv are saved
- basic_controller.py: Openloop controller script to run for the robot arm using the Dynamixel XM430-W350 motors.
- dynamixel_controller.py: Module to help control the motors