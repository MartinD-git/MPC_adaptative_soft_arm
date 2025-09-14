import casadi as ca
import numpy as np

# Parameters
from parameters import arm_parameters, mpc_parameters
from arm_model import dyn_kin_functions

if __name__ == "__main__":

    opti= ca.Opti()

    #Declare decision variables
    