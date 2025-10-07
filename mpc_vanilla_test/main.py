import traceback
import casadi as ca
import numpy as np
from tqdm import tqdm
import time

# Parameters
from pcc_arm import PCCSoftArm
from parameters import ARM_PARAMETERS, MPC_PARAMETERS, SIM_PARAMETERS
from visualisation import history_plot
from utils import generate_total_trajectory

def main():
    start_time = time.time()
    N=MPC_PARAMETERS['N']

    pcc_arm = PCCSoftArm(ARM_PARAMETERS)
    pcc_arm.current_state=SIM_PARAMETERS['x0']

    #generate circular trajectory (N,4*num_segments)
    print("Generating trajectory")
    q_tot_traj, xyz_circular_traj = generate_total_trajectory(pcc_arm,SIM_PARAMETERS['T'],SIM_PARAMETERS['dt'],SIM_PARAMETERS['x0'],N,stabilizing_time=1.0, loop_time=4.0)
    print("Trajectory is generated")
    opti= ca.Opti()

    #Declare decision variables
    X= opti.variable(4*pcc_arm.num_segments,N+1)
    #q = X[:6,:]
    #q_dot = X[6:,:]
    
    u = opti.variable(2*pcc_arm.num_segments,N)
    q_goal = opti.parameter(4*pcc_arm.num_segments,N+1)
    x0 = opti.parameter(4*pcc_arm.num_segments)
    q0 = opti.parameter(4*pcc_arm.num_segments)


    # Create integrator
    pcc_arm.create_integrator(SIM_PARAMETERS['dt'])
    F = pcc_arm.integrator
    #F = F.expand() # may be faster but needs more memory



    # Objective
    objective = 0
    for i in range(N):
        '''e_pos = pos_err_vec(X[:2*pcc_arm.num_segments, i], q_goal[:2*pcc_arm.num_segments, i], pcc_arm.num_segments)
        e_vel = X[2*pcc_arm.num_segments:, i] - q_goal[2*pcc_arm.num_segments:, i]
        e = ca.vertcat(e_pos, e_vel)

        objective += ca.mtimes([e.T, MPC_PARAMETERS['Q'], e]) + ca.mtimes([u[:,i].T, MPC_PARAMETERS['R'], u[:,i]])
        '''
        objective += ca.mtimes([(X[:,i]-q_goal[:,i]).T, MPC_PARAMETERS['Q'], (X[:,i]-q_goal[:,i])]) + ca.mtimes([u[:,i].T, MPC_PARAMETERS['R'], u[:,i]])
        
        if i>0:
            du = u[:,i] - u[:,i-1]
            objective += ca.mtimes([du.T, 0.1*np.eye(2*pcc_arm.num_segments), du]) #smooth input changes

    objective += ca.mtimes([(X[:,N]-q_goal[:,N]).T, MPC_PARAMETERS['Qf'], (X[:,N]-q_goal[:,N])]) #final cost

    opti.minimize(objective)

    # Constraints
    opti.subject_to(X[:,0] == x0) # initial condition
    for i in range(N):
        opti.subject_to(X[:, i+1] ==F(x0=X[:, i], u=u[:, i], q0=q0)['xf']) # system dynamics
    u_bound=MPC_PARAMETERS['u_bound']
    opti.subject_to(opti.bounded(-u_bound, u, u_bound)) # input constraints

    #solver
    opti.solver(
        'ipopt',
        {
            'expand': True,  
            'jit': False,          
            'ipopt': {
                'print_level': 0,
                'tol': 1e-3,
                'acceptable_tol': 5e-3,
                'acceptable_iter': 3,
                'mu_strategy': 'adaptive',
                'hessian_approximation': 'limited-memory',
                'limited_memory_max_history': 100, 
                'limited_memory_initialization': 'scalar2',
                'linear_solver': 'mumps',
                'mumps_mem_percent': 5000, 
                'mumps_pivtol': 1e-6,
                'mumps_pivtolmax': 1e-1,
                'warm_start_init_point': 'yes',
                'bound_mult_init_method': 'mu-based',
                'warm_start_bound_push': 1e-6,
                'warm_start_mult_bound_push': 1e-6
            },
            #'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'
        }
    )


    # Simu loop
    num_iter = int(SIM_PARAMETERS['T']/SIM_PARAMETERS['dt'])
    with tqdm(total=num_iter*SIM_PARAMETERS['dt'], desc="MPC loop",bar_format = '{l_bar}{bar}| {n:.2f}/{total_fmt} ''[{elapsed}<{remaining}]') as pbar:
        for t in range(num_iter):

            q_goal_value = q_tot_traj[t:t+N+1,:].T
            
            opti.set_value(x0, pcc_arm.current_state)
            opti.set_value(q_goal, q_goal_value)
            opti.set_value(q0, pcc_arm.current_state)

            if t == 0:
                opti.set_initial(X, np.tile(pcc_arm.current_state.reshape(-1, 1), (1, N+1)))
                opti.set_initial(u, np.zeros((2*pcc_arm.num_segments, N)))
            else:
                opti.set_initial(u, np.hstack((sol.value(u)[:,1:], sol.value(u)[:,-1:])))
            
            # solve the problem
            try:
                sol = opti.solve()

                # WARM START OPTI
                u_sol = sol.value(u)
                X_sol = sol.value(X)

                # predict one step to fill the last state (use last control)
                u_last = u_sol[:, -1]
                x_pred = F(x0=X_sol[:, -1], u=u_last, q0 = pcc_arm.current_state)['xf'].full().flatten()

                # shifted X init
                X_init = np.hstack([X_sol[:, 1:], x_pred.reshape(-1, 1)])
                opti.set_initial(X, X_init)

                # warm-start duals (Ipopt)
                opti.set_initial(opti.lam_g, sol.value(opti.lam_g))
                # WARM START OPTI

                # apply the first control input to the real system
                pcc_arm.next_step(sol.value(u)[:,0])

                pcc_arm.log_history(sol.value(u)[:,0], q_goal_value[:,0])
                pbar.update(SIM_PARAMETERS['dt'])
            except:
                traceback.print_exc()
                break

    print("--- %s seconds ---" % (time.time() - start_time))
    history_plot(pcc_arm,MPC_PARAMETERS['u_bound'],xyz_circular_traj)

def angle_err(a, b):
    # shortest signed angle difference
    return ca.atan2(ca.sin(a-b), ca.cos(a-b))

def pos_err_vec(xq, gq, nseg):
    terms = []
    for k in range(2*nseg):
        if k % 2 == 0:   # φ indices: 0,2,(4)...
            terms.append(angle_err(xq[k], gq[k]))
        else:            # θ: plain difference
            terms.append(xq[k] - gq[k])
    return ca.vertcat(*terms)

if __name__ == "__main__":
    main()
