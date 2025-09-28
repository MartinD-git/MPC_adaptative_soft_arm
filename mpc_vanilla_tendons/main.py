import casadi as ca
import numpy as np
from tqdm import tqdm
import time
import traceback

# Parameters
from pcc_arm import PCCSoftArm
from parameters import ARM_PARAMETERS, MPC_PARAMETERS, SIM_PARAMETERS
from visualisation import history_plot

def main():
    start_time = time.time()
    N=MPC_PARAMETERS['N']
    

    pcc_arm = PCCSoftArm(
        L_segs = ARM_PARAMETERS['L_segs'],
        m = ARM_PARAMETERS['m'],
        d_eq = ARM_PARAMETERS['d_eq'],
        K = ARM_PARAMETERS['K'],
        num_segments=ARM_PARAMETERS['num_segments'],
        r_d=ARM_PARAMETERS['r_d'],
        sigma_k=ARM_PARAMETERS['sigma_k'],
    ) 
    pcc_arm.current_state=SIM_PARAMETERS['x0']

    opti= ca.Opti()

    #Declare decision variables
    X= opti.variable(4*pcc_arm.num_segments,N+1)
    #q = X[:6,:]
    #q_dot = X[6:,:]
    
    u = opti.variable(3*pcc_arm.num_segments,N)
    q_goal = opti.parameter(4*pcc_arm.num_segments)
    x0 = opti.parameter(4*pcc_arm.num_segments)
    q0 = opti.parameter(4*pcc_arm.num_segments)


    # Create integrator
    pcc_arm.create_integrator(SIM_PARAMETERS['dt'])
    F = pcc_arm.integrator
    F = F.expand() # may be faster but needs more memory

    # Objective
    objective = 0
    for i in range(N):
        objective += ca.mtimes([(X[:,i]-q_goal).T, MPC_PARAMETERS['Q'], (X[:,i]-q_goal)]) + ca.mtimes([u[:,i].T, MPC_PARAMETERS['R'], u[:,i]])
        
        if i>0:
            du = u[:,i] - u[:,i-1]
            objective += ca.mtimes([du.T, 0.1*np.eye(3*pcc_arm.num_segments), du]) #smooth input changes

    objective += ca.mtimes([(X[:,N]-q_goal).T, MPC_PARAMETERS['Qf'], (X[:,N]-q_goal)]) #final cost

    opti.minimize(objective)

    # Constraints
    opti.subject_to(X[:,0] == x0) # initial condition
    for i in range(N):
        opti.subject_to(X[:, i+1] ==F(x0=X[:, i], u=u[:, i], q0=q0)['xf']) # system dynamics
    u_bound=MPC_PARAMETERS['u_bound']
    opti.subject_to(opti.bounded(0, u, u_bound)) # input constraints

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
            }
        }
    )


    # Simu loop
    num_iter = int(SIM_PARAMETERS['T']/SIM_PARAMETERS['dt'])
    with tqdm(total=num_iter*SIM_PARAMETERS['dt'], desc="MPC loop",bar_format = '{l_bar}{bar}| {n:.2f}/{total_fmt} ''[{elapsed}<{remaining}]') as pbar:
        for t in range(num_iter):

            # set the goal
            if t*SIM_PARAMETERS['dt'] > SIM_PARAMETERS['T']/3:
                if pcc_arm.num_segments ==2:
                    q_goal_value = np.array([
                        0, np.deg2rad(-90), 0, np.deg2rad(+120),
                        0, 0, 0, 0
                    ])

                elif pcc_arm.num_segments ==3:
                    q_goal_value = np.array([
                        0, np.deg2rad(-90), 0, np.deg2rad(+120), 0, np.deg2rad(-120),
                        0, 0, 0, 0, 0, 0
                    ])

                else:
                    raise ValueError("num_segments must be 2 or 3")
                
            else:
                q_goal_value = SIM_PARAMETERS['x0']
            
            opti.set_value(x0, pcc_arm.current_state)
            opti.set_value(q_goal, q_goal_value)
            opti.set_value(q0, pcc_arm.current_state)

            if t == 0:
                opti.set_initial(X, np.tile(pcc_arm.current_state.reshape(-1, 1), (1, N+1)))
                opti.set_initial(u, np.zeros((3*pcc_arm.num_segments, N))+0.5)
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

                pcc_arm.log_history(sol.value(u)[:,0],q_goal_value)
                pbar.update(SIM_PARAMETERS['dt'])
            except Exception:
                traceback.print_exc()
                break

    print("--- %s seconds ---" % (time.time() - start_time))
    history_plot(pcc_arm,MPC_PARAMETERS['u_bound'])

if __name__ == "__main__":
    main()
