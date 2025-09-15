import casadi as ca
import numpy as np

# Parameters
from pcc_arm import PCCSoftArm
from parameters import ARM_PARAMETERS, MPC_PARAMETERS, SIM_PARAMETERS

def main():
    N=MPC_PARAMETERS['N']

    pcc_arm = PCCSoftArm(
        L_segs = ARM_PARAMETERS['L_segs'],
        m = ARM_PARAMETERS['m'],
        d_eq = ARM_PARAMETERS['d_eq'],
        K = ARM_PARAMETERS['K']
    )
    pcc_arm.current_state=SIM_PARAMETERS['x0']
    pcc_arm.history.append(pcc_arm.current_state)

    opti= ca.Opti()

    #Declare decision variables
    X= opti.variable(12,N+1)
    #q = X[:6,:]
    #q_dot = X[6:,:]
    u = opti.variable(6,N+1)
    q_goal = opti.parameter(12)
    m_par   = opti.parameter()
    d_par   = opti.parameter(3)
    K_par   = opti.parameter(6, 6)

    # Create integrator
    pcc_arm.create_integrator(SIM_PARAMETERS['dt'])
    F = pcc_arm.integrator
    #F = F.expand() # may be faster but needs more memory 

    # Objective
    objective = 0
    for i in range(N):
        objective += ca.mtimes([(X[:,i]-q_goal).T, MPC_PARAMETERS['Q'], (X[:,i]-q_goal)]) + ca.mtimes([u[:,i].T, MPC_PARAMETERS['R'], u[:,i]])
    
    objective += ca.mtimes([(X[:,N]-q_goal).T, MPC_PARAMETERS['Qf'], (X[:,N]-q_goal)]) #final cost

    opti.minimize(objective)

    # Constraints
    opti.subject_to(X[:,0] == SIM_PARAMETERS['x0']) # initial condition
    for i in range(N):
        p_i  = ca.vertcat(u[:, i], m_par, d_par, ca.reshape(K_par, 36, 1))
        opti.subject_to(X[:, i+1] ==F(x0=X[:, i], p=p_i)['xf']) # system dynamics
        opti.subject_to(opti.bounded(-MPC_PARAMETERS['u_bound'], u[:, i], MPC_PARAMETERS['u_bound'])) # input constraints

    #solver
    opti.solver('ipopt')

    # Simu loop
    for t in range(int(SIM_PARAMETERS['T']/SIM_PARAMETERS['dt'])):
        print(f"Time: {t*SIM_PARAMETERS['dt']:.2f}s / {SIM_PARAMETERS['T']}s", end='\r')

        # set the goal
        if t*SIM_PARAMETERS['dt'] > SIM_PARAMETERS['T']/3:
            q_goal_value = np.array([
                0, np.deg2rad(-90), 0, np.deg2rad(+120), 0, np.deg2rad(-120),
                0, 0, 0, 0, 0, 0
            ])
        else:
            q_goal_value = SIM_PARAMETERS['x0']
        
        opti.set_value(q_goal, q_goal_value)
        opti.set_value(m_par, ARM_PARAMETERS['m'])
        opti.set_value(d_par, ARM_PARAMETERS['d_eq'])
        opti.set_value(K_par, ARM_PARAMETERS['K'])

        # initial guess
        if t == 0:
            opti.set_initial(X, np.tile(SIM_PARAMETERS['x0'].reshape(-1,1), N+1))
            opti.set_initial(u, np.zeros((6,N+1)))
        else:
            opti.set_initial(X, np.hstack((sol.value(X)[:,1:], sol.value(X)[:,-1:])))
            opti.set_initial(u, np.hstack((sol.value(u)[:,1:], sol.value(u)[:,-1:])))
        
        # solve the problem
        sol = opti.solve()

        # apply the first control input to the real system
        pcc_arm.next_step(sol.value(u)[:,0])

        pcc_arm.log_history(sol.value(u)[:,0],q_goal_value)

    print("Finished ")


if __name__ == "__main__":
    main()
