import numpy as np
import matplotlib.pyplot as plt 

def exp_smooth(x, dt, tau):
    '''
    making step-wise function into exponential growth and decay
    x : step-wise function to make change
    dt : time step
    tau : time contant used in exponential growth and decay
    return changed function
    '''
    for m in range(np.shape(x)[1]):
        for n in range(1, np.shape(x)[0]):
            x[n,m] = x[n,m] + (x[n-1, m] - x[n,m])*np.exp(-(dt/tau))
    
    return x
            
def h_static(y, TAU_D, TAU_F, U):
    '''
    recevie sigmoid transduction function g(h_i) and making G_E(h_i)
    mimicking the effect of short-term synaptic plasiticity
    y : sigmoid activation function values at each time step
    TAU_D : depression recovery time constant
    TAU_F : facilitation recovery time constant
    U : synaptic release probability
    return G_E(h_i), see equation (5) in the original paper
    '''
    u_hat = (U + U * TAU_F * y) / (1 + U * TAU_F * y)
    x_hat = 1 / (1 + u_hat * TAU_D * y)

    y_new = u_hat * x_hat * y
    return y_new

def ALM_attractor_figs(r_mat, pv_vec, t_vec, dt, t_stim_start, t_stim_end, t_delay_end, n_trials):
    '''
    plot figure in Inagki et al(2019)
    '''
    t_vec_plt = (t_vec - t_delay_end * dt)/1000
    
    bin_3 = np.nonzero(t_vec_plt < -1.25)[0][-1]
    bin_4 = np.nonzero(t_vec_plt < -0.05)[0][-1]
    
    right_trials = np.arange(int(n_trials/2))
    left_trials = np.arange(int(n_trials/2), int(n_trials))

    N_right_trials = len(right_trials)
    N_left_trials = len(left_trials)
    
    correct_trials_right  = []
    correct_trials_left = []

    correct_trials_right.append(np.nonzero(r_mat[0, right_trials, int(t_delay_end), 0]>r_mat[0, right_trials, int(t_delay_end), 1])[0])
    correct_trials_left.append(int(n_trials/2) + np.nonzero(r_mat[0, left_trials, int(t_delay_end), 0]<r_mat[0, left_trials, int(t_delay_end), 1])[0])

    ## Compute Coding direction
    # smoothing window
    win_ms = 100

    r_CD = 0 # TODO
    print(correct_trials_left[0])
    
    correct_trials_right = 0
    correct_trials_left = 0
    r_right_pj_c = 0
    r_left_pj_c = 0
    return [correct_trials_right, correct_trials_left, r_right_pj_c, r_left_pj_c]
