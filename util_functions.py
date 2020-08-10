import numpy as np

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
