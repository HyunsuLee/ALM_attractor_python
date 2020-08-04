import numpy as np

def exp_smooth(x, dt, tau):
    
    for m in range(np.shape(x)[1]):
        for n in range(1, np.shape(x)[0]):
            x[n,m] = x[n,m] + (x[n-1, m] - x[n,m])*np.exp(-(dt/tau))
    
    return x
            

