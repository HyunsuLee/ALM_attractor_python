import numpy as np

def exp_smooth(x, dt, tau):
    T = 1000*dt # 왜? dt는 실제로 받는게 fs라는 값인데, fs=1000/dt다. 1sec안의 bin 수인 fs에 왜 또 1000을 곱하는가? 이해 안됨. 
    # exp smooth function자체에 대해 알아볼 필요가 있음.
    
    for m in range(np.shape(x)[1]):
        for n in range(1, np.shape(x)[0]):
            
            

