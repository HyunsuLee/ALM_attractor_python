# discrete_attractors.py implements code for generate Inagaki et al. 2019
# original author provided MATLAB code at github repository.
# https://github.com/fontolanl/ALM_attractors/blob/master/ALM_attractors_code/discrete_attractors.m

# load library
# preserve for scipy or numpy, activate gym environemt
import numpy as np
import util_functions as uf

# Network parameters

N_TRIALS = 20

# Membrane time constant
TAU_C = 100 # 정확히 뭔지 알아낼 것.
TAU_N = 100 # 마찬가지

DT = 1 # 1ms

T_END = 6000
T_VEC = np.arange(0, T_END, DT)

TAU_S = np.transpose([TAU_C, TAU_N, TAU_N, TAU_C])

BSLN_T = 1500/DT # 무슨 의미? sample period 중 auditory cue 주기 전 기간인듯.

T_DUR_STIM = 1000/DT
T_DUR_PV = 600/DT # pertubation인가?
T_DUR_DELAY = 2000/DT

# sample epoch start and end (coincides with start of delay epoch)
T_STIM_START = BSLN_T + 1 # DT얼마이든 1DT만큼 더 가서 한다는 의미?
T_STIM_END = T_STIM_START + T_DUR_STIM -1 

# Photostimulation
T_PV_START = T_STIM_END + 1
T_PV_END = T_PV_START + T_DUR_PV - 1

# End of Delay epoch
T_DELAY_END = T_STIM_END + T_DUR_DELAY
T_RAMP_START = T_STIM_START

# Value of ALPHA in the transduction function
# f(x) = ALPHA * log(1+exp(x/ALPHA))
ALPHA = 4.5

# network connectivity and parameters
# 1) 'one_hemi_multi_fp_step'
# 2) 'one_hemi_single_fp'
# 3) 'two_hemi_ramp' - RAMP_TYPE '2s' or 'step' must be selected
# 4) 'two_hemi_internal'

NETWORK_STR = 'one_hemi_multi_fp_step'
# RAMP_TYPE = '2s' ## for 'two_hemi_ramp' network architecture

if NETWORK_STR == 'one_hemi_multi_fp_step':
    # one hemi Fig1b right, EDF1p-v, supple table #1
    STIM_SIGMA = 0.1 # SD of selective cue intensity
    STIM_AMP = 0.8   # mean of selective cue intensity
        
    # amplitude of fast noise
    SIGMA_NOISE = 9
        
    # synaptic weights(changed according to supple tables)
    W_LL = 5.8
    W_RR = W_LL
    W_LI = 5 # pre:Inh post:Exc
    W_IL = 1.08 # pre:Exc post:Inh
    W_RI = W_LI
    W_IR = W_IL
    W_LR = 0.9 # between left-right Exc
    W_RL = W_LR
    W_II = 2
        
    # input currents
    I_L = 2.27
    I_R = I_L
    I_I = 1
    I_I_BSL = I_I # 뭔지 모르겠음.
    
    # cross-hemisphere excitation
    W_HEMI = 0.0
        
    # static nonlinearity parameters
    TAU_D = 0.14 # sec, depression recovery 
    TAU_F = 0.8 # sec, facilitation recovery 
    U = 0.05 # synaptic release probability 
    
    # Step-like ramp
    T_RAMP_END = T_RAMP_START + 10/DT
        
    # amplitude of ramping at the end of delay
    RAMP_AMP = 2
        
    # amplitude of PV perturbation
    PV_VEC = [0,1,2,6]

elif NETWORK_STR == 'one_hemi_single_fp':
    # one hemi Fig1b middel, EDF1i-o, supple table #2
    STIM_SIGMA = 0.05 # SD of selective cue intensity
    STIM_AMP = 0.5   # mean of selective cue intensity
        
    # amplitude of fast noise
    SIGMA_NOISE = 14
        
    # synaptic weights(changed according to supple tables)
    W_LL = 16
    W_RR = W_LL
    W_LI = 8 # pre:Inh post:Exc
    W_IL = 4 # pre:Exc post:Inh
    W_RI = W_LI
    W_IR = W_IL
    W_LR = 1 # between left-right Exc
    W_RL = W_LR
    W_II = 1.13
        
    # input currents
    I_L = 16
    I_R = I_L
    I_I = 1.07
    I_I_BSL = I_I # 뭔지 모르겠음.
    
    # cross-hemisphere excitation
    W_HEMI = 0.0
        
    # static nonlinearity parameters
    TAU_D = 0.0 # sec, depression recovery 
    TAU_F = 0.05 # sec, facilitation recovery 
    U = 0.05 # synaptic release probability 
    
    # Step-like ramp
    T_RAMP_END = T_RAMP_START + 10/DT
        
    # amplitude of ramping at the end of delay
    RAMP_AMP = 2
        
    # amplitude of PV perturbation
    PV_VEC = [0,0.4,2,12]

elif NETWORK_STR == 'two_hemi_ramp':
    # EDF9b-h - supple table#4(ramp 2s), EDF9p-q - supple table#5(ramp step)
    # amplitude of fast noise
    SIGMA_NOISE = 8
        
    # synaptic weights(changed according to supple tables)
    W_LL = 6.9
    W_RR = W_LL
    W_LI = 5 # pre:Inh post:Exc
    W_IL = 2.64 # pre:Exc post:Inh
    W_RI = W_LI
    W_IR = W_IL
    W_LR = 2.2 # between left-right Exc
    W_RL = W_LR
    W_II = 2
        
    # input currents
    I_L = 1
    I_R = I_L
    I_I = 1.2
    I_I_BSL = I_I # 뭔지 모르겠음.
    
    # cross-hemisphere excitation
    W_HEMI = 0.58
        
    # static nonlinearity parameters
    TAU_D = 0.1 # sec, depression recovery 
    TAU_F = 0.8 # sec, facilitation recovery 
    U = 0.1 # synaptic release probability 
    
    # amplitude of ramping at the end of delay
    RAMP_AMP = 4

    if RAMP_TYPE == '2s':
        STIM_SIGMA = 0.05 # SD of selective cue intensity
        STIM_AMP = 0.6   # mean of selective cue intensity    
        # Step-like ramp
        T_RAMP_END = T_RAMP_START + T_DUR_DELAY + T_DUR_STIM
        # amplitude of PV perturbation
        PV_VEC = [0,0.5,1,3.2]
    
    elif RAMP_TYPE == 'step':
        STIM_SIGMA = 0.01 # SD of selective cue intensity
        STIM_AMP = 1   # mean of selective cue intensity    
        # Step-like ramp
        T_RAMP_END = T_RAMP_START + 10/DT
        # amplitude of PV perturbation
        PV_VEC = [0,1.5,2.5,3.8,6.8]

    else:
        print('ERROR: no valid paratemer for ramping has been selected')
       
elif NETWORK_STR == 'two_hemi_internal':
    # EDF9i-o, supple table #6
    STIM_SIGMA = 0.01 # SD of selective cue intensity
    STIM_AMP = 0.4   # mean of selective cue intensity
        
    # amplitude of fast noise
    SIGMA_NOISE = 9

    # change presample epoch duration
    # already stated above but if you want to change the condition, do here.
    BSLN_T = 1500/DT 

    T_DUR_STIM = 1000/DT
    T_DUR_PV = 600/DT
    T_DUR_DELAY = 2000/DT

    # sample epoch start and end (coincies with start of delay epoch)
    # also, already stated above
    T_STIM_START = BSLN_T + 1
    T_STIM_END = T_STIM_START + T_DUR_STIM - 1

    # Photostimulation
    T_PV_START = T_STIM_END + 1
    T_PV_END = T_PV_START + T_DUR_PV - 1

    # End of Delay epoch
    T_DELAY_END = T_STIM_END + T_DUR_DELAY
    T_RAMP_START = T_STIM_START

    # synaptic weights(changed according to supple tables)
    W_LL = 9.75
    W_RR = W_LL
    W_LI = 2.5 # pre:Inh post:Exc
    W_IL = 4 # pre:Exc post:Inh
    W_RI = W_LI
    W_IR = W_IL
    W_LR = 2.5 # between left-right Exc
    W_RL = W_LR
    W_II = 1
        
    # input currents
    I_L = 2.3 # or 2
    I_R = I_L
    I_I = 1
    I_I_BSL = I_I # 뭔지 모르겠음.
    
    # cross-hemisphere excitation
    W_HEMI = 0.4
        
    # static nonlinearity parameters
    TAU_D = 0.12 # sec, depression recovery 
    TAU_F = 0.1 # sec, facilitation recovery 
    U = 0.1 # synaptic release probability 
    
    ALPHA = 2.5

    # Step-like ramp
    T_RAMP_END = T_RAMP_START + 10/DT
        
    # amplitude of ramping at the end of delay
    RAMP_AMP = 3.15
        
    # amplitude of PV perturbation
    PV_VEC = [0,1.3,2.3,7]

else:
    print('ERROR: no valid network structure has been selected')

# Perturbation type
PV_TYPE = 'bilateral'

# Stimulus smooth
SIM_LEN = int(T_END/DT)
TAU_EXT = 20 # ms

STIM_T = np.zeros((SIM_LEN, 1))
SMOOTHED_START = int(T_STIM_START + TAU_EXT/DT)
SMOOTHED_END = int(T_STIM_END - TAU_EXT/DT)
SMOOTHED_DUR = int(SMOOTHED_END-SMOOTHED_START) 
STIM_T[SMOOTHED_START:SMOOTHED_END] = np.ones((SMOOTHED_DUR, 1))

STIM_SMOOTHED = uf.exp_smooth(STIM_T, DT, TAU_EXT)

# PV perturbation

STIM_T = np.zeros((SIM_LEN, 1))
PV_SMOOTHED_START = int(T_PV_START)
PV_SMOOTHED_END = int(T_PV_END - TAU_EXT/DT)
PV_SMOOTHED_DUR = int(PV_SMOOTHED_END - PV_SMOOTHED_START)
STIM_T[PV_SMOOTHED_START:PV_SMOOTHED_END] = np.ones((PV_SMOOTHED_DUR, 1))

PV_SMOOTHED = uf.exp_smooth(STIM_T, DT, TAU_EXT)

# ramping stimulus
RAMP_TEMP = (RAMP_AMP/((T_RAMP_END-T_RAMP_START)/1000)) * np.ones((1,4))
RAMP_VEC = np.zeros((SIM_LEN, 4))
for t_idx in range(int(T_RAMP_START-1), int(T_RAMP_END)):
    RAMP_VEC[t_idx,:] = RAMP_TEMP * ((t_idx + 1 - T_RAMP_START) /1000)
RAMP_VEC[int(T_RAMP_END):, :]= RAMP_VEC[int(T_RAMP_END-1),:]

# Dynamics
PV_idx = 0
h_mat = np.zeros((len(PV_VEC), N_TRIALS, len(T_VEC),4))
r_mat = np.zeros((len(PV_VEC), N_TRIALS, len(T_VEC),4))

for PV_intensity in PV_VEC:
    
    # Scaling weigths
    Scaled_W_LL = W_LL - (W_IL*W_LI)/(1+W_II)
    Scaled_W_RR = W_RR - (W_IR*W_RI)/(1+W_II)
    Scaled_W_LR = W_LR - (W_LI*W_IR)/(1+W_II)
    Scaled_W_RL = W_RL - (W_RI*W_IL)/(1+W_II)

    # Scaling input currents
    Scaled_I_L = I_L - (W_LI*I_I)/(1+W_II)
    Scaled_I_R = I_R - (W_RI*I_I)/(1+W_II)

    # Synaptic Matrix
    P_MATRIX = np.array([[Scaled_W_LL, Scaled_W_LR, 0, 0], \
                        [Scaled_W_RL, Scaled_W_RR, 0, 0], \
                        [0, 0, Scaled_W_RR, Scaled_W_RL], \
                        [0, 0, Scaled_W_LR, Scaled_W_LL]])

    INPUT_MAT = np.zeros((4, int(T_END/DT), N_TRIALS))

    # Noise
    noise_input = np.zeros((4, int(T_END/DT), N_TRIALS))
    for trial in range(0, N_TRIALS):
        stim_amp_distr = STIM_AMP + STIM_SIGMA*np.random.randn()
        
        ramp_in = RAMP_VEC

        if trial < N_TRIALS/2:
            # for contralateral trial
            stim_in = stim_amp_distr * np.array([1,0,1,0]) * STIM_SMOOTHED
            if NETWORK_STR == 'one_hemi_single_fp':
                ramp_in[:,1] = 0
        else:
            # for ipsilateral trial
            stim_in = stim_amp_distr * np.array([0,1,0,1]) * STIM_SMOOTHED
            if NETWORK_STR == 'one_hemi_single_fp':
                ramp_in[:,0] = 0
        
        PV_I_I = I_I + PV_intensity * I_I * PV_SMOOTHED
        
        # Scaling weights
        I_L_PV = I_L - (W_LI * PV_I_I)/(1+W_II)
        I_R_PV = I_R - (W_RI * PV_I_I)/(1+W_II)
        
        if PV_TYPE == 'bilateral':
            PV_in = np.concatenate((I_L_PV, I_R_PV, I_R_PV, I_L_PV), axis = 1)
        elif PV_TYPE == 'ipsilateral':
            PV_in = np.concatenate((I_L_PV, I_R_PV, Scaled_I_R * np.ones((SIM_LEN, 1)),\
                 Scaled_I_L * np.ones((SIM_LEN, 1))), axis = 1)
        elif PV_TYPE == 'contralateral':
            PV_in = np.concatenate((Scaled_I_L * np.ones((SIM_LEN, 1)), Scaled_I_R * np.ones((SIM_LEN, 1)),\
                 I_R_PV, I_L_PV), axis = 1)
        
        # Dynamics
        y = np.zeros((SIM_LEN, 4)) # applied sigmoid function
        h = np.zeros((SIM_LEN, 4)) # h_i(t) value
        r = np.zeros((SIM_LEN, 4))
        h[0,:] = np.array([1,1,1,1])
        
        for t_idx in range(int(T_END/DT)-1):
            y[t_idx,:] = ALPHA * np.log(1+np.exp(h[t_idx,:]/ALPHA))
            noise_input[:, t_idx, trial] = SIGMA_NOISE * np.random.randn(4)
            INPUT_MAT[:, t_idx, trial] = np.matmul(P_MATRIX, uf.h_static(y[t_idx, :], TAU_D, TAU_F, U)) + \
                 W_HEMI * uf.h_static(np.array([y[t_idx,2], y[t_idx,3], y[t_idx, 0], y[t_idx, 1]]), TAU_D, TAU_F, U) + \
                     PV_in[t_idx, :] + noise_input[:, t_idx, trial] + stim_in[t_idx, :] + ramp_in[t_idx,:]

            # euler method
            h[t_idx + 1, :] = h[t_idx, :] + (-h[t_idx, :] + INPUT_MAT[:, t_idx, trial])/TAU_S*DT
            r[t_idx + 1, :] = (uf.h_static(y[t_idx, :], TAU_D, TAU_F, U) / y[t_idx, :]) * ALPHA * \
                np.log(1 + np.exp(h[t_idx, :]/ALPHA))
        
        h_mat[PV_idx, trial, :, :] = h
        r_mat[PV_idx, trial, :, :] = r

    PV_idx += 1

[correct_trials_right, correct_trials_left, r_right_pj_c, r_left_pf_c] = \
    uf.ALM_attractor_figs(r_mat, PV_VEC, T_VEC, DT, T_STIM_START, T_STIM_END, T_DELAY_END, N_TRIALS)

