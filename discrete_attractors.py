# discrete_attractors.py implements code for generate Inagaki et al. 2019
# original author provided MATLAB code at github repository.

# load library
# preserve for scipy or numpy, activate gym environemt
import numpy as np

# Network parameters

N_TRIALS = 1000

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

NETWORK_STR = 'one_hemi_single_fp'
# RAMP_TYPE = '2s' ## for 'two_hemi_ramp' network architecture

if NETWORK_STR == 'one_hemi_multi_fp_step':
    # one hemi Fig1b right, EDF1p-v
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
    TAU_f = 0.8 # sec, facilitation recovery 
    U = 0.05 # synaptic release probability 
    
    # Step-like ramp
    T_RAMP_END = T_RAMP_START + 10/DT
        
    # amplitude of ramping at the end of delay
    RAMP_AMP = 2
        
    # amplitude of PV perturbation
    PV_VEC = [0,1,2,6]

elif NETWORK_STR == 'one_hemi_single_fp':
    # one hemi Fig1b right, EDF1p-v
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
    TAU_f = 0.05 # sec, facilitation recovery 
    U = 0.05 # synaptic release probability 
    
    # Step-like ramp
    T_RAMP_END = T_RAMP_START + 10/DT
        
    # amplitude of ramping at the end of delay
    RAMP_AMP = 2
        
    # amplitude of PV perturbation
    PV_VEC = [0,0.4,2,12]

 
print(PV_VEC)