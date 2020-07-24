# discrete_attractors.py implements code for generate Inagaki et al. 2019
# original author provided MATLAB code at github repository.

# load library
# preserve for scipy or numpy, activate gym environemt
import numpy as np

# Network parameters

N_trials = 1000

# Membrane time constant
tau_c = 100 # 정확히 뭔지 알아낼 것.
tau_n = 100 # 마찬가지

dt = 1 # 1ms

t_end = 6000
t_vec = np.arange(0, t_end, dt)

taus = np.transpose([tau_c, tau_n, tau_n, tau_c])

bsln_t = 1500/dt # 무슨 의미? sample period 중 auditory cue 주기 전 기간인듯.

t_dur_stim = 1000/dt
t_dur_PV = 600/dt # pertubation인가?
t_dur_delay = 2000/dt

# sample epoch start and end (coincides with start of delay epoch)
t_stim_start = bsln_t + 1 # dt얼마이든 1dt만큼 더 가서 한다는 의미?
t_stim_end = t_stim_start + t_dur_stim -1 

# Photostimulation
t_PV_start = t_stim_end + 1
t_PV_end = t_PV_start + t_dur_PV - 1

# End of Delay epoch
t_delay_end = t_stim_end + t_dur_delay

t_ramp_start = t_stim_start

# Value of alpha in the transduction function
# f(x) = alpha * log(1+exp(x/alpha))
alpha = 4.5

# network connectivity and parameters
# 1) 'one_hemi_multi_fp_step'
# 2) 'one_hemi_single_fp'
# 3) 'two_hemi_ramp' - ramp_type '2s' or 'step' must be selected
# 4) 'two_hemi_internal'

network_str = 'one_hemi_multi_fp_step'
# ramp_type = '2s' ## for 'two_hemi_ramp' network architecture

if network_str == 'one_hemi_multi_fp_step':
    # one hemi Fig1b right, EDF1p-v
    stim_sigma = 0.1 # SD of selective cue intensity
    stim_amp = 0.8   # mean of selective cue intensity
        
    # amplitude of fast noise
    sigma_noise = 9
        
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
    i_L = 2.27
    i_R = i_L
    i_I = 1
    iI_bsl = i_I # 뭔지 모르겠음.
    
    # cross-hemisphere excitation
    JH = 0.0
        
    # static nonlinearity parameters
    tau_D = 0.14 # sec, depression recovery 
    tau_f = 0.8 # sec, facilitation recovery 
    U = 0.05 # synaptic release probability 
    
    # Step-like ramp
    t_ramp_end = t_ramp_start + 10/dt
        
    # amplitude of ramping at the end of delay
    ramp_amp = 2
        
    # amplitude of PV perturbation
    PV_vec = [0,1,2,6]


 
