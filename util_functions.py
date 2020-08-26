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

    CD_rt_1st = smooth(np.squeeze(np.mean(np.squeeze(r_mat[0, correct_trials_right[0], :, 0]), axis = 0)), np.round(win_ms/dt))
    CD_lt_1st = smooth(np.squeeze(np.mean(np.squeeze(r_mat[0, correct_trials_left[0], :, 0]), axis = 0)), np.round(win_ms/dt))
    CD_rt_2nd = smooth(np.squeeze(np.mean(np.squeeze(r_mat[0, correct_trials_right[0], :, 1]), axis = 0)), np.round(win_ms/dt))
    CD_lt_2nd = smooth(np.squeeze(np.mean(np.squeeze(r_mat[0, correct_trials_left[0], :, 1]), axis = 0)), np.round(win_ms/dt))
    r_CD = np.array([CD_rt_1st - CD_lt_1st, CD_rt_2nd - CD_lt_2nd])

    CD_delay = np.mean(r_CD[:, int((t_delay_end-400)/dt):int(t_delay_end)], axis = 1)

    ## Compute endpoints and projected trajectories
    r_right_c = []
    r_left_c = []
    r_right_pj_c = []
    r_left_pj_c = []

    r_right_c.append(np.squeeze(r_mat[0, correct_trials_right[0], :, 0:2]))
    r_left_c.append(np.squeeze(r_mat[0, correct_trials_left[0], :, 0:2]))

    r_right_pj_c.append(np.zeros((len(correct_trials_right[0]), len(t_vec))))
    r_left_pj_c.append(np.zeros((len(correct_trials_left[0]), len(t_vec))))

    endpoints_right = np.zeros((len(correct_trials_right[0]), 1))
    endpoints_left = np.zeros((len(correct_trials_left[0]), 1))
    
    delay_endpoint_start = int((t_delay_end-400)/dt)
    delay_endpoint_end = int(t_delay_end)

    for idx in range(len(endpoints_right)):
        endpoints_right[idx] = np.mean(np.matmul(r_right_c[0][idx][delay_endpoint_start:delay_endpoint_end][:], \
            CD_delay))
        r_right_pj_c[0][idx][:] = np.matmul(r_right_c[0][idx][:][:], CD_delay)

    for idx in range(len(endpoints_left)):
        endpoints_left[idx] = np.mean(np.matmul(r_left_c[0][idx][delay_endpoint_start:delay_endpoint_end][:], \
            CD_delay))
        r_left_pj_c[0][idx][:] = np.matmul(r_left_c[0][idx][:][:], CD_delay)      
        
    # Median of endpoints
    right_norm = np.median(endpoints_right)
    left_norm = np.median(endpoints_left)
    
    # Normalization
    r_right_pj_c[0] = (r_right_pj_c[0] - left_norm)/(right_norm - left_norm)
    r_left_pj_c[0] = (r_left_pj_c[0] - left_norm)/(right_norm - left_norm)

    # Smooth the vectors
    r_smooth_right_pj = []
    r_smooth_left_pj = []
    r_smooth_right_pj_std = []
    r_smooth_left_pj_std = []

    r_smooth_right_pj.append(smooth(np.mean(r_right_pj_c[0], axis = 0), np.round(win_ms/dt)))
    r_smooth_left_pj.append(smooth(np.mean(r_left_pj_c[0], axis = 0), np.round(win_ms/dt)))

    r_smooth_right_pj_std.append(smooth(np.std(r_right_pj_c[0], axis = 0), np.round(win_ms/dt)))
    r_smooth_left_pj_std.append(smooth(np.std(r_left_pj_c[0], axis = 0), np.round(win_ms/dt)))

    Data_at_each_bin = []
    Data_at_each_bin.append(np.zeros((2,2)))

    Data_at_each_bin[0][0,0] = r_smooth_right_pj[0][bin_3]
    Data_at_each_bin[0][0,1] = r_smooth_right_pj[0][bin_4]
    Data_at_each_bin[0][1,0] = r_smooth_left_pj[0][bin_3]
    Data_at_each_bin[0][1,1] = r_smooth_left_pj[0][bin_4]

    mean_proj = []
    mean_proj.append(r_smooth_right_pj[0])
    mean_proj.append(r_smooth_left_pj[0])
    
    # generate the unperturbed correct trials fig
    fig1, ax1 = plt.subplots()

    ax1.set_xlabel('Time to movement onset (s)')
    ax1.set_ylabel('Proj. to CD')
    ax1.set_title('Unperturbed correct trials')
    ax1.plot(t_vec_plt, r_smooth_right_pj[0], color = 'tab:blue')
    ax1.fill_between(t_vec_plt, r_smooth_right_pj[0] - r_smooth_right_pj_std[0], \
        r_smooth_right_pj[0] + r_smooth_right_pj_std[0], alpha =0.2)
    ax1.plot(t_vec_plt, r_smooth_left_pj[0], color = 'tab:red')
    ax1.fill_between(t_vec_plt, r_smooth_left_pj[0] - r_smooth_left_pj_std[0], \
        r_smooth_left_pj[0] + r_smooth_left_pj_std[0], alpha =0.2)

    ax1.axvline((t_stim_start - t_delay_end)*dt/1000, color = "black", linestyle = "--")
    ax1.axvline((t_stim_end - t_delay_end)*dt/1000, color = "black", linestyle = "--")
    ax1.axvline(0, color = "black", linestyle = "--")
    
    fig1.tight_layout()

    plt.savefig('./figures/unperturbed_correct_trial.png')
    plt.close()
    
    ## Distribution of end points - unperturbed correct trials fig

    hist_rt, bin_rt = np.histogram(r_right_pj_c[0][:, int(t_delay_end)], bins = \
        np.arange(-1.5, 2.5, 0.1) + 0.05)
    hist_rt_center = bin_rt[0:-1] + (bin_rt[1] -bin_rt[0])/2
    hist_rt_width = (bin_rt[1] - bin_rt[0])*0.8
    
    hist_lt, bin_lt = np.histogram(r_left_pj_c[0][:, int(t_delay_end)], bins = \
        np.arange(-1.5, 2.5, 0.1) + 0.05)
    hist_lt_center = bin_lt[0:-1] + (bin_lt[1] - bin_lt[0])/2
    hist_lt_width = (bin_lt[1] - bin_lt[0])*0.8

    fig2, ax2 = plt.subplots()
    
    ax2.bar(hist_rt_center, hist_rt/N_right_trials, align= 'center',width = hist_rt_width, \
        color = 'tab:blue', label = 'right')
    ax2.bar(hist_lt_center, hist_lt/N_left_trials, align = 'center', width = hist_lt_width, \
        color = 'tab:red', label = 'left')
    
    ax2.set_xlabel('Fraction of correct trials (unperturbed)')
    ax2.set_ylabel('Proj. to CD')
    ax2.set_title('Endpoint distribution - unpertubed correct trials')
    ax2.legend()
    fig2.tight_layout()

    plt.savefig('./figures/unperturbed_correct_trial_hist.png')
    plt.close()
    
    # Projected Activity - Perturbed correct trials

    fig3, ax3 = plt.subplots()

    for pv_idx in range(1, len(pv_vec)):
        correct_trials_right.append(np.nonzero(r_mat[pv_idx, right_trials, int(t_delay_end), 0] > \
             r_mat[pv_idx, right_trials, int(t_delay_end), 1])[0])
        correct_trials_left.append(int(n_trials/2) + np.nonzero(r_mat[pv_idx, left_trials, int(t_delay_end), 0] < \
             r_mat[pv_idx, left_trials, int(t_delay_end), 1])[0])
        
        r_right_c.append(np.squeeze(r_mat[pv_idx, correct_trials_right[pv_idx], :, 0:2]))
        r_left_c.append(np.squeeze(r_mat[pv_idx, correct_trials_left[pv_idx], :, 0:2]))

        r_right_pj_c.append(np.zeros((len(correct_trials_right[pv_idx]), len(t_vec))))
        r_left_pj_c.append(np.zeros((len(correct_trials_left[pv_idx]), len(t_vec))))

        for idx in range(len(correct_trials_right[pv_idx])):
            r_right_pj_c[pv_idx][idx, :] = (np.matmul(r_right_c[pv_idx][idx][:][:], CD_delay) - left_norm) / \
                (right_norm - left_norm)

        for idx in range(len(correct_trials_left[pv_idx])):
            r_left_pj_c[pv_idx][idx, :] = (np.matmul(r_left_c[pv_idx][idx][:][:], CD_delay) - left_norm) / \
                (right_norm - left_norm)
        
        r_smooth_right_pj.append(smooth(np.mean(r_right_pj_c[pv_idx], axis = 0), np.round(win_ms/dt)))
        r_smooth_left_pj.append(smooth(np.mean(r_left_pj_c[pv_idx], axis = 0), np.round(win_ms/dt)))

        ax3.plot(t_vec_plt, r_smooth_right_pj[pv_idx], \
            color = (0.1, 0.3, (6 - pv_idx)/(len(pv_vec) + 3), (8 - pv_idx)/(len(pv_vec) + 4)))
        ax3.plot(t_vec_plt, r_smooth_left_pj[pv_idx], \
            color = ((6 - pv_idx)/(len(pv_vec) + 3), 0.1, 0.4, (8 - pv_idx)/(len(pv_vec) + 4)))
        
        Data_at_each_bin.append(np.zeros((2,2)))
        Data_at_each_bin[pv_idx][0,0] = r_smooth_right_pj[pv_idx][bin_3]
        Data_at_each_bin[pv_idx][0,1] = r_smooth_right_pj[pv_idx][bin_4]
        Data_at_each_bin[pv_idx][1,0] = r_smooth_left_pj[pv_idx][bin_3]
        Data_at_each_bin[pv_idx][1,1] = r_smooth_left_pj[pv_idx][bin_4]

        mean_proj.append(r_smooth_right_pj[pv_idx])
        mean_proj.append(r_smooth_left_pj[pv_idx])
    
    ax3.plot(t_vec_plt, r_smooth_right_pj[0], color = 'tab:blue', linestyle = '--')
    ax3.plot(t_vec_plt, r_smooth_left_pj[0], color = 'tab:red', linestyle = '--')

    ax3.axvline((t_stim_start - t_delay_end)*dt/1000, color = "black", linestyle = "--")
    ax3.axvline((t_stim_end - t_delay_end)*dt/1000, color = "black", linestyle = "--")
    ax3.axvline(0, color = "black", linestyle = "--")

    ax3.set_ylabel('Proj. to CD')
    ax3.set_xlabel('Time to movement onset (s)')
    ax3.set_title('Perturbed correct trials')
    fig3.tight_layout()
    plt.savefig('./figures/perturbed_correct.png')
    plt.close()
    
    r_right_pj_c_mat = np.concatenate(r_right_pj_c[1:], axis = 0)
    r_left_pj_c_mat = np.concatenate(r_left_pj_c[1:], axis = 0)

    hist_pv_rt, bin_pv_rt = np.histogram(r_right_pj_c_mat[:, int(t_delay_end)], bins = \
        np.arange(-1.5, 2.5, 0.1) + 0.05)
    hist_pv_rt_center = bin_pv_rt[0:-1] + (bin_pv_rt[1] - bin_pv_rt[0])/2
    hist_pv_rt_width = (bin_pv_rt[1] - bin_pv_rt[0])*0.8
    
    hist_pv_lt, bin_pv_lt = np.histogram(r_left_pj_c_mat[:, int(t_delay_end)], bins = \
        np.arange(-1.5, 2.5, 0.1) + 0.05)
    hist_pv_lt_center = bin_pv_lt[0:-1] + (bin_pv_lt[1] - bin_pv_lt[0])/2
    hist_pv_lt_width = (bin_pv_lt[1] - bin_pv_lt[0])*0.8

    fig4, ax4 = plt.subplots()
    
    ax4.bar(hist_pv_rt_center, hist_pv_rt/np.shape(r_right_pj_c_mat)[0], align= 'center', \
        width = hist_pv_rt_width, color = 'tab:blue', label = 'right')
    ax4.bar(hist_pv_lt_center, hist_pv_lt/np.shape(r_left_pj_c_mat)[0], align = 'center', \
        width = hist_pv_lt_width, color = 'tab:red', label = 'left')
    
    ax4.set_xlabel('Fraction of correct trials')
    ax4.set_ylabel('Proj. to CD')
    ax4.set_title('Endpoint distribution - pertubed correct trials')
    ax4.legend()
    fig4.tight_layout()

    plt.savefig('./figures/perturbed_correct_trial_hist.png')
    plt.close()

    # Projected Activity - Perturbed error trials

    fig5, ax5 = plt.subplots()

    error_trials_right = [0] # to pass zero perturbation intensity in error trials 
    error_trials_left = [0]
    
    r_right_e = [0]
    r_left_e = [0]

    r_right_pj_e = [0]
    r_left_pj_e = [0]

    r_right_pj_e_avg = [0]
    r_left_pj_e_avg = [0]

    mean_cor_proj = []

    for pv_idx in range(1, len(pv_vec)):
        error_trials_right.append(np.nonzero(r_mat[pv_idx, right_trials, int(t_delay_end), 0] < \
             r_mat[pv_idx, right_trials, int(t_delay_end), 1])[0])
        error_trials_left.append(int(n_trials/2) + np.nonzero(r_mat[pv_idx, left_trials, int(t_delay_end), 0] > \
             r_mat[pv_idx, left_trials, int(t_delay_end), 1])[0])

        r_right_e.append(np.squeeze(r_mat[pv_idx, error_trials_right[pv_idx], :, 0:2])) 
        r_left_e.append(np.squeeze(r_mat[pv_idx, error_trials_left[pv_idx], :, 0:2]))

        r_right_pj_e.append(np.zeros((len(error_trials_right[pv_idx]), len(t_vec)))) 
        r_left_pj_e.append(np.zeros((len(error_trials_left[pv_idx]), len(t_vec))))

        r_right_pj_e_avg.append(np.zeros((len(t_vec), 1)))
        r_left_pj_e_avg.append(np.zeros((len(t_vec), 1)))

        for idx in range(len(error_trials_right[pv_idx])):
            r_right_pj_e[pv_idx][idx, :] = (np.matmul(r_right_e[pv_idx][idx][:][:], CD_delay) - left_norm) / \
                (right_norm - left_norm)

        for idx in range(len(error_trials_left[pv_idx])):
            r_left_pj_e[pv_idx][idx, :] = (np.matmul(r_left_e[pv_idx][idx][:][:], CD_delay) - left_norm) / \
                (right_norm - left_norm)

        if len(error_trials_right[pv_idx]) > 1:
            r_right_pj_e_avg.append(smooth(np.mean(r_right_pj_e[pv_idx], axis = 0), np.round(win_ms/dt)))
            ax5.plot(t_vec_plt, r_right_pj_e_avg[pv_idx], \
                color = (0.1, 0.3, (6 - pv_idx)/(len(pv_vec) + 3), (8 - pv_idx)/(len(pv_vec) + 4)))
        elif len(error_trials_right[pv_idx]) == 1:
            r_right_pj_e_avg.append(smooth(np.squeeze(r_right_pj_e[pv_idx], axis=0), np.round(win_ms/dt)))
        
        if len(error_trials_left[pv_idx]) > 1:
            r_left_pj_e_avg.append(smooth(np.mean(r_left_pj_e[pv_idx], axis = 0), np.round(win_ms/dt)))
            ax5.plot(t_vec_plt, r_left_pj_e_avg[pv_idx], \
                color = ((6 - pv_idx)/(len(pv_vec) + 3), 0.1, 0.4, (8 - pv_idx)/(len(pv_vec) + 4)))
        elif len(error_trials_left[pv_idx]) == 1:
            r_left_pj_e_avg.append(smooth(np.squeeze(r_left_pj_e[pv_idx], axis=0), np.round(win_ms/dt)))

        Data_at_each_bin.append(np.zeros((2,2)))
        Data_at_each_bin[pv_idx + 3][0,0] = r_right_pj_e_avg[pv_idx][bin_3]
        Data_at_each_bin[pv_idx + 3][0,1] = r_right_pj_e_avg[pv_idx][bin_4]
        Data_at_each_bin[pv_idx + 3][1,0] = r_left_pj_e_avg[pv_idx][bin_3]
        Data_at_each_bin[pv_idx + 3][1,1] = r_left_pj_e_avg[pv_idx][bin_4]
        mean_proj.append(r_right_pj_e_avg[pv_idx])
        mean_proj.append(r_left_pj_e_avg[pv_idx])

        bin_pre_stim = np.nonzero(t_vec_plt>-2.05)[0][0]

        cor_right = r_right_pj_e[pv_idx][:, bin_pre_stim] > 0.5
        cor_left = r_left_pj_e[pv_idx][:, bin_pre_stim] < 0.5
        
        r_right_temp = smooth(np.mean(r_right_pj_e[pv_idx][cor_right, :], axis = 0), np.round(win_ms/dt))
        r_left_temp = smooth(np.mean(r_left_pj_e[pv_idx][cor_left, :], axis = 0), np.round(win_ms/dt))

        right_data_tmp = r_right_temp[::int(1/dt)]
        left_data_tmp = r_left_temp[::int(1/dt)]

        mean_cor_proj.append(right_data_tmp)
        mean_cor_proj.append(left_data_tmp)

    ax5.plot(t_vec_plt, r_smooth_right_pj[0], color = 'tab:blue', linestyle = '--')
    ax5.plot(t_vec_plt, r_smooth_left_pj[0], color = 'tab:red', linestyle = '--')

    ax5.axvline((t_stim_start - t_delay_end)*dt/1000, color = "black", linestyle = "--")
    ax5.axvline((t_stim_end - t_delay_end)*dt/1000, color = "black", linestyle = "--")
    ax5.axvline(0, color = "black", linestyle = "--")

    ax5.set_ylabel('Proj. to CD')
    ax5.set_xlabel('Time to movement onset (s)')
    ax5.set_title('Perturbed error trials')
    fig5.tight_layout()
    
    plt.savefig('./figures/perturbed_error.png')
    plt.close()
    
    # Error trial histogram
    r_right_pj_e_mat = np.concatenate(r_right_pj_e[1:], axis = 0)
    r_left_pj_e_mat = np.concatenate(r_left_pj_e[1:], axis = 0)

    hist_pv_rt, bin_pv_rt = np.histogram(r_right_pj_e_mat[:, int(t_delay_end)], bins = \
        np.arange(-1.5, 2.5, 0.1) + 0.05)
    hist_pv_rt_center = bin_pv_rt[0:-1] + (bin_pv_rt[1] - bin_pv_rt[0])/2
    hist_pv_rt_width = (bin_pv_rt[1] - bin_pv_rt[0])*0.8
    
    hist_pv_lt, bin_pv_lt = np.histogram(r_left_pj_e_mat[:, int(t_delay_end)], bins = \
        np.arange(-1.5, 2.5, 0.1) + 0.05)
    hist_pv_lt_center = bin_pv_lt[0:-1] + (bin_pv_lt[1] - bin_pv_lt[0])/2
    hist_pv_lt_width = (bin_pv_lt[1] - bin_pv_lt[0])*0.8

    fig6, ax6 = plt.subplots()
    
    ax6.bar(hist_pv_rt_center, hist_pv_rt/np.shape(r_right_pj_e_mat)[0], align= 'center', \
        width = hist_pv_rt_width, color = 'tab:blue', label = 'right')
    ax6.bar(hist_pv_lt_center, hist_pv_lt/np.shape(r_left_pj_e_mat)[0], align = 'center', \
        width = hist_pv_lt_width, color = 'tab:red', label = 'left')
    
    ax6.set_xlabel('Fraction of error trials')
    ax6.set_ylabel('Proj. to CD')
    ax6.set_title('Endpoint distribution - pertubed error trials')
    ax6.legend()
    fig6.tight_layout()

    plt.savefig('./figures/perturbed_error_trial_hist.png')
    plt.close()

    # Trial-to-trial fluctuations
    norm_var_right = np.std(r_right_pj_c[0], axis=0) / np.std(r_right_pj_c[0][:, int(t_stim_end)], axis=0)
    norm_var_left = np.std(r_left_pj_c[0], axis=0) / np.std(r_left_pj_c[0][:, int(t_stim_end)], axis=0)
    
    fig7, ax7 = plt.subplots()

    ax7.plot(t_vec_plt, norm_var_right, color = 'tab:blue')
    ax7.plot(t_vec_plt, norm_var_left, color = 'tab:red')
    ax7.set_xlabel('Time to movement onset (s)')
    ax7.set_ylabel('Changes in across-trials fluc. of proj. to CD')
    ax7.set_title('Trial-averaged fluctuations')
    fig7.tight_layout()

    plt.savefig('./figures/trial_fluctions.png')
    plt.close()

    # plot change in Delta

    title_list = ['0.1mW correct', '0.2mW correct', '0.3mW correct', 'incorrect']
    fig8, ax8 = plt.subplots(1, 4, sharey=True)

    for idx in range(4):
        if idx < 3:
            data_1 = (abs(Data_at_each_bin[idx + 1][0, 0] - Data_at_each_bin[0][0, 0]) + \
                abs(Data_at_each_bin[idx +1][1, 0] - Data_at_each_bin[0][1,0])) / 2
            data_2 = (abs(Data_at_each_bin[idx + 1][0, 1] - Data_at_each_bin[0][0,1]) + \
                abs(Data_at_each_bin[idx + 1][1, 1] - Data_at_each_bin[0][1,1])) / 2
        else:
            data_bin1_C_tmp = np.nanmean([Data_at_each_bin[4][0,0], Data_at_each_bin[5][0,0], \
                Data_at_each_bin[6][0,0]])
            data_bin1_I_tmp = np.nanmean([Data_at_each_bin[4][1,0], Data_at_each_bin[5][1,0], \
                Data_at_each_bin[6][1,0]])
            data_bin2_C_tmp = np.nanmean([Data_at_each_bin[4][0,1], Data_at_each_bin[5][1,0], \
                Data_at_each_bin[6][1,0]])
            data_bin2_I_tmp = np.nanmean([Data_at_each_bin[4][1,1], Data_at_each_bin[5][1,1], \
                Data_at_each_bin[6][1,1]])

            data_1 = (abs(data_bin1_C_tmp - Data_at_each_bin[0][0,0]) + \
                abs(data_bin1_I_tmp - Data_at_each_bin[0][1,0])) / 2
            data_2 = (abs(data_bin2_C_tmp - Data_at_each_bin[0][0,1]) + \
                abs(data_bin2_I_tmp - Data_at_each_bin[0][1,1])) / 2
        ax8[idx].set_title(title_list[idx])
        ax8[idx].plot([data_1, data_2], 'ko-')
        ax8[idx].set_ylim([0, 1])
        
    ax8[0].set_ylabel('Delta proj')
    
    fig8.tight_layout()
    plt.savefig('./figures/delta_proj.png')
    plt.close()
    
    # TODO, plot phase line, there's seperated m script file in original code.
    return [correct_trials_right, correct_trials_left, r_right_pj_c, r_left_pj_c]




def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    # source: https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-numpy-python
    if WSZ%2 == 0:
        WSZ = int(WSZ) - 1
    else:
        WSZ = int(WSZ)
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))