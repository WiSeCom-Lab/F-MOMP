import scipy.linalg
from src import *
import winsound
tqdm = partial(tqdm, position=0, leave=True)


if __name__ == "__main__":
    
    for scene in [2]:
    
    #   # TODO: Parameters to be set
        # scene = 1
        N_UE, N_AP, N_taps, Q, p_t_dBm = 12, 16, 32, 36, 45
        num_measurement = 40
        reconf_period = 4
        
    #   # TODO: ends
        
            
        commSys = CommSysInfo(
            # * Dataset related; track_T: tracking period; number samples to estimate (as each car has 4 arrays so 4x)
            Scene = scene, track_T = 10e-3, traj_start=0, traj_pts = 150, arr_loc=np.array([[2.5, 0, 0], [-2.5, 0, 0], [0, -1, 0], [0, 1, 0]]),
            # * Comm system hardware related; N_U/N_AP: #antenna elements; K: # taps
            link='up', N_RF_UE = 4, N_RF_AP = 4, N_UE=N_UE, N_AP=N_AP, K=N_taps, 
            # * Communication related
            f_c = 73e9, B=1e9, T=15+273.1, c=299792458, p_t_dBm=p_t_dBm, Q=Q, 
            # * Channel est/track related
            K_res=64, K_res_lr=2, N_est=5, # Channel est/track relate
        )
            
        
        
        with open(f"./data/ds{scene}/Info_selected.txt") as f:
            Rays = [pywarraychannels.em.Geometric([[float(p) for p in line.split()] for line in ue_block.split("\n")], bool_flip_RXTX=commSys.link=="up") for ue_block in f.read()[:-1].split("\n<ue>\n")]
        orisPerShot = np.squeeze(sio.loadmat(f'./data/ds{scene}/orisPershot.mat')['ori']) # in rad
        
        
        traj_rec_sigs = {'Y_M': [], 'whi_Y_M': [], 'whi_WH_M': [], 'F_M': [], 'Linv_M': []}
        link_cur = 'uplink' if commSys.link in 'up' else 'downlink'
        
        traj_len = len(Rays) // 4
        for arr_ind in range(0,4): # index of array on the vehicle
            traj_chan_track = {'descrb': f'{link_cur} estimated channels in shape: N_est x # chan params x traj_len', 'est_chan': []}

            for shot_id in tqdm(range(traj_len), position=0, leave=1, desc=f'Tracking channel arr {arr_ind} =========================>>>>>>>>>>>>>'): # tqdm(range(traj_len), desc='Tracking channel ing................'):
                # print(f'==========={shot_id+1}/{traj_len}=============>>>>>>>>>>>>>>>>>>>>>>>>>>')
                rays = Rays[shot_id * 4 + arr_ind] #! phase, delay, power(dBm), ang_az_B, ang_el_B, ang_az_v, ang_el_v
                # print(rays.ray_info[:10, -4:])
                p_dBm_max = np.max(rays.ray_info[:, 2])
                if p_dBm_max <= -100:
                    commSys.p_t = 10 ** ((p_t_dBm + (-90-p_dBm_max) - 30)/10)
                H_taps = commSys.build_chan(rays, arr_ind) # N_\rmr x N_\rmt*N_\rmd
                
                #! Precoder/combiner design
                F_cb, W_cb = [], []
                if shot_id % reconf_period == 0:
                    rdm_add_ang = np.random.normal(0, np.deg2rad(.75), size=(commSys.N_est, 4))
                    rdm_add_t = np.random.normal(0, 5e-11, size=(commSys.N_est))
                for path_ii in range(commSys.N_est): # decide precoder/combiner based on previous estimates
                    if shot_id % reconf_period == 0:
                        Aaz_mid, Ael_mid, Daz_mid, Del_mid = \
                            np.radians(rays.ray_info[path_ii, 3])+rdm_add_ang[path_ii, 0], np.radians(rays.ray_info[path_ii, 4])+rdm_add_ang[path_ii, 1], np.radians(rays.ray_info[path_ii, 5])+rdm_add_ang[path_ii, 2], np.radians(rays.ray_info[path_ii, 6])+rdm_add_ang[path_ii, 3]
                    else:
                        Aaz_mid, Ael_mid, Daz_mid, Del_mid = est_Aaz_mids[path_ii], est_Ael_mids[path_ii], est_Daz_mids[path_ii], est_Del_mids[path_ii]
                    F, W = commSys.design_codebook_perpath(Aaz_mid, Ael_mid, Daz_mid, Del_mid, num_beam_az=3, num_beam_el=3)
                    F_cb.append(F)
                    W_cb.append(W)
                F_cb_inOne, W_cb_inOne, F_M_colIds,  W_M_colIds, Linv_M = commSys.cb_with_Linv(F_cb_list=F_cb, W_cb_list=W_cb, num_Mea=num_measurement)
                
                #! Receive signals: Now we have F_cb, W_cb, pair_mats (where all the pairs for each cb are given), and self.LLinv and we receive the signals (measurements)
                Y_M, whi_Y_M, whi_WH_M, F_M = commSys.get_Y(F_cb_inOne, W_cb_inOne, F_M_colIds, W_M_colIds, Linv_M, H_taps)

                #! F-MOMPp algorithms
                # define dictionaries
                if shot_id % reconf_period == 0: # define dictionaries
                    Aaz_mids, Ael_mids, Daz_mids, Del_mids, t_mids = \
                        np.radians(rays.ray_info[:commSys.N_est, 3])+rdm_add_ang[:, 0], np.radians(rays.ray_info[:commSys.N_est, 4])+rdm_add_ang[:, 1], np.radians(rays.ray_info[:commSys.N_est, 5])+rdm_add_ang[:, 2], np.radians(rays.ray_info[:commSys.N_est, 6])+rdm_add_ang[:, 3], commSys.true_chan_params['tdoa'][:commSys.N_est]+rdm_add_t
                    
                else:
                    Aaz_mids, Ael_mids, Daz_mids, Del_mids, t_mids = est_Aaz_mids, est_Ael_mids, est_Daz_mids, est_Del_mids, est_t_mids
                # if np.isnan(np.sum(Aaz_mids)):
                #     break
                all_est_paths, col_names = commSys.F_MOMP(Y_M, whi_Y_M, whi_WH_M, F_M, Linv_M, Aaz_mids, Ael_mids, Daz_mids, Del_mids, t_mids) 
                # Update next dictionary centers
                est_Aaz_mids, est_Ael_mids, est_Daz_mids, est_Del_mids, est_t_mids = all_est_paths[:, 3], all_est_paths[:, 4], all_est_paths[:, 5], all_est_paths[:, 6], all_est_paths[:, 2] - np.min(rays.ray_info[:, 1])
                
                
                traj_chan_track['est_chan'].append(all_est_paths)
                
                # # Display results
                # np.set_printoptions(precision=3, suppress=1, formatter={'float': lambda x: '{:e}'.format(x)})
                # print(np.hstack([20*np.log10(all_est_paths[:, 1:2])+30, all_est_paths[:, 2:3], np.rad2deg(np.around(all_est_paths[:, 3:7], 4))]))
                # print(np.hstack([rays.ray_info[:, np.array([2, 1, 3,4,5,6])]]))
                # print()
                getattr(tqdm, '_instances', {}).clear()
            traj_chan_track['est_chan'] = np.stack(traj_chan_track['est_chan'], axis=2)
            traj_chan_track['col_names'] = col_names
            
            #! Write to files
            # sio.savemat(f'./data/ds{scene}/F-MOMP_arr{arr_ind}_{N_UE}UEele_{N_AP}APele_{N_taps}taps_{Q}pilots_{num_measurement}measrs_{p_t_dBm}dBm.mat', traj_chan_track)
        
