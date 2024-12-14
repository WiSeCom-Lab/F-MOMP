# Mathmetcial packages
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import scipy.io as sio
import math
import random as rdm

# System packages
import os, sys, json, argparse, time, itertools
from time import time
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)

import libs.pywarraychannels as pywarraychannels

class CommSysInfo:
    def __init__(self,  *args, **kwargs) -> None:
        self.init_params = {}
        for key, value in kwargs.items():
            self.__dict__[key] = value
            
        # self.init_dataset()
        #* Transmitted signal, precoders/combiners, channel geo information
        self.to_data_trajs()
        self.init_communication() # design transimitted signals, get noise level
        
        
        
    
    def to_data_trajs(self):
        set = f"ds{self.Scene}"
        self.set = set    
        # Load data
        with open("data/{}/AP_pos.txt".format(set)) as f:
            self.AP_pos = [[float(el) for el in line.split()] for line in f.read().split("\n")[1:-1]]
        with open("data/{}/UE_pos.txt".format(set)) as f:
            self.UE_pos = [[float(el) for el in line.split()] for line in f.read().split("\n")[1:-1]] # including all array positions ⚠ (not the strongest)
        with open("data/{}/Info_selected.txt".format(set)) as f:
            self.Rays = [pywarraychannels.em.Geometric([[float(p) for p in line.split()] for line in ue_block.split("\n")], bool_flip_RXTX=self.link=="up") for ue_block in f.read()[:-1].split("\n<ue>\n")]
        self.true_num_inters = sio.loadmat(f'data/{set}/Num_inters.mat')
        self.arr_ori_compensate = np.array([0., -np.pi, np.pi/2, -np.pi/2]) # front, back, right, left arr orientation compensate, i.e., aoa/aod + self.arr_ori_compensate
        self.ori_AP = self.arr_ori_compensate[3] if np.squeeze(self.AP_pos)[1] <=-20. else self.arr_ori_compensate[2]
        self.p_max_chans = np.zeros([len(self.Rays) // 4, 2]).astype(np.int32) # maximal power among the 4 arrays
        for sim_id in range(len(self.Rays) // 4):
            chan_ids = np.arange(sim_id * 4, (sim_id+1) * 4)
            powers = [np.max(self.Rays[ii].ray_info[:, 2]) for ii in chan_ids]
            self.p_max_chans[sim_id, :] = np.argsort(powers)[-2:][::-1]

    def load_trajInfo(self, traj_file=None):
        self.traj_info = sio.loadmat(traj_file)
        pass
        
    def init_communication(self):
        k_B = 1.38064852e-23
        Q = 2 ** np.ceil(np.log2(self.Q))
        self.p_t = np.power(10, (self.p_t_dBm-30)/10) # transimitted power
        self.p_n = k_B*self.T*self.B # noise power
        print("Noise level: {:.2f}dBm".format(10*np.log10(self.p_n)+30))
        self.N_s = np.min([self.N_RF_UE, self.N_RF_AP])
        # Pilot = np.concatenate([scipy.linalg.hadamard(Q)[:self.N_s], np.zeros((self.N_s, self.K//2))], axis=1)/np.sqrt(self.N_s)
        
        Pilot = np.concatenate([np.zeros([self.N_s, self.K-1]), scipy.linalg.hadamard(Q)[:self.N_s, :self.Q]], axis=1) / np.sqrt(self.N_s)
        self.Pilot = Pilot[:, self.K-1:] # N_s x Q
        real_Q = Pilot.shape[1] # Including prefix 0's
        S_mat = np.concatenate([Pilot[:, (self.K-1)-x: (self.K-1)-x + self.Q-1 + 1] for x in range(self.K)], axis=0)
        self.S_mat = S_mat

        
    def design_codebook_perpath(self, Aaz_mid=0, Ael_mid=0, Daz_mid=0, Del_mid=0, num_beam_az=3, num_beam_el=3, ang_res=np.deg2rad(10)): #* angle_mid means the center of the sector for the precoder/combiners
        """_summary_
        All angles are in radians
        #! Actually ang resolution should be depending on number of antennaa elements with quantized phase shifters, but here for simplicity we use just a fixed degree
        Args:
            load (int, optional): if write the precoder/combiner to dataset folder. Defaults to 0.
        """
        F_ele = self.N_UE if self.link in 'up' else self.N_AP
        W_ele = self.N_AP if self.link in 'up' else self.N_UE
        b_a, b_e = num_beam_az//2, num_beam_el//2
        
        A_azs = Aaz_mid + np.arange(-b_a, b_a+1)*ang_res
        A_els = Ael_mid + np.arange(-b_e, b_e+1)*ang_res
        D_azs = Daz_mid + np.arange(-b_a, b_a+1)*ang_res
        D_els = Del_mid + np.arange(-b_e, b_e+1)*ang_res
        
        
        W_prob_angs = np.kron(np.cos(A_els), np.sin(A_azs))*(-np.pi)
        W_angs_x = np.linspace(np.min(W_prob_angs), np.max(W_prob_angs), num_beam_az)
        W_angs_y = np.sin(A_els)*(-np.pi)
        F_prob_angs = np.kron(np.cos(D_els), np.sin(D_azs))*(-np.pi)
        F_angs_x = np.linspace(np.min(F_prob_angs), np.max(F_prob_angs), num_beam_az)
        F_angs_y = np.sin(D_els)*(-np.pi)
        
        cb_Fx = np.exp(1j * F_angs_x[np.newaxis, :] * np.arange(F_ele)[:, np.newaxis])
        cb_Fy = np.exp(1j * F_angs_y[np.newaxis, :] * np.arange(F_ele)[:, np.newaxis])
        cb_Wx = np.exp(1j * W_angs_x[np.newaxis, :] * np.arange(W_ele)[:, np.newaxis])
        cb_Wy = np.exp(1j * W_angs_y[np.newaxis, :] * np.arange(W_ele)[:, np.newaxis])
        
        F = np.kron(cb_Fx, cb_Fy) # whole codebook
        W = np.kron(cb_Wx, cb_Wy)
        num_FW_comb = F.shape[1]*W.shape[1]
        FW_pair_id = np.zeros([2, num_FW_comb])
        for Fi in range(F.shape[1]):
            FW_pair_id[0, Fi*W.shape[1]: (Fi+1)*W.shape[1]] = Fi
            FW_pair_id[1, Fi*W.shape[1]: (Fi+1)*W.shape[1]] = np.arange(0, W.shape[1])
        return F, W
    
    def cb_with_Linv(self, F_cb_list, W_cb_list, num_Mea = 40): # whiten noises
        F_cb_inOne = np.concatenate(F_cb_list, axis=1)
        W_cb_inOne = np.concatenate(W_cb_list, axis=1)
        F_M_colIds, W_M_colIds, Linv_M = [], [], []
        F_cb_numCol, W_cb_numCol = F_cb_list[0].shape[1], W_cb_list[0].shape[1]
        for n_est in range(self.N_est):
            for m in range(num_Mea//self.N_est):
                F_sel_nums, W_sel_nums = range(n_est*F_cb_numCol, (n_est+1)*F_cb_numCol), range(n_est*W_cb_numCol, (n_est+1)*F_cb_numCol)
                F_m_cols = np.random.choice(F_sel_nums, size=self.N_s, replace=0)
                W_m_cols = np.random.choice(W_sel_nums, size=self.N_s, replace=0)
                W_m = W_cb_inOne[:, W_m_cols]
                try:
                    L = np.linalg.cholesky(W_m.conj().T @ W_m)
                except:
                    continue
                Linv = np.linalg.inv(L)
                Linv_M.append(Linv)
                
                F_M_colIds.append(F_m_cols)
                W_M_colIds.append(W_m_cols)
                
            
            
        return F_cb_inOne, W_cb_inOne, F_M_colIds,  W_M_colIds, Linv_M
        
        
    
    def build_chan(self, rays, arr_id):
        paths_info = rays.ray_info # UL: 0 phase, 1 tau, 2 power, 3 doa_az, 4 doa_el, 5 dod_az, 6 dod_el
        if self.link == "up":
            numEle_R, numEle_T = self.N_AP, self.N_UE
        else:
            numEle_R, numEle_T = self.N_UE, self.N_AP
        phase, doa_az, doa_el, dod_az, dod_el = np.radians(paths_info[:, 0]), np.radians(paths_info[:, 3]), \
                np.radians(paths_info[:, 4]), np.radians(paths_info[:, 5]), np.radians(paths_info[:, 6])   # Transform to radians
        doa = np.vstack([np.cos(doa_el)*np.cos(doa_az), np.cos(doa_el)*np.sin(doa_az), np.sin(doa_el)]).T                   
        dod = np.vstack([np.cos(dod_el)*np.cos(dod_az), np.cos(dod_el)*np.sin(dod_az), np.sin(dod_el)]).T   
        tau_min = np.min(paths_info[:, 1])
        tdoa = paths_info[:, 1] - tau_min
        t_response = self.rcosfilter(num_taps=self.K, delays=tdoa, beta=0.2)
        complex_gain = np.sqrt(np.power(10, (paths_info[:, 2]-30)/10)) * np.exp(1j*phase)
        arrival_mat, departure_mat = [], []
        for path_id in range(len(tdoa)):
            arrival_mat.append(self.array_respnse(ang_az=doa_az[path_id], ang_el=doa_el[path_id], ele_per_dim=numEle_R))
            departure_mat.append(self.array_respnse(ang_az=dod_az[path_id], ang_el=dod_el[path_id], ele_per_dim=numEle_T))
        arrival_mat, departure_mat = np.array(arrival_mat).T, np.array(departure_mat).T
        
        self.true_chan_params = {}
        self.true_chan_params['Amat_arrive'] = arrival_mat
        self.true_chan_params['Amat_depart'] = departure_mat
        self.true_chan_params['P_mat'] = t_response
        self.true_chan_params['complex_gain'] = complex_gain
        self.true_chan_params['tdoa'] = tdoa
        self.true_chan_params['Aaz'] = doa_az
        self.true_chan_params['Ael'] = doa_el
        self.true_chan_params['Daz'] = dod_az
        self.true_chan_params['Del'] = dod_el
        self.true_chan_params['toa_min'] = tau_min
        
        H_allTaps = []
        for d_ii in range(self.K): # loop each tap
            CGxPulseShape = complex_gain.reshape(-1, 1) * t_response[d_ii, :].reshape(-1, 1)
            gain_mat = np.zeros([len(tdoa), len(tdoa)]).astype(np.complex128)
            np.fill_diagonal(gain_mat, np.squeeze(CGxPulseShape))
            H_tap_ii = np.dot(np.dot(arrival_mat, gain_mat), departure_mat.conj().T)
            H_allTaps.append(H_tap_ii)
        
        return np.concatenate(H_allTaps, axis=1) # list ewhere the i-th element is the i-th tap of the channel
    def array_respnse(self, ang_az, ang_el, ele_per_dim):
        resp_vec = np.kron(np.exp(-1j*np.pi*np.arange(0, ele_per_dim)*np.cos(ang_el)*np.sin(ang_az)), \
            np.exp(-1j*np.pi*np.arange(0, ele_per_dim)*np.sin(ang_el)))
        return resp_vec
    def rcosfilter(self, num_taps, delays, beta=0.2, upSample=1):
        T_samp = 1/(self.B * upSample)
        t_response = np.zeros([num_taps, len(delays)]) 
        for ii, delay in zip(range(len(delays)), delays):
            rela_t = np.arange(0, num_taps)*T_samp - delay
            loc_inval = abs(abs(rela_t)-T_samp/(beta*2))<=1e-20
            t_response[loc_inval, ii] = np.pi/(4*T_samp) * np.sinc(1/(beta*2)) 
            t_response[~loc_inval, ii] =  1/T_samp*np.sinc(rela_t[~loc_inval]/T_samp) * np.cos(np.pi*beta*rela_t[~loc_inval]/T_samp) / (1-np.power(2*beta*rela_t[~loc_inval]/T_samp, 2))
            t_response[:, ii] /= np.linalg.norm(t_response[:, ii])

        return t_response

    def get_Y(self, F_cb_inOne, W_cb_inOne, F_M_colIds, W_M_colIds, Linv_M, chan_taps):
        
        Y_M, whi_Y_M, whi_WH_M, F_M = [], [], [], []
        
        for mea_id in range(len(Linv_M)): #tqdm(range(len(Linv_list)), desc=f'Getting measurements ---------------------->>>>>>>>>>>>>>>>>>'):
            F_use = F_cb_inOne[:, F_M_colIds[mea_id]]
            F_use /= np.sqrt(F_use.shape[0])
            W_use = W_cb_inOne[:, W_M_colIds[mea_id]]
            W_use /= np.sqrt(W_use.shape[0])
            Linv = Linv_M[mea_id]
            
            add_N = np.sqrt(self.p_n / W_use.shape[0]) * (1/np.sqrt(2) * (np.random.randn(W_use.shape[0], self.Q) + 1j*np.random.randn(W_use.shape[0], self.Q)))
            IkronF_S = np.kron(np.eye(self.K), F_use) @ self.S_mat
            Y_m = np.sqrt(self.p_t) * (W_use.conj().T @ chan_taps) @ IkronF_S + W_use.conj().T @ add_N
                
            whi_Y_m = Linv @ Y_m
            whi_WH = Linv @ W_use.conj().T
            if np.isnan(np.sum(whi_WH)):
                continue
            
            Y_M.append(Y_m)
            whi_Y_M.append(whi_Y_m)
            whi_WH_M.append(whi_WH)
            F_M.append(F_use)
        
        
        return Y_M, whi_Y_M, whi_WH_M, F_M
        



            
    
    def F_MOMP(self, Y_M, whi_Y_M, whi_WH_M, F_M, Linv_M, Aaz_mids, Ael_mids, Daz_mids, Del_mids, t_mids, dic_ang_rslu=np.deg2rad(0.25), srch_grid = 8, N_iter = 4):

        
        whi_Y_M_vec = np.expand_dims(np.hstack([np.expand_dims(whi_Y_M[ii].flatten(order='F'), axis=1) for ii in range(len(Y_M))]).flatten(order='F'), axis=1)
        whi_Y_M_vec_res = whi_Y_M_vec
        psi_t, psi_AngDsprl, psi_AngDbot, psi_AngAsprl, psi_AngAbot = [], [], [], [], []
        num_Mea = len(whi_Y_M)
        
        if self.link=="up":
            numEle_R, numEle_T = self.N_AP, self.N_UE
        else:
            numEle_R, numEle_T = self.N_UE, self.N_AP
        
        
        Sup = []
        ChanGeo_ests = {}
        best_xi_set = [] # for line 22 in Algorithm 1
        sprs_x_est = []
        for n_est in range(self.N_est): # loop for N_est
            #* for delay values
            delay_grids = t_mids[n_est] + np.arange(-srch_grid, srch_grid+1)*2.5e-10
            # delay_grids = delay_grids[delay_grids>=0]
            psi_t_temp = self.rcosfilter(num_taps=self.K, delays=delay_grids, beta=0.2)
            # psi_t.append(psi_t_temp)
            #* for phi^\sprl values
            cosElVals = np.reshape(np.cos( Del_mids[n_est] + np.arange(-srch_grid, srch_grid+1)*dic_ang_rslu), [-1, 1])
            sinAzVals = np.reshape(np.sin( Daz_mids[n_est] + np.arange(-srch_grid, srch_grid+1)*dic_ang_rslu), [1, -1] )
            v = cosElVals @ sinAzVals
            v_min, v_max = np.min(v), np.max(v)
            Dsprl_grids = np.linspace(v_min, v_max, int(2*srch_grid+1))
            psi_Dsprl_tp = np.exp(np.reshape(np.arange(0, numEle_T) * (-1j*np.pi), [-1, 1]) @ np.reshape(Dsprl_grids, [1, -1]))
            # psi_AngDsprl.append(psi_Dsprl_tp)
            #* for phi^\bot values
            Dbot_grids = np.sin(Del_mids[n_est] + np.arange(-srch_grid, srch_grid+1)*dic_ang_rslu)
            psi_Dbot_tp = np.exp(np.reshape(np.arange(0, numEle_T) * (-1j*np.pi), [-1, 1]) @ np.reshape(Dbot_grids, [1, -1]))
            # psi_AngDbot.append(psi_Dsprl_tp)
            #* for theta^\sprl values
            cosElVals = np.reshape(np.cos( Ael_mids[n_est] + np.arange(-srch_grid, srch_grid+1)*dic_ang_rslu), [-1, 1])
            sinAzVals = np.reshape(np.sin( Aaz_mids[n_est] + np.arange(-srch_grid, srch_grid+1)*dic_ang_rslu), [1, -1] )
            v = cosElVals @ sinAzVals
            v_min, v_max = np.min(v), np.max(v)
            Asprl_grids = np.linspace(v_min, v_max, int(2*srch_grid+1))
            psi_Asprl_tp = np.exp(np.reshape(np.arange(0, numEle_R) * (-1j*np.pi), [-1, 1]) @ np.reshape(Asprl_grids, [1, -1]))
            #* for theta^\bot values
            Abot_grids = np.sin(Ael_mids[n_est] + np.arange(-srch_grid, srch_grid+1)*dic_ang_rslu)
            psi_Abot_tp = np.exp(np.reshape(np.arange(0, numEle_R) * (-1j*np.pi), [-1, 1]) @ np.reshape(Abot_grids, [1, -1]))
            
            sup = np.zeros([5]).astype(np.int8) + srch_grid # all temp suppoprt are the center column
            Pilot_new = np.hstack([np.zeros([self.N_s, self.K-1]), self.Pilot]) # Add 0's ahead
            for n_iter in range(N_iter):
                
                #! Find dim2 support for departure angle sprl
                Xi_S = np.sqrt(self.p_t)*np.squeeze(np.stack([psi_t_temp[:, sup[0]:sup[0]+1].T @ Pilot_new[:, np.arange(ii, ii+self.K)[::-1]].T for ii in range(self.Q)], axis=0)) # axis=1 is only 1 -> Xi_S's shape Q x N_s
                Xi_F = np.concatenate([np.expand_dims(F_M[ii].T @ (np.kron(psi_Dsprl_tp, psi_Dbot_tp[:, sup[2]:sup[2]+1]).conj()), axis=1) for ii in range(num_Mea)], axis=1) # N_s x M x N_2^\rma
                Xi_W = np.concatenate([whi_WH_M[ii] @ (np.kron(psi_Asprl_tp[:, sup[3]:sup[3]+1], psi_Abot_tp[:, sup[4]:sup[4]+1])) for ii in range(num_Mea)], axis=1)
                corMax_Dsprl = -1
                for ii in range(Xi_F.shape[2]):
                    xi_F = np.reshape((scipy.linalg.khatri_rao( Xi_S @ Xi_F[:, :, ii], Xi_W)).flatten(order='F'), [-1, 1])
                    cor_F = abs(xi_F.conj().T @ whi_Y_M_vec_res) / scipy.linalg.norm(xi_F) 
                    if cor_F > corMax_Dsprl:
                        corMax_Dsprl = cor_F
                        sup[1] = ii 
                        
                #! Find dim3 support for departure angle bot
                Xi_F = np.concatenate([np.expand_dims(F_M[ii].T @ (np.kron(psi_Dsprl_tp[:, sup[1]:sup[1]+1], psi_Dbot_tp).conj()), axis=1) for ii in range(num_Mea)], axis=1) # N_s x M x N_3^\rma
                corMax_Dbot = -1
                for ii in range(Xi_F.shape[2]):
                    xi_F = np.reshape((scipy.linalg.khatri_rao( Xi_S @ Xi_F[:, :, ii], Xi_W)).flatten(order='F'), [-1, 1])
                    cor_F = abs(xi_F.conj().T @ whi_Y_M_vec_res) / scipy.linalg.norm(xi_F) 
                    if cor_F > corMax_Dbot:
                        corMax_Dbot = cor_F
                        sup[2] = ii 
                
                #! Find dim4 support for arrival angle sprl
                Xi_F = np.concatenate([F_M[ii].T @ (np.kron(psi_Dsprl_tp[:, sup[1]:sup[1]+1], psi_Dbot_tp[:, sup[2]:sup[2]+1]).conj()) for ii in range(num_Mea)], axis=1) # N_s x M 
                Xi_W = np.concatenate([np.expand_dims(whi_WH_M[ii] @ (np.kron(psi_Asprl_tp, psi_Abot_tp[:, sup[4]:sup[4]+1])), axis=1) for ii in range(num_Mea)], axis=1) # N_s x M x N_4^\rma
                corMax_Asprl = -1
                for ii in range(Xi_W.shape[2]):
                    xi_W = np.reshape((scipy.linalg.khatri_rao( Xi_S @ Xi_F, Xi_W[:, :, ii])).flatten(order='F'), [-1, 1])
                    cor_W = abs(xi_W.conj().T @ whi_Y_M_vec_res) / scipy.linalg.norm(xi_W) 
                    if cor_W > corMax_Asprl:
                        corMax_Asprl = cor_W
                        sup[3] = ii 
                        
                #! Find dim5 support for arrival angle bot
                Xi_W = np.concatenate([np.expand_dims(whi_WH_M[ii] @ (np.kron(psi_Asprl_tp[:, sup[3]:sup[3]+1], psi_Abot_tp)), axis=1) for ii in range(num_Mea)], axis=1) # N_s x M x N_4^\rma
                corMax_Abot = -1
                for ii in range(Xi_W.shape[2]):
                    xi_W = np.reshape((scipy.linalg.khatri_rao( Xi_S @ Xi_F, Xi_W[:, :, ii])).flatten(order='F'), [-1, 1])
                    cor_W = abs(xi_W.conj().T @ whi_Y_M_vec_res) / scipy.linalg.norm(xi_W) 
                    if cor_W > corMax_Abot:
                        corMax_Abot = cor_W
                        sup[4] = ii 
                
                #! Find dim1 support for t
                Xi_S = np.sqrt(self.p_t)*np.stack([psi_t_temp.T @ Pilot_new[:, np.arange(ii, ii+self.K)[::-1]].T for ii in range(self.Q)], axis=0) # Q x N_1^\rma x N_s
                Xi_W = np.concatenate([whi_WH_M[ii] @ (np.kron(psi_Asprl_tp[:, sup[3]:sup[3]+1], psi_Abot_tp[:, sup[4]:sup[4]+1])) for ii in range(num_Mea)], axis=1)
                corMax_t = -1
                for ii in range(Xi_S.shape[1]):
                    xi_t = np.reshape((scipy.linalg.khatri_rao( Xi_S[:, ii,:] @ Xi_F, Xi_W)).flatten(order='F'), [-1, 1])
                    cor_t = abs(xi_t.conj().T @ whi_Y_M_vec_res) / scipy.linalg.norm(xi_t) 
                    if cor_t > corMax_t:
                        corMax_t = cor_t 
                        sup[0] = ii 
                # print(sup)
                    
            Sup.append(sup)
            est_t = delay_grids[sup[0]]
            
            est_Dbot = Dbot_grids[sup[2]]
            est_Dele = np.arcsin(est_Dbot)
            est_Dsprl = Dsprl_grids[sup[1]]
            phi_sprl = est_Dsprl/np.cos(est_Dele)
            phi_sprl = phi_sprl if abs(phi_sprl)<1. else np.sign(phi_sprl)*0.9999
            est_Daz = np.arcsin(phi_sprl)
            if abs(Daz_mids[n_est] - est_Daz) > np.deg2rad(8):
                est_Daz = np.pi - est_Daz
                est_Daz = (est_Daz + np.pi) % (2 * np.pi) - np.pi # wrap to [-pi, pi]
            est_Abot = Abot_grids[sup[4]]
            est_Aele = np.arcsin(est_Abot)
            est_Asprl = Asprl_grids[sup[3]]
            theta_sprl = est_Asprl/np.cos(est_Aele)
            theta_sprl = theta_sprl if abs(theta_sprl)<1. else np.sign(theta_sprl) * 0.9999
            est_Aaz = np.arcsin(theta_sprl)
            if abs(Aaz_mids[n_est] - est_Aaz) > np.deg2rad(8):
                est_Aaz = np.pi-est_Aaz
                est_Aaz = (est_Aaz + np.pi) % (2 * np.pi) - np.pi # wrap to [-pi, pi]
            ChanGeo_ests[f'{n_est}'] = np.array([est_t, est_Aaz, est_Aele, est_Daz, est_Dele])

            
            #! Projection and residual
            best_Xi_S = np.sqrt(self.p_t)*np.squeeze(np.stack([psi_t_temp[:, sup[0]:sup[0]+1].T @ Pilot_new[:, np.arange(ii, ii+self.K)[::-1]].T for ii in range(self.Q)], axis=0))
            best_Xi_F = np.concatenate([F_M[ii].T @ (np.kron(psi_Dsprl_tp[:, sup[1]:sup[1]+1], psi_Dbot_tp[:, sup[2]:sup[2]+1]).conj()) for ii in range(num_Mea)], axis=1)
            best_Xi_W = np.concatenate([whi_WH_M[ii] @ (np.kron(psi_Asprl_tp[:, sup[3]:sup[3]+1], psi_Abot_tp[:, sup[4]:sup[4]+1])) for ii in range(num_Mea)], axis=1)
            be_xi = np.reshape((scipy.linalg.khatri_rao( best_Xi_S @ best_Xi_F, best_Xi_W)).flatten(order='F'), [-1, 1])
            
            
            best_xi_set.append(be_xi)
            spars_x_cur = np.linalg.pinv(np.hstack(best_xi_set)) @ whi_Y_M_vec
            sprs_x_est.append(spars_x_cur[-1, 0])
            whi_Y_M_vec_res = whi_Y_M_vec - np.hstack(best_xi_set) @ np.reshape(np.array(sprs_x_est), [-1, 1])
        
        
        # Gather all channel info
        sprs_x_est_vec = np.reshape(np.array(sprs_x_est), [-1, 1])
        all_est_paths = np.vstack([ChanGeo_ests[f'{n_est}'] for n_est in range(self.N_est)])
        all_est_paths[:, 0] += self.true_chan_params['toa_min']
        all_est_paths = np.hstack([np.angle(sprs_x_est_vec), np.abs(sprs_x_est_vec), all_est_paths]) #! phase, |Gain|, delay, ang_az_B, ang_el_B, ang_az_v, ang_el_v
        col_names = ['phase', '|Gain|', 'toa', 'angAaz', 'angAel', 'angDaz', 'angDel']
            
        return all_est_paths, col_names
            
            
            
        
        
        
        pass

