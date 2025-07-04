# -*- coding: utf-8 -*-
"""
Created on Tue May 27 11:59:53 2025

@author: Iman R
"""

import numpy as np
from scipy.signal import hilbert
from scipy.io import loadmat
from typing import Tuple
import sys
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from pathlib import Path

# Relevant Articles
# Hruska DP, Oelze ML. Improved parameter estimates based on the homodyned K distribution. IEEE Trans Ultrason Ferroelectr Freq Control. 2009
# Zhou Z, Gao A, Wu W, Tai DI, Tseng JH, Wu S, Tsui PH. Parameter estimation of the homodyned K distribution based on an artificial neural network for ultrasound tissue characterization. Ultrasonics. 2021
# Destrempes F, Porée J, Cloutier G. ESTIMATION METHOD OF THE HOMODYNED K-DISTRIBUTION BASED ON THE MEAN INTENSITY AND TWO LOG-MOMENTS. SIAM J Imaging Sci. 2013
# Liu Y, Zhang Y, He B, Li Z, Lang X, Liang H, Chen J. An Improved Parameter Estimator of the Homodyned K Distribution Based on the Maximum Likelihood Method for Ultrasound Tissue Characterization. Ultrason Imaging. 2022

class RSK_HKD_Estimator:
        
    def __init__(self):
        rsk_npz_path = Path(__file__).parent / 'LUData.npz'
        data = np.load(rsk_npz_path)
        self.RSK_Maps = data['RSK_Maps']
        self.k_vals = data['k_vals'].ravel()
        self.Mu_vals = data['Mu_vals'].ravel()
        self.Nu_vals = data['Nu_vals'].ravel()
        self.grid_size = len(self.k_vals)

    def compute_rsk_statistics(self, env, Nu):
        env_Nu = env ** Nu
        R = np.mean(env_Nu) / np.std(env_Nu)
        S = skew(env_Nu, bias=False)
        K = kurtosis(env_Nu, bias=False)
        return R, S, K

    def compute_all_statistics(self, env):
        stats = []
        for Nu in self.Nu_vals:
            R, S, K = self.compute_rsk_statistics(env, Nu)
            stats.append([R, S, K])
        return np.array(stats)

    def search_level_curve(self, n, p, value):
        indices = []
        for l in range(self.grid_size):
            for m in range(self.grid_size - 1):
                v1 = self.RSK_Maps[l, m, n, p]
                v2 = self.RSK_Maps[l, m + 1, n, p]
                if (v1 - value) * (v2 - value) < 0:
                    indices.append((l, m))
        for m in range(self.grid_size):
            for l in range(self.grid_size - 1):
                v1 = self.RSK_Maps[l, m, n, p]
                v2 = self.RSK_Maps[l + 1, m, n, p]
                if (v1 - value) * (v2 - value) < 0:
                    indices.append((l, m))
        return np.array(indices)

    def find_best_match(self, stats):
        curve_points = []
        for n in range(len(self.Nu_vals)):
            for p in range(3):
                val = stats[n, p]
                pts = self.search_level_curve(n, p, val)
                #curve_points.append(pts)
                if pts.size > 0:
                    curve_points.append(pts)
                    
        if not curve_points:
            raise ValueError("No level curve intersections found. Your input data may be out of expected parameter range.")


        best_err = np.inf
        best_idx = (0, 0)
        for l in range(self.grid_size):
            for m in range(self.grid_size):
                err = 0
                for pts in curve_points:
                    dists = np.sum((pts - np.array([l, m]))**2, axis=1)
                    err += np.min(dists)
                if err < best_err:
                    best_err = err
                    best_idx = (l, m)
        return best_idx, best_err
    
    def check_params(self, param, nu, p, param_min, param_max):
        if np.isnan(param) or param < param_min or param > param_max:
            return -1
        return 0
    
    def closer_RSK_pt(self,pt1, pt2, RSK_nu, index):
        
        
        val1 = self.RSK_Maps[pt1[0], pt1[1], pt1[2], pt1[3]]
        val2 = self.RSK_Maps[pt2[0], pt2[1], pt2[2], pt2[3]]
        
        if abs(val2 - RSK_nu) > abs(val1 - RSK_nu):
            return pt1[index]
        else:
            return pt2[index]

    def level_curve_dist(self, lm, curve_pts, num_nu):
        # lm: [l, m]
        err_vect = np.zeros((num_nu, 3))
        
        for n in range(num_nu):
            for p in range(3):
                points = curve_pts[n][p]  # assuming curve_pts is a 2D list: curve_pts[num_nu][3]
                if len(points) == 0:
                    # no points on curve, skip or treat distance as large?
                    continue
                diffs = points - lm  # vectorized difference
                dists_sq = np.sum(diffs**2, axis=1)
                err_vect[n, p] = np.min(dists_sq)
        
        dist2 = np.sum(err_vect)
        return dist2

    def search_vert(self,n, p, RSK_nu, grid_size):     
        indices = []
        for m in range(grid_size):
            # sign change detection along l dimension
            diff_sign = np.diff(np.sign(self.RSK_Maps[:, m, n, p] - RSK_nu))
            sign_changes = np.where(diff_sign != 0)[0]  # indices of sign changes
            for sc in sign_changes:
                l = self.closer_RSK_pt((sc, m, n, p),(sc + 1, m, n, p),RSK_nu, 0)
                indices.append([l, m])
        
        if len(indices) == 0:
            raise ValueError('General parameter out of range (search_vert)')

        return np.array(indices)

    def search_horiz(self, n, p, RSK_nu, grid_size):
        indices = []
        for l in range(grid_size):
            # sign change detection along m dimension
            diff_sign = np.diff(np.sign(self.RSK_Maps[l, :, n, p] - RSK_nu))
            sign_changes = np.where(diff_sign != 0)[0]
            for sc in sign_changes:
                m = self.closer_RSK_pt((l, sc, n, p), (l, sc + 1, n, p),RSK_nu,1)
                indices.append([l, m])

        return np.array(indices)
    
    def kurtosis(x, flag=1, dim=None):
        x = np.array(x, dtype=np.float64)

        # Determine default dim if not specified
        if dim is None:
            if x.ndim == 1:
                dim = 0
            else:
                dim = next((i for i, s in enumerate(x.shape) if s > 1), 0)

        # Remove NaNs for moment calculation
        x0 = x - np.nanmean(x, axis=dim, keepdims=True)
        s2 = np.nanmean(x0 ** 2, axis=dim)     # biased variance
        m4 = np.nanmean(x0 ** 4, axis=dim)     # 4th central moment

        k = m4 / (s2 ** 2)

        if flag == 0:  # unbiased correction
            n = np.sum(~np.isnan(x), axis=dim)
            with np.errstate(invalid='ignore', divide='ignore'):
                correction = ((n + 1) * k - 3 * (n - 1)) * (n - 1) / ((n - 2) * (n - 3)) + 3
                correction[n < 4] = np.nan  # undefined for n < 4
                k = correction
                
        return k   
    def estimate_parameters(self, rf_data):
        env = np.abs(hilbert(rf_data, axis=1)).flatten()
        env = env.flatten()
        stats = self.compute_all_statistics(env)
        (l, m), err = self.find_best_match(stats)

        
        grid_size = len(self.k_vals)
        num_nu = len(self.Nu_vals)
        RSK_nu_values = np.zeros((num_nu, 3))

        
        for n in range(num_nu):
            nu = self.Nu_vals[n]
            ROI_nu = env ** nu
            RSK_nu_values[n, 0] = np.mean(ROI_nu) / np.std(ROI_nu)
            RSK_nu_values[n, 1] = skew(ROI_nu.flatten(), bias=False)
            RSK_nu_values[n, 2] = kurtosis(ROI_nu.flatten(), bias=False, fisher=True)+3  # Pearson kurtosis (normal = 3)

            if self.check_params(RSK_nu_values[n, 2], self.Nu_vals[n], 3,
                            np.min(self.RSK_Maps[:, :, n, 2]), 
                            np.max(self.RSK_Maps[:, :, n, 2])) == -1:
                return  -1, -1

            # Level curves
            curve_pts = [[[] for _ in range(3)] for _ in range(num_nu)]
        for n in range(num_nu):
            for p in range(3):
                RSK_nu = RSK_nu_values[n, p]
                h_ind = self.search_horiz(n, p, RSK_nu, grid_size)
                if h_ind.size == 0:
                    return -2, -2
                v_ind = self.search_vert(n, p, RSK_nu, grid_size)
                curve_pts[n][p] = np.unique(np.vstack((h_ind, v_ind)), axis=0)

        # Grid search
        num_grid_pts = 11
        l_min, l_max = 0, grid_size - 1
        m_min, m_max = 0, grid_size - 1

        while True:
            l_vals = np.round(np.linspace(l_min, l_max, num_grid_pts)).astype(int)
            m_vals = np.round(np.linspace(m_min, m_max, num_grid_pts)).astype(int)
            dist2 = []
            points = []

            for l in l_vals:
                for m in m_vals:
                    points.append([l, m])
                    dist2.append(self.level_curve_dist(np.array([l, m]), curve_pts, num_nu))

            dist2 = np.array(dist2)
            min_idx = np.argmin(dist2)
            l, m = points[min_idx]

            if (np.ptp(l_vals) <= num_grid_pts) and (np.ptp(m_vals) <= num_grid_pts):
                break

            new_box_size = int(np.ptp(l_vals) / 4)
            l_min, l_max = max(0, l - new_box_size), min(grid_size - 1, l + new_box_size)
            m_min, m_max = max(0, m - new_box_size), min(grid_size - 1, m + new_box_size)

        hk = self.k_vals[l]
        Mu = self.Mu_vals[m]
        alpha = Mu       
        kappa = np.sqrt(2*hk)
        return kappa, alpha
    