#!/usr/bin/env python3
# Author: Roberto Serino
# Date: 2026-03-14
# Description: Discrimination parameter evaluation through a sim fit of 
# the LY1 and LY2 distributions in energy intervals, using a global NLL approach.


import ROOT
import os
import argparse
from collections import defaultdict
from array import array
import numpy as np
import math
import gc

import matplotlib.pyplot as plt
from lmfit import Parameters, Minimizer
from lmfit.lineshapes import gaussian
import warnings
from scipy.optimize import curve_fit, minimize
from scipy.special import erf

import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
from ROOT import TObjString
from ROOT import TString
import inspect

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument("--cross", required=True)
parser.add_argument("--runs", nargs="+", required=True)
parser.add_argument("--outdir", default=".")
parser.add_argument("--channel", type=int, default=None)
parser.add_argument("--channelLD1", type=int, default=None)
parser.add_argument("--channelLD2", type=int, default=None)
parser.add_argument("--outPlot", default=".")

ROOT.gROOT.SetBatch(True)


args = parser.parse_args()

# =============================================================================
# SECTION 0 — Histogram configuration
# =============================================================================
 
N_HIST_BINS = 60
LY_MIN, LY_MAX = 0.0002, 0.01





def format_run(run):
    run = str(run)

    if run.startswith("RUN"):
        return run

    # convert 102 -> RUN000102
    return f"RUN{int(run):06d}"


# --- LY1 ---------------------------------------------------------------------
 
# def ly1_func_mean(E, a, b):
#     """Linear:  mu(E) = a + b*E"""
#     return a + b * E

# def ly1_func_mean(E, a, b):
#     """sqrt:  sigma(E) = sqrt(a + b*E)"""
#     return np.sqrt(a + b*E) 

# def ly1_func_mean(E, a, b, c):
#     ''' Michaelis-Menten a · E / (b + E) '''
#     return (a*E)/(b+E) + c


def ly1_func_mean(E, a, b, c):
    ''' Logarithmic fit: a * ln(E + b) + c '''
    # Using np.log handles arrays (E_range) automatically
    return a * np.log(E + b) + c


# def ly1_func_mean(E, a, b, c):
#     return a + b*E + c*E**2  
 
# def ly1_func_sigma(E, c, d):
#     """Linear:  sigma(E) = c + d*E"""
#     return c + d * E

# def ly1_func_sigma(E, a, b):
#     # Using np.log handles arrays (E_range) automatically
#     return a/(E + b)

def ly1_func_sigma(E, c, d):
    return c * np.power(E, d)   # d ~ -0.3 to -0.5

# def ly1_func_sigma(E, c, d):
#     """sqrt:  sigma(E) = sqrt(c + d*E)"""
#     return np.sqrt(c + d*E) 
 
# --- LY2 ---------------------------------------------------------------------
 
# def ly2_func_mean(E, a, b):
#     """Linear:  mu(E) = a + b*E"""
#     return a + b * E
 
# def ly2_func_sigma(E, c, d):
#     """Linear:  sigma(E) = c + d*E"""
#     return c + d * E


def ly2_func_mean(E, a, b, c):
    ''' Logarithmic fit: a * ln(E + b) + c '''
    # Using np.log handles arrays (E_range) automatically
    return a * np.log(E + b) + c

def ly2_func_sigma(E, c, d):
    return c * np.power(E, d)   # d ~ -0.3 to -0.5
 
# Other examples:
#
                        # quadratic
#
# def ly1_func_sigma(E, a, b):
#     return a * np.power(np.abs(E), b)                # power law
#
# def ly1_func_sigma(E, a, b):
#     return np.sqrt(a**2 * E**2 + b**2 * E)          # Poisson-like
 


 
 
# =============================================================================
# SECTION 3 — Energy interval construction
# =============================================================================
 
def build_energy_limits(E_start: float = 700., E_max: float = 2700.,
                        base_step: float = 50., growth: float = 1.2) -> list:
    limit = [E_start]
    i = 0
    while True:
        next_edge = limit[i] + base_step * (growth ** i)
        if next_edge > E_max:
            break
        limit.append(next_edge)
        i += 1
    return limit
 
 
def fill_map_ly_intervals(vectorEnergy, vectorStab, vectorLY1, vectorLY2, limit: list) -> dict:
    """
    Sort events into intervals based on vectorEnergy.
 
    mapLYIntervals[j][0]  ->  vectorEnergy values in interval j
    mapLYIntervals[j][1]  ->  LY1 values
    mapLYIntervals[j][2]  ->  LY2 values
    """
    n_intervals    = len(limit) - 1
    mapLYIntervals = {j: [[], [], []] for j in range(n_intervals)}
 
    for i in range(len(vectorEnergy)):
        energy = vectorEnergy[i]
        for j in range(n_intervals):
            if limit[j] <= energy < limit[j + 1]:
                mapLYIntervals[j][0].append(vectorStab[i])
                mapLYIntervals[j][1].append(vectorLY1[i])
                mapLYIntervals[j][2].append(vectorLY2[i])
                break
 
    return mapLYIntervals
 
 
# =============================================================================
# SECTION 4 — Robust per-interval mu/sigma estimation from histogram
# =============================================================================
def histogram_mu_sigma(
    ly_values: np.ndarray,
    n_bins: int = N_HIST_BINS,
    zero_exclusion_frac: float = 0.05,
    fit_window_sigma: float = 2.5,
    min_fit_points: int = 7,
):
    """
    Estimate mu and sigma from a histogram peak in a robust way.

    Improvements over the original:
      - Independent of fixed bin index windows
      - Uses data-driven peak width estimate (FWHM)
      - More stable initial sigma
      - Weighted fit using Poisson uncertainties
      - More precise sub-bin peak estimate
      - Better fallback if fit fails
      - Uses 80% of counts closest to maximum for robust fitting
    """

    def gauss(x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # -------------------------------------------------
    # Histogram
    # -------------------------------------------------
    counts, edges = np.histogram(
        ly_values,
        bins=n_bins,
        range=(LY_MIN, LY_MAX)
    )

    ctrs = 0.5 * (edges[:-1] + edges[1:])
    bw = edges[1] - edges[0]

    if counts.sum() == 0:
        return np.nan, np.nan, counts, edges

    # -------------------------------------------------
    # Ignore near-zero region
    # -------------------------------------------------
    zero_threshold = zero_exclusion_frac * LY_MAX
    valid = ctrs > zero_threshold

    if valid.any():
        masked_counts = np.where(valid, counts, 0)
        peak_idx = np.argmax(masked_counts)
    else:
        peak_idx = np.argmax(counts)

    # -------------------------------------------------
    # Sub-bin peak estimate using weighted centroid
    # -------------------------------------------------
    i0 = max(0, peak_idx - 1)
    i1 = min(len(ctrs), peak_idx + 2)

    local_c = counts[i0:i1].astype(float)
    local_x = ctrs[i0:i1]

    if local_c.sum() > 0:
        mu0 = np.sum(local_x * local_c) / np.sum(local_c)
    else:
        mu0 = ctrs[peak_idx]

    # -------------------------------------------------
    # Width estimate from RIGHT-side HWHM only
    # (left side may be contaminated by satellite peaks)
    # -------------------------------------------------
    half_max = counts[peak_idx] / 2.0

    right = peak_idx
    while right < len(counts) - 1 and counts[right] > half_max:
        right += 1

    right_hwhm = max((right - peak_idx) * bw, bw / 2)
    sigma0 = right_hwhm / (2.0 * np.log(2.0)) ** 0.5  # HWHM -> sigma

    # Safety floor
    sigma0 = max(sigma0, bw)

    # Constrain sigma so that peak - sigma > 0
    if mu0 > 0:
        sigma0 = min(sigma0, mu0 * 0.99)

    # -------------------------------------------------
    # Dynamic fit region using 80% of counts closest to maximum
    # -------------------------------------------------
    # Find cumulative counts sorted by distance to peak
    distances = np.abs(ctrs - mu0)
    
    # Create a list of (distance, count, x) triples and sort by distance
    bins_info = list(zip(distances, counts, ctrs))
    bins_info.sort(key=lambda x: x[0])  # Sort by distance to mu0
    
    # Calculate cumulative sum of counts
    cumsum_counts = 0
    total_counts = counts.sum()
    target_80_percent = 0.8 * total_counts
    
    fit_mask = np.zeros(len(ctrs), dtype=bool)
    max_distance_for_fit = 0
    
    # Include bins until we reach 80% of total counts
    for dist, cnt, x in bins_info:
        if cnt > 0:  # Only consider bins with counts
            cumsum_counts += cnt
            fit_mask[np.where(ctrs == x)[0][0]] = True
            max_distance_for_fit = dist
            if cumsum_counts >= target_80_percent:
                break
    
    # Optionally, also include neighboring bins within a safety margin
    # Add bins within 20% of the max distance to ensure continuity
    safety_margin = max_distance_for_fit * 1.2
    for i, (dist, cnt, x) in enumerate(bins_info):
        if dist <= safety_margin and not fit_mask[np.where(ctrs == x)[0][0]]:
            fit_mask[np.where(ctrs == x)[0][0]] = True
    
    # Apply the mask
    xfit = ctrs[fit_mask]
    yfit = counts[fit_mask]
    
    # Ensure we have enough points
    if len(xfit) < min_fit_points:
        # Fallback to sigma-based window
        fit_min = mu0 - fit_window_sigma * sigma0
        fit_max = mu0 + fit_window_sigma * sigma0
        mask = (ctrs >= fit_min) & (ctrs <= fit_max)
        
        if mask.sum() < min_fit_points:
            fit_min = mu0 - 4 * sigma0
            fit_max = mu0 + 4 * sigma0
            mask = (ctrs >= fit_min) & (ctrs <= fit_max)
        
        xfit = ctrs[mask]
        yfit = counts[mask]
    
    # Need enough populated bins
    good = yfit > 0
    xfit = xfit[good]
    yfit = yfit[good]

    if len(xfit) < 3:
        return mu0, sigma0, counts, edges

    # -------------------------------------------------
    # Poisson uncertainties
    # -------------------------------------------------
    sigma_y = np.sqrt(yfit)
    sigma_y[sigma_y == 0] = 1.0

    # -------------------------------------------------
    # Fit
    # -------------------------------------------------
    try:
        p0 = [counts[peak_idx], mu0, sigma0]

        bounds = (
            [0, LY_MIN, bw / 10],
            [np.inf, LY_MAX, (LY_MAX - LY_MIN)]
        )

        popt, _ = curve_fit(
            gauss,
            xfit,
            yfit,
            p0=p0,
            sigma=sigma_y,
            absolute_sigma=True,
            bounds=bounds,
            maxfev=20000
        )

        mu_est = popt[1]
        sigma_est = abs(popt[2])

    except Exception:
        mu_est = mu0
        sigma_est = sigma0

    return mu_est, sigma_est, counts, edges


# =============================================================================
# SECTION 5 — Data structure
# =============================================================================
 
@dataclass
class EnergyInterval:
    j:             int
    E_low:         float
    E_high:        float
    E_mean:        float        
    weight:        float  
    n_events:      int
    # robust per-interval estimates (used for auto p0 only)
    ly1_mu_est:    float
    ly1_sigma_est: float
    ly2_mu_est:    float
    ly2_sigma_est: float


    # histograms (built once, reused by NLL)
    ly1_counts:    np.ndarray
    ly1_edges:     np.ndarray
    ly2_counts:    np.ndarray
    ly2_edges:     np.ndarray

    ly1_raw: np.ndarray   
    ly2_raw: np.ndarray   

    ly1_mu_err:    float = 0.
    ly1_sigma_err: float = 0.
    ly2_mu_err:    float = 0.
    ly2_sigma_err: float = 0.

    ly1_peak_center: float = 0.   
    ly2_peak_center: float = 0.  


def build_intervals(mapLYIntervals: dict, limit: list,
                    n_hist_bins: int = N_HIST_BINS) -> List[EnergyInterval]:
    intervals = []
    
    # First pass: collect all sigma and mu estimates
    temp_intervals = []
    sig1_list = []
    sig2_list = []
    mu1_list = []
    mu2_list = []
    weight_list = []  # Add this
    
    for j in sorted(mapLYIntervals.keys()):
        energy = np.asarray(mapLYIntervals[j][0], dtype=float)
        ly1    = np.asarray(mapLYIntervals[j][1], dtype=float)
        ly2    = np.asarray(mapLYIntervals[j][2], dtype=float)

        if len(energy) < 5:
            print(f"[WARNING] Interval {j} [{limit[j]:.1f}, {limit[j+1]:.1f}) "
                  f"has only {len(energy)} events — skipping.")
            continue

        n_hist_bins_adj = int(n_hist_bins * 1.5)

        mu1, sig1, c1, e1 = histogram_mu_sigma(ly1, n_hist_bins_adj)
        mu2, sig2, c2, e2 = histogram_mu_sigma(ly2, n_hist_bins_adj)

        # Calculate weight based on energy
        E_mean = float(np.mean(energy))
        # Gentle weighting: linear with energy, normalized to [0.5, 1.5] range
        # Adjust based on your energy range (700-2700 keV)
        # Number of valid intervals
        n_total = len(limit) - 1

        # plateau starts 3 intervals before the end
        plateau_start = limit[-4]

        if E_mean >= plateau_start:
            x = (plateau_start - 700) / (2700 - 700)
        else:
            x = (E_mean - 700) / (2700 - 700)

        x = max(0.0, min(1.0, x))

        w_min = 0.2
        w_max = 1.0

        weight = w_min + (w_max - w_min) * np.sqrt(x)
        
        # Alternative: sqrt weighting (even gentler)
        # weight = np.sqrt(E_mean / 700)  # Range: 1 to ~1.96
        
        # Alternative: quadratic but limited (more aggressive but capped)
        # weight = min(2.0, 1.0 + ((E_mean - 700) / (2700 - 700))**2)

        # Peak bin center — used as window anchor in global_nll
        ctrs1 = 0.5 * (e1[:-1] + e1[1:])
        ctrs2 = 0.5 * (e2[:-1] + e2[1:])
        peak_center1 = float(ctrs1[np.argmax(c1)])
        peak_center2 = float(ctrs2[np.argmax(c2)])
        
        # Calculate separation quality optional
        if not np.isnan(sig1) and not np.isnan(sig2) and sig1 > 0 and sig2 > 0:
            separation = abs(mu2 - mu1) / (sig1 + sig2)
            # Combine with energy weight
            weight = weight * (1.0 + 0.5 * min(2.0, separation))
        
        temp_intervals.append({
            'j': j,
            'E_low': limit[j],
            'E_high': limit[j+1],
            'E_mean': E_mean,
            'weight': weight,  
            'n_events': len(energy),
            'ly1_mu_est': mu1,
            'ly1_sigma_est': sig1,
            'ly2_mu_est': mu2,
            'ly2_sigma_est': sig2,
            'ly1_counts': c1,
            'ly1_edges': e1,
            'ly2_counts': c2,
            'ly2_edges': e2,
            'ly1_raw': ly1,
            'ly2_raw': ly2,
            'ly1_peak_center': peak_center1,
            'ly2_peak_center': peak_center2,
        })
        
        if not np.isnan(sig1):
            sig1_list.append(sig1)
            mu1_list.append(mu1)
        if not np.isnan(sig2):
            sig2_list.append(sig2)
            mu2_list.append(mu2)

        weight_list.append(weight)
    
    
    # Calculate reference values (median) for sigmas and mus
        # Calculate weighted reference values (median) for sigmas and mus
    if len(sig1_list) > 0:
        # Use weighted median for reference values
        weight_arr = np.array(weight_list[:len(sig1_list)])
        
        # Weighted median for mu1
        idx = np.argsort(mu1_list)
        sorted_mu1 = np.array(mu1_list)[idx]
        sorted_weights_mu1 = weight_arr[idx]
        cumsum_mu1 = np.cumsum(sorted_weights_mu1)
        ref_mu1 = sorted_mu1[np.searchsorted(cumsum_mu1, 0.5 * cumsum_mu1[-1])]
        
        # Weighted median for sig1
        sorted_sig1 = np.array(sig1_list)[idx]
        sorted_weights_sig1 = weight_arr[idx]
        cumsum_sig1 = np.cumsum(sorted_weights_sig1)
        ref_sig1 = sorted_sig1[np.searchsorted(cumsum_sig1, 0.5 * cumsum_sig1[-1])]
        
        # Similarly for LY2
        idx2 = np.argsort(mu2_list)
        sorted_mu2 = np.array(mu2_list)[idx2]
        sorted_weights_mu2 = weight_arr[idx2]
        cumsum_mu2 = np.cumsum(sorted_weights_mu2)
        ref_mu2 = sorted_mu2[np.searchsorted(cumsum_mu2, 0.5 * cumsum_mu2[-1])]
        
        # Weighted median for sig1
        sorted_sig2 = np.array(sig2_list)[idx2]
        sorted_weights_sig2 = weight_arr[idx2]
        cumsum_sig2 = np.cumsum(sorted_weights_sig2)
        ref_sig2 = sorted_sig2[np.searchsorted(cumsum_sig2, 0.5 * cumsum_sig2[-1])]

        last = temp_intervals[-1]

        c1   = last['ly1_counts']
        e1   = last['ly1_edges']
        ctr1 = 0.5 * (e1[:-1] + e1[1:])
        win1 = (ctr1 >= ref_mu1 - ref_sig1) & (ctr1 <= ref_mu1 + ref_sig1)
        ref_peak_bin1 = float(ctr1[win1][np.argmax(c1[win1])]) if win1.any() else ref_mu1

        c2   = last['ly2_counts']
        e2   = last['ly2_edges']
        ctr2 = 0.5 * (e2[:-1] + e2[1:])
        win2 = (ctr2 >= ref_mu2 - ref_sig2) & (ctr2 <= ref_mu2 + ref_sig2)
        ref_peak_bin2 = float(ctr2[win2][np.argmax(c2[win2])]) if win2.any() else ref_mu2


        # Apply consistency check
        corrected_count = 0
        for interval in temp_intervals:
            sig1 = interval['ly1_sigma_est']
            sig2 = interval['ly2_sigma_est']
            
            # Check sig1 against reference
            if not np.isnan(sig1) and not (0.8 * ref_sig1 <= sig1 <= 1.2 * ref_sig1):
                counts = interval['ly1_counts']
                edges = interval['ly1_edges']
                if len(counts) > 0:
                    # Calculate bin centers
                    ctrs = 0.5 * (edges[:-1] + edges[1:])
                    # Define window around reference mu using reference sigma
                    window = (ctrs >= ref_mu1 - ref_sig1) & (ctrs <= ref_mu1 + ref_sig1)
                    
                    if window.any():
                        # Find peak within window
                        peak_bin = np.argmax(counts[window]) + np.where(window)[0][0]
                        peak_mu = ctrs[peak_bin]
                    else:
                        # Fallback: use the global maximum
                        peak_bin = np.argmax(counts)
                        peak_mu = ctrs[peak_bin]
                    
                    interval['ly1_sigma_est'] = ref_sig1
                    interval['ly1_mu_est'] = peak_mu
                    interval['ly1_peak_center'] = peak_mu
                else:
                    # Fallback to reference if histogram is empty
                    interval['ly1_sigma_est'] = ref_sig1
                    interval['ly1_mu_est'] = ref_mu1
                    interval['ly1_peak_center'] = ref_peak_bin1
                corrected_count += 1

                # Check sig2 against reference
                if not np.isnan(sig2) and not (0.5 * ref_sig2 <= sig2 <= 1.5 * ref_sig2):
                    counts = interval['ly2_counts']
                    edges = interval['ly2_edges']
                    if len(counts) > 0:
                        # Calculate bin centers
                        ctrs = 0.5 * (edges[:-1] + edges[1:])
                        # Define window around reference mu using reference sigma
                        window = (ctrs >= ref_mu2 - ref_sig2) & (ctrs <= ref_mu2 + ref_sig2)
                        
                        if window.any():
                            # Find peak within window
                            peak_bin = np.argmax(counts[window]) + np.where(window)[0][0]
                            peak_mu = ctrs[peak_bin]
                        else:
                            # Fallback: use the global maximum
                            peak_bin = np.argmax(counts)
                            peak_mu = ctrs[peak_bin]
                        
                        interval['ly2_sigma_est'] = ref_sig2
                        interval['ly2_mu_est'] = peak_mu
                        interval['ly2_peak_center'] = peak_mu
                    else:
                        # Fallback to reference if histogram is empty
                        interval['ly2_sigma_est'] = ref_sig2
                        interval['ly2_mu_est'] = ref_mu2
                        interval['ly2_peak_center'] = ref_peak_bin2
                    corrected_count += 1

            if corrected_count > 0:
                print(f"  Corrected {corrected_count} intervals (replaced mu and sigma with reference values)")
        
        # Second pass: create final EnergyInterval objects
        for interval in temp_intervals:
            intervals.append(EnergyInterval(
                j=interval['j'],
                E_low=interval['E_low'],
                E_high=interval['E_high'],
                E_mean=interval['E_mean'],
                weight=interval['weight'], 
                n_events=interval['n_events'],
                ly1_mu_est=interval['ly1_mu_est'],
                ly1_sigma_est=interval['ly1_sigma_est'],
                ly2_mu_est=interval['ly2_mu_est'],
                ly2_sigma_est=interval['ly2_sigma_est'],
                ly1_counts=interval['ly1_counts'],
                ly1_edges=interval['ly1_edges'],
                ly2_counts=interval['ly2_counts'],
                ly2_edges=interval['ly2_edges'],
                ly1_raw=interval['ly1_raw'],
                ly2_raw=interval['ly2_raw'],
                ly1_peak_center=interval['ly1_peak_center'],
                ly2_peak_center=interval['ly2_peak_center'],
            ))

            print(f"LY1:  Interval {interval['j']}: E=[{interval['E_low']:.1f}, {interval['E_high']:.1f}), "
                f"mu={interval['ly1_mu_est']:.8f}, sigma={interval['ly1_sigma_est']:.8f}, "
                f"bins={len(interval['ly1_counts'])}, peak={interval['ly1_peak_center']:.8f}")

            print(f"LY2:  Interval {interval['j']}: E=[{interval['E_low']:.1f}, {interval['E_high']:.1f}), "
                f"mu={interval['ly2_mu_est']:.8f}, sigma={interval['ly2_sigma_est']:.8f}, "
                f"bins={len(interval['ly2_counts'])}, peak={interval['ly2_peak_center']:.8f}")

            

        print(f"  -> {len(intervals)} valid intervals out of {len(mapLYIntervals)}")
        return intervals


# =============================================================================
# SECTION 6 — Auto p0 from per-interval histogram estimates
# =============================================================================

def auto_p0(intervals, func_mean, func_sigma, ly_key, n_sig=4):


    def gauss(x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    E_vals, mu_indep, sig_indep = [], [], []

    prev_mu = None
    prev_sigma = None

    for iv in intervals:
        counts = iv.ly1_counts if ly_key == 1 else iv.ly2_counts
        edges  = iv.ly1_edges  if ly_key == 1 else iv.ly2_edges
        ctrs   = 0.5 * (edges[:-1] + edges[1:])
        mu0    = iv.ly1_mu_est    if ly_key == 1 else iv.ly2_mu_est
        sig0   = iv.ly1_sigma_est if ly_key == 1 else iv.ly2_sigma_est
        raw    = iv.ly1_raw       if ly_key == 1 else iv.ly2_raw


        # Check if sigma is too small (below threshold)
        if sig0 < 1e-6:
            # Use previous valid values if available
            if prev_sigma is not None and prev_mu is not None:
                mu0 = prev_mu
                sig0 = prev_sigma
            # If no previous values, try to use a reasonable default based on bin width
            else:
                bw = edges[1] - edges[0] if len(edges) > 1 else 1.0
                sig0 = max(bw, 0.01)  # Fallback to bin width or 0.01
                # Keep mu0 as is (or could set to peak of histogram)
        
        # Update previous valid values for next iteration
        # Only update if current sigma is valid (or we used previous valid ones)
        if sig0 >= 1e-6:
            prev_mu = mu0
            prev_sigma = sig0


        try:
            p, pcov = curve_fit(
                gauss, ctrs, counts,
                p0=[counts.max(), mu0, sig0],
                maxfev=10_000,
            )
            mu_fit  = p[1]
            sig_fit = abs(p[2])
            # print(f"  Interval {iv.j}: gauss fit -> mu={mu_fit:.2e}, sigma={sig_fit:.2e}")
            perr    = np.sqrt(np.diag(pcov))
            mu_err, sig_err = perr[1], perr[2]
        except Exception as e:
            print(f"  [WARNING] interval {iv.j} gauss fit failed: {e}")
            mu_fit, sig_fit = mu0, sig0
            mu_err, sig_err = 0., 0.

        # Rebin the histogram using the fitted mu/sigma as window
        new_min = mu_fit - n_sig * sig_fit
        new_max = mu_fit + n_sig * sig_fit

        if new_min < LY_MIN:
            new_min = LY_MIN
        
        if new_max > LY_MAX:
            new_max = LY_MAX

        new_counts, new_edges = np.histogram(raw, bins=N_HIST_BINS,
                                             range=(new_min, new_max))

        if ly_key == 1:
            iv.ly1_mu_est    = mu_fit
            iv.ly1_sigma_est = sig_fit
            iv.ly1_counts    = new_counts
            iv.ly1_edges     = new_edges
            iv.ly1_mu_err    = mu_err
            iv.ly1_sigma_err = sig_err
        else:
            iv.ly2_mu_est    = mu_fit
            iv.ly2_sigma_est = sig_fit
            iv.ly2_counts    = new_counts
            iv.ly2_edges     = new_edges
            iv.ly2_mu_err    = mu_err
            iv.ly2_sigma_err = sig_err

        E_vals.append(iv.E_mean)
        mu_indep.append(mu_fit)
        sig_indep.append(sig_fit)

    E         = np.array(E_vals)
    mu_indep  = np.array(mu_indep)
    sig_indep = np.array(sig_indep)

    n_pm = len(inspect.signature(func_mean).parameters)  - 1
    n_ps = len(inspect.signature(func_sigma).parameters) - 1

    mu_scale  = np.median(mu_indep)
    sig_scale = np.median(sig_indep)
    E_scale   = np.median(E)

    p0_mean_init  = np.full(n_pm, mu_scale / max(n_pm, 1))
    p0_sigma_init = np.full(n_ps, sig_scale / max(n_ps, 1))
    if n_pm > 1: p0_mean_init[-1]  = mu_scale  / E_scale
    if n_ps > 1: p0_sigma_init[-1] = sig_scale / E_scale

    # When fitting initial parameters, use weights
    weights = np.array([iv.weight for iv in intervals])
    
    # Detect which parameter index is b (index 1) for log functions
    try:
        # bounds: a=free, b>0, c=free
        mean_bounds = (-np.inf * np.ones(n_pm), np.inf * np.ones(n_pm))
        mean_bounds[0][1] = 0.0   # lower bound on b
        pm, _ = curve_fit(func_mean, E, mu_indep, p0=p0_mean_init,
                        sigma=1.0/np.clip(weights, 1e-6, None),
                        absolute_sigma=True,
                        bounds=(mean_bounds[0], mean_bounds[1]),
                        maxfev=10_000)
    except Exception as e:
        warnings.warn(f"auto_p0 mean failed ({e}) — using heuristic")
        pm = p0_mean_init

    # Force b positive after fit just in case
    pm[1] = abs(pm[1])

    try:
        ps, _ = curve_fit(func_sigma, E, sig_indep, p0=p0_sigma_init,
                         sigma=1.0/np.clip(weights, 1e-6, None),
                         absolute_sigma=True, maxfev=10_000)
    except Exception as e:
        warnings.warn(f"auto_p0 sigma failed ({e}) — using heuristic")
        ps = p0_sigma_init

    print(f"  auto p0 mean  : {pm}")
    print(f"  auto p0 sigma : {ps}")
    return pm, ps

# =============================================================================
# SECTION 7 — Global NLL
# =============================================================================
 
def _gauss_cdf(x, mu, sigma):
    return 0.5 * (1.0 + erf((x - mu) / (sigma * np.sqrt(2.0))))
 
 
def _bin_prob(edges, mu, sigma):
    p = _gauss_cdf(edges[1:], mu, sigma) - _gauss_cdf(edges[:-1], mu, sigma)
    return np.clip(p, 1e-300, None)


def global_nll(params, intervals, func_mean, func_sigma, n_params_mean, ly_key,
               left_n_sig=4, right_n_sig=3.0):
    pm = params[:n_params_mean]
    ps = params[n_params_mean:]
    total = 0.0
    total_weight = 0.0  # For normalization

    for iv in intervals:
        mu    = func_mean( iv.E_mean, *pm)
        sigma = func_sigma(iv.E_mean, *ps)

        if sigma <= 0.0:
            return 1e15

        counts = iv.ly1_counts if ly_key == 1 else iv.ly2_counts
        edges  = iv.ly1_edges  if ly_key == 1 else iv.ly2_edges
        ctrs   = 0.5 * (edges[:-1] + edges[1:])

        sig_ref    = iv.ly1_sigma_est    if ly_key == 1 else iv.ly2_sigma_est
        peak_center = iv.ly1_peak_center if ly_key == 1 else iv.ly2_peak_center  

        min_counts_thr = int(counts.max() * 0.2)


        # if ly_key == 2:
        #     left_n_sig=1.0

        mask = (
            (counts >= min_counts_thr) &
            (ctrs >= peak_center - left_n_sig  * sig_ref) &  
            (ctrs <= peak_center + right_n_sig * sig_ref)
        )

        if not np.any(mask):
            continue

        p = _bin_prob(edges, mu, sigma)

        p_masked = p[mask]
        p_sum    = p_masked.sum()
        if p_sum <= 0:
            continue
        p_masked = p_masked / p_sum

        counts_masked = counts[mask]
        if counts_masked.sum() <= 0:
            continue

        # Weighted NLL contribution
        weight = getattr(iv, 'weight', 1.0)  # Default to 1.0 if not set
        term = -np.sum(counts_masked * np.log(np.clip(p_masked, 1e-300, None)))
        total += weight * term
        total_weight += weight

    # Return weighted average (or just total, depending on your preference)
    return total / total_weight if total_weight > 0 else total


# =============================================================================
# SECTION 8 — Fit runner
# =============================================================================
def make_scaled_nll(intervals, func_mean, func_sigma, npm, ly_key, scales):
    """Wraps global_nll so the optimizer works on O(1) parameters."""
    def scaled_nll(p_scaled):
        p_real = p_scaled * scales
        return global_nll(p_real, intervals, func_mean, func_sigma, npm, ly_key)
    return scaled_nll


def run_fit(
    intervals:  List[EnergyInterval],
    ly_key:     int,
    func_mean:  Callable,
    func_sigma: Callable,
    method:     str = "L-BFGS-B",
    bounds:     Optional[List[Tuple]] = None,
) -> dict:
    """
    1. Estimate p0 from histogram peak/HWHM via curve_fit  (auto_p0)
    2. Minimise the global binned NLL from that p0

    ly_key = 1  ->  fit LY1
    ly_key = 2  ->  fit LY2
    """
    label = f"LY{ly_key}"

    print(f"\n{'='*55}")
    print(f"  Fitting {label}  |  {len(intervals)} intervals  |  method={method}")

    # --- step 1: auto p0 -----------------------------------------------------
    pm0, ps0 = auto_p0(intervals, func_mean, func_sigma, ly_key)

    # Force b (index 1 in log mean) to be positive
    pm0[1] = abs(pm0[1])
    # Force a (index 0 in sigma) to be positive
    ps0[0] = abs(ps0[0])

    p0  = np.concatenate([pm0, ps0])
    npm = len(pm0)
    nps = len(ps0)

    nll_at_p0 = global_nll(p0, intervals, func_mean, func_sigma, npm, ly_key)
    print(f"  NLL at p0    : {nll_at_p0:.6f}")
    print(f"  p0 mean      : {pm0}")
    print(f"  p0 sigma     : {ps0}")
    print(f"  {len(p0)} free params")
    print(f"{'='*55}")

    # --- step 2: parameter scaling -------------------------------------------
    scales    = np.where(np.abs(p0) > 1e-30, np.abs(p0), 1.0)
    p0_scaled = p0 / scales
    obj       = make_scaled_nll(intervals, func_mean, func_sigma, npm, ly_key, scales)

    # --- step 3: build bounds ------------------------------------------------
    # mean params:  (a=free, b>0, c=free)   -> index 1 bounded below by 0
    # sigma params: (a>0,    b=free)         -> index npm bounded below by 0
    lo = [-np.inf] * (npm + nps)
    hi = [ np.inf] * (npm + nps)

    lo[1]    = 0.0   # b in log mean must be > 0  so that log(E+b) is defined for E>=0
    lo[npm]  = 0.0   # a in sigma (noise floor) must be > 0

    # Convert bounds to scaled space
    lo_scaled = [lo[i] / scales[i] if np.isfinite(lo[i]) else -np.inf
                 for i in range(len(lo))]
    hi_scaled = [hi[i] / scales[i] if np.isfinite(hi[i]) else  np.inf
                 for i in range(len(hi))]
    bounds_scaled = list(zip(lo_scaled, hi_scaled))

    # --- step 4: minimisation ------------------------------------------------
    best_result = None
    best_nll    = np.inf

    methods_to_try = ["L-BFGS-B", "Powell", "Nelder-Mead"]

    for meth in methods_to_try:
        print(f"\n  Trying {meth}...")
        try:
            res = minimize(
                obj,
                x0      = p0_scaled,
                method  = meth,
                bounds  = bounds_scaled if meth == "L-BFGS-B" else None,
                options = {
                    "maxiter": 50000,
                    "ftol"   : 1e-12,   # L-BFGS-B
                    "gtol"   : 1e-8,    # L-BFGS-B
                    "xatol"  : 1e-10,   # Nelder-Mead / Powell
                    "fatol"  : 1e-10,   # Nelder-Mead / Powell
                    "disp"   : True,
                },
            )
        except Exception as e:
            print(f"    {meth} raised: {e}")
            continue

        if not np.isfinite(res.fun):
            print(f"    {meth} returned non-finite NLL — skipping")
            continue

        print(f"    converged={res.success}  NLL={res.fun:.6f}")

        if res.fun < best_nll:
            best_nll    = res.fun
            best_result = res

        if res.success:
            break   # good enough — stop trying

    if best_result is None:
        print("  [WARNING] all optimisers failed — returning p0")
        best_result      = type("R", (), {})()   # dummy object
        best_result.x    = p0_scaled
        best_result.fun  = nll_at_p0
        best_result.success = False

    # --- step 5: unscale -----------------------------------------------------
    best_result.x = best_result.x * scales

    # Safety: re-enforce b > 0 after unscaling (Powell / Nelder-Mead ignore bounds)
    best_result.x[1]   = abs(best_result.x[1])    # b_mean
    best_result.x[npm] = abs(best_result.x[npm])  # a_sigma

    pm = best_result.x[:npm]
    ps = best_result.x[npm:]

    print(f"\n  Converged    : {best_result.success}")
    print(f"  NLL at p0    : {nll_at_p0:.6f}")
    print(f"  NLL final    : {best_result.fun:.6f}")
    print(f"  delta NLL    : {nll_at_p0 - best_result.fun:.6f}")
    print(f"  params_mean  : {pm}")
    print(f"  params_sigma : {ps}")

    return dict(
        params_mean  = pm,
        params_sigma = ps,
        p0_mean      = pm0,
        p0_sigma     = ps0,
        nll_p0       = nll_at_p0,
        nll_final    = best_result.fun,
        result       = best_result,
    )


# =============================================================================
# SECTION 9 — Plotting
# =============================================================================
def plot_fit(
    intervals: List[EnergyInterval],
    fit_output: dict,
    func_mean: Callable,
    func_sigma: Callable,
    ly_key: int,
    max_cols: int = 4,
    output_plot: Optional[str] = None,
    channel_info: Optional[str] = None,  # Add channel info
    runs_list: Optional[List[str]] = None,  # Add runs list
):
    
    pm    = fit_output["params_mean"]
    ps    = fit_output["params_sigma"]
    label = f"LY{ly_key}"
    
    # Create channel string for filename
    if channel_info:
        channel_str = f"_ch{channel_info}"
    else:
        channel_str = ""
    
    # Create runs string for display
    if runs_list:
        runs_display = ", ".join(runs_list)
        runs_text = f"Runs: {runs_display}"
    else:
        runs_text = ""
    
    E_vals  = np.array([iv.E_mean for iv in intervals])
    E_range = np.linspace(E_vals.min() * 0.95, E_vals.max() * 1.05, 400)
    
    n_iv   = len(intervals)
    n_cols = min(max_cols, n_iv)
    n_rows = (n_iv + n_cols - 1) // n_cols
    
    # Add extra space at bottom for runs text (increase figure height slightly)
    fig = plt.figure(figsize=(5 * n_cols, 5.0 + 3.5 * n_rows))  # Increased from 4.5 to 5.0
    gs  = gridspec.GridSpec(
        n_rows + 2, n_cols,  # Changed from n_rows+1 to n_rows+2
        height_ratios=[1.8] + [1.0] * n_rows + [0.2],  # Add small row at bottom for text
        hspace=0.60, wspace=0.38,
    )
    
    # --- summary panels ------------------------------------------------------
    ax_mu  = fig.add_subplot(gs[0, : n_cols // 2])
    ax_sig = fig.add_subplot(gs[0, n_cols // 2 :])
    
    # fitted curve
    ax_mu.plot(E_range, func_mean(E_range, *pm),  "r-", lw=2, label="Simoultaneous fit")
    ax_sig.plot(E_range, func_sigma(E_range, *ps), "r-", lw=2, label="Simoultaneous fit")

    # if ly_key==2:
    #     ax_sig.set_ylim(0.00003, 0.00015) 
    #     ax_mu.set_ylim(0.00016, 0.00022) 
    
    # per-interval NLL-fitted values as dots
    ax_mu.plot(E_vals,  func_mean(E_vals,  *pm),  "ro", ms=5)
    ax_sig.plot(E_vals, func_sigma(E_vals, *ps),  "ro", ms=5)
    
    # per-interval histogram estimates as crosses with error bars
    mu_est  = [iv.ly1_mu_est    if ly_key==1 else iv.ly2_mu_est    for iv in intervals]
    sig_est = [iv.ly1_sigma_est if ly_key==1 else iv.ly2_sigma_est for iv in intervals]
    mu_err  = [iv.ly1_mu_err    if ly_key==1 else iv.ly2_mu_err    for iv in intervals]
    sig_err = [iv.ly1_sigma_err if ly_key==1 else iv.ly2_sigma_err for iv in intervals]
    
    ax_mu.errorbar(E_vals, mu_est,  yerr=mu_err,  fmt="b+", ms=8,
                   capsize=3, elinewidth=1, label="Indipendent fit")
    ax_sig.errorbar(E_vals, sig_est, yerr=sig_err, fmt="b+", ms=8,
                    capsize=3, elinewidth=1, label="Indipendent fit")
    
    for ax, ylabel, title in [
        (ax_mu,  f"Mean {label}",  f"{label} — Mean vs Energy"),
        (ax_sig, f"Sigma {label}", f"{label} — Sigma vs Energy"),
    ]:
        ax.set_xlabel("Energy (a.u.)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # --- per-interval histograms ---------------------------------------------
    for idx, iv in enumerate(intervals):
        row = idx // n_cols + 1
        col = idx  % n_cols
        ax  = fig.add_subplot(gs[row, col])
        
        counts = iv.ly1_counts if ly_key == 1 else iv.ly2_counts
        edges  = iv.ly1_edges  if ly_key == 1 else iv.ly2_edges
        bw     = np.diff(edges)[0]
        ctrs   = 0.5 * (edges[:-1] + edges[1:])
        
        ax.bar(ctrs, counts, width=bw, color="steelblue", alpha=0.6)
        
        # --- NLL global fit (red) ---
        mu_j  = func_mean( iv.E_mean, *pm)
        sig_j = func_sigma(iv.E_mean, *ps)
        x_g   = np.linspace(edges[0], edges[-1], 400)
        gauss_nll = (counts.sum() * bw / (sig_j * np.sqrt(2 * np.pi))
                    * np.exp(-0.5 * ((x_g - mu_j) / sig_j) ** 2))
        ax.plot(x_g, gauss_nll, "r-", lw=1.8,
                label=f"Simoultaneous fit")
        
        # --- independent per-interval fit (blue) ---
        mu_ind  = iv.ly1_mu_est    if ly_key == 1 else iv.ly2_mu_est
        sig_ind = iv.ly1_sigma_est if ly_key == 1 else iv.ly2_sigma_est
        gauss_ind = (counts.sum() * bw / (sig_ind * np.sqrt(2 * np.pi))
                    * np.exp(-0.5 * ((x_g - mu_ind) / sig_ind) ** 2))
        ax.plot(x_g, gauss_ind, "b--", lw=1.4,
                label=f"Independent fit")
        
        ax.set_title(
            f"j={iv.j}  [{iv.E_low:.0f}, {iv.E_high:.0f})\n"
            f"E={iv.E_mean:.1f}  n={iv.n_events}",
            fontsize=7,
        )
        ax.set_xlabel(label, fontsize=7)
        ax.set_ylabel("Counts", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)
        
    
    # Add runs text at the bottom
    if runs_text:
        ax_text = fig.add_subplot(gs[-1, :])  # Use the last row for text
        ax_text.axis('off')  # Turn off axes
        ax_text.text(0.5, 0.5, runs_text, 
                    transform=ax_text.transAxes,
                    fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Create title with channel info
    if channel_info:
        title_text = f"Simultaneous Gaussian fit — {label} — Channel {channel_info}"
    else:
        title_text = f"Simultaneous Gaussian fit — {label}"
    
    plt.suptitle(title_text, fontsize=13, y=0.995)
    plt.tight_layout()
    
    # Save with channel name in filename
    out = f"{output_plot}/{label}_ch{channel_info}.png"

    
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)  # Close the figure instead of showing
    print(f"  Plot saved -> {out}")
# =============================================================================
# SECTION 10 — Main
# =============================================================================

def EnergiesLaw(vectorEnergy, vectorStab, vectorLY1, vectorLY2, output_plot, channel_info=None, runs_list=None):

    limit = build_energy_limits(E_start=700., E_max=2700.,
                                base_step=50., growth=1.2)
    print(f"{len(limit)-1} energy intervals  "
          f"[{limit[0]:.0f}, {limit[-1]:.0f}) keV")

    mapLYIntervals = fill_map_ly_intervals(
        vectorEnergy, vectorStab, vectorLY1, vectorLY2, limit
    )

    print("\nBuilding intervals ...")
    intervals = build_intervals(mapLYIntervals, limit, n_hist_bins=N_HIST_BINS)

    fit_ly1 = run_fit(
        intervals  = intervals,
        ly_key     = 1,
        func_mean  = ly1_func_mean,
        func_sigma = ly1_func_sigma,
        method     = "Nelder-Mead",
    )

    fit_ly2 = run_fit(
        intervals  = intervals,
        ly_key     = 2,
        func_mean  = ly2_func_mean,
        func_sigma = ly2_func_sigma,
        method     = "Nelder-Mead",
    )

    plot_fit(intervals, fit_ly1, ly1_func_mean, ly1_func_sigma, ly_key=1, output_plot=output_plot, channel_info=channel_info, runs_list=runs_list)
    plot_fit(intervals, fit_ly2, ly2_func_mean, ly2_func_sigma, ly_key=2, output_plot=output_plot, channel_info=channel_info, runs_list=runs_list)

    # Return the 4 callables with baked-in parameters
    pm1, ps1 = fit_ly1["params_mean"], fit_ly1["params_sigma"]
    pm2, ps2 = fit_ly2["params_mean"], fit_ly2["params_sigma"]

    mean_LD1  = lambda E: ly1_func_mean( E, *pm1)
    sigma_LD1 = lambda E: ly1_func_sigma(E, *ps1)
    mean_LD2  = lambda E: ly2_func_mean( E, *pm2)
    sigma_LD2 = lambda E: ly2_func_sigma(E, *ps2)

    return mean_LD1, sigma_LD1, mean_LD2, sigma_LD2


def FindCutCorrelation(ch_corr, chain_pulser):
    
    # compute median correlation for pulser events to find a cut value
    correlations = []
    
    for i in range(chain_pulser.GetEntries()):
        chain_pulser.GetEntry(i)
        
        ispulser = getattr(chain_pulser, 'heat_IsHeater', 0)
        
        if not ispulser:
            continue

        ch_corr.GetEntry(i)
        corr = getattr(ch_corr, 'heat_correlation', 0)
        

        if corr<0.8:
            continue
        
        correlations.append(corr)
    
    if len(correlations) > 0:
        # Calculate median
        correlations.sort()
        n = len(correlations)
        if n % 2 == 0:
            # Even number of entries: average of two middle values
            median_corr = (correlations[n//2 - 1] + correlations[n//2]) / 2.0
        else:
            # Odd number of entries: middle value
            median_corr = correlations[n//2]
    else:
        median_corr = 0.9  # default value if no pulser events found
    
    return median_corr

        

            
 


def create_ly_tree(chain_Stab, chain_optim, ch_corr, ch_module, chain_pulser, chRoughEnergy, output_file, output_plot, channel_info=None, runs_list=None, mode="OF"):
    """
    Create a separate LY tree with the same number of entries as the timestamp chain.
    
    Parameters:
    - chain: The TChain containing the timestamp tree from all merged files
    - output_file: The ROOT file where the LY tree will be written
    - output_plot: The directory where the plot will be saved
    
    Returns:
    - ly_tree: The created LY tree
    """
    

    # Create a new tree for LY values
    ly_tree = ROOT.TTree("LY", "Light Yield values")

    # Create branches for LY and associated variables
    ly_value = array('d', [0.])
    ly_LD1_value = array('d', [0.])
    ly_LD2_value = array('d', [0.])

    ly_tree.Branch("LD1_LY", ly_LD1_value, "LD1_LY/D")
    ly_tree.Branch("LD2_LY", ly_LD2_value, "LD2_LY/D")

    # define vectors of Ly used or energy law function with 700 < Energy < 2600

    EnergyVal = []     # wll be filled with stab values (proportional to energy)
    LY1Val = []
    LY2Val = []
    StabVal = []


    #find cut on correlation going over the pulser
    corrcut = FindCutCorrelation(ch_corr, chain_pulser)

    corrcut = corrcut * (1+1.1e-8)

    print(f"Correlation cut value determined from pulser events: {corrcut:.3f}")
    

    chain_used = None

    if mode == "OF":
        chain_used = chain_optim
        print("Using OF amplitudes for LY calculation")
    elif mode == "RoughEnergy":
        chain_used = chRoughEnergy
        print("Using RoughEnergy for LY calculation")
    elif mode == "Stab":
        chain_used = chain_Stab
        print("Using Stab for LY calculation")


    nentries = chain_used.GetEntries()    

    print(f"Calculating LY for {nentries} entries...")




    for i in range(nentries):
        chain_used.GetEntry(i)
        chain_optim.GetEntry(i)
        chRoughEnergy.GetEntry(i)
        
        # Get amplitudes from the chain
        amp_stab = getattr(chain_used, 'heat_amplitude', 0)
        amp_opt_LD1 = getattr(chain_optim, 'LD1_amplitude', 0)
        amp_opt_LD2 = getattr(chain_optim, 'LD2_amplitude', 0)
        calEner = getattr(chRoughEnergy, 'heat_amplitude', 0)


        # Calculate LY values and assign to arrays
        ly_LD1_value[0] = amp_opt_LD1 *1000 / amp_stab if amp_stab != 0 else 0
        ly_LD2_value[0] = amp_opt_LD2 *1000 / amp_stab if amp_stab != 0 else 0

        # print(f"Entry {i}: Energy={calEner:.1f}, Stab={amp_stab:.1f}, LY1={ly_LD1_value[0]:.3f}, LY2={ly_LD2_value[0]:.3f}")

        ly_tree.Fill()


        if calEner > 700 and calEner < 2600:
            ch_corr.GetEntry(i)
            chain_pulser.GetEntry(i)
            
            corr = getattr(ch_corr, 'heat_correlation', 0)
            ispulser = getattr(chain_pulser, 'heat_IsHeater', 0)
            
            
            if corr < corrcut  or ispulser:
                continue
            
            EnergyVal.append(calEner)
            StabVal.append(amp_stab)
            LY1Val.append(ly_LD1_value[0])
            LY2Val.append(ly_LD2_value[0])

        
        # Progress indicator
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{nentries} entries")

    # Write LY tree to file
    output_file.cd()
    ly_tree.Write()

    print(f"LY tree created with {ly_tree.GetEntries()} entries")


    mean_LD1, sigma_LD1, mean_LD2, sigma_LD2 = EnergiesLaw(
        EnergyVal, StabVal, LY1Val, LY2Val, output_plot,
        channel_info=channel_info, 
        runs_list=runs_list
    )

    # Now create second tree using the LY values
    print("Creating Discrimination parameter tree")
    second_tree = ROOT.TTree("DiscrPar", "Analysis based on LY values")

    # Create branches for second tree
    ld1_d1 = array('d', [0.])
    ld2_d2 = array('d', [0.])
    DiscrPar = array('d', [0.])

    second_tree.Branch("DiscrParLD1", ld1_d1, "DiscrParLD1/D")
    second_tree.Branch("DiscrParLD2", ld2_d2, "DiscrParLD2/D")
    second_tree.Branch("DiscrPar", DiscrPar, "DiscrPar/D")

    # # Fill second tree using the in-memory arrays from first tree
    # # Method A: Use the arrays directly (they still contain the last values)
    # # You need to loop through entries again

    # Now loop through each entry to analyze
    for i in range(nentries):
        # Get the reference amplitude for this entry
        chain_used.GetEntry(i)
        ref_amp = getattr(chain_used, 'heat_amplitude', 0)
        
        # Get the LY values for this entry
        ly_tree.GetEntry(i)
        LY1_ref = ly_LD1_value[0]
        LY2_ref = ly_LD2_value[0]

        ly_LD1_mean  = float(mean_LD1(ref_amp))
        ly_LD2_mean  = float(mean_LD2(ref_amp))
        ly_LD1_sigma = float(sigma_LD1(ref_amp))
        ly_LD2_sigma = float(sigma_LD2(ref_amp))

        n1 = (LY1_ref - ly_LD1_mean)/ly_LD1_sigma if ly_LD1_sigma > 0 else 1000
        n2 = (LY2_ref - ly_LD2_mean)/ly_LD2_sigma if ly_LD2_sigma > 0 else 1000

        ld1_d1[0] = n1
        ld2_d2[0] = n2  

        DiscrPar[0] = math.sqrt(n1**2 + n2**2)



        second_tree.Fill()

    
    second_tree.Write()
        
        
    
    # Clear histograms for this reference point? Or accumulate?
    # If you want fresh histograms for each ref_amp, create new ones here
    # But that might be inefficient. Better to bin by amp_stab as above.


    


        



    #     # Fill second tree with results
    #     ld1_ly_analysis[0] = mean1 if 'mean1' in locals() else LY1_ref
    #     # ld2_ly_analysis[0] = mean2 if 'mean2' in locals() else LY2_ref
    #     # ratio_ly[0] = mean1 / mean2 if mean2 != 0 else 0
        
    #     second_tree.Fill()

    # # After the loop, write all histograms to file
    # output_file.cd()


    # # Write the second tree
    # second_tree.Write()
    # print(f"\nAnalysis complete! Processed {nentries} entries")


    return ly_tree



def merge_runs(cross_dir, runs, channel=None, channelLD1=None, channelLD2=None, output_dir=".", outputPlot=""):
    

    if len(runs) < 2:
        raise ValueError("Need at least 2 runs")

    print("Merging runs:", runs)
    print("Channel:", channel if channel is not None else "ALL")

    files_by_channel = defaultdict(list)

    # --------------------------------------------
    # Collect files
    # --------------------------------------------
    for run in runs:
        run_str = str(run)
        if run_str.startswith("RUN"):
            run_number = int(run_str.replace("RUN", ""))
        else:
            run_number = int(run_str)

        run_fmt = f"RUN{run_number:06d}"
        ofdata_path = os.path.join(cross_dir, run_fmt, "Coincidence")

        if not os.path.isdir(ofdata_path):
            print(f"Warning: missing {ofdata_path}")
            continue

        for fname in os.listdir(ofdata_path):
            if not fname.endswith(".root"):
                continue

            if channel is None or channelLD1 is None or channelLD2 is None:
                continue

            parts = fname.replace(".root", "").split("_")

            ch_name = ""
            run_in_file = ""

            requested_ch = ""

            if len(parts) == 4:
                ch_name = f"{parts[0]}_{parts[1]}_{parts[2]}"
                run_in_file = int(parts[3])
                requested_ch = f"{channel}_{channelLD1}_{channelLD2}"
            elif len(parts) == 3:
                ch_name = f"{parts[0]}_{parts[1]}"
                run_in_file = int(parts[2])
                requested_ch = f"{channel}_{channelLD1}"
            else:
                continue

            if run_in_file != run_number:
                continue

            if ch_name != requested_ch:
                continue

            fullpath = os.path.join(ofdata_path, fname)
            print(fullpath)
            files_by_channel[ch_name].append(fullpath)

    if not files_by_channel:
        print("No matching files found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    out_base = f"{runs[0]}_{runs[-1]}" if len(runs) > 1 else runs[0]

    # --------------------------------------------
    # Merge with cumulative time
    # --------------------------------------------
    for ch, flist in sorted(files_by_channel.items()):
        if len(flist) != len(runs):
            print(f"Skipping channel {ch} (missing in some runs)")
            continue

        print(f"\nMerging channel {ch}")

        f0 = ROOT.TFile.Open(flist[0])
        tree_names = [k.GetName() for k in f0.GetListOfKeys() if k.GetClassName() == "TTree"]
        f0.Close()

        if not tree_names:
            print("No TTrees found")
            continue

        

        outname = os.path.join(output_dir, f"{out_base}_ch{ch}.root")
        outfile = ROOT.TFile(outname, "RECREATE")

        for treename in tree_names:
            chain = ROOT.TChain(treename)

            if treename == "global":
                continue  # skip global tree for now

            for f in flist:
                chain.Add(f)

            # Only for timestamp tree, add cumulative branch
            if treename == "timestamp":
                # Clone structure but empty
                newtree = chain.CloneTree(0)

                # New branches
                time_cum = array('d', [0.])
                run_number = array('i', [0])  # Store run number as integer
                
                newtree.Branch("time_cumulative", time_cum, "time_cumulative/D")
                newtree.Branch("run_number", run_number, "run_number/I")  # /I for unsigned integer

                # Precompute last time of each file and entries
                entries_per_file = []
                last_times = []
                file_run_numbers = []  # Store run numbers instead of names

                for idx, f in enumerate(flist):
                    tf = ROOT.TFile.Open(f)
                    t = tf.Get("timestamp")
                    entries = t.GetEntries()
                    entries_per_file.append(entries)
                    
                    # Get the last timestamp from this file
                    t.GetEntry(entries - 1)
                    last_times.append(t.heat_timefromstartrun)
                    
                    # Get the run number as integer
                    if idx < len(runs):
                        # Extract number from RUN format (e.g., "RUN000102" -> 102)
                        run_str = runs[idx]
                        if run_str.startswith("RUN"):
                            run_num = int(run_str.replace("RUN", ""))
                        else:
                            run_num = int(run_str)
                    else:
                        # Alternative: extract from filename
                        import re
                        match = re.search(r'RUN(\d+)', f)
                        if match:
                            run_num = int(match.group(1))
                        else:
                            run_num = idx  # Fallback to index
                    
                    file_run_numbers.append(run_num)
                    tf.Close()

                cumulative_offset = 0
                file_index = 0
                entry_start = 0
                entry_end = entries_per_file[file_index]

                # Loop over all entries
                for i, event in enumerate(chain):
                    # Check if we need to move to next file
                    if i >= entry_end and file_index < len(flist) - 1:
                        cumulative_offset += last_times[file_index]  
                        file_index += 1
                        entry_start = entry_end
                        entry_end += entries_per_file[file_index]
                    
                    # Fill cumulative time
                    time_cum[0] = event.heat_timefromstartrun + cumulative_offset
                    
                    # Fill run number
                    run_number[0] = file_run_numbers[file_index]
                    
                    newtree.Fill()

                outfile.cd()
                newtree.Write()
        
            else:
                # Merge all other trees normally
                chain.Merge(outfile, 0, "keep")
        
        chain_usedilization = ROOT.TChain("stabilization_all")
        chain_op = ROOT.TChain("optimumfilter_all")
        chRoughEn = ROOT.TChain("calibration_rough")

        chain_corr = ROOT.TChain("correlation_corr")
        chain_module = ROOT.TChain("module")
        chain_flag = ROOT.TChain("flagpropagator_heater")

        for f in flist:
            chain_usedilization.Add(f)
            chain_op.Add(f)
            chain_module.Add(f)
            chain_corr.Add(f)
            chain_flag.Add(f)
            chRoughEn.Add(f)

        create_ly_tree(
            chain_usedilization, chain_op, chain_corr, chain_module, 
            chain_flag, chRoughEn, outfile, outputPlot,
            channel_info=ch,  # Pass the channel name
            runs_list=runs,    # Pass the runs list
            mode="RoughEnergy"          # Pass the mode
        )

        globalTree = ROOT.TTree("global", "Empty global tree")
        outfile.cd()
        globalTree.Write()

        for key in outfile.GetListOfKeys():
            if key.GetClassName() == "TTree" and key.GetName() != "global":
                globalTree.AddFriend(key.GetName())

        globalTree.Write("", ROOT.TObject.kOverwrite)
        del globalTree
        gc.collect()
        
        outfile.Close()
        print("  ->", outname)

    print("\nDone.")


os.makedirs(args.outdir, exist_ok=True)
os.makedirs(args.outPlot, exist_ok=True)

merge_runs(
    cross_dir=args.cross,
    runs=args.runs,
    channel=args.channel,
    channelLD1=args.channelLD1,
    channelLD2=args.channelLD2,
    output_dir=args.outdir,
    outputPlot=args.outPlot
)


