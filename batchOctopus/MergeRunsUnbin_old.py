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
import csv

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

# ── Per-detector LY axis limits ──────────────────────────────────────────────
# Adjust these independently for LD1 (top) and LD2 (bottom) as needed.
LY_MIN_LD1, LY_MAX_LD1 = 0.0007, 0.01
LY_MIN_LD2, LY_MAX_LD2 = 0.0007, 0.01


# ── NLL fit window (number of sigma around the per-interval peak to include) ─
NLL_LEFT_N_SIGMA_LD1  = 1.5   
NLL_RIGHT_N_SIGMA_LD1 = 2.5   

NLL_LEFT_N_SIGMA_LD2  = 1.5  
NLL_RIGHT_N_SIGMA_LD2 = 2.5   

# ── LOGICA GESTIONE CSV ──────────────────────────────────────────────────────
PARAMS_DB = {}

def load_params_csv(filepath):
    """Legge il CSV in modo robusto (gestisce spazi, punto e virgola, BOM)."""
    if not os.path.exists(filepath):
        print(f"\n[WARNING] File CSV '{filepath}' non trovato! Uso i default.\n")
        return
    
    # utf-8-sig ignora eventuali caratteri invisibili all'inizio del file (BOM)
    with open(filepath, mode='r', encoding='utf-8-sig') as f:
        # Rileva automaticamente se il separatore è virgola o punto e virgola
        primo_rigo = f.readline()
        delim = ';' if ';' in primo_rigo else ','
        f.seek(0)
        
        reader = csv.DictReader(f, delimiter=delim)
        
        # Pulisce eventuali spazi accidentali nei nomi delle colonne
        if reader.fieldnames:
            reader.fieldnames = [col.strip() for col in reader.fieldnames]
            
        for row in reader:
            if 'Channel' not in row or not row['Channel']:
                continue
            
            ch = row['Channel'].strip()
            try:
                PARAMS_DB[ch] = {
                    'LY_MIN_LD1': float(row['LY_MIN_LD1']),
                    'LY_MAX_LD1': float(row['LY_MAX_LD1']),
                    'LY_MIN_LD2': float(row['LY_MIN_LD2']),
                    'LY_MAX_LD2': float(row['LY_MAX_LD2']),
                    'NLL_LEFT_N_SIGMA_LD1': float(row['NLL_LEFT_N_SIGMA_LD1']),
                    'NLL_RIGHT_N_SIGMA_LD1': float(row['NLL_RIGHT_N_SIGMA_LD1']),
                    'NLL_LEFT_N_SIGMA_LD2': float(row['NLL_LEFT_N_SIGMA_LD2']),
                    'NLL_RIGHT_N_SIGMA_LD2': float(row['NLL_RIGHT_N_SIGMA_LD2'])
                }
            except KeyError as e:
                print(f"\n[ERRORE CSV] Colonna non trovata: {e}. Controlla l'intestazione!\n")
                return
            except ValueError as e:
                print(f"\n[ERRORE CSV] Valore non numerico per il canale {ch}: {e}\n")
                return
                
    print(f"\n[OK] Caricati parametri per i canali: {list(PARAMS_DB.keys())}\n")

def set_channel_params(ch):
    """Aggiorna le variabili globali con i parametri specifici del canale."""
    global LY_MIN_LD1, LY_MAX_LD1, LY_MIN_LD2, LY_MAX_LD2
    global NLL_LEFT_N_SIGMA_LD1, NLL_RIGHT_N_SIGMA_LD1
    global NLL_LEFT_N_SIGMA_LD2, NLL_RIGHT_N_SIGMA_LD2

    if ch in PARAMS_DB:
        p = PARAMS_DB[ch]
        LY_MIN_LD1 = p['LY_MIN_LD1']
        LY_MAX_LD1 = p['LY_MAX_LD1']
        LY_MIN_LD2 = p['LY_MIN_LD2']
        LY_MAX_LD2 = p['LY_MAX_LD2']
        NLL_LEFT_N_SIGMA_LD1  = p['NLL_LEFT_N_SIGMA_LD1']
        NLL_RIGHT_N_SIGMA_LD1 = p['NLL_RIGHT_N_SIGMA_LD1']
        NLL_LEFT_N_SIGMA_LD2  = p['NLL_LEFT_N_SIGMA_LD2']
        NLL_RIGHT_N_SIGMA_LD2 = p['NLL_RIGHT_N_SIGMA_LD2']
        print(f"Applicati parametri personalizzati per il canale {ch}")
    else:
        print(f"Nessun parametro nel CSV per il canale {ch}, uso i valori attuali/default.")

# Helper: select the right limits given ly_key (1 or 2)
def ly_range(ly_key: int) -> Tuple[float, float]:
    return (LY_MIN_LD1, LY_MAX_LD1) if ly_key == 1 else (LY_MIN_LD2, LY_MAX_LD2)

ONLYTOPLD = False


def format_run(run):
    run = str(run)
    if run.startswith("RUN"):
        return run
    return f"RUN{int(run):06d}"


# --- LY1 ---------------------------------------------------------------------

def ly1_func_mean(E, a, b, c):
    ''' Logarithmic fit: a * ln(E + b) + c '''
    return a * np.log(E + b) + c

def ly1_func_sigma(E, c, d):
    return c * np.power(E, d)   # d ~ -0.3 to -0.5

# --- LY2 ---------------------------------------------------------------------

def ly2_func_mean(E, a, b, c):
    ''' Logarithmic fit: a * ln(E + b) + c '''
    return a * np.log(E + b) + c

def ly2_func_sigma(E, c, d):
    return c * np.power(E, d)   # d ~ -0.3 to -0.5


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


def fill_map_ly_intervals(vectorEnergy, vectorStab, vectorLY1, vectorLY2,
                           limit: list) -> dict:
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
    ly_key: int,                        # NEW — selects the right axis range
    n_bins: int = N_HIST_BINS,
    zero_exclusion_frac: float = 0.05,
    fit_window_sigma: float = 2.5,
    min_fit_points: int = 7,
):
    """
    Estimate mu and sigma from a histogram peak in a robust way.
    Uses per-detector LY axis limits (ly_key selects LD1 or LD2).
    """

    LY_MIN, LY_MAX = ly_range(ly_key)

    def gauss(x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    counts, edges = np.histogram(
        ly_values,
        bins=n_bins,
        range=(LY_MIN, LY_MAX)
    )

    ctrs = 0.5 * (edges[:-1] + edges[1:])
    bw = edges[1] - edges[0]

    if counts.sum() == 0:
        return np.nan, np.nan, counts, edges

    # Ignore near-zero region
    zero_threshold = zero_exclusion_frac * LY_MAX
    valid = ctrs > zero_threshold

    if valid.any():
        masked_counts = np.where(valid, counts, 0)
        peak_idx = np.argmax(masked_counts)
    else:
        peak_idx = np.argmax(counts)

    # Sub-bin peak estimate
    i0 = max(0, peak_idx - 1)
    i1 = min(len(ctrs), peak_idx + 2)
    local_c = counts[i0:i1].astype(float)
    local_x = ctrs[i0:i1]
    mu0 = (np.sum(local_x * local_c) / np.sum(local_c)
           if local_c.sum() > 0 else ctrs[peak_idx])

    # Width from right-side HWHM
    half_max  = counts[peak_idx] / 2.0
    right     = peak_idx
    while right < len(counts) - 1 and counts[right] > half_max:
        right += 1
    right_hwhm = max((right - peak_idx) * bw, bw / 2)
    sigma0     = right_hwhm / (2.0 * np.log(2.0)) ** 0.5
    sigma0     = max(sigma0, bw)
    if mu0 > 0:
        sigma0 = min(sigma0, mu0 * 0.99)

    # Dynamic fit region using 80 % of counts closest to peak
    distances  = np.abs(ctrs - mu0)
    bins_info  = sorted(zip(distances, counts, ctrs), key=lambda x: x[0])
    cumsum_counts = 0
    total_counts  = counts.sum()
    target        = 0.8 * total_counts
    fit_mask      = np.zeros(len(ctrs), dtype=bool)
    max_dist      = 0.0

    for dist, cnt, x in bins_info:
        if cnt > 0:
            cumsum_counts += cnt
            fit_mask[np.where(ctrs == x)[0][0]] = True
            max_dist = dist
            if cumsum_counts >= target:
                break

    safety = max_dist * 1.2
    for dist, cnt, x in bins_info:
        idx_arr = np.where(ctrs == x)[0]
        if dist <= safety and len(idx_arr) and not fit_mask[idx_arr[0]]:
            fit_mask[idx_arr[0]] = True

    xfit = ctrs[fit_mask]
    yfit = counts[fit_mask]

    if len(xfit) < min_fit_points:
        mask = (ctrs >= mu0 - fit_window_sigma * sigma0) & \
               (ctrs <= mu0 + fit_window_sigma * sigma0)
        if mask.sum() < min_fit_points:
            mask = (ctrs >= mu0 - 4 * sigma0) & (ctrs <= mu0 + 4 * sigma0)
        xfit = ctrs[mask]
        yfit = counts[mask]

    good = yfit > 0
    xfit = xfit[good]
    yfit = yfit[good]

    if len(xfit) < 3:
        return mu0, sigma0, counts, edges

    sigma_y = np.sqrt(yfit)
    sigma_y[sigma_y == 0] = 1.0

    try:
        popt, _ = curve_fit(
            gauss, xfit, yfit,
            p0=[counts[peak_idx], mu0, sigma0],
            sigma=sigma_y, absolute_sigma=True,
            bounds=([0, LY_MIN, bw / 10], [np.inf, LY_MAX, LY_MAX - LY_MIN]),
            maxfev=20000
        )
        mu_est    = popt[1]
        sigma_est = abs(popt[2])
    except Exception:
        mu_est    = mu0
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
    ly1_mu_est:    float
    ly1_sigma_est: float
    ly2_mu_est:    float
    ly2_sigma_est: float
    ly1_counts:    np.ndarray
    ly1_edges:     np.ndarray
    ly2_counts:    np.ndarray
    ly2_edges:     np.ndarray
    ly1_raw:       np.ndarray
    ly2_raw:       np.ndarray
    ly1_mu_err:    float = 0.
    ly1_sigma_err: float = 0.
    ly2_mu_err:    float = 0.
    ly2_sigma_err: float = 0.
    ly1_peak_center: float = 0.
    ly2_peak_center: float = 0.


def build_intervals(mapLYIntervals: dict, limit: list,
                    n_hist_bins: int = N_HIST_BINS) -> List[EnergyInterval]:
    intervals = []
    global ONLYTOPLD

    temp_intervals = []
    sig1_list, sig2_list = [], []
    mu1_list,  mu2_list  = [], []
    weight_list          = []

    for j in sorted(mapLYIntervals.keys()):
        energy = np.asarray(mapLYIntervals[j][0], dtype=float)
        ly1    = np.asarray(mapLYIntervals[j][1], dtype=float)
        ly2    = np.asarray(mapLYIntervals[j][2], dtype=float)

        if len(energy) < 5:
            print(f"[WARNING] Interval {j} [{limit[j]:.1f}, {limit[j+1]:.1f}) "
                  f"has only {len(energy)} events — skipping.")
            continue

        n_hist_bins_adj = int(n_hist_bins * 1.5)

        # Pass ly_key so each detector uses its own range
        mu1, sig1, c1, e1 = histogram_mu_sigma(ly1, ly_key=1, n_bins=n_hist_bins_adj)
        mu2, sig2, c2, e2 = histogram_mu_sigma(ly2, ly_key=2, n_bins=n_hist_bins_adj)

        E_mean = float(np.mean(energy))
        plateau_start = limit[-4]
        x = (min(E_mean, plateau_start) - 700) / (2700 - 700)
        x = max(0.0, min(1.0, x))
        weight = 0.2 + 0.8 * np.sqrt(x)

        ctrs1 = 0.5 * (e1[:-1] + e1[1:])
        ctrs2 = 0.5 * (e2[:-1] + e2[1:])
        peak_center1 = float(ctrs1[np.argmax(c1)])
        peak_center2 = float(ctrs2[np.argmax(c2)])

        if not np.isnan(sig1) and not np.isnan(sig2) and sig1 > 0 and sig2 > 0:
            separation = abs(mu2 - mu1) / (sig1 + sig2)
            weight = weight * (1.0 + 0.5 * min(2.0, separation))
            
        #weight = 1

        temp_intervals.append(dict(
            j=j, E_low=limit[j], E_high=limit[j+1], E_mean=E_mean,
            weight=weight, n_events=len(energy),
            ly1_mu_est=mu1, ly1_sigma_est=sig1,
            ly2_mu_est=mu2, ly2_sigma_est=sig2,
            ly1_counts=c1, ly1_edges=e1,
            ly2_counts=c2, ly2_edges=e2,
            ly1_raw=ly1, ly2_raw=ly2,
            ly1_peak_center=peak_center1,
            ly2_peak_center=peak_center2,
        ))

        if not np.isnan(sig1): sig1_list.append(sig1); mu1_list.append(mu1)
        if not np.isnan(sig2): sig2_list.append(sig2); mu2_list.append(mu2)
        weight_list.append(weight)

    if not sig1_list:
        return intervals

    # ---- Weighted-median reference values -----------------------------------
    def weighted_median(vals, weights):
        idx    = np.argsort(vals)
        sv     = np.array(vals)[idx]
        sw     = np.array(weights)[idx]
        cumsum = np.cumsum(sw)
        return sv[np.searchsorted(cumsum, 0.5 * cumsum[-1])]

    w_arr   = np.array(weight_list[:len(sig1_list)])
    ref_mu1  = weighted_median(mu1_list,  w_arr)
    ref_sig1 = weighted_median(sig1_list, w_arr)

    if not ONLYTOPLD:
        w_arr2   = np.array(weight_list[:len(sig2_list)])
        ref_mu2  = weighted_median(mu2_list,  w_arr2)
        ref_sig2 = weighted_median(sig2_list, w_arr2)

    def _peak_in_window(counts, edges, ref_mu, ref_sig):
        ctrs = 0.5 * (edges[:-1] + edges[1:])
        win  = (ctrs >= ref_mu - ref_sig) & (ctrs <= ref_mu + ref_sig)
        if win.any():
            return float(ctrs[win][np.argmax(counts[win])])
        return float(ctrs[np.argmax(counts)])

    last = temp_intervals[-1]
    ref_peak_bin1 = _peak_in_window(last['ly1_counts'], last['ly1_edges'],
                                    ref_mu1, ref_sig1)
    if not ONLYTOPLD:
        ref_peak_bin2 = _peak_in_window(last['ly2_counts'], last['ly2_edges'],
                                        ref_mu2, ref_sig2)

    # ---- Consistency correction ---------------------------------------------
    corrected_count = 0
    for iv in temp_intervals:
        sig1 = iv['ly1_sigma_est']
        if not np.isnan(sig1) and not (0.8 * ref_sig1 <= sig1 <= 1.2 * ref_sig1):
            pk = _peak_in_window(iv['ly1_counts'], iv['ly1_edges'],
                                 ref_mu1, ref_sig1)
            iv['ly1_sigma_est']   = ref_sig1
            iv['ly1_mu_est']      = pk
            iv['ly1_peak_center'] = pk
            corrected_count += 1

        if not ONLYTOPLD:
            sig2 = iv['ly2_sigma_est']
            if not np.isnan(sig2) and not (0.5 * ref_sig2 <= sig2 <= 1.5 * ref_sig2):
                pk = _peak_in_window(iv['ly2_counts'], iv['ly2_edges'],
                                     ref_mu2, ref_sig2)
                iv['ly2_sigma_est']   = ref_sig2
                iv['ly2_mu_est']      = pk
                iv['ly2_peak_center'] = pk
                corrected_count += 1

    if corrected_count:
        print(f"  Corrected {corrected_count} intervals")

    # ---- Build final objects -------------------------------------------------
    for iv in temp_intervals:
        intervals.append(EnergyInterval(**{k: iv[k] for k in iv}))

        print(f"LY1  j={iv['j']}: E=[{iv['E_low']:.1f},{iv['E_high']:.1f}) "
              f"mu={iv['ly1_mu_est']:.6e}  sigma={iv['ly1_sigma_est']:.6e}  "
              f"peak={iv['ly1_peak_center']:.6e}")
        if not ONLYTOPLD:
            print(f"LY2  j={iv['j']}: E=[{iv['E_low']:.1f},{iv['E_high']:.1f}) "
                  f"mu={iv['ly2_mu_est']:.6e}  sigma={iv['ly2_sigma_est']:.6e}  "
                  f"peak={iv['ly2_peak_center']:.6e}")

    print(f"  -> {len(intervals)} valid intervals out of {len(mapLYIntervals)}")
    return intervals


# =============================================================================
# SECTION 6 — Auto p0 from per-interval histogram estimates
# =============================================================================

def auto_p0(intervals, func_mean, func_sigma, ly_key, n_sig=4):

    if ONLYTOPLD and ly_key == 2:   # ← fixed (was: `not ONLYTOPLD`)
        return None, None

    LY_MIN, LY_MAX = ly_range(ly_key)

    def gauss(x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    E_vals, mu_indep, sig_indep = [], [], []
    prev_mu, prev_sigma = None, None

    prevmin, prevmax = 0, 0

    for iv in intervals:
        counts = iv.ly1_counts if ly_key == 1 else iv.ly2_counts
        edges  = iv.ly1_edges  if ly_key == 1 else iv.ly2_edges
        ctrs   = 0.5 * (edges[:-1] + edges[1:])
        mu0    = iv.ly1_mu_est    if ly_key == 1 else iv.ly2_mu_est
        sig0   = iv.ly1_sigma_est if ly_key == 1 else iv.ly2_sigma_est
        raw    = iv.ly1_raw       if ly_key == 1 else iv.ly2_raw

        if sig0 < 1e-6:
            if prev_sigma is not None:
                mu0, sig0 = prev_mu, prev_sigma
            else:
                bw   = edges[1] - edges[0] if len(edges) > 1 else 1.0
                sig0 = max(bw, 0.01)

        if sig0 >= 1e-6:
            prev_mu    = mu0
            prev_sigma = sig0

        try:
            p, pcov = curve_fit(
                gauss, ctrs, counts,
                p0=[counts.max(), mu0, sig0],
                maxfev=10_000,
            )
            mu_fit  = p[1]
            sig_fit = abs(p[2])
            perr    = np.sqrt(np.diag(pcov))
            mu_err, sig_err = perr[1], perr[2]
        except Exception as e:
            print(f"  [WARNING] interval {iv.j} gauss fit failed ({e})")
            mu_fit, sig_fit = mu0, sig0
            mu_err, sig_err = 0., 0.

        # Rebin around fitted peak
        new_min = max(mu_fit - n_sig * sig_fit, LY_MIN)
        new_max = min(mu_fit + n_sig * sig_fit, LY_MAX)


        if np.isnan(new_min):
            new_min = prevmin
        else:
            prevmin = new_min

        if np.isnan(new_max):
            new_max = prevmax
        else:
            prevmax = new_max

        if new_min > new_max:
            new_min, new_max = new_max, new_min


    
        new_counts, new_edges = np.histogram(
            raw,
            bins=N_HIST_BINS,
            range=(new_min, new_max)
        )
    

        if ly_key == 1:
            iv.ly1_mu_est    = mu_fit;  iv.ly1_sigma_est = sig_fit
            iv.ly1_counts    = new_counts; iv.ly1_edges  = new_edges
            iv.ly1_mu_err    = mu_err;  iv.ly1_sigma_err = sig_err
        else:
            iv.ly2_mu_est    = mu_fit;  iv.ly2_sigma_est = sig_fit
            iv.ly2_counts    = new_counts; iv.ly2_edges  = new_edges
            iv.ly2_mu_err    = mu_err;  iv.ly2_sigma_err = sig_err

        E_vals.append(iv.E_mean)
        mu_indep.append(mu_fit)
        sig_indep.append(sig_fit)

    E        = np.array(E_vals)
    mu_arr   = np.array(mu_indep)
    sig_arr  = np.array(sig_indep)
    weights  = np.array([iv.weight for iv in intervals])

    n_pm = len(inspect.signature(func_mean).parameters)  - 1
    n_ps = len(inspect.signature(func_sigma).parameters) - 1

    mu_scale  = np.median(mu_arr)
    sig_scale = np.median(sig_arr)
    E_scale   = np.median(E)

    p0_mean_init  = np.full(n_pm, mu_scale  / max(n_pm, 1))
    p0_sigma_init = np.full(n_ps, sig_scale / max(n_ps, 1))
    if n_pm > 1: p0_mean_init[-1]  = mu_scale  / E_scale
    if n_ps > 1: p0_sigma_init[-1] = sig_scale / E_scale

    try:
        mean_bounds = (-np.inf * np.ones(n_pm), np.inf * np.ones(n_pm))
        mean_bounds[0][1] = 0.0
        pm, _ = curve_fit(func_mean, E, mu_arr, p0=p0_mean_init,
                          sigma=1.0 / np.clip(weights, 1e-6, None),
                          absolute_sigma=True,
                          bounds=(mean_bounds[0], mean_bounds[1]),
                          maxfev=10_000)
    except Exception as e:
        warnings.warn(f"auto_p0 mean failed ({e})")
        pm = p0_mean_init

    pm[1] = abs(pm[1])

    try:
        ps, _ = curve_fit(func_sigma, E, sig_arr, p0=p0_sigma_init,
                          sigma=1.0 / np.clip(weights, 1e-6, None),
                          absolute_sigma=True, maxfev=10_000)
    except Exception as e:
        warnings.warn(f"auto_p0 sigma failed ({e})")
        ps = p0_sigma_init

    print(f"  auto p0 mean  : {pm}")
    print(f"  auto p0 sigma : {ps}")
    return pm, ps


# =============================================================================
# SECTION 7 — GLOBAL UNBINNED NLL
# =============================================================================

LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def gaussian_nll(x, mu, sigma):
    sigma = np.clip(sigma, 1e-12, None)
    z     = (x - mu) / sigma
    return LOG_SQRT_2PI + np.log(sigma) + 0.5 * z * z


def global_unbinned_nll(
    params,
    intervals,
    func_mean,
    func_sigma,
    n_params_mean,
    ly_key,
    left_n_sig=None,
    right_n_sig=None,
):

    if ly_key == 1: 
        if left_n_sig  is None: left_n_sig  = NLL_LEFT_N_SIGMA_LD1
        if right_n_sig is None: right_n_sig = NLL_RIGHT_N_SIGMA_LD1

    if ly_key == 2: 
        if left_n_sig  is None: left_n_sig  = NLL_LEFT_N_SIGMA_LD2
        if right_n_sig is None: right_n_sig = NLL_RIGHT_N_SIGMA_LD2

    pm = params[:n_params_mean]
    ps = params[n_params_mean:]

    total_nll    = 0.0
    total_weight = 0.0

    for iv in intervals:
        data = iv.ly1_raw if ly_key == 1 else iv.ly2_raw
        if len(data) == 0:
            continue

        energies = np.full(len(data), iv.E_mean)
        mu       = func_mean( energies, *pm)
        sigma    = func_sigma(energies, *ps)

        if not np.all(np.isfinite(mu))    : return 1e30
        if not np.all(np.isfinite(sigma)) : return 1e30
        if np.any(sigma <= 0)             : return 1e30

        mu_ref  = iv.ly1_mu_est    if ly_key == 1 else iv.ly2_mu_est
        sig_ref = iv.ly1_sigma_est if ly_key == 1 else iv.ly2_sigma_est

        mask = ((data >= mu_ref - left_n_sig  * sig_ref) &
                (data <= mu_ref + right_n_sig * sig_ref))

        data_fit  = data[mask]
        if len(data_fit) < 5:
            continue

        term   = np.sum(gaussian_nll(data_fit, mu[mask], sigma[mask]))
        weight = getattr(iv, "weight", 1.0)
        total_nll    += weight * term
        total_weight += weight

    return total_nll / total_weight if total_weight > 0 else 1e30


# =============================================================================
# SECTION 8 — FIT RUNNER
# =============================================================================

def make_scaled_nll(intervals, func_mean, func_sigma, npm, ly_key, scales):
    def scaled_nll(p_scaled):
        return global_unbinned_nll(
            p_scaled * scales, intervals, func_mean, func_sigma, npm, ly_key
        )
    return scaled_nll


def run_fit(
    intervals:  List[EnergyInterval],
    ly_key:     int,
    func_mean:  Callable,
    func_sigma: Callable,
    method:     str = "L-BFGS-B",
    bounds:     Optional[List[Tuple]] = None,
) -> dict:

    label = f"LY{ly_key}"
    print(f"\n{'='*60}\n  UNBINNED FIT {label}\n  intervals : {len(intervals)}\n{'='*60}")

    pm0, ps0 = auto_p0(intervals, func_mean, func_sigma, ly_key)

    pm0[1] = abs(pm0[1])
    ps0[0] = abs(ps0[0])

    p0  = np.concatenate([pm0, ps0])
    npm = len(pm0)
    nps = len(ps0)

    nll_at_p0 = global_unbinned_nll(p0, intervals, func_mean, func_sigma, npm, ly_key)
    print(f"Initial NLL : {nll_at_p0:.6f}")

    scales    = np.where(np.abs(p0) > 1e-30, np.abs(p0), 1.0)
    p0_scaled = p0 / scales
    obj       = make_scaled_nll(intervals, func_mean, func_sigma, npm, ly_key, scales)

    lo = [-np.inf] * (npm + nps)
    hi = [ np.inf] * (npm + nps)
    lo[1]   = 0.0   # log(E+b): b > 0
    lo[npm] = 0.0   # sigma prefactor > 0

    bounds_scaled = [
        (lo[i] / scales[i] if np.isfinite(lo[i]) else -np.inf,
         hi[i] / scales[i] if np.isfinite(hi[i]) else  np.inf)
        for i in range(len(lo))
    ]

    best_result = None
    best_nll    = np.inf

    for meth in ["L-BFGS-B", "Powell", "Nelder-Mead"]:
        print(f"\nTrying minimizer: {meth}")
        try:
            res = minimize(
                obj, x0=p0_scaled, method=meth,
                bounds=bounds_scaled if meth == "L-BFGS-B" else None,
                options={"maxiter": 50000, "ftol": 1e-12, "gtol": 1e-8,
                         "xatol": 1e-10, "fatol": 1e-10, "disp": True},
            )
        except Exception as e:
            print(f"Minimizer failed: {e}"); continue

        if not np.isfinite(res.fun):
            print("Non-finite NLL"); continue

        print(f"Converged : {res.success}  NLL : {res.fun:.6f}")
        if res.fun < best_nll:
            best_nll, best_result = res.fun, res
        if res.success:
            break

    if best_result is None:
        print("All minimizers failed — returning p0.")
        best_result         = type("R", (), {})()
        best_result.x       = p0_scaled
        best_result.fun     = nll_at_p0
        best_result.success = False

    best_result.x     = best_result.x * scales
    best_result.x[1]  = abs(best_result.x[1])
    best_result.x[npm]= abs(best_result.x[npm])

    pm = best_result.x[:npm]
    ps = best_result.x[npm:]

    print(f"\n{'='*60}\nFINAL RESULT {label}")
    print(f"Converged : {best_result.success}")
    print(f"NLL p0    : {nll_at_p0:.6f}  NLL final : {best_result.fun:.6f}  "
          f"Delta : {nll_at_p0 - best_result.fun:.6f}")
    print("Mean params  :", "  ".join(f"p{i}={v:.6e}" for i, v in enumerate(pm)))
    print("Sigma params :", "  ".join(f"p{i}={v:.6e}" for i, v in enumerate(ps)))
    print('='*60)

    return dict(params_mean=pm, params_sigma=ps,
                p0_mean=pm0, p0_sigma=ps0,
                nll_p0=nll_at_p0, nll_final=best_result.fun,
                result=best_result)


# =============================================================================
# SECTION 9 — Plotting
# =============================================================================

def plot_fit(
    intervals:    List[EnergyInterval],
    fit_output:   dict,
    func_mean:    Callable,
    func_sigma:   Callable,
    ly_key:       int,
    max_cols:     int = 4,
    output_plot:  Optional[str] = None,
    channel_info: Optional[str] = None,
    runs_list:    Optional[List[str]] = None,
):
    pm    = fit_output["params_mean"]
    ps    = fit_output["params_sigma"]
    label = f"LY{ly_key}"

    runs_text = f"Runs: {', '.join(runs_list)}" if runs_list else ""

    E_vals  = np.array([iv.E_mean for iv in intervals])
    E_range = np.linspace(E_vals.min() * 0.95, E_vals.max() * 1.05, 400)

    n_iv   = len(intervals)
    n_cols = min(max_cols, n_iv)
    n_rows = (n_iv + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 5.0 + 3.5 * n_rows))
    gs  = gridspec.GridSpec(
        n_rows + 2, n_cols,
        height_ratios=[1.8] + [1.0] * n_rows + [0.2],
        hspace=0.60, wspace=0.38,
    )

    ax_mu  = fig.add_subplot(gs[0, : n_cols // 2])
    ax_sig = fig.add_subplot(gs[0, n_cols // 2 :])

    ax_mu.plot(E_range, func_mean( E_range, *pm), "r-", lw=2, label="Simultaneous fit")
    ax_sig.plot(E_range, func_sigma(E_range, *ps), "r-", lw=2, label="Simultaneous fit")
    ax_mu.plot(E_vals,  func_mean( E_vals,  *pm), "ro", ms=5)
    ax_sig.plot(E_vals, func_sigma(E_vals,  *ps), "ro", ms=5)

    mu_est  = [iv.ly1_mu_est    if ly_key == 1 else iv.ly2_mu_est    for iv in intervals]
    sig_est = [iv.ly1_sigma_est if ly_key == 1 else iv.ly2_sigma_est for iv in intervals]
    mu_err  = [iv.ly1_mu_err    if ly_key == 1 else iv.ly2_mu_err    for iv in intervals]
    sig_err = [iv.ly1_sigma_err if ly_key == 1 else iv.ly2_sigma_err for iv in intervals]

    ax_mu.errorbar( E_vals, mu_est,  yerr=mu_err,  fmt="b+", ms=8,
                    capsize=3, elinewidth=1, label="Independent fit")
    ax_sig.errorbar(E_vals, sig_est, yerr=sig_err, fmt="b+", ms=8,
                    capsize=3, elinewidth=1, label="Independent fit")

    for ax, ylabel, title in [
        (ax_mu,  f"Mean {label}",  f"{label} — Mean vs Energy"),
        (ax_sig, f"Sigma {label}", f"{label} — Sigma vs Energy"),
    ]:
        ax.set_xlabel("Energy (a.u.)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx, iv in enumerate(intervals):
        row = idx // n_cols + 1
        col = idx  % n_cols
        ax  = fig.add_subplot(gs[row, col])

        counts = iv.ly1_counts if ly_key == 1 else iv.ly2_counts
        edges  = iv.ly1_edges  if ly_key == 1 else iv.ly2_edges
        bw     = np.diff(edges)[0]
        ctrs   = 0.5 * (edges[:-1] + edges[1:])

        ax.bar(ctrs, counts, width=bw, color="steelblue", alpha=0.6)

        mu_j  = func_mean( iv.E_mean, *pm)
        sig_j = func_sigma(iv.E_mean, *ps)
        x_g   = np.linspace(edges[0], edges[-1], 400)

        gauss_nll = (counts.sum() * bw / (sig_j * np.sqrt(2 * np.pi))
                     * np.exp(-0.5 * ((x_g - mu_j) / sig_j) ** 2))
        ax.plot(x_g, gauss_nll, "r-", lw=1.8, label="Simultaneous fit")

        mu_ind  = iv.ly1_mu_est    if ly_key == 1 else iv.ly2_mu_est
        sig_ind = iv.ly1_sigma_est if ly_key == 1 else iv.ly2_sigma_est
        gauss_ind = (counts.sum() * bw / (sig_ind * np.sqrt(2 * np.pi))
                     * np.exp(-0.5 * ((x_g - mu_ind) / sig_ind) ** 2))
        ax.plot(x_g, gauss_ind, "b--", lw=1.4, label="Independent fit")

        ax.set_title(
            f"j={iv.j}  [{iv.E_low:.0f}, {iv.E_high:.0f})\n"
            f"E={iv.E_mean:.1f}  n={iv.n_events}", fontsize=7)
        ax.set_xlabel(label, fontsize=7)
        ax.set_ylabel("Counts", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)

    if runs_text:
        ax_text = fig.add_subplot(gs[-1, :])
        ax_text.axis('off')
        ax_text.text(0.5, 0.5, runs_text, transform=ax_text.transAxes,
                     fontsize=10, ha='center', va='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    title_text = (f"Simultaneous Gaussian fit — {label} — Channel {channel_info}"
                  if channel_info else f"Simultaneous Gaussian fit — {label}")
    plt.suptitle(title_text, fontsize=13, y=0.995)
    plt.tight_layout()

    out = f"{output_plot}/{label}_ch{channel_info}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved -> {out}")


# =============================================================================
# SECTION 10 — Main
# =============================================================================

def EnergiesLaw(vectorEnergy, vectorStab, vectorLY1, vectorLY2,
                output_plot, channel_info=None, runs_list=None):

    global ONLYTOPLD

    limit = build_energy_limits(E_start=700., E_max=2700.,
                                base_step=50., growth=1.2)
    print(f"{len(limit)-1} energy intervals  [{limit[0]:.0f}, {limit[-1]:.0f}) keV")

    mapLYIntervals = fill_map_ly_intervals(
        vectorEnergy, vectorStab, vectorLY1, vectorLY2, limit
    )

    print("\nBuilding intervals ...")
    intervals = build_intervals(mapLYIntervals, limit, n_hist_bins=N_HIST_BINS)

    fit_ly1 = run_fit(intervals=intervals, ly_key=1,
                      func_mean=ly1_func_mean, func_sigma=ly1_func_sigma,
                      method="Nelder-Mead")

    if not ONLYTOPLD:
        fit_ly2 = run_fit(intervals=intervals, ly_key=2,
                          func_mean=ly2_func_mean, func_sigma=ly2_func_sigma,
                          method="Nelder-Mead")

    plot_fit(intervals, fit_ly1, ly1_func_mean, ly1_func_sigma, ly_key=1,
             output_plot=output_plot, channel_info=channel_info, runs_list=runs_list)

    if not ONLYTOPLD:
        plot_fit(intervals, fit_ly2, ly2_func_mean, ly2_func_sigma, ly_key=2,
                 output_plot=output_plot, channel_info=channel_info, runs_list=runs_list)

    pm1, ps1 = fit_ly1["params_mean"], fit_ly1["params_sigma"]

    if not ONLYTOPLD:
        pm2, ps2 = fit_ly2["params_mean"], fit_ly2["params_sigma"]
    else:
        pm2 = [0, 0, 0]
        ps2 = [0, 0]

    mean_LD1  = lambda E: ly1_func_mean( E, *pm1)
    sigma_LD1 = lambda E: ly1_func_sigma(E, *ps1)
    mean_LD2  = lambda E: ly2_func_mean( E, *pm2)
    sigma_LD2 = lambda E: ly2_func_sigma(E, *ps2)

    return mean_LD1, sigma_LD1, mean_LD2, sigma_LD2


def FindCutCorrelation(ch_corr, chain_pulser):
    correlations = []
    for i in range(chain_pulser.GetEntries()):
        chain_pulser.GetEntry(i)
        if not getattr(chain_pulser, 'heat_IsHeater', 0):
            continue
        ch_corr.GetEntry(i)
        corr = getattr(ch_corr, 'heat_correlation', 0)
        if corr < 0.8:
            continue
        correlations.append(corr)

    if correlations:
        correlations.sort()
        n = len(correlations)
        return (correlations[n // 2 - 1] + correlations[n // 2]) / 2.0 \
               if n % 2 == 0 else correlations[n // 2]
    return 0.9


def create_ly_tree(chain_Stab, chain_optim, ch_corr, ch_module, chain_pulser,
                   chRoughEnergy, output_file, output_plot,
                   channel_info=None, runs_list=None, mode="OF"):

    ly_tree    = ROOT.TTree("LY", "Light Yield values")
    ly_LD1_value = array('d', [0.])
    ly_LD2_value = array('d', [0.])
    ly_tree.Branch("LD1_LY", ly_LD1_value, "LD1_LY/D")
    ly_tree.Branch("LD2_LY", ly_LD2_value, "LD2_LY/D")

    EnergyVal, LY1Val, LY2Val, StabVal = [], [], [], []

    corrcut = FindCutCorrelation(ch_corr, chain_pulser)
    corrcut = corrcut * (1 + 1.1e-8)
    if np.isnan(corrcut):
        corrcut = 0.9
    print(f"Correlation cut: {corrcut:.6f}")

    chain_used = {"OF": chain_optim, "RoughEnergy": chRoughEnergy,
                  "Stab": chain_Stab}[mode]
    print(f"Using mode: {mode}")

    nentries = chain_used.GetEntries()
    print(f"Calculating LY for {nentries} entries...")

    for i in range(nentries):
        chain_used.GetEntry(i)
        chain_optim.GetEntry(i)
        chRoughEnergy.GetEntry(i)

        amp_stab     = getattr(chain_used,   'heat_amplitude', 0)
        amp_opt_LD1  = getattr(chain_optim,  'LD1_amplitude',  0)
        amp_opt_LD2  = getattr(chain_optim,  'LD2_amplitude',  0)
        calEner      = getattr(chRoughEnergy,'heat_amplitude',  0)

        ly_LD1_value[0] = amp_opt_LD1 * 1000 / amp_stab if amp_stab != 0 else 0
        ly_LD2_value[0] = amp_opt_LD2 * 1000 / amp_stab if amp_stab != 0 else 0
        ly_tree.Fill()

        if 700 < calEner < 2600:
            ch_corr.GetEntry(i)
            chain_pulser.GetEntry(i)
            corr     = getattr(ch_corr,      'heat_correlation', 0)
            ispulser = getattr(chain_pulser,  'heat_IsHeater',    0)
            if corr < corrcut or ispulser:
                continue
            EnergyVal.append(calEner)
            StabVal.append(amp_stab)
            LY1Val.append(ly_LD1_value[0])
            LY2Val.append(ly_LD2_value[0])

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{nentries} entries")

    output_file.cd()
    ly_tree.Write()
    print(f"LY tree created with {ly_tree.GetEntries()} entries")


    mean_LD1, sigma_LD1, mean_LD2, sigma_LD2 = EnergiesLaw(
        EnergyVal, StabVal, LY1Val, LY2Val, output_plot,
        channel_info=channel_info, runs_list=runs_list
    )

    print("Creating Discrimination parameter tree")
    second_tree = ROOT.TTree("DiscrPar", "Analysis based on LY values")
    ld1_d1  = array('d', [0.])
    ld2_d2  = array('d', [0.])
    DiscrPar = array('d', [0.])
    second_tree.Branch("DiscrParLD1", ld1_d1,   "DiscrParLD1/D")
    second_tree.Branch("DiscrParLD2", ld2_d2,   "DiscrParLD2/D")
    second_tree.Branch("DiscrPar",    DiscrPar,  "DiscrPar/D")

    for i in range(nentries):
        chain_used.GetEntry(i)
        ref_amp = getattr(chain_used, 'heat_amplitude', 0)

        ly_tree.GetEntry(i)
        LY1_ref = ly_LD1_value[0]
        LY2_ref = ly_LD2_value[0]

        ly_LD1_mean  = float(mean_LD1(ref_amp))
        ly_LD2_mean  = float(mean_LD2(ref_amp))
        ly_LD1_sigma = float(sigma_LD1(ref_amp))
        ly_LD2_sigma = float(sigma_LD2(ref_amp))

        n1 = (LY1_ref - ly_LD1_mean) / ly_LD1_sigma if ly_LD1_sigma > 0 else 0
        n2 = (LY2_ref - ly_LD2_mean) / ly_LD2_sigma if ly_LD2_sigma > 0 else 0

        ld1_d1[0]   = n1
        ld2_d2[0]   = n2
        DiscrPar[0] = math.sqrt(n1**2 + n2**2)
        second_tree.Fill()

    second_tree.Write()
    return ly_tree


def merge_runs(cross_dir, runs, channel=None, channelLD1=None, channelLD2=None,
               output_dir=".", outputPlot=""):

    global ONLYTOPLD


    if len(runs) < 2:
        raise ValueError("Need at least 2 runs")

    print("Merging runs:", runs)
    print("Channel:", channel if channel is not None else "ALL")

    if channelLD2 == -1:
        ONLYTOPLD = True

    files_by_channel = defaultdict(list)

    for run in runs:
        run_str    = str(run)
        run_number = int(run_str.replace("RUN", "")) if run_str.startswith("RUN") else int(run_str)
        run_fmt    = f"RUN{run_number:06d}"
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

            if len(parts) == 4:
                ch_name      = f"{parts[0]}_{parts[1]}_{parts[2]}"
                run_in_file  = int(parts[3])
                requested_ch = f"{channel}_{channelLD1}_{channelLD2}"
            elif len(parts) == 3:
                ch_name      = f"{parts[0]}_{parts[1]}"
                run_in_file  = int(parts[2])
                requested_ch = f"{channel}_{channelLD1}"
            else:
                continue

            if run_in_file != run_number or ch_name != requested_ch:
                continue

            fullpath = os.path.join(ofdata_path, fname)
            print(fullpath)
            files_by_channel[ch_name].append(fullpath)

    if not files_by_channel:
        print("No matching files found.")
        return


    os.makedirs(output_dir, exist_ok=True)
    out_base = f"{runs[0]}_{runs[-1]}" if len(runs) > 1 else runs[0]

    for ch, flist in sorted(files_by_channel.items()):
        if len(flist) != len(runs):
            print(f"Skipping channel {ch} (missing in some runs)")
            continue

        print(f"\nMerging channel {ch}")
        
        set_channel_params(ch.split('_')[0])

        f0         = ROOT.TFile.Open(flist[0])
        tree_names = [k.GetName() for k in f0.GetListOfKeys()
                      if k.GetClassName() == "TTree"]
        f0.Close()

        if not tree_names:
            print("No TTrees found"); continue

        outname = os.path.join(output_dir, f"{out_base}_ch{ch}.root")
        outfile = ROOT.TFile(outname, "RECREATE")

        muon_branch = "LD1_IsMuon"   

        for treename in tree_names:
            chain = ROOT.TChain(treename)

            if treename == "global":
                continue

            for f in flist:
                chain.Add(f)

            if treename == "timestamp":
                newtree   = chain.CloneTree(0)
                time_cum  = array('d', [0.])
                run_number_br = array('i', [0])
                newtree.Branch("time_cumulative", time_cum, "time_cumulative/D")
                newtree.Branch("run_number",      run_number_br, "run_number/I")

                entries_per_file, last_times, file_run_numbers = [], [], []
                for idx, f in enumerate(flist):
                    tf = ROOT.TFile.Open(f)
                    t  = tf.Get("timestamp")
                    entries_per_file.append(t.GetEntries())
                    t.GetEntry(t.GetEntries() - 1)
                    last_times.append(t.heat_timefromstartrun)
                    run_str = runs[idx]
                    run_num = (int(run_str.replace("RUN", ""))
                               if run_str.startswith("RUN") else int(run_str))
                    file_run_numbers.append(run_num)
                    tf.Close()

                cumulative_offset = 0
                file_index        = 0
                entry_end         = entries_per_file[0]

                for i, event in enumerate(chain):
                    if i >= entry_end and file_index < len(flist) - 1:
                        cumulative_offset += last_times[file_index]
                        file_index        += 1
                        entry_end         += entries_per_file[file_index]

                    time_cum[0]       = event.heat_timefromstartrun + cumulative_offset
                    run_number_br[0]  = file_run_numbers[file_index]
                    newtree.Fill()

                outfile.cd()
                newtree.Write()
            elif treename == "flagpropagator_muon":
                muon_tree  = ROOT.TTree("flagpropagator_muon", "Muon veto flag")
                is_muon_br = array('i', [0])
                muon_tree.Branch("IsMuon", is_muon_br, "IsMuon/I")

                chain.SetBranchStatus("*", 0)          # skip heat_ and LD2_ branches
                chain.SetBranchStatus(muon_branch, 1)  # read LD1_IsMuon only

                for event in chain:
                    is_muon_br[0] = int(getattr(event, muon_branch, 0))
                    muon_tree.Fill()
                
                outfile.cd()
                muon_tree.Write()

            elif treename == "flagpropagator_heater":
                # (source_branch, output_name, root_type, array_typecode)
                HEATER_VARS = [
                    ("heat_IsHeater",      "IsHeater",      "I", 'i'),
                    ("heat_delay",         "delay",         "D", 'd'),
                    ("heat_CheckedSample", "CheckedSample", "D", 'd'),
                ]
 
                heater_tree = ROOT.TTree("flagpropagator_heater", "Heater flag")
                buffers = {}
                for src_branch, out_name, type_char, arr_code in HEATER_VARS:
                    buf = array(arr_code, [0])
                    buffers[src_branch] = (buf, arr_code)
                    heater_tree.Branch(out_name, buf, f"{out_name}/{type_char}")
 
                chain.SetBranchStatus("*", 0)
                for src_branch, *_ in HEATER_VARS:
                    chain.SetBranchStatus(src_branch, 1)
 
                for event in chain:
                    for src_branch, _, _, arr_code in HEATER_VARS:
                        buf, ac = buffers[src_branch]
                        raw = getattr(event, src_branch, 0)
                        buf[0] = float(raw) if ac == 'd' else int(raw)
                    heater_tree.Fill()
 
                outfile.cd()
                heater_tree.Write("", ROOT.TObject.kOverwrite)
 
            elif treename == "flagpropagator_led":
                LED_VARS = [
                    ("heat_delay",         "delay",         "D", 'd'),
                    ("heat_IsLED",         "IsLED",         "I", 'i'),
                    ("heat_CheckedSample", "CheckedSample", "D", 'd'),
                ]
 
                led_tree = ROOT.TTree("flagpropagator_led", "LED flag")
                buffers = {}
                for src_branch, out_name, type_char, arr_code in LED_VARS:
                    buf = array(arr_code, [0])
                    buffers[src_branch] = (buf, arr_code)
                    led_tree.Branch(out_name, buf, f"{out_name}/{type_char}")
 
                chain.SetBranchStatus("*", 0)
                for src_branch, *_ in LED_VARS:
                    chain.SetBranchStatus(src_branch, 1)
 
                for event in chain:
                    for src_branch, _, _, arr_code in LED_VARS:
                        buf, ac = buffers[src_branch]
                        raw = getattr(event, src_branch, 0)
                        buf[0] = float(raw) if ac == 'd' else int(raw)
                    led_tree.Fill()
 
                outfile.cd()
                led_tree.Write("", ROOT.TObject.kOverwrite)

            elif treename == "module":

                MODULE_VARS = [
                    ("heat_triggersample",      "triggersample",      "I"),
                    ("heat_isselftrigger",       "isselftrigger",      "I"),
                    ("heat_isnoise",             "isnoise",            "I"),
                    ("heat_iscoincidence",       "iscoincidence",      "I"),
                    ("heat_triggerchannel",      "triggerchannel",     "I"),
                    ("heat_issignal",            "issignal",           "I"),
                ]
 
                mod_tree = ROOT.TTree("module", "Module flags (heat_ copy only)")
                buffers  = {}
                for src_branch, out_name, type_char in MODULE_VARS:
                    buf = array('i', [0])
                    buffers[src_branch] = buf
                    mod_tree.Branch(out_name, buf, f"{out_name}/{type_char}")
 
                chain.SetBranchStatus("*", 0)
                for src_branch, _, _ in MODULE_VARS:
                    chain.SetBranchStatus(src_branch, 1)
 
                for event in chain:
                    for src_branch, _, _ in MODULE_VARS:
                        buffers[src_branch][0] = int(getattr(event, src_branch, 0))
                    mod_tree.Fill()
 
                outfile.cd()
                mod_tree.Write("", ROOT.TObject.kOverwrite)
 
            else:
                chain.Merge(outfile, 0, "keep")

        chain_stab   = ROOT.TChain("stabilization_all")
        chain_op     = ROOT.TChain("optimumfilter_all")
        chRoughEn    = ROOT.TChain("calibration_rough")
        chain_corr   = ROOT.TChain("correlation_corr")
        chain_module = ROOT.TChain("module")
        chain_flag   = ROOT.TChain("flagpropagator_heater")

        for f in flist:
            chain_stab.Add(f);   chain_op.Add(f)
            chain_module.Add(f); chain_corr.Add(f)
            chain_flag.Add(f);   chRoughEn.Add(f)

        create_ly_tree(
            chain_stab, chain_op, chain_corr, chain_module,
            chain_flag, chRoughEn, outfile, outputPlot,
            channel_info=ch, runs_list=runs, mode="RoughEnergy"
        )

        globalTree = ROOT.TTree("global", "Empty global tree")
        outfile.cd()
        globalTree.Write()

        for key in outfile.GetListOfKeys():
            if key.GetClassName() == "TTree" and key.GetName() != "global":
                globalTree.AddFriend(key.GetName())

        globalTree.Write("", ROOT.TObject.kOverwrite)
        del globalTree
        import gc
        gc.collect()

        outfile.Purge() 
        outfile.Close()
        print("  ->", outname)

    print("\nDone.")


os.makedirs(args.outdir, exist_ok=True)
os.makedirs(args.outPlot, exist_ok=True)

# Percorso fisso del file CSV (cambialo con il percorso reale che vuoi usare)
CSV_PATH = "/data/users/azanelli/octopus_work/batchOctopus/parametri_canali.csv" 

# Carica i parametri
load_params_csv(CSV_PATH)

merge_runs(
    cross_dir=args.cross,
    runs=args.runs,
    channel=args.channel,
    channelLD1=args.channelLD1,
    channelLD2=args.channelLD2,
    output_dir=args.outdir,
    outputPlot=args.outPlot,
)