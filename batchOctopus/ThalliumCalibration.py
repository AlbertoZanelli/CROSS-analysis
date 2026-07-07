#!/usr/bin/env python3
"""
ThalliumCalibration.py  (batch analysis + optional interactive GUI)
===================================================================
PARALLEL version of ThalliumStabilization.py.

Difference from the stabilization script
-----------------------------------------
  The baseline-dependent linear stabilization

        a_stab_cal = TARGET_ENERGY * amp / (q_0 + slope * baseline)

  is REMOVED. There is no robust pol1 baseline fit and no per-event
  baseline correction. Instead, the thallium peak is located, cleaned and
  Gaussian-fitted exactly as before, and the mean of that "clean peak"
  Gaussian fit (mean_amp_clean) is used to build a SINGLE multiplicative
  calibration factor:

        cal_factor = TARGET_ENERGY / mean_amp_clean

  The WHOLE stabilization_all spectrum is then multiplied by this factor
  so that the thallium peak lands on TARGET_ENERGY (keV):

        a_cal = cal_factor * heat_amplitude          (every event)

  The calibrated amplitude of every event of "stabilization_all" is written
  to the new TTree "calibrated_heater_thallium" in the output ROOT file.

Pipeline overview
-----------------
  1. Discover every ROOT file in BASE_DIR (files containing "calibrated" /
     "stabilized" in the name are skipped). The detector channel is parsed
     from the number that follows "ch" in the file name.

  2. For each file, bulk-read the main tree "stabilization_all" with its
     friend trees via RDataFrame + AsNumpy. Quality filters (single trigger,
     good interval, signal flag) are applied at the RDataFrame level.

  3. Build a DYNAMIC correlation cut (CORR_CUT_PERCENTILE-th percentile of the
     valid-correlation distribution) to reject poorly reconstructed pulses.

  4. Light-Yield analysis (LY tree, leaves LD1_LY / LD2_LY): fit thallium and
     (when present) alpha in each LD, pick the LD with the larger DF, derive
     the LY acceptance window [cut_min, cut_max].

  5. Select thallium-peak events surviving correlation + LY cuts, clean
     outliers around the peak, and perform a Gaussian fit of the clean peak.
     mean_amp_clean = mean of that fit.

  6. Calibrate the whole spectrum with cal_factor = TARGET_ENERGY/mean_amp_clean
     (NO baseline term). Build the calibrated spectrum and measure its
     resolution (FWHM) from a Gaussian fit.

  7. [optional] Write a copy of the input file to a dedicated folder and append
     the TTree "calibrated_heater_thallium" (one calibrated amplitude per event
     of "stabilization_all").

  8. Render a 3x3 summary canvas. In BATCH mode it is saved as JPEG; in GUI mode
     it is shown on screen and the user can tweak the manual cuts, then
     recompute / accept (save JPEG + ROOT) / quit.

Modes / Channel selection / Flags
----------------------------------
  Same conventions as ThalliumStabilization.py (see __main__).

Output
------
  ThalliumCalibrationDebug/ch<N>_calibration_overview.jpg  (3x3 overview)
  CorrelationCut/ch<N>_correlation_cut.jpg                 (correlation cut)
  ThalliumCalibratedAmp/<stem>_thallium_calibrated.root    (copy + new TTree)
  All output folders are created (if missing) inside BASE_DIR.

Execution
---------
  python3 ThalliumCalibration.py                 # uses CHANNELS_TO_PROCESS
  python3 ThalliumCalibration.py 24 51           # overrides with channels 24, 51
"""

# ===========================================================================
# IMPORTS
# ===========================================================================

# -- standard library --------------------------------------------------------
import sys
import os
import math
import time
import traceback
import argparse
import re
from array import array

# -- third-party -------------------------------------------------------------
import numpy as np

# -- domain-specific ---------------------------------------------------------
import ROOT

# -- GUI (only used in interactive mode) -------------------------------------
import tkinter as tk
from tkinter import ttk


# ===========================================================================
# GLOBAL SETUP
# ===========================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Histograms are detached from any TFile so they survive file.Close().
ROOT.TH1.AddDirectory(False)

# Keeps ROOT objects alive in interactive mode (see run_calibration). The
# Python garbage collector would otherwise free histograms/graphs/fits and the
# on-screen canvas would go blank at the first repaint.
GLOBAL_KEEPALIVE = []

# NB: the ROOT batch mode (off-screen rendering) is decided in __main__ from the
# run mode: ON for batch analysis, OFF when the interactive GUI is active.


# ===========================================================================
# PARAMETERS  <- tune these
# ===========================================================================

# --- Output folder names (created if missing, inside BASE_DIR) --------------
CAL_ROOT_DIR_NAME = os.path.join("..", "ThalliumCalibratedAmp")            # calibrated ROOT files

SUMMARY_DIR_NAME  = os.path.join("..", "ThalliumCalibratedAmp/ThalliumCalibrationDebug")  # 3x3 overview JPEGs
CORR_DIR_NAME     = os.path.join("..", "ThalliumCalibratedAmp/CorrelationCut")             # correlation-cut JPEGs

# --- Plotting ---------------------------------------------------------------
# Max number of markers drawn in any overview scatter (TGraph). Decimation is
# for DISPLAY ONLY -- the analysis and all fits always use the full dataset.
MAX_SCATTER_POINTS = 4000

# --- Calibration-energy windows (rough-calibration units) -------------------
CAL_CORR_MIN, CAL_CORR_MAX = 2400.0, 2700.0   # window for the conversion factor
CAL_LY_MIN,   CAL_LY_MAX   = 2350.0, 2700.0   # window for the light-yield study
CAL_STAB_MIN, CAL_STAB_MAX = 2400.0, 2700.0   # window for the clean-peak selection

# --- Thallium reference -----------------------------------------------------
TARGET_ENERGY = 2614.511   # keV, 208-Tl line used as the calibration anchor

# --- Correlation cut --------------------------------------------------------
CORR_VALID_MIN      = 0.999   # events below this correlation are ignored
CORR_CUT_PERCENTILE = 0.10    # dynamic cut at this percentile of valid corr

# --- Peak-selection half-widths (in sigma) ----------------------------------
LY_N_SIGMA        = 4.0   # thallium acceptance half-width in the LY spectrum
HEAT_CLEAN_NSIGMA = 3.0   # pre-cleaning half-width around the thallium peak

# --- Gaussian -> FWHM conversion constant: 2*sqrt(2*ln2) --------------------
SIGMA_TO_FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))


# ===========================================================================
# PLOTTING / IO HELPERS
# ===========================================================================

def make_scatter_graph(x_arr, y_arr, max_points=MAX_SCATTER_POINTS):
    """
    Build a TGraph from two NumPy arrays, uniformly decimating when the point
    count exceeds *max_points*. Decimation affects DISPLAY ONLY.

    Returns an (possibly empty) ROOT.TGraph.
    """
    n = len(x_arr)
    if n == 0:
        return ROOT.TGraph()
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(np.int64)
        x_arr = x_arr[idx]
        y_arr = y_arr[idx]
    return ROOT.TGraph(len(x_arr),
                       np.ascontiguousarray(x_arr, np.double),
                       np.ascontiguousarray(y_arr, np.double))


def save_canvas_jpeg(canvas, out_path):
    """
    Save a canvas to JPEG robustly (useful in batch mode). On failure a PNG with
    the same stem is attempted. Returns True on success, False otherwise.
    """
    try:
        canvas.Modified()
        canvas.Update()
        canvas.SaveAs(out_path)
        print(f">>> Saved JPEG: {out_path}")
        return True
    except Exception as e:
        print(f"  [!] JPEG save failed for {out_path}: {e}", file=sys.stderr)
        try:
            png_path = os.path.splitext(out_path)[0] + ".png"
            canvas.SaveAs(png_path)
            print(f">>> Saved PNG (fallback): {png_path}")
            return True
        except Exception as e2:
            print(f"  [!] PNG fallback also failed: {e2}", file=sys.stderr)
            return False


# ===========================================================================
# GUI  (interactive manual-cut editor)
# ===========================================================================

class ParamEditorApp:
    """
    Tk dialog used in interactive mode to override the automatic cuts.

    The user can leave fields empty (keep the automatic value) or type a value
    for: chosen light detector (1/2), LY window [min, max], and the heat
    pre-cleaning window [min, max]. The three buttons map to the actions
    'recalc', 'accept' and 'quit', returned by run().
    """
    def __init__(self, ch_id, current_cuts):
        self.action = None
        self.manual_cuts = current_cuts.copy() if current_cuts else {}
        self.tk_root = tk.Tk()
        self.tk_root.title(f"Parametri - Ch {ch_id}")
        self.tk_root.geometry("380x380")

        frame = ttk.LabelFrame(self.tk_root, text="Tagli Manuali (lascia vuoto per default)")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.entries = {}
        fields = [
            ("chosen_ld", "Canale LD (1 o 2):"),
            ("ly_cut_min", "LY cut min:"),
            ("ly_cut_max", "LY cut max:"),
            ("heat_cut_min", "Heat cleaning min:"),
            ("heat_cut_max", "Heat cleaning max:")
        ]

        for key, label in fields:
            f = ttk.Frame(frame)
            f.pack(fill="x", pady=5, padx=5)
            ttk.Label(f, text=label, width=18).pack(side="left")
            e = ttk.Entry(f)
            e.pack(side="right", fill="x", expand=True)
            if key in self.manual_cuts:
                e.insert(0, str(self.manual_cuts[key]))
            self.entries[key] = e

        btn_frame = ttk.Frame(self.tk_root)
        btn_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(btn_frame, text="Ricalcola", command=lambda: self.set_action('recalc')).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Accetta e Salva (ROOT + JPEG)", command=lambda: self.set_action('accept')).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Esci dallo Script", command=lambda: self.set_action('quit')).pack(fill="x", pady=10)

        self.tk_root.protocol("WM_DELETE_WINDOW", lambda: self.set_action('quit'))

    def get_values(self):
        """Read the entry fields into self.manual_cuts (empty -> remove key)."""
        for key, e in self.entries.items():
            val = e.get().strip()
            if val: self.manual_cuts[key] = float(val) if 'ly' in key or 'heat' in key else int(val)
            elif key in self.manual_cuts: del self.manual_cuts[key]

    def set_action(self, act):
        """Store the chosen action, commit field values and close the window."""
        self.get_values()
        self.action = act
        self.tk_root.destroy()

    def run(self):
        """
        Pump the Tk and ROOT event loops until the user picks an action.
        Returns (action, manual_cuts).
        """
        while self.action is None:
            try:
                self.tk_root.update_idletasks()
                self.tk_root.update()
            except tk.TclError:
                if self.action is None: self.action = 'quit'
                break
            ROOT.gSystem.ProcessEvents()
            time.sleep(0.02)
        return self.action, self.manual_cuts


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================

class LYResult:
    """Container for one light detector's LY fit results and acceptance cut."""
    def __init__(self):
        self.df_value  = -1.0    # discrimination factor (alpha vs thallium)
        self.cut_min   = 0.0     # lower LY acceptance bound
        self.cut_max   = 0.0     # upper LY acceptance bound
        self.fit_Tl    = None    # TF1, thallium Gaussian fit
        self.fit_alpha = None    # TF1, alpha Gaussian fit (may stay None)


class BinningParams:
    """Robust binning descriptor: centre, scale, visible range and bin count."""
    def __init__(self, median=0.0, robust_sigma=5.0, vis_min=0.0, vis_max=0.0, bins=50):
        self.median       = median
        self.robust_sigma = robust_sigma
        self.vis_min      = vis_min
        self.vis_max      = vis_max
        self.bins         = bins


# ===========================================================================
# BINNING HELPERS
# ===========================================================================

def GetCenteredBinning(vals_input, fallback_median):
    """
    Compute a robust, peak-centred binning for a Gaussian-like sample.
    Returns a BinningParams instance.
    """
    p = BinningParams(median=fallback_median, robust_sigma=5.0,
                      vis_min=fallback_median - 65.0, vis_max=fallback_median + 65.0, bins=50)
    if vals_input is None or len(vals_input) == 0:
        return p

    arr = np.asarray(vals_input, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return p

    p.median  = float(np.median(arr))
    max_dev   = float(np.max(np.abs(arr - p.median)))

    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    p.robust_sigma = (q75 - q25) / 1.349
    if p.robust_sigma <= 0:
        p.robust_sigma = max_dev / 3.0 if max_dev > 0 else 1.0

    # Visible range: the Gaussian fit spans median +/- 5 sigma; widen to
    # ~7.5 sigma for white margins, but always contain every point (max_dev).
    half = max(7.5 * p.robust_sigma, max_dev * 1.10)
    p.vis_min = p.median - half
    p.vis_max = p.median + half

    # Wider bins (less jagged): width ~ 0.7 robust sigma.
    if p.robust_sigma > 0:
        p.bins = math.ceil((p.vis_max - p.vis_min) / (p.robust_sigma * 0.7))
    p.bins = max(12, min(p.bins, 80))
    return p


def calcRobustLimitsAndBins(vals):
    """
    Robust (limits, bin count) for a generic 1-D sample, based on percentiles.
    Returns (n_bins, min, max).
    """
    if len(vals) == 0:
        return 40, -0.05, 0.15

    arr = np.sort(np.asarray(vals, dtype=np.float64))
    n   = len(arr)

    q02 = arr[int(n * 0.05)]
    q98 = arr[int(n * 0.95)]
    vis_range = max(q98 - q02, 1e-6) if (q98 - q02) >= 1e-6 else 0.1

    min_out = q02 - vis_range * 0.4
    max_out = q98 + vis_range * 0.4

    IQR          = arr[int(n * 0.75)] - arr[int(n * 0.25)]
    robust_sigma = IQR / 1.349
    if robust_sigma <= 1e-6:
        robust_sigma = vis_range / 6.0
        if robust_sigma <= 1e-6:
            robust_sigma = 0.1

    n_bins = math.ceil((max_out - min_out) / (robust_sigma / 2.5))
    n_bins = max(40, min(n_bins, 500))
    return n_bins, float(min_out), float(max_out)


# ===========================================================================
# LIGHT-YIELD ANALYSIS
# ===========================================================================

def AnalyzeLightYield(h_ly, name_ext):
    """
    Find and fit the thallium (and, if present, alpha) peak in an LY histogram.
    Returns an LYResult.
    """
    res = LYResult()
    print("=" * 50)
    print(f"--- Peak Analysis {h_ly.GetTitle()} ---")

    xMin = h_ly.GetXaxis().GetXmin()
    xMax = h_ly.GetXaxis().GetXmax()
    hist_range = xMax - xMin
    fit_window = hist_range * 0.12

    spec   = ROOT.TSpectrum(10)
    nPeaks = spec.Search(h_ly, 2, "goff", 0.02)
    peakTl_X, peakAlpha_X = -999.0, -999.0

    if nPeaks > 0:
        xpeaks_buf      = spec.GetPositionX()
        peaks_with_area = []
        int_window      = hist_range * 0.04

        for i in range(nPeaks):
            x_val = xpeaks_buf[i]
            if x_val <= 0.0: continue
            bin_min = h_ly.GetXaxis().FindBin(x_val - int_window)
            bin_max = h_ly.GetXaxis().FindBin(x_val + int_window)
            peaks_with_area.append((x_val, h_ly.Integral(bin_min, bin_max)))

        if not peaks_with_area:
            # Only noise (non-positive peaks): fall back to the global maximum.
            peakTl_X = h_ly.GetXaxis().GetBinCenter(h_ly.GetMaximumBin())
            print("  [!] Found only noise (negative peaks). Fallback on global maximum.")
        else:
            # Keep the two largest, well-separated peaks; order them in x.
            peaks_with_area.sort(key=lambda p: p[1], reverse=True)
            valid_peaks = []
            min_dist    = hist_range * 0.08
            for p in peaks_with_area:
                if not any(abs(p[0] - vp[0]) < min_dist for vp in valid_peaks):
                    valid_peaks.append(p)
                if len(valid_peaks) == 2: break
            valid_peaks.sort(key=lambda p: p[0])

            if len(valid_peaks) == 2:
                peakAlpha_X, area_alpha = valid_peaks[0]
                peakTl_X,   area_tl    = valid_peaks[1]
                print(f"Found 2 positive peaks. Alpha at: {peakAlpha_X:.4f} (Area: {area_alpha:.0f}), "
                      f"Thallium at: {peakTl_X:.4f} (Area: {area_tl:.0f})")
            else:
                peakTl_X = valid_peaks[0][0]
                print(f"Found 1 positive peak. Thallium at: {peakTl_X:.4f} (Area: {valid_peaks[0][1]:.0f})")
    else:
        peakTl_X = h_ly.GetXaxis().GetBinCenter(h_ly.GetMaximumBin())

    # --- Thallium Gaussian fit ----------------------------------------------
    fit_Tl_min = peakTl_X - fit_window
    fit_Tl_max = peakTl_X + fit_window
    if peakAlpha_X != -999.0:
        # Narrow the fit toward the thallium side if alpha is close by.
        dist = peakTl_X - peakAlpha_X
        if dist < fit_window * 1.5:
            fit_Tl_min = peakTl_X - dist * 0.4

    res.fit_Tl = ROOT.TF1(f"fit_Tl_{name_ext}", "gaus", fit_Tl_min, fit_Tl_max)
    res.fit_Tl.SetLineColor(ROOT.kBlue)
    res.fit_Tl.SetParameters(h_ly.GetBinContent(h_ly.GetXaxis().FindBin(peakTl_X)),
                              peakTl_X, (fit_Tl_max - fit_Tl_min) / 4.0)
    h_ly.Fit(res.fit_Tl, "Q0 R L")

    mean_Tl  = res.fit_Tl.GetParameter(1)
    sigma_Tl = res.fit_Tl.GetParameter(2)
    mean_alpha, sigma_alpha = -999.0, 0.0

    # --- Alpha Gaussian fit + discrimination factor -------------------------
    if peakAlpha_X != -999.0:
        fit_alpha_min = peakAlpha_X - fit_window
        fit_alpha_max = peakAlpha_X + fit_window
        dist = peakTl_X - peakAlpha_X
        if dist < fit_window * 1.5:
            fit_alpha_max = peakAlpha_X + dist * 0.4

        res.fit_alpha = ROOT.TF1(f"fit_alpha_{name_ext}", "gaus", fit_alpha_min, fit_alpha_max)
        res.fit_alpha.SetLineColor(ROOT.kGreen + 2)
        res.fit_alpha.SetParameters(h_ly.GetBinContent(h_ly.GetXaxis().FindBin(peakAlpha_X)),
                                     peakAlpha_X, (fit_alpha_max - fit_alpha_min) / 4.0)
        h_ly.Fit(res.fit_alpha, "Q0+ R L")

        mean_alpha  = res.fit_alpha.GetParameter(1)
        sigma_alpha = res.fit_alpha.GetParameter(2)
        sigma_tot   = math.sqrt(sigma_Tl**2 + sigma_alpha**2)
        res.df_value = abs(mean_Tl - mean_alpha) / sigma_tot if sigma_tot > 0 else -1.0

    # --- Acceptance window --------------------------------------------------
    res.cut_max      = mean_Tl + LY_N_SIGMA * sigma_Tl
    standard_cut_min = mean_Tl - LY_N_SIGMA * sigma_Tl

    if peakAlpha_X != -999.0 and res.fit_alpha is not None and (sigma_alpha + sigma_Tl) > 0:
        # Lower bound at the sigma-weighted alpha/thallium valley, but never
        # closer than 1.5 sigma to the thallium mean, nor below the standard cut.
        valley_point = (mean_alpha * sigma_Tl + mean_Tl * sigma_alpha) / (sigma_alpha + sigma_Tl)
        valley_point = min(valley_point, mean_Tl - 1.5 * sigma_Tl)
        res.cut_min  = max(valley_point, standard_cut_min)
    else:
        res.cut_min = standard_cut_min

    print(f"  -> Calculated LY cuts for {name_ext}: [{res.cut_min:.4f}, {res.cut_max:.4f}]")
    return res


def CreateLYBox(res, title_prefix):
    """Build an NDC TPaveText summarising the LY thallium/alpha fit results."""
    pt = ROOT.TPaveText(0.15, 0.30, 0.58, 0.88, "NDC")
    pt.SetFillColor(ROOT.kWhite); pt.SetBorderSize(1)
    pt.SetTextAlign(12); pt.SetTextFont(42); pt.SetTextSize(0.04)
    if res.df_value > 0:
        t = pt.AddText(f"{title_prefix} DF (Alpha-Tl) = {res.df_value:.2f}")
        t.SetTextColor(ROOT.kBlack); t.SetTextFont(62); pt.AddText("")

    t_tl = pt.AddText("THALLIUM FIT")
    t_tl.SetTextColor(ROOT.kBlue); t_tl.SetTextFont(62)
    pt.AddText(f"#chi^{{2}}/ndf = {res.fit_Tl.GetChisquare():.1f}/{res.fit_Tl.GetNDF()} "
               f"(P:{res.fit_Tl.GetProb():.2f})")
    pt.AddText(f"#mu = {res.fit_Tl.GetParameter(1):.4f} #pm {res.fit_Tl.GetParError(1):.4f}")
    pt.AddText(f"#sigma = {res.fit_Tl.GetParameter(2):.5f} #pm {res.fit_Tl.GetParError(2):.5f}")
    pt.AddText("")

    if res.fit_alpha:
        t_al = pt.AddText("ALPHA FIT")
        t_al.SetTextColor(ROOT.kGreen + 2); t_al.SetTextFont(62)
        pt.AddText(f"#chi^{{2}}/ndf = {res.fit_alpha.GetChisquare():.1f}/{res.fit_alpha.GetNDF()} "
                   f"(P:{res.fit_alpha.GetProb():.2f})")
        pt.AddText(f"#mu = {res.fit_alpha.GetParameter(1):.4f} #pm {res.fit_alpha.GetParError(1):.4f}")
        pt.AddText(f"#sigma = {res.fit_alpha.GetParameter(2):.5f} #pm {res.fit_alpha.GetParError(2):.5f}")
    return pt


# ===========================================================================
# FIT-RESULT BOX + PEAK FINDER
# ===========================================================================

def CreateFitBox(fit, header, header_color=ROOT.kBlack,
                 x1=0.13, y1=0.60, x2=0.50, y2=0.88):
    """
    NDC TPaveText with a Gaussian fit's results: chi2/ndf, mu, sigma, FWHM and
    the percentage resolution (FWHM/mu*100). Returns the TPaveText (a
    placeholder text is shown when *fit* is None).
    """
    pt = ROOT.TPaveText(x1, y1, x2, y2, "NDC")
    pt.SetFillColor(ROOT.kWhite); pt.SetBorderSize(1)
    pt.SetTextAlign(12); pt.SetTextFont(42); pt.SetTextSize(0.035)

    t_h = pt.AddText(header)
    t_h.SetTextColor(header_color); t_h.SetTextFont(62)

    if fit is None:
        pt.AddText("(fit non disponibile)")
        return pt

    mu     = fit.GetParameter(1)
    sigma  = abs(fit.GetParameter(2))
    sig_e  = fit.GetParError(2)
    fwhm   = SIGMA_TO_FWHM * sigma
    fwhm_e = SIGMA_TO_FWHM * sig_e
    ndf    = fit.GetNDF()

    if ndf > 0:
        pt.AddText(f"#chi^{{2}}/ndf = {fit.GetChisquare():.1f}/{ndf} "
                   f"(P:{fit.GetProb():.2f})")
    pt.AddText(f"#mu = {mu:.4f} #pm {fit.GetParError(1):.4f}")
    pt.AddText(f"#sigma = {sigma:.5f} #pm {sig_e:.5f}")
    pt.AddText(f"FWHM = {fwhm:.5f} #pm {fwhm_e:.5f}")
    if abs(mu) > 1e-12:
        pt.AddText(f"Risoluzione = {100.0 * fwhm / abs(mu):.2f} %")
    return pt


def FindThalliumPeak(h_heat_orig):
    """
    Locate the thallium peak by scanning a sliding integration window and
    returning the (x, height) of the tallest bin inside the most populated
    window. Operates on a ROOT histogram.
    """
    n_bins_total = h_heat_orig.GetNbinsX()
    window_width = max(3, int(n_bins_total * 0.08))
    max_integral, best_peak_bin = -1, 1

    for b in range(1, n_bins_total - window_width + 1):
        current_integral = h_heat_orig.Integral(b, b + window_width)
        if current_integral > max_integral:
            max_integral    = current_integral
            max_val_in_win  = -1
            for w in range(b, b + window_width + 1):
                val = h_heat_orig.GetBinContent(w)
                if val > max_val_in_win:
                    max_val_in_win = val; best_peak_bin = w

    return h_heat_orig.GetXaxis().GetBinCenter(best_peak_bin), h_heat_orig.GetBinContent(best_peak_bin)


# ===========================================================================
# ROBUST PEAK FIT
# ===========================================================================

def robust_peak_gaussian_fit(hist, func_name, seed_mean, seed_sigma,
                             init_nsigma=3.0, core_nsigma=2.0,
                             max_iter=5, tol=1e-3):
    """
    Iterative Gaussian fit focused on the peak CORE, robust against the tails
    and the side events (background, neighbouring structures) that bias a plain
    wide-window fit.

    Strategy
    --------
      1. Seed mu/sigma from a robust estimate (median + IQR-based sigma).
      2. First pass over a moderately wide window [mu +/- init_nsigma*sigma] to
         lock onto the peak.
      3. Re-fit iteratively in a NARROW core window [mu +/- core_nsigma*sigma],
         re-centring on the latest fitted mu/sigma each time. As the window
         shrinks the lateral events drop out and stop pulling the Gaussian.
         Iterate until mu AND sigma are stable (relative change < tol) or until
         max_iter is reached.
      4. Guard against pathological iterations (sigma <= 0, NaN, runaway): the
         last good (amplitude, mu, sigma) is restored and the loop stops.

    The log-likelihood option ("L") is kept: it handles the low-statistics bins
    near the peak edges better than a plain chi^2.

    Returns the fitted ROOT.TF1 (its range is left set to the final core window,
    so when drawn the curve spans only the fitted core).
    """
    seed_sigma = max(abs(seed_sigma), 1e-6)

    fit = ROOT.TF1(func_name, "gaus",
                   seed_mean - init_nsigma * seed_sigma,
                   seed_mean + init_nsigma * seed_sigma)
    fit.SetParameters(hist.GetMaximum(), seed_mean, seed_sigma)
    hist.Fit(fit, "Q0 R L")

    mu, sigma = fit.GetParameter(1), abs(fit.GetParameter(2))
    if not (np.isfinite(mu) and np.isfinite(sigma)) or sigma <= 0:
        mu, sigma = seed_mean, seed_sigma

    last_good = (fit.GetParameter(0), mu, sigma)

    for _ in range(max_iter):
        fit.SetRange(mu - core_nsigma * sigma, mu + core_nsigma * sigma)
        fit.SetParameters(last_good[0], mu, sigma)
        hist.Fit(fit, "Q0 R L")

        new_mu, new_sigma = fit.GetParameter(1), abs(fit.GetParameter(2))

        # Reject a pathological iteration: restore the last good fit and stop.
        if not (np.isfinite(new_mu) and np.isfinite(new_sigma)) or new_sigma <= 0:
            fit.SetRange(last_good[1] - core_nsigma * last_good[2],
                         last_good[1] + core_nsigma * last_good[2])
            fit.SetParameters(*last_good)
            hist.Fit(fit, "Q0 R L")
            break

        d_mu    = abs(new_mu - mu)       / (abs(mu) + 1e-12)
        d_sigma = abs(new_sigma - sigma) / (sigma + 1e-12)
        last_good = (fit.GetParameter(0), new_mu, new_sigma)
        mu, sigma = new_mu, new_sigma
        if d_mu < tol and d_sigma < tol:
            break

    return fit


# ===========================================================================
# CORE ANALYSIS  (one file)
# ===========================================================================

def run_calibration(
    filename,
    save_summary_jpeg=True, save_corr_jpeg=True, create_root_file=True,
    show_canvas=False, manual_cuts=None, output_dir=None
):
    """
    Full calibration pipeline for a single ROOT file (NO baseline stabilization).

    Parameters
    ----------
    filename          : path of the input ROOT file.
    save_summary_jpeg : write the 3x3 overview JPEG.
    save_corr_jpeg    : write the correlation-analysis JPEG.
    create_root_file  : write the calibrated copy with the new TTree.
    show_canvas       : draw the canvases on screen (interactive mode).
    manual_cuts       : optional dict overriding the automatic cuts
                        (chosen_ld, ly_cut_min/max, heat_cut_min/max).
    output_dir        : base folder for the output sub-folders; defaults to the
                        folder containing *filename*.

    Returns
    -------
    gaussian_counts (float): area of the calibrated thallium peak, or
    -1.0 on a fatal read error.
    """
    # Output base folder for JPEGs and ROOT; defaults to the file's folder.
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(filename))

    print(f"\n{'='*60}")
    print(f" PROCESSING FILE: {os.path.basename(filename)}")

    base_name = os.path.basename(filename)
    # Channel parsed from the number following "ch" in the file name.
    match = re.search(r"ch(\d+)", base_name)
    ch_id = match.group(1) if match else "0"

    if not os.path.exists(filename):
        print(f"Error: File not found at {filename}", file=sys.stderr)
        return -1.0

    file = ROOT.TFile.Open(filename, "READ")
    if not file or file.IsZombie():
        return -1.0

    tree_cal = file.Get("calibration_rough") or file.Get("calibration_all")
    if not tree_cal or not hasattr(tree_cal, "GetEntries"):
        print(f"  [!] Missing calibration tree in {os.path.basename(filename)}", file=sys.stderr)
        file.Close(); return -1.0

    # Main tree: the (heater-)stabilized amplitude, always present.
    tree_main = file.Get("stabilization_all")
    tree_mod  = file.Get("module")
    tree_bad  = file.Get("badinterval")
    tree_trig = file.Get("numberoftriggers")
    tree_corr = file.Get("correlation_corr")
    tree_ly   = file.Get("LY")

    if not tree_main or not hasattr(tree_main, "GetEntries"):
        print(f"  [!] Missing 'stabilization_all' tree in {os.path.basename(filename)}",
              file=sys.stderr)
        file.Close(); return -1.0

    tree_for_calibration = tree_main

    print(f" MODE: HEATER + THALLIUM (linear calibration, no baseline term)")
    print(f"{'='*60}\n")

    # --- Attach friend trees to the main tree -------------------------------
    friend_list = [tree_mod, tree_bad, tree_trig, tree_corr, tree_cal]
    if tree_ly is not None:
        already = {tree_for_calibration.GetName()}
        already.update(t.GetName() for t in friend_list)
        if tree_ly.GetName() not in already:
            friend_list.append(tree_ly)
    for t in friend_list:
        tree_for_calibration.AddFriend(t)

    # Local aliases for the calibration windows (see PARAMETERS).
    cal_corr_min, cal_corr_max = CAL_CORR_MIN, CAL_CORR_MAX
    cal_ly_min,   cal_ly_max   = CAL_LY_MIN,   CAL_LY_MAX
    cal_stab_min, cal_stab_max = CAL_STAB_MIN, CAL_STAB_MAX

    # ======================================================================
    # OPTIONAL-BRANCH DETECTION
    # ======================================================================
    _ly_own_branches = ({b.GetName() for b in tree_ly.GetListOfBranches()}
                        if tree_ly is not None else set())
    has_ld2_branch       = "LD2_LY" in _ly_own_branches
    has_heat_badinterval = bool(tree_for_calibration.GetLeaf("heat_badinterval"))
    has_heat_issignal    = bool(tree_for_calibration.GetLeaf("heat_issignal"))
    cal_tree_name        = tree_cal.GetName()
    ly_tree_name         = tree_ly.GetName() if tree_ly is not None else None

    # The LY cut is on by default; it is disabled if LD1_LY is missing.
    apply_ly_cut = True
    if "LD1_LY" not in _ly_own_branches:
        print("  [!] 'LY' tree missing or without LD1_LY branch. LY cut DISABLED.")
        apply_ly_cut = False

    # ======================================================================
    # RDATAFRAME: C++-level filters + bulk read with AsNumpy
    # ======================================================================
    print("Phase 1: Reading events via RDataFrame + NumPy...")

    rdf = ROOT.RDataFrame(tree_for_calibration)

    # --- Quality filters (run at C++ speed) ---------------------------------
    rdf_f = rdf.Filter("heat_numberoftriggers == 1")
    if has_heat_badinterval:
        rdf_f = rdf_f.Filter("heat_badinterval == 0")
    if has_heat_issignal:
        rdf_f = rdf_f.Filter("heat_issignal != 0")

    rdf_f = rdf_f.Define("_cal_rough_", f"{cal_tree_name}.heat_amplitude")

    # --- Light Yield: TREE-QUALIFIED access from the "LY" tree --------------
    if apply_ly_cut:
        if ly_tree_name == tree_for_calibration.GetName():
            _ld1_ly_expr, _ld2_ly_expr = "LD1_LY", "LD2_LY"
        else:
            _ld1_ly_expr = f"{ly_tree_name}.LD1_LY"
            _ld2_ly_expr = f"{ly_tree_name}.LD2_LY"
        rdf_f = rdf_f.Define("_ld1_ly_", _ld1_ly_expr)
        if has_ld2_branch:
            rdf_f = rdf_f.Define("_ld2_ly_", _ld2_ly_expr)

    # --- Columns to read ----------------------------------------------------
    # NOTE: heat_baseline is NO LONGER read: there is no baseline-dependent term.
    cols = ["heat_amplitude", "_cal_rough_", "heat_correlation"]
    if apply_ly_cut:
        cols.append("_ld1_ly_")
        if has_ld2_branch:
            cols.append("_ld2_ly_")

    # --- Single bulk read ---------------------------------------------------
    np_data = rdf_f.AsNumpy(cols)
    N = len(np_data["heat_amplitude"])

    print(f"  Read {N} events after quality filters.")

    # --- Extract typed arrays -----------------------------------------------
    ha        = np_data["heat_amplitude"].astype(np.float64)
    cal_rough = np_data["_cal_rough_"].astype(np.float64)
    corr      = np_data["heat_correlation"].astype(np.float64)

    # Light Yield read DIRECTLY from the LD1_LY / LD2_LY leaves.
    ld1_ly   = np_data["_ld1_ly_"].astype(np.float64)          if apply_ly_cut     else np.zeros(N, np.float64)
    ld2_ly   = (np_data["_ld2_ly_"].astype(np.float64)
                if (apply_ly_cut and has_ld2_branch) else np.zeros(N, np.float64))

    # --- Vectorised NaN/Inf removal -----------------------------------------
    valid = np.isfinite(ha) & np.isfinite(cal_rough) & np.isfinite(corr)
    ha        = ha[valid];   cal_rough = cal_rough[valid]
    corr      = corr[valid]
    ld1_ly    = ld1_ly[valid]; ld2_ly = ld2_ly[valid]
    N = len(ha)

    # ======================================================================
    # DYNAMIC CORRELATION CUT  (vectorised)
    # ======================================================================
    mask_corr_valid = corr > CORR_VALID_MIN
    corr_above      = corr[mask_corr_valid]
    ha_above        = ha[mask_corr_valid]
    corr_sorted     = np.sort(corr_above)
    n_corr          = len(corr_sorted)

    corr_hist_min    = float(corr_sorted[int(n_corr * 0.01)]) if n_corr > 0 else CORR_VALID_MIN
    corr_cut_dynamic = (float(corr_sorted[int(n_corr * CORR_CUT_PERCENTILE)])
                        if n_corr > 0 else 0.9995)

    print(f">>> Correlation Cut ({int(CORR_CUT_PERCENTILE*100)}th percentile): {corr_cut_dynamic:.6f}")

    # Scatter (decimated for display) and distribution of the correlation.
    g_corr_vs_heat = make_scatter_graph(ha_above, corr_above)
    g_corr_vs_heat.SetName(f"g_corr_vs_heat_{ch_id}")
    g_corr_vs_heat.SetTitle(
        f"Ch {ch_id}: Correlation vs Heat Amplitude; Heat Amplitude; Correlation")

    h_corr = ROOT.TH1F(f"h_corr_{ch_id}",
                        f"Ch {ch_id}: Correlation Distribution; Correlation; Counts",
                        100, corr_hist_min, 1.00005)
    if n_corr > 0:
        h_corr.FillN(n_corr, corr_sorted.astype(np.double), np.ones(n_corr, np.double))

    # ======================================================================
    # CONVERSION FACTOR  (vectorised) -- only used to scale display axes/lines
    # ======================================================================
    mask_main = corr > corr_cut_dynamic
    mask_conv = mask_main & (cal_rough > cal_corr_min) & (cal_rough < cal_corr_max) & (cal_rough > 0)
    cnt_conv  = int(mask_conv.sum())

    if cnt_conv > 0:
        conv_raw = float(np.sum(ha[mask_conv] / cal_rough[mask_conv]) / cnt_conv)
    else:
        conv_raw = 1.0

    conv_factor = conv_raw

    print(f"Estimated conversion factor (clean events): Raw/Rough = {conv_raw:.7f}")

    # ======================================================================
    # SPECTRAL HISTOGRAMS  (bulk FillN)
    # ======================================================================
    n_main  = int(mask_main.sum())
    _w_main = np.ones(n_main, np.double)

    h_cal_rough = ROOT.TH1F(f"h_cal_rough_{ch_id}",
                             f"Ch {ch_id}: Calibration Rough after Correlation Cut; Amplitude; Counts",
                             80, 2300, 2800)
    if n_main > 0:
        h_cal_rough.FillN(n_main, cal_rough[mask_main].astype(np.double), _w_main)

    h_raw = ROOT.TH1F(f"h_raw_{ch_id}",
                       f"Ch {ch_id}: Heater Stabilized after Correlation Cut; Amplitude; Counts",
                       80, 2300 * conv_raw, 2800 * conv_raw)
    if n_main > 0:
        h_raw.FillN(n_main, ha[mask_main].astype(np.double), _w_main)

    h2_full = ROOT.TH1F(f"h2_full_{ch_id}",
                         f"Ch {ch_id}: Heater Stabilized after Correlation and LY Cut; Amplitude; Counts",
                         80, 2300 * conv_factor, 2800 * conv_factor)
    h2_full.SetLineColor(ROOT.kBlack)
    h2_full.SetFillColorAlpha(ROOT.kGreen + 1, 0.5)

    # ======================================================================
    # LIGHT-YIELD ANALYSIS  (vectorised) -- LY read directly from LD1_LY/LD2_LY
    # ======================================================================
    if apply_ly_cut:
        mask_ly_range = mask_main & (cal_rough > cal_ly_min) & (cal_rough < cal_ly_max)

        ly1_all = ld1_ly
        ly2_all = ld2_ly

        mask_ld1_ly = mask_ly_range & np.isfinite(ly1_all)
        mask_ld2_ly = mask_ly_range & np.isfinite(ly2_all) if has_ld2_branch else np.zeros(N, bool)

        vals_ly1_np = ly1_all[mask_ld1_ly];  heat_ly1_np = ha[mask_ld1_ly]
        vals_ly2_np = ly2_all[mask_ld2_ly];  heat_ly2_np = ha[mask_ld2_ly]

        bins_ly1, ly1_min, ly1_max = calcRobustLimitsAndBins(vals_ly1_np.tolist())
        bins_ly2, ly2_min, ly2_max = calcRobustLimitsAndBins(vals_ly2_np.tolist())

        h_ly1 = ROOT.TH1F(f"h_ly1_{ch_id}",
                           f"Ch {ch_id}: LD1 Light Yield; Light Yield; Counts",
                           bins_ly1, ly1_min, ly1_max)
        h_ly2 = ROOT.TH1F(f"h_ly2_{ch_id}",
                           f"Ch {ch_id}: LD2 Light Yield; Light Yield; Counts",
                           bins_ly2, ly2_min, ly2_max)

        n_ly1, n_ly2 = len(vals_ly1_np), len(vals_ly2_np)
        if n_ly1 > 0:
            h_ly1.FillN(n_ly1, vals_ly1_np.astype(np.double), np.ones(n_ly1, np.double))
        if n_ly2 > 0:
            h_ly2.FillN(n_ly2, vals_ly2_np.astype(np.double), np.ones(n_ly2, np.double))

        # LY-vs-heat scatter graphs (decimated for display).
        g_ly1_vs_heat = make_scatter_graph(heat_ly1_np, vals_ly1_np)
        g_ly1_vs_heat.SetName(f"g_ly1_vs_heat_{ch_id}")
        g_ly2_vs_heat = make_scatter_graph(heat_ly2_np, vals_ly2_np)
        g_ly2_vs_heat.SetName(f"g_ly2_vs_heat_{ch_id}")

        res1 = AnalyzeLightYield(h_ly1, f"LD1_{ch_id}") if h_ly1.GetEntries() > 0 else LYResult()
        res2 = AnalyzeLightYield(h_ly2, f"LD2_{ch_id}") if h_ly2.GetEntries() > 0 else LYResult()

        # Pick the LD with the higher discrimination factor (DF).
        chosen_ld        = 2 if res2.df_value > res1.df_value else 1
        ly_cut_min_final = res1.cut_min if chosen_ld == 1 else res2.cut_min
        ly_cut_max_final = res1.cut_max if chosen_ld == 1 else res2.cut_max

        # Manual overrides (interactive mode).
        if manual_cuts:
            if manual_cuts.get('chosen_ld')  is not None: chosen_ld        = manual_cuts['chosen_ld']
            if manual_cuts.get('ly_cut_min') is not None: ly_cut_min_final = manual_cuts['ly_cut_min']
            if manual_cuts.get('ly_cut_max') is not None: ly_cut_max_final = manual_cuts['ly_cut_max']
            print(f"\n>>> USING CUTS: LD{chosen_ld} | "
                  f"LY Range: [{ly_cut_min_final:.4f}, {ly_cut_max_final:.4f}] <<<")
        else:
            print(f"\n>>> AUTOMATIC CHOICE: LD{chosen_ld} SELECTED (Higher DF) <<<")
    else:
        print("\n>>> LY Cut OFF: Light Yield analysis ignored. <<<")

    # ======================================================================
    # VALID-EVENT MASK  (vectorised)
    # ======================================================================
    amp_for_analysis = ha   # the (heater-)stabilized amplitude

    if apply_ly_cut:
        ly_sel = ld1_ly if chosen_ld == 1 else ld2_ly
        mask_ly_pass = (mask_main
                        & np.isfinite(ly_sel)
                        & (ly_sel >= ly_cut_min_final)
                        & (ly_sel <= ly_cut_max_final))
    else:
        mask_ly_pass = mask_main

    # h2_full: all events passing correlation + LY.
    n_h2 = int(mask_ly_pass.sum())
    if n_h2 > 0:
        h2_full.FillN(n_h2,
                      amp_for_analysis[mask_ly_pass].astype(np.double),
                      np.ones(n_h2, np.double))

    # Sub-sample for the clean-peak fit (also restricted to the cal_stab window).
    mask_stab      = mask_ly_pass & (cal_rough > cal_stab_min) & (cal_rough < cal_stab_max)
    amps_for_stab  = amp_for_analysis[mask_stab]

    sufficient_events = len(amps_for_stab) >= 1
    if not sufficient_events:
        print(f"  [!] Warning: Only {len(amps_for_stab)} events survived the cuts. "
              f"Final fit impossible.")
        gaussian_counts = 0.0
    else:
        # ==================================================================
        # PRELIMINARY FIT
        # ==================================================================
        bins_heat, heat_min, heat_max = calcRobustLimitsAndBins(amps_for_stab.tolist())
        n_stab  = len(amps_for_stab)
        _w_stab = np.ones(n_stab, np.double)

        h_heat_orig = ROOT.TH1F(f"h_heat_orig_{ch_id}",
                                 f"Ch {ch_id}: Spectrum after LY cut - Pre-cleaning and with Thallium Peak selection;Amplitude;Counts",
                                 bins_heat, heat_min, heat_max)
        h_heat_orig.FillN(n_stab, amps_for_stab.astype(np.double), _w_stab)

        peak_x, peak_y = FindThalliumPeak(h_heat_orig)
        fit_window = (heat_max - heat_min) * 0.1
        fit_prelim = ROOT.TF1(f"fit_prelim_{ch_id}", "gaus",
                               peak_x - fit_window, peak_x + fit_window)
        fit_prelim.SetParameters(peak_y, peak_x, h_heat_orig.GetRMS() * 0.1)
        h_heat_orig.Fit(fit_prelim, "Q0 R L", "", peak_x - fit_window, peak_x + fit_window)

        mean_heat_prelim  = fit_prelim.GetParameter(1)
        sigma_heat_prelim = fit_prelim.GetParameter(2)
        if sigma_heat_prelim > fit_window or sigma_heat_prelim <= 0:
            mean_heat_prelim  = peak_x
            sigma_heat_prelim = h_heat_orig.GetRMS() * 0.1

        heat_cut_min = mean_heat_prelim - HEAT_CLEAN_NSIGMA * sigma_heat_prelim
        heat_cut_max = mean_heat_prelim + HEAT_CLEAN_NSIGMA * sigma_heat_prelim
        if manual_cuts:
            if manual_cuts.get('heat_cut_min') is not None: heat_cut_min = manual_cuts['heat_cut_min']
            if manual_cuts.get('heat_cut_max') is not None: heat_cut_max = manual_cuts['heat_cut_max']

        # ==================================================================
        # OUTLIER CLEANING
        # ==================================================================
        clean_mask     = (amps_for_stab >= heat_cut_min) & (amps_for_stab <= heat_cut_max)
        clean_amps_np  = amps_for_stab[clean_mask]
        n_clean        = len(clean_amps_np)

        params_clean = GetCenteredBinning(clean_amps_np.tolist(),
                                          heat_min + (heat_max - heat_min) / 2.0)
        h_heat_clean = ROOT.TH1F(f"h_heat_clean_{ch_id}",
                                  f"Ch {ch_id}: Thallium Peak BEFORE calibration ;Amplitude;Counts",
                                  params_clean.bins, params_clean.vis_min, params_clean.vis_max)

        if n_clean > 0:
            h_heat_clean.FillN(n_clean, clean_amps_np.astype(np.double),
                                np.ones(n_clean, np.double))

        # Gaussian fit of the clean thallium peak: its MEAN drives the calibration.
        fit_clean = ROOT.TF1(f"fit_clean_{ch_id}", "gaus",
                              params_clean.median - 5.0 * params_clean.robust_sigma,
                              params_clean.median + 5.0 * params_clean.robust_sigma)
        fit_clean.SetParameters(h_heat_clean.GetMaximum(),
                                 params_clean.median, params_clean.robust_sigma)
        h_heat_clean.Fit(fit_clean, "Q0 R L")
        mean_amp_clean = fit_clean.GetParameter(1)

        # ==================================================================
        # CALIBRATION FACTOR  (NO baseline stabilization)
        # ==================================================================
        # A SINGLE multiplicative factor rescales the whole spectrum so that the
        # clean-peak mean lands exactly on TARGET_ENERGY. No q_0 + slope*baseline.
        target_energy = TARGET_ENERGY
        if np.isfinite(mean_amp_clean) and abs(mean_amp_clean) > 1e-12:
            cal_factor = target_energy / mean_amp_clean
        else:
            cal_factor = 1.0
            print("  [!] Warning: invalid clean-peak mean; cal_factor forced to 1.0")

        print(f">>> Calibration factor (TARGET_ENERGY / clean-peak mean): {cal_factor:.7f}")

        # Apply the linear calibration to the clean thallium events (for display).
        a_cal     = clean_amps_np * cal_factor
        a_cal     = a_cal[np.isfinite(a_cal)]
        n_cal_evt = len(a_cal)

        # ==================================================================
        # CALIBRATED HISTOGRAM
        # ==================================================================
        params_cal = GetCenteredBinning(a_cal.tolist(), TARGET_ENERGY)
        h_heat_cal = ROOT.TH1F(f"h_heat_cal_{ch_id}",
                                f"Ch {ch_id}: Calibrated Thallium Peak ;Energy (keV);Counts",
                                params_cal.bins, params_cal.vis_min, params_cal.vis_max)
        h_heat_cal.SetDirectory(0)
        if n_cal_evt > 0:
            h_heat_cal.FillN(n_cal_evt, a_cal.astype(np.double),
                              np.ones(n_cal_evt, np.double))

        # Robust iterative CORE fit: a first pass on +/-3 sigma locks the peak,
        # then re-fits shrink to the +/-2 sigma core so the lateral events stop
        # biasing mu/sigma (and hence the resolution).
        fit_cal = robust_peak_gaussian_fit(
            h_heat_cal, f"fit_cal_{ch_id}",
            seed_mean=params_cal.median, seed_sigma=params_cal.robust_sigma,
            init_nsigma=3.0, core_nsigma=2.0)

        # Display window for the filled histogram: wider than the fit core, so
        # the peak shape (and some tails) stays visible under the fitted curve.
        mu_cal    = fit_cal.GetParameter(1)
        sigma_cal = abs(fit_cal.GetParameter(2))
        disp_min_cal = mu_cal - 4.0 * sigma_cal
        disp_max_cal = mu_cal + 4.0 * sigma_cal

        # Truncated histogram (drawn filled): the +/-4 sigma display region.
        h_heat_cal_final = h_heat_cal.Clone(f"h_heat_cal_final_{ch_id}")
        h_heat_cal_final.Reset()
        mask_cal_win = (a_cal >= disp_min_cal) & (a_cal <= disp_max_cal)
        n_cal_win    = int(mask_cal_win.sum())
        if n_cal_win > 0:
            h_heat_cal_final.FillN(n_cal_win,
                                    a_cal[mask_cal_win].astype(np.double),
                                    np.ones(n_cal_win, np.double))

        # Gaussian area = amplitude * sigma * sqrt(2pi) / bin_width.
        if n_clean > 3 and h_heat_cal.GetEntries() > 0:
            fit_cal_func    = h_heat_cal.GetFunction(f"fit_cal_{ch_id}")
            gaussian_counts = (fit_cal_func.GetParameter(0)
                               * fit_cal_func.GetParameter(2)
                               * math.sqrt(2 * math.pi)
                               / h_heat_cal.GetXaxis().GetBinWidth(1))
        else:
            gaussian_counts = float(n_clean)

        # ==================================================================
        # WRITE CALIBRATED .ROOT FILE
        # ==================================================================
        if create_root_file:
            print("\nCreating output file by cloning the original file...")
            pure_filename = os.path.basename(filename)
            base_filename = pure_filename.rsplit('.', 1)[0] if '.' in pure_filename else pure_filename

            out_dir = os.path.join(output_dir, CAL_ROOT_DIR_NAME)
            os.makedirs(out_dir, exist_ok=True)

            out_filename     = os.path.join(out_dir, f"{base_filename}_thallium_calibrated.root")
            source_tree_name = "stabilization_all"
            new_tree_name    = "calibrated_heater_thallium"
            new_tree_title   = "Heater + Thallium Calibrated (single linear factor)"

            ROOT.gSystem.CopyFile(filename, out_filename, ROOT.kTRUE)
            file_out = ROOT.TFile(out_filename, "UPDATE")

            # 'stabilization_all' owns heat_amplitude as a proper branch:
            # read it directly, no friend, no name ambiguity. NO baseline needed.
            tree_source_out = file_out.Get(source_tree_name)
            rdf_out  = ROOT.RDataFrame(tree_source_out)
            out_data = rdf_out.AsNumpy(["heat_amplitude"])
            raw_amps_out = out_data["heat_amplitude"].astype(np.float64)

            # Vectorised calibrated amplitude for EVERY event of stabilization_all.
            calibrated_amps = raw_amps_out * cal_factor

            new_tree           = ROOT.TTree(new_tree_name, new_tree_title)
            final_heat_amp_arr = array('d', [0.0])
            new_tree.Branch("heat_amplitude", final_heat_amp_arr, "heat_amplitude/D")
            for amp_val in calibrated_amps:
                final_heat_amp_arr[0] = float(amp_val)
                new_tree.Fill()

            new_tree.Write()
            global_tree = file_out.Get("global")
            if global_tree:
                global_tree.AddFriend(new_tree_name)
                global_tree.Write("", ROOT.TObject.kOverwrite)
            file_out.Close()
            print(f">>> Successfully saved '{new_tree_name}' tree to {out_filename}")

    # ======================================================================
    # CANVASES  (saved to JPEG, or shown on screen in interactive mode)
    # ======================================================================
    global_lines, canvases = [], []

    def drawLines(ymax, conv, color, style, min_val, max_val):
        """Draw the two vertical range lines (min, max) scaled by *conv*."""
        l_min = ROOT.TLine(min_val * conv, 0, min_val * conv, ymax)
        l_max = ROOT.TLine(max_val * conv, 0, max_val * conv, ymax)
        for l, c, s in [(l_min, color, style), (l_max, color, style)]:
            l.SetLineColor(c); l.SetLineWidth(2); l.SetLineStyle(s); l.Draw("same")
        global_lines.extend([l_min, l_max])

    # -------------------------------------------------------------- SUMMARY
    make_summary = show_canvas or save_summary_jpeg
    if make_summary:
        # ==================================================================
        # CALIBRATION OVERVIEW  (3x3 grid)
        #
        #  Row 1 :  Correlation hist  | Calibration Rough     | Heater Stabilized
        #  Row 2 :  LD1 Light Yield   | LD2 Light Yield       | After Corr + LY Cut
        #  Row 3 :  Original Thallium | Clean Peak (+fit+mean)| Calibrated Peak (+FWHM)
        # ==================================================================
        c_rough = ROOT.TCanvas(f"c_rough_{ch_id}",
                                f"Calibration Overview Ch {ch_id}",
                                500 * 3, 400 * 3)
        c_rough.Divide(3, 3)

        # ---------- ROW 1 : correlation histogram + spectra ----------
        # pad 1 : correlation distribution with the cut line
        c_rough.cd(1); ROOT.gPad.SetGrid()
        h_corr.SetLineColor(ROOT.kBlack); h_corr.SetFillColorAlpha(ROOT.kBlack, 0.3)
        h_corr.Draw(); c_rough.Update()
        lcr = ROOT.TLine(corr_cut_dynamic, ROOT.gPad.GetUymin(),
                         corr_cut_dynamic, ROOT.gPad.GetUymax())
        lcr.SetLineColor(ROOT.kRed); lcr.SetLineWidth(2); lcr.SetLineStyle(2)
        lcr.Draw("same"); global_lines.append(lcr)

        # pad 2 : Calibration Rough (after the correlation cut)
        c_rough.cd(2); ROOT.gPad.SetGrid()
        h_cal_rough.SetLineColor(ROOT.kBlack); h_cal_rough.SetFillColorAlpha(ROOT.kGray, 0.5)
        h_cal_rough.Draw(); c_rough.Update()
        drawLines(ROOT.gPad.GetUymax(), 1.0, ROOT.kBlue,    2, cal_ly_min,   cal_ly_max)
        drawLines(ROOT.gPad.GetUymax(), 1.0, ROOT.kRed,     1, cal_corr_min, cal_corr_max)
        drawLines(ROOT.gPad.GetUymax(), 1.0, ROOT.kGreen+2, 3, cal_stab_min, cal_stab_max)
        leg_rough = ROOT.TLegend(0.15, 0.70, 0.45, 0.85)
        leg_rough.SetBorderSize(1); leg_rough.SetFillColor(0)
        d_ly   = ROOT.TLine(); d_ly.SetLineColor(ROOT.kBlue);     d_ly.SetLineWidth(2);   d_ly.SetLineStyle(2)
        d_corr = ROOT.TLine(); d_corr.SetLineColor(ROOT.kRed);    d_corr.SetLineWidth(2); d_corr.SetLineStyle(1)
        d_stab = ROOT.TLine(); d_stab.SetLineColor(ROOT.kGreen+2); d_stab.SetLineWidth(3); d_stab.SetLineStyle(3)
        leg_rough.AddEntry(d_ly,   "LY Range",   "l")
        leg_rough.AddEntry(d_corr, "Corr Range", "l")
        leg_rough.AddEntry(d_stab, "Cal Range",  "l")
        leg_rough.Draw("same")
        global_lines.extend([d_ly, d_corr, d_stab, leg_rough])

        # pad 3 : Heater Stabilized (raw spectrum after the correlation cut)
        c_rough.cd(3); ROOT.gPad.SetGrid()
        h_raw.SetLineColor(ROOT.kBlack); h_raw.SetFillColorAlpha(ROOT.kOrange+1, 0.5)
        h_raw.Draw(); c_rough.Update()
        drawLines(ROOT.gPad.GetUymax(), conv_raw, ROOT.kBlue,    2, cal_ly_min,   cal_ly_max)
        drawLines(ROOT.gPad.GetUymax(), conv_raw, ROOT.kRed,     1, cal_corr_min, cal_corr_max)
        drawLines(ROOT.gPad.GetUymax(), conv_raw, ROOT.kGreen+2, 3, cal_stab_min, cal_stab_max)

        # ---------- ROW 2 : light yield + After Corr + LY Cut spectrum ----------
        if apply_ly_cut:
            c_rough.cd(4); ROOT.gPad.SetGrid()
            if h_ly1.GetEntries() > 0:
                h_ly1.SetStats(0); h_ly1.SetLineColor(ROOT.kRed)
                h_ly1.SetFillColorAlpha(ROOT.kRed, 0.3); h_ly1.Draw()
                if res1.fit_Tl:    res1.fit_Tl.Draw("same")
                if res1.fit_alpha: res1.fit_alpha.Draw("same")
                c_rough.Update()
                if res1.fit_Tl:
                    ll1 = ROOT.TLine(res1.cut_min, 0, res1.cut_min, ROOT.gPad.GetUymax())
                    lr1 = ROOT.TLine(res1.cut_max, 0, res1.cut_max, ROOT.gPad.GetUymax())
                    for l in (ll1, lr1):
                        l.SetLineColor(ROOT.kBlue); l.SetLineWidth(2)
                        l.SetLineStyle(2); l.Draw("same")
                    pt1 = CreateLYBox(res1, "LD1"); pt1.Draw()
                    global_lines.extend([ll1, lr1, pt1])

            c_rough.cd(5); ROOT.gPad.SetGrid()
            if h_ly2.GetEntries() > 0:
                h_ly2.SetStats(0); h_ly2.SetLineColor(ROOT.kRed)
                h_ly2.SetFillColorAlpha(ROOT.kRed, 0.3); h_ly2.Draw()
                if res2.fit_Tl:    res2.fit_Tl.Draw("same")
                if res2.fit_alpha: res2.fit_alpha.Draw("same")
                c_rough.Update()
                if res2.fit_Tl:
                    ll2 = ROOT.TLine(res2.cut_min, 0, res2.cut_min, ROOT.gPad.GetUymax())
                    lr2 = ROOT.TLine(res2.cut_max, 0, res2.cut_max, ROOT.gPad.GetUymax())
                    for l in (ll2, lr2):
                        l.SetLineColor(ROOT.kBlue); l.SetLineWidth(2)
                        l.SetLineStyle(2); l.Draw("same")
                    pt2 = CreateLYBox(res2, "LD2"); pt2.Draw()
                    global_lines.extend([ll2, lr2, pt2])

        c_rough.cd(6); ROOT.gPad.SetGrid()
        h2_full.Draw(); c_rough.Update()
        drawLines(ROOT.gPad.GetUymax(), conv_factor, ROOT.kBlue,    2, cal_ly_min,   cal_ly_max)
        drawLines(ROOT.gPad.GetUymax(), conv_factor, ROOT.kRed,     1, cal_corr_min, cal_corr_max)
        drawLines(ROOT.gPad.GetUymax(), conv_factor, ROOT.kGreen+2, 3, cal_stab_min, cal_stab_max)

        # ---------- ROW 3 : original peak, clean peak (+fit+mean), calibrated peak ----------
        c_rough.cd(7); ROOT.gPad.SetGrid()
        if sufficient_events:
            h_heat_orig.SetLineColor(ROOT.kBlue); h_heat_orig.SetStats(0)
            h_heat_orig.Draw()
            fn = h_heat_orig.GetFunction(f"fit_prelim_{ch_id}")
            if fn: fn.SetLineColor(ROOT.kRed); fn.SetLineWidth(2); fn.Draw("same")

        c_rough.cd(8); ROOT.gPad.SetGrid()
        if sufficient_events:
            h_heat_clean.SetStats(0); h_heat_clean.SetLineColor(ROOT.kBlack)
            h_heat_clean.SetFillColorAlpha(ROOT.kMagenta, 0.5); h_heat_clean.Draw()
            fn = h_heat_clean.GetFunction(f"fit_clean_{ch_id}")
            if fn: fn.SetLineColor(ROOT.kMagenta+2); fn.SetLineWidth(2); fn.Draw("same")
            box_clean = CreateFitBox(fn, "CLEAN PEAK (mean -> calibration)", ROOT.kMagenta+2)
            box_clean.Draw(); global_lines.append(box_clean)

        c_rough.cd(9); ROOT.gPad.SetGrid()
        if sufficient_events:
            h_heat_cal.SetLineColor(ROOT.kBlack); h_heat_cal.SetLineWidth(1)
            h_heat_cal.SetFillStyle(0); h_heat_cal.SetStats(0); h_heat_cal.Draw()
            h_heat_cal_final.SetLineColor(ROOT.kBlack)
            h_heat_cal_final.SetFillColorAlpha(ROOT.kRed + 1, 0.5); h_heat_cal_final.Draw("same")
            fn = h_heat_cal.GetFunction(f"fit_cal_{ch_id}")
            if fn: fn.SetLineColor(ROOT.kBlue); fn.SetLineWidth(2); fn.Draw("same")
            box_cal = CreateFitBox(fn, "CALIBRATED Tl RESOLUTION", ROOT.kRed + 1)
            box_cal.Draw(); global_lines.append(box_cal)
            # Small note: the multiplicative factor and the clean-peak mean.
            pt_factor = ROOT.TPaveText(0.55, 0.13, 0.88, 0.30, "NDC")
            pt_factor.SetFillColor(ROOT.kWhite); pt_factor.SetBorderSize(1)
            pt_factor.SetTextAlign(12); pt_factor.SetTextFont(42); pt_factor.SetTextSize(0.032)
            pt_factor.AddText(f"cal factor = {cal_factor:.6f}")
            pt_factor.AddText(f"clean #mu = {mean_amp_clean:.3f}")
            pt_factor.Draw(); global_lines.append(pt_factor)

        c_rough.Update()
        if save_summary_jpeg:
            summary_dir = os.path.join(output_dir, SUMMARY_DIR_NAME)
            os.makedirs(summary_dir, exist_ok=True)
            out_jpg = os.path.join(summary_dir, f"ch{ch_id}_calibration_overview.jpg")
            save_canvas_jpeg(c_rough, out_jpg)
        canvases.append(c_rough)

    # ----------------------------------------------------------- CORRELATION
    make_corr = show_canvas or save_corr_jpeg
    if make_corr:
        c_corr = ROOT.TCanvas(f"c_corr_{ch_id}", f"Correlation Analysis Ch {ch_id}", 1200, 600)
        c_corr.Divide(2, 1)
        c_corr.cd(1); ROOT.gPad.SetGrid()
        g_corr_vs_heat.SetMarkerStyle(20); g_corr_vs_heat.SetMarkerSize(0.4)
        g_corr_vs_heat.SetMarkerColor(ROOT.kBlack); g_corr_vs_heat.Draw("AP"); c_corr.Update()
        lc1 = ROOT.TLine(ROOT.gPad.GetUxmin(), corr_cut_dynamic,
                          ROOT.gPad.GetUxmax(), corr_cut_dynamic)
        lc1.SetLineColor(ROOT.kRed); lc1.SetLineWidth(2); lc1.SetLineStyle(2)
        lc1.Draw("same"); global_lines.append(lc1)
        c_corr.cd(2); ROOT.gPad.SetGrid()
        h_corr.SetLineColor(ROOT.kBlack); h_corr.SetFillColorAlpha(ROOT.kBlack, 0.3)
        h_corr.Draw(); c_corr.Update()
        lc2 = ROOT.TLine(corr_cut_dynamic, ROOT.gPad.GetUymin(),
                          corr_cut_dynamic, ROOT.gPad.GetUymax())
        lc2.SetLineColor(ROOT.kRed); lc2.SetLineWidth(2); lc2.SetLineStyle(2)
        lc2.Draw("same"); global_lines.append(lc2)
        c_corr.Update()
        if save_corr_jpeg:
            corr_dir = os.path.join(output_dir, CORR_DIR_NAME)
            os.makedirs(corr_dir, exist_ok=True)
            out_jpg = os.path.join(corr_dir, f"ch{ch_id}_correlation_cut.jpg")
            save_canvas_jpeg(c_corr, out_jpg)
        canvases.append(c_corr)

    # --- MEMORY PROTECTION: only when canvases are shown interactively ------
    if show_canvas:
        global GLOBAL_KEEPALIVE
        GLOBAL_KEEPALIVE.extend(canvases)
        GLOBAL_KEEPALIVE.extend(global_lines)

        keep = [h_cal_rough, h_raw, h_corr, g_corr_vs_heat, h2_full]
        if apply_ly_cut:
            keep.extend([h_ly1, h_ly2, g_ly1_vs_heat, g_ly2_vs_heat,
                         res1.fit_Tl, res1.fit_alpha, res2.fit_Tl, res2.fit_alpha])
        if sufficient_events:
            keep.extend([h_heat_orig, h_heat_clean, h_heat_cal, h_heat_cal_final])
        GLOBAL_KEEPALIVE.extend([o for o in keep if o is not None])

        for canvas in canvases:
            ROOT.SetOwnership(canvas, False)
            canvas.Modified(); canvas.Update()

    file.Close()
    return gaussian_counts


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Calibrate (single linear factor, no baseline stabilization) "
                    "the .root files in BASE_DIR for the channels in "
                    "CHANNELS_TO_PROCESS (or the channels given on the command line).")
    parser.add_argument("channels", nargs="*", default=[],
                        help="Channels to analyse (e.g. 24 51). If omitted, "
                             "CHANNELS_TO_PROCESS is used.")
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # FLAGS  (top-level switches)
    # --------------------------------------------------------------------------
    SAVE_SUMMARY_JPEG = True    # save the overview JPEGs   (ThalliumCalibrationDebug)
    SAVE_CORR_JPEG    = False   # save the correlation JPEGs (CorrelationCut)
    CREATE_ROOT_FILE  = True    # write the calibrated ROOT files (ThalliumCalibratedAmp)
    GUI_MANUAL_CUTS   = False   # interactive manual-cut GUI.
                                # Active ONLY when EXACTLY ONE channel is selected.

    # --------------------------------------------------------------------------
    # CHANNEL LIST  (in-code selection)
    # --------------------------------------------------------------------------
    #   []            -> process ALL files in BASE_DIR, in batch (no GUI).
    #   [N]           -> single channel; GUI if GUI_MANUAL_CUTS is True.
    #   [N, M, ...]   -> several channels, ALWAYS processed in batch (no GUI).
    # Command-line channels, if given, OVERRIDE this list (same GUI rule).
    CHANNELS_TO_PROCESS = [25, 26, 27, 28, 29, 55, 56, 57, 58, 59, 60]

    # The command line wins over the in-code list when channels are passed.
    if args.channels:
        target_channels = {str(c) for c in args.channels}
    else:
        target_channels = {str(c) for c in CHANNELS_TO_PROCESS}

    # GUI is active only when the flag is on AND exactly one channel is selected.
    gui_active = GUI_MANUAL_CUTS and len(target_channels) == 1

    # Folder containing the .root files to analyse.
    BASE_DIR = "/data/users/azanelli/octopus_work/CROSS/MergedRuns"

    if not os.path.isdir(BASE_DIR):
        print(f"[!] Error: folder not found: {BASE_DIR}")
        sys.exit(1)

    # --- Run-mode banner --------------------------------------------------------
    if not target_channels:
        print(">>> Channel selection: ALL files in folder (batch mode).")
    else:
        print(f">>> Channel selection: {', '.join(sorted(target_channels, key=int))}  "
              f"({'GUI' if gui_active else 'batch'} mode).")

    # Batch mode (off-screen rendering): ON except when the GUI is active.
    ROOT.gROOT.SetBatch(not gui_active)

    if gui_active:
        # Warm up tkinter BEFORE any TCanvas (needed on macOS).
        try:
            _tk_warmup = tk.Tk()
            _tk_warmup.withdraw()
            _tk_warmup.update()
            _tk_warmup.destroy()
        except tk.TclError:
            pass

    # Collect the valid .root files (skip already-calibrated / -stabilized outputs).
    root_files = sorted(f for f in os.listdir(BASE_DIR)
                        if f.endswith(".root")
                        and "calibrated" not in f and "stabilized" not in f)

    if not root_files:
        print(f"[!] No .root file found in {BASE_DIR}")
        sys.exit(1)

    any_success = False
    processed   = 0

    for fname in root_files:
        # Channel parsed from the file name (number after "ch").
        m = re.search(r"ch(\d+)", fname)
        ch_id = m.group(1) if m else None

        # Filter by the selected channels (empty set => take every file).
        if target_channels and (ch_id is None or ch_id not in target_channels):
            continue

        full_path = os.path.join(BASE_DIR, fname)
        print(f"\n>>> File: {fname}  (channel {ch_id})")

        try:
            if gui_active:
                # --------------------------------------------------------------
                # INTERACTIVE GUI MODE  (only reachable with a single channel)
                # --------------------------------------------------------------
                manual_cuts_dict = {}
                counts = -1.0
                while True:
                    counts = run_calibration(
                        filename=full_path,
                        save_summary_jpeg=False, save_corr_jpeg=False,
                        create_root_file=False,
                        show_canvas=True,
                        manual_cuts=manual_cuts_dict if manual_cuts_dict else None,
                        output_dir=BASE_DIR,
                    )

                    param_editor = ParamEditorApp(ch_id, manual_cuts_dict)
                    action, manual_cuts_dict = param_editor.run()

                    if action == 'quit':
                        print("\n[!] Exiting the script...")
                        sys.exit(0)

                    elif action == 'recalc':
                        for obj in GLOBAL_KEEPALIVE:
                            if isinstance(obj, ROOT.TCanvas): obj.Close()
                        GLOBAL_KEEPALIVE.clear()
                        print("\n>>> Recomputing with the new parameters...\n")
                        continue

                    elif action == 'accept':
                        # Close the current interactive canvases.
                        for obj in GLOBAL_KEEPALIVE:
                            if isinstance(obj, ROOT.TCanvas): obj.Close()
                        GLOBAL_KEEPALIVE.clear()
                        print("\n>>> Parameters accepted! Saving JPEG and calibrated ROOT...")
                        # Temporary batch: writing the JPEGs must not pop windows.
                        ROOT.gROOT.SetBatch(True)
                        counts = run_calibration(
                            filename=full_path,
                            save_summary_jpeg=SAVE_SUMMARY_JPEG,
                            save_corr_jpeg=SAVE_CORR_JPEG,
                            create_root_file=CREATE_ROOT_FILE,
                            show_canvas=False,
                            manual_cuts=manual_cuts_dict if manual_cuts_dict else None,
                            output_dir=BASE_DIR,
                        )
                        ROOT.gROOT.SetBatch(False)
                        break
            else:
                # --------------------------------------------------------------
                # BATCH MODE  (whole folder, empty list, or more than one channel)
                # --------------------------------------------------------------
                counts = run_calibration(
                    filename=full_path,
                    save_summary_jpeg=SAVE_SUMMARY_JPEG,
                    save_corr_jpeg=SAVE_CORR_JPEG,
                    create_root_file=CREATE_ROOT_FILE,
                    show_canvas=False,
                    output_dir=BASE_DIR,
                )
        except SystemExit:
            raise
        except Exception as e:
            # An error on one file must not stop the analysis of the others.
            print(f"\n[!] ERROR while analysing {fname}: {e}", file=sys.stderr)
            traceback.print_exc()
            counts = -1.0

        processed += 1
        if counts != -1.0 and counts >= 0:
            print(f"\n>>> Channel {ch_id} done! Counts under the peak: {counts:.2f} <<<")
            any_success = True

    if processed == 0:
        if target_channels:
            print(f"\n[!] No file matching the selected channels: "
                  f"{', '.join(sorted(target_channels, key=int))}")
        else:
            print("\n[!] No file processed.")
        sys.exit(1)

    if any_success:
        print("\n" + "=" * 50)
        print("All requested files have been processed.")
        if SAVE_SUMMARY_JPEG:
            print(f"  Overview JPEGs    -> {os.path.join(BASE_DIR, SUMMARY_DIR_NAME)}")
        if SAVE_CORR_JPEG:
            print(f"  Correlation JPEGs -> {os.path.join(BASE_DIR, CORR_DIR_NAME)}")
        if CREATE_ROOT_FILE:
            print(f"  Calibrated ROOT   -> {os.path.join(BASE_DIR, CAL_ROOT_DIR_NAME)}")
        print("=" * 50 + "\n")
    else:
        print("\n>>> No channel processed successfully. <<<\n")