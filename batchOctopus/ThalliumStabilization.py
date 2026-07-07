#!/usr/bin/env python3
"""
ThalliumStabilization.py  (batch analysis + optional interactive GUI)
=====================================================================
Pipeline overview
-----------------
  1. Discover every ROOT file in BASE_DIR (files containing "stabilized" in the
     name are skipped). The detector channel is parsed from the number that
     follows "ch" in the file name (e.g. ..._ch50_... -> channel 50). The set of
     channels to analyse is taken from CHANNELS_TO_PROCESS (in-code list) or,
     when given, from the command line (which overrides the list).

  2. For each file, bulk-read the main tree "corrected_amplitude" together with
     its friend trees via RDataFrame + AsNumpy (one C++-side pass, no per-event
     Python loop). Quality filters (single trigger, good interval, signal flag)
     are applied at the RDataFrame level.

  3. Build a DYNAMIC correlation cut: keep events whose heat_correlation lies
     above the CORR_CUT_PERCENTILE-th percentile of the valid-correlation
     distribution. This rejects poorly reconstructed pulses.

  4. Light-Yield analysis (read directly from the "LY" tree, leaves LD1_LY /
     LD2_LY): fit the thallium and (when present) the alpha peak in each light
     detector, pick the LD with the larger discrimination factor (DF), and
     derive the LY acceptance window [cut_min, cut_max].

  5. Select thallium-peak events surviving correlation + LY cuts, clean
     outliers around the peak, and perform a ROBUST pol1 fit of amplitude vs
     baseline (ROOT "ROB=0.90" estimator).

  6. Apply the per-event linear stabilization to the thallium reference energy:

         a_stab_cal = TARGET_ENERGY * amp / (q_0 + slope * baseline)

     Build the calibrated spectrum and measure its resolution (FWHM) from a
     Gaussian fit.

  7. [optional] Write a full copy of the input file to a dedicated folder and
     append the TTree "stabilized_heater_thallium" (one calibrated amplitude
     per event of "corrected_amplitude").

  8. Render two summary canvases. In BATCH mode they are saved as JPEG; in GUI
     mode they are shown on screen and the user can tweak the manual cuts, then
     recompute / accept (save JPEG + ROOT) / quit.

Modes
-----
  BATCH  (default; used for the whole folder, an empty channel list, OR a list
         with MORE THAN ONE channel):
         ROOT runs head-less (SetBatch(True)); canvases are written to JPEG.
  GUI    (only when GUI_MANUAL_CUTS is True AND exactly ONE channel is selected):
         canvases are shown interactively and a Tk dialog edits the cuts.

Channel selection  (set in __main__)
------------------------------------
  CHANNELS_TO_PROCESS : in-code list of channels (numbers after "ch").
                        []  -> analyse ALL files in BASE_DIR, in batch.
                        [N] -> single channel (GUI if GUI_MANUAL_CUTS=True).
                        [N, M, ...] -> several channels, ALWAYS batch.
  Command-line channels (if given) OVERRIDE CHANNELS_TO_PROCESS and follow the
  same GUI rule (GUI only when exactly one channel is requested).

Flags  (set in __main__)
------
  SAVE_SUMMARY_JPEG : save the stabilization-overview JPEGs
  SAVE_CORR_JPEG    : save the correlation-analysis JPEGs
  CREATE_ROOT_FILE  : write the stabilized ROOT files
  GUI_MANUAL_CUTS   : enable the interactive manual-cut GUI (single channel only)

Input
-----
  ROOT files in BASE_DIR. Each file must contain at least the trees:
    corrected_amplitude (main: heat_amplitude, heat_baseline, ...),
    calibration_rough (or calibration_all), correlation_corr, baseline,
    module, badinterval, numberoftriggers, and LY (leaves LD1_LY / LD2_LY).

Output
------
  ThalliumStabilizationDebug/ch<N>_stabilization_overview.jpg  (4x3 overview)
  CorrelationCut/ch<N>_correlation_cut.jpg                     (correlation cut)
  Stabilized_Output/<stem>_thallium_stabilized.root           (copy + new TTree)
  All output folders are created (if missing) inside BASE_DIR.

Execution
---------
  python3 ThalliumStabilization.py                 # uses CHANNELS_TO_PROCESS
  python3 ThalliumStabilization.py 24 51           # overrides with channels 24, 51
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

# Keeps ROOT objects alive in interactive mode (see run_stabilization). The
# Python garbage collector would otherwise free histograms/graphs/fits and the
# on-screen canvas would go blank at the first repaint.
GLOBAL_KEEPALIVE = []

# NB: the ROOT batch mode (off-screen rendering) is decided in __main__ from the
# run mode: ON for batch analysis, OFF when the interactive GUI is active.


# ===========================================================================
# RUN CONFIGURATION  <- edit here (everything the run needs)
# ===========================================================================

# --- Analysis mode ----------------------------------------------------------
# "mergedrun"     : files in BASE_DIR named like 000097_000128_ch25_73_74.root
#                   (channel = the number after "ch").
# "run"           : a single run; files in <CROSS_DIR>/RUN<NNNNNN>/Coincidence
#                   named like 25_73_74_000096.root (channel = the FIRST number).
#                   The run folder is built from RUN_NUMBER, zero-padded to 6 digits.
# "calibrationrun": same files/paths/parsing as "run", but the LY cut uses the
#                   thallium-dominant peak finder (AnalyzeLightYieldRun), suited
#                   to calibration runs where Tl >> alpha and alpha can be at
#                   negative LY. "run" and "mergedrun" use the standard finder.
ANALYSIS_MODE = "mergedrun"     # "mergedrun", "run" or "calibrationrun"

# mergedrun mode: folder containing the input .root files.
BASE_DIR = "/data/users/azanelli/octopus_work/CROSS/MergedRuns/CorrectedAmp"

# run mode: CROSS folder holding the RUNxxxxxx sub-folders, and the run number.
CROSS_DIR  = "/data/users/azanelli/octopus_work/CROSS"
RUN_NUMBER = 96                 # e.g. 96 -> folder RUN000096, sub-folder Coincidence

# --- Channels to analyse (number after "ch" in the file name) ---------------
#   []          -> process ALL files in BASE_DIR (batch).
#   [N]         -> single channel; GUI if GUI_MANUAL_CUTS is True.
#   [N, M, ...] -> several channels, always batch.
# Command-line channels (if given) OVERRIDE this list.
CHANNELS_TO_PROCESS = [25, 26, 27, 28, 29, 30, 55, 56, 57, 58, 59, 60] 

# --- Output / GUI switches --------------------------------------------------
SAVE_SUMMARY_JPEG   = True    # per-channel debug JPEGs (global, partitions, before/after)
SAVE_CORR_JPEG      = False   # correlation-analysis JPEGs
SAVE_PARTITION_JPEG = True    # baseline-partition debug JPEG
CREATE_ROOT_FILE    = True    # write the stabilized ROOT files
GUI_MANUAL_CUTS     = False   # interactive manual-cut GUI (active only with a single channel)

# Baseline partitioning: when False the dataset is NOT split by baseline and a
# single stabilization is performed over the whole baseline range.
ENABLE_BASELINE_PARTITIONS = True

# --- Stabilization-line I/O -------------------------------------------------
# Each partition's stabilization is a line  amp = q0 + slope * baseline.
#   SAVE_STAB_LINES        : write these lines (one row per partition: channel,
#                            partition, baseline range, slope, q0) into the
#                            output ROOT file, in the STAB_LINES_TREE_NAME tree.
#   USE_EXTERNAL_STAB_LINE : apply the lines (and their baseline intervals) read
#                            from EXTERNAL_STAB_LINE_FILE instead of fitting them
#                            from the data; lines are matched by channel.
#   EXTERNAL_STAB_LINE_FILE: ROOT file holding a STAB_LINES_TREE_NAME tree.
SAVE_STAB_LINES          = False
USE_EXTERNAL_STAB_LINE   = False
EXTERNAL_STAB_LINE_DIR   = "/data/users/azanelli/octopus_work/CROSS/RUN000096/ThalliumStabilizedAmp"
STAB_LINES_TREE_NAME     = "stabilization_lines"


# --- Heat-amplitude source --------------------------------------------------
# Default: 'corrected_amplitude.heat_amplitude' when that tree exists, otherwise
# 'stabilization_all.heat_amplitude'. For the channels in OPTIMUM_FILTER_CHANNELS
# the amplitude is taken from 'optimumfilter_all.heat_amplitude' instead. The chosen
# source is printed for every file. When the 'LY' tree is absent, the light yield
# is computed as LDx_amplitude / heat_amplitude from the optimum-filter tree.
AMP_TREE_DEFAULT        = "corrected_amplitude"
AMP_TREE_FALLBACK       = "stabilization_all"
AMP_TREE_OPTIMUM        = "optimumfilter_all"
OPTIMUM_FILTER_CHANNELS = [25, 59]   # channels using AMP_TREE_OPTIMUM.heat_amplitude


# ===========================================================================
# PARAMETERS  <- tune these
# ===========================================================================

# --- Output folder names (created if missing, inside BASE_DIR) --------------
STAB_ROOT_DIR_NAME = os.path.join("..", "ThalliumStabilizedAmp")            # stabilized ROOT files

SUMMARY_DIR_NAME   = os.path.join("..", "ThalliumStabilizedAmp/ThalliumStabilizationDebug")  # 4x3 overview JPEGs
CORR_DIR_NAME      = os.path.join("..", "ThalliumStabilizedAmp/CorrelationCut")               # correlation-cut JPEGs
# All per-channel debug JPEGs (baseline partitions, global overview, per-partition
# stabilization, before/after comparison) go to <SUMMARY_DIR>/ch<N>/.

# --- Plotting ---------------------------------------------------------------
# Max number of markers drawn in any overview scatter (TGraph). Rasterising a
# scatter is ~linear in the marker count, so tens of thousands of points make
# the JPEG SaveAs extremely slow in batch mode. The decimation is for DISPLAY
# ONLY -- the analysis and all fits always use the full dataset.
MAX_SCATTER_POINTS = 4000

# --- Calibration-energy windows (rough-calibration units) -------------------
CAL_CORR_MIN, CAL_CORR_MAX = 2400.0, 2700.0   # window for the conversion factor
CAL_LY_MIN,   CAL_LY_MAX   = 2350.0, 2700.0   # window for the light-yield study
CAL_STAB_MIN, CAL_STAB_MAX = 2400.0, 2700.0   # window for the stabilization fit

# --- Thallium reference -----------------------------------------------------
TARGET_ENERGY = 2614.511   # keV, 208-Tl line used as the calibration anchor

# --- Correlation cut --------------------------------------------------------
CORR_VALID_MIN      = 0.999   # events below this correlation are ignored
CORR_CUT_PERCENTILE = 0.10    # dynamic cut at this percentile of valid corr

# --- Peak-selection half-widths (in sigma) ----------------------------------
LY_N_SIGMA        = 4.0   # thallium acceptance half-width in the LY spectrum
HEAT_CLEAN_NSIGMA = 1.5   # pre-cleaning half-width around the thallium peak

# --- Baseline partitioning --------------------------------------------------
# The amplitude-vs-baseline scatter splits into several clusters along the
# baseline axis. Partitions separate ONLY blocks that are clearly DETACHED in
# the baseline histogram: a boundary is placed across a gap of consecutive
# EMPTY bins. Close peaks with counts between them stay in the same partition.
PART_N_BINS          = 200    # bins of the baseline histogram used for the search
PART_MIN_GAP_BINS    = 4      # a separation needs at least this many consecutive low bins
PART_GAP_HEIGHT_FRAC = 0.03   # a bin belongs to a GAP when its height is below this fraction
                              # of the tallest block: gaps are judged by RELATIVE height, not
                              # raw counts (a few stray counts in a deep valley still separate)
PART_MIN_BLOCK_FRAC  = 0.15   # blocks below this fraction of the MOST POPULATED block are
                              # absorbed into the nearest (in baseline) block: small isolated
                              # peaks join the closest partition instead of forming their own

# --- Thallium-peak search hint (multi-partition) ----------------------------
# The thallium peak is first located on the COMBINED spectrum (all partitions
# merged): that position is reused as a SEARCH HINT for every partition, whose
# peak finder is restricted to [hint - halfwidth, hint + halfwidth]. This keeps
# low-statistics partitions from locking the peak onto a noise fluctuation. The
# amplitude shift between partitions (baseline slope) is small, so a generous
# window is safe.
PEAK_HINT_NSIGMA   = 8.0    # search half-width in units of the combined-peak sigma
PEAK_HINT_MIN_FRAC = 0.20   # ...but at least this fraction of the partition heat range


# --- Gaussian -> FWHM conversion constant: 2*sqrt(2*ln2) --------------------
SIGMA_TO_FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))


# ===========================================================================
# PATH / NAMING HELPERS  (analysis-mode dependent)
# ===========================================================================

def resolve_scan_dir():
    """
    Folder to scan for input .root files, depending on ANALYSIS_MODE:
      "mergedrun"               -> BASE_DIR
      "run" / "calibrationrun"  -> <CROSS_DIR>/RUN<NNNNNN>/Coincidence  (6 digits)
    """
    if ANALYSIS_MODE in ("run", "calibrationrun"):
        return os.path.join(CROSS_DIR, f"RUN{int(RUN_NUMBER):06d}", "Coincidence")
    return BASE_DIR


def parse_channel_id(filename):
    """
    Channel id parsed from the file name, depending on ANALYSIS_MODE:
      "mergedrun"               -> the number after "ch" (..._ch25_73_74.root -> 25)
      "run" / "calibrationrun"  -> the FIRST number      (25_73_74_000096.root -> 25)
    Returns the channel string, or None when it cannot be parsed.
    """
    base = os.path.basename(filename)
    if ANALYSIS_MODE in ("run", "calibrationrun"):
        m = re.match(r"(\d+)_", base)
    else:
        m = re.search(r"ch(\d+)", base)
    return m.group(1) if m else None


# ===========================================================================
# PLOTTING / IO HELPERS
# ===========================================================================

def make_scatter_graph(x_arr, y_arr, max_points=MAX_SCATTER_POINTS):
    """
    Build a TGraph from two NumPy arrays, uniformly decimating when the point
    count exceeds *max_points*.

    The uniform sub-sampling preserves the visual shape of the distribution
    while keeping the JPEG rasterisation fast. Decimation affects DISPLAY ONLY.

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
    Save a canvas to JPEG robustly (useful in batch mode).

    Forces a repaint before the SaveAs and swallows graphics-backend errors so
    that a single failed write never aborts the whole run. If the JPEG write
    fails, a PNG with the same stem is attempted as a fallback.

    Returns True on success, False otherwise.
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

    The median sets the centre and the inter-quartile range gives a robust
    sigma. The visible range is widened to ~7.5 sigma so that the +/-5 sigma
    Gaussian fit has white space on both sides (always containing all points
    via the max_dev * 1.10 floor). Bin width ~ 0.7 sigma keeps the profile from
    looking jagged.

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

    Uses the 5th/95th percentiles (padded by 40 % of their span) for the range
    and the IQR-derived sigma for the bin width. Returns (n_bins, min, max).
    """
    if len(vals) == 0:
        return 40, -0.05, 0.15

    arr = np.sort(np.asarray(vals, dtype=np.float64))
    n   = len(arr)

    q02 = arr[int(n * 0.05)]
    q98 = arr[int(n * 0.95)]
    vis_range = max(q98 - q02, 1e-6) if (q98 - q02) >= 1e-6 else 0.1

    min_out = q02 - vis_range * 0.1
    max_out = q98 + vis_range * 0.1

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

    Workflow:
      - TSpectrum peak search; rank candidates by integrated area and keep at
        most two well-separated positive peaks.
      - Gaussian fit of the thallium peak (and alpha, when two peaks survive).
      - Discrimination factor DF = |mu_Tl - mu_alpha| / sqrt(sig_Tl^2+sig_alpha^2).
      - Acceptance window: upper bound mu_Tl + LY_N_SIGMA*sig_Tl; lower bound the
        alpha/thallium valley when alpha exists, else mu_Tl - LY_N_SIGMA*sig_Tl.

    Operates on a ROOT histogram. Returns an LYResult.
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


def _ly_limits_run(vals):
    """
    LY-histogram (n_bins, min, max) for CALIBRATION-RUN mode.

    A calibration run is dominated by thallium, so the robust 5-95 percentile
    range can exclude the rare alpha peak, which sits low (~LY_Tl/5) and can
    extend to NEGATIVE LY. Extend the lower edge down to frame it. Tuning values
    are kept LOCAL here (not exposed as global parameters).
    """
    RATIO, NEG_MARGIN = 5.0, 0.30          # LY_Tl/LY_alpha, negative-tail margin
    n_bins, lo, hi = calcRobustLimitsAndBins(vals)
    if len(vals) == 0:
        return n_bins, lo, hi
    tl_guess = float(np.median(vals))      # Tl dominates -> median ~ LY_Tl
    if tl_guess <= 0:
        return n_bins, lo, hi
    alpha_lo = (tl_guess / RATIO) - NEG_MARGIN * tl_guess     # may be < 0
    new_lo   = alpha_lo - (hi - alpha_lo) * 0.05
    if new_lo < lo:
        width  = (hi - lo) / max(n_bins, 1)
        n_bins = min(500, max(n_bins, int(math.ceil((hi - new_lo) / max(width, 1e-9)))))
        lo     = new_lo
    return n_bins, float(lo), float(hi)


def AnalyzeLightYieldRun(h_ly, name_ext):
    """
    CALIBRATION-RUN variant of AnalyzeLightYield.

    Robust when the thallium peak dominates and a light-poor noise pile near/below
    zero can outnumber it: thallium is taken as the highest-LY of the few largest-
    area peaks (NOT simply the richest), and the alpha peak is searched BELOW it,
    guided by LY_alpha ~ LY_Tl / 5 but allowed to reach NEGATIVE LY. A failed or
    degenerate acceptance window disables the cut (full range) instead of starving
    the stabilization. All tuning values are LOCAL to this function.
    """
    # --- local tuning (kept out of the global parameter block) --------------
    RATIO         = 5.0     # expected LY_Tl / LY_alpha
    SEARCH_TOL    = 0.40    # alpha window upper edge at expected*(1 + this)
    NEG_MARGIN    = 0.30    # ...lower edge at expected - this*LY_Tl (-> negative LY)
    MIN_HEIGHT    = 0.02    # min alpha height (frac. of Tl) for a direct-scan alpha
    TSPEC_THR     = 0.01    # TSpectrum threshold (small: keep a tiny alpha)
    MAX_CAND      = 3       # Tl = rightmost among this many largest-area peaks
    MIN_AREA_FRAC = 0.05    # drop flukes below this frac. of the richest peak
    MIN_KEEP_FRAC = 0.05    # below this kept fraction the LY cut is disabled

    res = LYResult()
    print("=" * 50)
    print(f"--- Peak Analysis (RUN) {h_ly.GetTitle()} ---")

    xMin = h_ly.GetXaxis().GetXmin()
    xMax = h_ly.GetXaxis().GetXmax()
    hist_range = xMax - xMin
    fit_window = hist_range * 0.12

    spec   = ROOT.TSpectrum(10)
    nPeaks = spec.Search(h_ly, 2, "goff", TSPEC_THR)
    peakTl_X, peakAlpha_X = -999.0, -999.0

    # Collect every candidate (positive OR negative LY) with its local area.
    peaks_with_area = []
    int_window      = hist_range * 0.04
    if nPeaks > 0:
        xpeaks_buf = spec.GetPositionX()
        for i in range(nPeaks):
            x_val = xpeaks_buf[i]
            if not math.isfinite(x_val): continue
            bmin = h_ly.GetXaxis().FindBin(x_val - int_window)
            bmax = h_ly.GetXaxis().FindBin(x_val + int_window)
            peaks_with_area.append((x_val, h_ly.Integral(bmin, bmax)))

    if not peaks_with_area:
        peakTl_X = h_ly.GetXaxis().GetBinCenter(h_ly.GetMaximumBin())
        print("  [!] No peaks found. Fallback on global maximum (thallium only).")
    else:
        # Thallium = highest-LY of the major peaks (noise/alpha sit lower).
        peaks_with_area.sort(key=lambda p: p[1], reverse=True)
        max_area = peaks_with_area[0][1]
        majors   = [p for p in peaks_with_area[:MAX_CAND]
                    if p[1] >= MIN_AREA_FRAC * max_area] or [peaks_with_area[0]]
        majors.sort(key=lambda p: p[0])
        peakTl_X, area_tl = majors[-1]

        # Alpha searched below Tl, guided by Tl/RATIO, down to NEGATIVE LY.
        alpha_expected = peakTl_X / RATIO
        win_hi   = alpha_expected * (1.0 + SEARCH_TOL)
        win_lo   = alpha_expected - NEG_MARGIN * peakTl_X
        min_dist = hist_range * 0.08
        cand = [p for p in peaks_with_area
                if win_lo <= p[0] <= win_hi and (peakTl_X - p[0]) >= min_dist]
        if cand:
            cand.sort(key=lambda p: abs(p[0] - alpha_expected))   # closest to expected
            peakAlpha_X, area_alpha = cand[0]
            print(f"Thallium at: {peakTl_X:.4f} (Area: {area_tl:.0f}); "
                  f"alpha at: {peakAlpha_X:.4f} (Area: {area_alpha:.0f}) "
                  f"[expected ~{alpha_expected:.4f}, window [{win_lo:.4f},{win_hi:.4f}]].")
        else:
            b_lo = max(1, h_ly.GetXaxis().FindBin(win_lo))
            b_hi = min(h_ly.GetNbinsX(), h_ly.GetXaxis().FindBin(win_hi))
            best_bin, best_val = -1, 0.0
            for b in range(b_lo, b_hi + 1):
                c = h_ly.GetBinContent(b)
                if c > best_val:
                    best_val, best_bin = c, b
            tl_height = h_ly.GetBinContent(h_ly.GetXaxis().FindBin(peakTl_X))
            if best_bin > 0 and best_val >= MIN_HEIGHT * tl_height:
                peakAlpha_X = h_ly.GetXaxis().GetBinCenter(best_bin)
                print(f"Thallium at: {peakTl_X:.4f} (Area: {area_tl:.0f}); "
                      f"alpha by direct scan at: {peakAlpha_X:.4f} [expected ~{alpha_expected:.4f}].")
            else:
                print(f"Thallium at: {peakTl_X:.4f} (Area: {area_tl:.0f}); "
                      f"no alpha in [{win_lo:.4f},{win_hi:.4f}] (expected ~{alpha_expected:.4f}).")

    # --- Thallium Gaussian fit ----------------------------------------------
    fit_Tl_min = peakTl_X - fit_window
    fit_Tl_max = peakTl_X + fit_window
    if peakAlpha_X != -999.0:
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

    # --- Acceptance window (lower bound may be NEGATIVE) --------------------
    res.cut_max      = mean_Tl + LY_N_SIGMA * sigma_Tl
    standard_cut_min = mean_Tl - LY_N_SIGMA * sigma_Tl
    if peakAlpha_X != -999.0 and res.fit_alpha is not None and (sigma_alpha + sigma_Tl) > 0:
        valley_point = (mean_alpha * sigma_Tl + mean_Tl * sigma_alpha) / (sigma_alpha + sigma_Tl)
        valley_point = min(valley_point, mean_Tl - 1.5 * sigma_Tl)
        res.cut_min  = max(valley_point, standard_cut_min)
    else:
        res.cut_min = standard_cut_min

    # --- Sanity guard: disable a failed / degenerate window -----------------
    total_evts = h_ly.Integral()
    bad_fit    = (sigma_Tl <= 0.0 or not (xMin <= mean_Tl <= xMax)
                  or not (res.cut_max > res.cut_min))
    keep_frac  = 0.0
    if not bad_fit and total_evts > 0:
        b_lo = max(1, h_ly.GetXaxis().FindBin(res.cut_min))
        b_hi = min(h_ly.GetNbinsX(), h_ly.GetXaxis().FindBin(res.cut_max))
        keep_frac = h_ly.Integral(b_lo, b_hi) / total_evts
    if bad_fit or keep_frac < MIN_KEEP_FRAC:
        res.cut_min, res.cut_max = xMin, xMax
        res.df_value = -1.0
        print(f"  [!] LY fit unreliable for {name_ext} "
              f"(mu={mean_Tl:.4g}, sigma={sigma_Tl:.4g}, keep={keep_frac:.1%}); "
              f"LY cut DISABLED (full range).")
    else:
        print(f"  -> Calculated LY cuts for {name_ext}: [{res.cut_min:.4f}, {res.cut_max:.4f}]")
    return res


def CreateLYBox(res, title_prefix):
    """Build an NDC TPaveText summarising the LY thallium/alpha fit results."""
    pt = ROOT.TPaveText(0.60, 0.55, 0.93, 0.85, "NDC")
    pt.SetFillColor(ROOT.kWhite); pt.SetBorderSize(1)
    pt.SetTextAlign(12); pt.SetTextFont(42); pt.SetTextSize(0.028)
    pt.AddText("")   # top padding: keep text off the box border
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
    pt.AddText("")   # bottom padding: keep text off the box border
    return pt


# ===========================================================================
# FIT-RESULT BOX + PEAK FINDER
# ===========================================================================

def CreateFitBox(fit, header, header_color=ROOT.kBlack,
                 x1=0.13, y1=0.60, x2=0.50, y2=0.88,
                 note=None, note_color=ROOT.kBlue):
    """
    NDC TPaveText with a Gaussian fit's results: chi2/ndf, mu, sigma, FWHM and
    the percentage resolution (FWHM/mu*100). An optional *note* line is added in
    *note_color* (used to flag that the calibration-run line was applied).
    Returns the TPaveText (a placeholder text is shown when *fit* is None).
    """
    pt = ROOT.TPaveText(x1, y1, x2, y2, "NDC")
    pt.SetFillColor(ROOT.kWhite); pt.SetBorderSize(1)
    pt.SetTextAlign(12); pt.SetTextFont(42); pt.SetTextSize(0.035)

    t_h = pt.AddText(header)
    t_h.SetTextColor(header_color); t_h.SetTextFont(62)

    if note:
        t_n = pt.AddText(note); t_n.SetTextColor(note_color); t_n.SetTextFont(62)

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


def CreateLineBox(f1, header="LINEAR FIT", header_color=ROOT.kRed,
                  x1=0.14, y1=0.74, x2=0.62, y2=0.88):
    """NDC TPaveText with a pol1 fit's intercept (q0) and slope."""
    pt = ROOT.TPaveText(x1, y1, x2, y2, "NDC")
    pt.SetFillColor(ROOT.kWhite); pt.SetBorderSize(1)
    pt.SetTextAlign(12); pt.SetTextFont(42); pt.SetTextSize(0.035)
    t_h = pt.AddText(header); t_h.SetTextColor(header_color); t_h.SetTextFont(62)
    if f1 is None:
        pt.AddText("(fit non disponibile)")
        return pt
    pt.AddText(f"q_{{0}} = {f1.GetParameter(0):.2f} #pm {f1.GetParError(0):.2f}")
    pt.AddText(f"slope = {f1.GetParameter(1):.2f} #pm {f1.GetParError(1):.2f}")
    return pt


def padded_view(x_arr, y_arr, x_pad_frac=0.45, y_pad_frac=0.55):
    """
    Return (x0, x1, y0, y1) padded display limits for a scatter, so the cloud
    is not drawn edge-to-edge: white margins reveal the linear trend. Returns
    None when there are no points.
    """
    if len(x_arr) == 0:
        return None
    xmin, xmax = float(np.min(x_arr)), float(np.max(x_arr))
    ymin, ymax = float(np.min(y_arr)), float(np.max(y_arr))
    xs = max(xmax - xmin, 1e-9)
    ys = max(ymax - ymin, 1e-9)
    return (xmin - x_pad_frac * xs, xmax + x_pad_frac * xs,
            ymin - y_pad_frac * ys, ymax + y_pad_frac * ys)


def build_combined_peak(vals, name, title, center=None, params_override=None):
    """
    Build a combined histogram from *vals*, Gaussian-fit it and return
    (h, h_truncated, fit). The truncated clone holds only the +/-5 sigma fit
    region (drawn filled). *center* seeds the robust binning (defaults to
    TARGET_ENERGY, suitable for already-calibrated samples; pass the raw median
    for an un-calibrated sample). *params_override* (a BinningParams) forces a
    given binning/fit range, so several histograms can SHARE the same axis (used
    for the before/after comparison). Returns (None, None, None) for an empty
    sample.
    """
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None, None

    if params_override is not None:
        params = params_override
    else:
        if center is None:
            center = TARGET_ENERGY
        params = GetCenteredBinning(vals.tolist(), center)
    h = ROOT.TH1F(name, title, params.bins, params.vis_min, params.vis_max)
    h.SetDirectory(0)
    h.FillN(vals.size, vals.astype(np.double), np.ones(vals.size, np.double))

    fmin = params.median - 5.0 * params.robust_sigma
    fmax = params.median + 5.0 * params.robust_sigma
    fit = ROOT.TF1(f"fit_{name}", "gaus", fmin, fmax)
    fit.SetParameters(h.GetMaximum(), params.median, params.robust_sigma)
    h.Fit(fit, "Q0 R L")

    h_final = h.Clone(f"{name}_final")
    h_final.Reset()
    m  = (vals >= fmin) & (vals <= fmax)
    nm = int(m.sum())
    if nm > 0:
        h_final.FillN(nm, vals[m].astype(np.double), np.ones(nm, np.double))
    return h, h_final, fit


def FindThalliumPeak(h_heat_orig, search_min=None, search_max=None):
    """
    Locate the thallium peak by scanning a sliding integration window and
    returning the (x, height) of the tallest bin inside the most populated
    window. Operates on a ROOT histogram.

    When *search_min* / *search_max* are given (amplitude units), the scan is
    restricted to that region: the window START bin is clamped to it. This is
    used to pass the peak position found in the richest partition as a hint to
    the others. If the restriction is empty/invalid the full range is scanned.
    """
    n_bins_total = h_heat_orig.GetNbinsX()
    window_width = max(3, int(n_bins_total * 0.08))

    b_lo = 1
    b_hi = n_bins_total - window_width + 1
    if search_min is not None:
        b_lo = max(b_lo, h_heat_orig.GetXaxis().FindBin(search_min))
    if search_max is not None:
        b_hi = min(b_hi, h_heat_orig.GetXaxis().FindBin(search_max))
    if b_hi < b_lo:                       # invalid window -> fall back to full range
        b_lo, b_hi = 1, n_bins_total - window_width + 1

    max_integral, best_peak_bin = -1, b_lo
    for b in range(b_lo, b_hi + 1):
        current_integral = h_heat_orig.Integral(b, b + window_width)
        if current_integral > max_integral:
            max_integral    = current_integral
            max_val_in_win  = -1
            for w in range(b, b + window_width + 1):
                val = h_heat_orig.GetBinContent(w)
                if val > max_val_in_win:
                    max_val_in_win = val; best_peak_bin = w

    return h_heat_orig.GetXaxis().GetBinCenter(best_peak_bin), h_heat_orig.GetBinContent(best_peak_bin)


def estimate_thallium_peak(amps, name):
    """
    Robust (center, sigma) of the thallium peak in an amplitude sample, via
    FindThalliumPeak followed by a narrow Gaussian fit. Used on the COMBINED
    (all-partition) spectrum to produce the search hint shared by every
    partition. Returns (0.0, 0.0) when the sample is empty.
    """
    amps = np.asarray(amps, dtype=np.float64)
    amps = amps[np.isfinite(amps)]
    if amps.size < 1:
        return 0.0, 0.0

    bins, lo, hi = calcRobustLimitsAndBins(amps.tolist())
    h = ROOT.TH1F(f"h_peakest_{name}", "peak estimate;Amplitude;Counts", bins, lo, hi)
    h.SetDirectory(0)
    h.FillN(amps.size, amps.astype(np.double), np.ones(amps.size, np.double))

    peak_x, peak_y = FindThalliumPeak(h)
    fit_window = (hi - lo) * 0.1
    f = ROOT.TF1(f"f_peakest_{name}", "gaus", peak_x - fit_window, peak_x + fit_window)
    f.SetParameters(peak_y, peak_x, h.GetRMS() * 0.1)
    h.Fit(f, "Q0 R L", "", peak_x - fit_window, peak_x + fit_window)

    mean, sigma = f.GetParameter(1), f.GetParameter(2)
    if sigma > fit_window or sigma <= 0:
        mean, sigma = peak_x, h.GetRMS() * 0.1
    return mean, sigma


# ===========================================================================
# OPTIMIZED PEAK FITS  (ported from AlphaStabilization.py)
# ===========================================================================
# Single-peak fit machinery shared by the BEFORE/AFTER comparison images: an
# optimized-binning peak finder, a robust sigma estimate to size a SHARED axis,
# and the final Gaussian-over-linear-background Poisson-likelihood fit. These give
# the thallium before/after peaks (combined and per partition) the same treatment
# as the alpha peaks: all events in the stabilization window (not only the clean
# subset), a linear background, and a shared range/binning.

def fit_peak_optimized(amps, win_lo, win_hi, tag, center_hint=None):
    """
    Single-peak Gaussian fit with binning optimized to the peak width (coarse
    histogram -> dominant peak via FindThalliumPeak -> preliminary Gaussian ->
    rebuild at bin width ~ sigma/4 -> final Gaussian). *center_hint*, when given,
    restricts the peak search to +/-15% of the range around it. Returns (hist, fit)
    or (None, None) for a too-small sample.
    """
    amps = np.asarray(amps, np.float64)
    amps = amps[np.isfinite(amps) & (amps >= win_lo) & (amps <= win_hi)]
    if amps.size < 5:
        return None, None

    def build(nb):
        h = ROOT.TH1F(f"h_pk_{tag}_{nb}", "", nb, win_lo, win_hi)
        h.SetDirectory(0)
        h.FillN(amps.size, amps.astype(np.double), np.ones(amps.size, np.double))
        return h

    h0 = build(80)
    s_min = center_hint - 0.15 * (win_hi - win_lo) if center_hint is not None else None
    s_max = center_hint + 0.15 * (win_hi - win_lo) if center_hint is not None else None
    peak_x, _ = FindThalliumPeak(h0, s_min, s_max)

    sg0 = (win_hi - win_lo) * 0.02
    g = ROOT.TF1(f"gp_{tag}", "gaus", peak_x - 3 * sg0, peak_x + 3 * sg0)
    g.SetParameters(h0.GetBinContent(h0.GetXaxis().FindBin(peak_x)), peak_x, sg0)
    h0.Fit(g, "Q0 R")
    A1, m1, s1 = g.GetParameter(0), g.GetParameter(1), abs(g.GetParameter(2))
    if s1 <= 0:
        s1 = (win_hi - win_lo) * 0.02
    sig_ref = max(s1, (win_hi - win_lo) * 0.004)
    nb = int(np.clip(round((win_hi - win_lo) / (sig_ref / 4.0)), 40, 300))
    h  = build(nb)
    ff = ROOT.TF1(tag, "gaus", m1 - 4 * s1, m1 + 4 * s1)
    ff.SetParameters(A1, m1, s1)
    h.Fit(ff, "Q0 R")
    return h, ff


def estimate_peak_sigma(energies, center, tag):
    """
    Robust sigma of the peak in *energies* (already rescaled/stabilized so the peak
    sits near *center*), via fit_peak_optimized. Returns the sigma or None. Used to
    size the SHARED before/after axis.
    """
    e = np.asarray(energies, np.float64)
    e = e[np.isfinite(e)]
    if e.size < 5:
        return None
    _, lo, hi = calcRobustLimitsAndBins(e.tolist())
    _, f = fit_peak_optimized(e, lo, hi, f"{tag}_sig", center_hint=center)
    if f is None:
        return None
    s = abs(f.GetParameter(2))
    return s if s > 0 else None


def fit_thallium_peak(energies, center, lo, hi, nb, sig_seed, tag):
    """
    FINAL thallium-peak fit for the before/after comparison.

    Histograms *energies* on the GIVEN shared axis (lo, hi, nb bins) -- so before
    and after use the SAME range and binning -- and fits the Tl peak with a Gaussian
    over a LINEAR background (gaus(0)+pol1(3)) using a Poisson maximum-likelihood
    fit ("L", correct for low statistics, natively handling empty bins). Seeded at
    *center* with width *sig_seed*. The Gaussian parameters stay at indices 0,1,2,
    so CreateFitBox reads mu/sigma directly. Returns (hist, fit) or (None, None).
    """
    e = np.asarray(energies, np.float64)
    e = e[np.isfinite(e)]
    if e.size < 5:
        return None, None
    h = ROOT.TH1F(tag, "", nb, lo, hi); h.SetDirectory(0)
    m  = (e >= lo) & (e <= hi)
    nm = int(m.sum())
    if nm > 0:
        h.FillN(nm, e[m].astype(np.double), np.ones(nm, np.double))
    seed = max(sig_seed, 1.0)
    ff = ROOT.TF1(f"fit_{tag}", "gaus(0)+pol1(3)", lo, hi)
    ff.SetParameters(max(h.GetMaximum(), 1.0), center, seed, 0.0, 0.0)
    # Keep the Gaussian width near the (reliable) seed so it fits the PEAK and lets
    # pol1 take the continuum, instead of ballooning into a broad blob that swallows
    # the background -- the failure mode on low-statistics partitions.
    ff.SetParLimits(2, seed * 0.3, seed * 2.0)
    h.Fit(ff, "Q0 R L")
    return h, ff


def shared_before_after_fits(before_e, after_e, center, tag, sig_hint=None):
    """
    Fit a BEFORE and an AFTER energy sample of the thallium peak on a SHARED axis
    (equal range + binning) with gaus(0)+pol1(3) Poisson-likelihood fits. The axis
    is window = center +/- 3.5*sigma_ref with bin width ~ sigma_ref/6. *after_e*
    may be None.

    *sig_hint*: a RELIABLE peak width (energy). When given (e.g. the partition's
    clean-peak sigma), it sets BOTH the window and the seed, giving a SMALL window
    tightly matched to the peak -- avoiding the case where a noisy low-statistics
    width estimate inflates the window and lets the background swallow the peak.
    When None (combined spectrum), the window uses the BROADER of the two measured
    peaks and the seed the narrower one.
    Returns (h_before, fit_before, h_after, fit_after) (fit_* may be None).
    """
    if sig_hint is not None and sig_hint > 0:
        sig_win = sig_seed = float(sig_hint)
    else:
        sig_b = estimate_peak_sigma(before_e, center, f"{tag}_b") if before_e is not None else None
        sig_a = estimate_peak_sigma(after_e,  center, f"{tag}_a") if after_e  is not None else None
        valid   = [s for s in (sig_b, sig_a) if s]
        sig_win  = max(valid) if valid else 15.0    # window: the BROADER peak
        sig_seed = min(valid) if valid else sig_win # seed: the narrower (reliable) one
    W  = 3.5 * sig_win
    lo, hi = center - W, center + W
    nb = int(np.clip(round((hi - lo) / max(sig_win / 6.0, 1e-9)), 12, 200))
    h_b = f_b = h_a = f_a = None
    if before_e is not None:
        h_b, f_b = fit_thallium_peak(before_e, center, lo, hi, nb, sig_seed, f"{tag}_before")
    if after_e is not None:
        h_a, f_a = fit_thallium_peak(after_e, center, lo, hi, nb, sig_seed, f"{tag}_after")
    return h_b, f_b, h_a, f_a


# ===========================================================================
# BASELINE PARTITIONING
# ===========================================================================

def FindBaselinePartitions(baseline_vals, ch_id, enabled=True):
    """
    Split the (correlation-cut) dataset into baseline-based partitions,
    separating ONLY blocks that are clearly DETACHED in the baseline histogram.

    When *enabled* is False the histogram is still built (for the debug plot)
    but a SINGLE partition spanning the whole data range is returned.

    The procedure is:

      - Build the baseline histogram over a robust range (1st/99th percentile,
        padded) so that a few far outliers do not flatten the dense clusters.
      - Treat the histogram as filled "blocks" separated by GAPS, judged by
        RELATIVE height: a bin belongs to a gap when its height is below
        PART_GAP_HEIGHT_FRAC of the tallest block. A boundary is placed only
        across a gap of at least PART_MIN_GAP_BINS bins, at the gap centre.
        Nearby peaks whose valley stays above that fraction therefore remain in
        the SAME partition (no boundary in a shallow, populated valley).
      - Absorb blocks holding less than PART_MIN_BLOCK_FRAC of the most
        populated block into the nearest (in baseline) block: small isolated
        peaks join the closest partition instead of forming their own.
      - The outer bounds extend to the global data extremes so that EVERY event
        (outliers included) falls into exactly one partition.

    Parameters
    ----------
    baseline_vals : 1-D array of baseline values surviving the correlation cut.
    ch_id         : channel id (used only to name the histogram).

    Returns
    -------
    (intervals, markers, h_base) where
      intervals : list of (low, high) baseline ranges, sorted by baseline;
                  together they tile [data_min, data_max].
      markers   : one representative baseline per partition (its tallest bin),
                  used only for the debug plot.
      h_base    : the ROOT.TH1F used for the search (None if no data), kept for
                  the debug drawing.
    """
    arr = np.asarray(baseline_vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return [(-np.inf, np.inf)], [], None

    data_min = float(arr.min())
    data_max = float(arr.max())

    # Robust visible range (a few far outliers must not dominate the binning).
    lo = float(np.percentile(arr, 1.0))
    hi = float(np.percentile(arr, 99.0))
    if hi <= lo:
        lo, hi = data_min, data_max
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    lo  -= 0.05 * span
    hi  += 0.05 * span

    h_base = ROOT.TH1F(f"h_base_part_{ch_id}",
                       f"Ch {ch_id}: Baseline Distribution after Correlation Cut;Baseline;Counts",
                       PART_N_BINS, lo, hi)
    in_range = arr[(arr >= lo) & (arr <= hi)]
    if len(in_range) > 0:
        h_base.FillN(len(in_range), in_range.astype(np.double),
                     np.ones(len(in_range), np.double))

    if not enabled:
        # Partitioning disabled: one partition over the whole data range.
        markers = ([h_base.GetXaxis().GetBinCenter(h_base.GetMaximumBin())]
                   if h_base.GetEntries() > 0 else [])
        return [(data_min, data_max)], markers, h_base

    nb = h_base.GetNbinsX()
    c  = np.array([h_base.GetBinContent(i) for i in range(1, nb + 1)], dtype=np.float64)  # 0-based

    # A bin belongs to a block when its height exceeds a fraction of the TALLEST
    # block (relative height), so gaps are deep valleys rather than strictly
    # empty bins: a few stray counts in a deep valley still mark a separation,
    # while a shallow valley with substantial counts keeps the blocks joined.
    cmax       = float(c.max())
    gap_thresh = PART_GAP_HEIGHT_FRAC * cmax
    high       = c > gap_thresh

    if not high.any():
        # No populated bin in range: a single partition spanning all the data.
        return [(data_min, data_max)], [], h_base

    first = int(np.argmax(high))                 # first block bin (0-based)
    last  = nb - 1 - int(np.argmax(high[::-1]))  # last  block bin (0-based)

    # --- Find internal gaps of consecutive low bins (>= PART_MIN_GAP_BINS) ---
    gaps = []           # list of (gap_start, gap_end) bin indices (0-based, inclusive)
    i = first + 1
    while i <= last:
        if not high[i]:
            j = i
            while j <= last and not high[j]:
                j += 1
            # low run is [i, j-1]; internal because bin j (<= last) is high
            if (j - i) >= PART_MIN_GAP_BINS:
                gaps.append((i, j - 1))
            i = j
        else:
            i += 1

    # --- Build blocks (bin ranges) from the qualifying gaps -----------------
    block_starts = [first] + [g[1] + 1 for g in gaps]
    block_ends   = [g[0] - 1 for g in gaps] + [last]
    blocks       = [[s, e] for s, e in zip(block_starts, block_ends)]

    # --- Absorb small isolated blocks into the nearest one ------------------
    # A block is "major" when it holds at least PART_MIN_BLOCK_FRAC of the most
    # populated block's counts. Minor blocks (small isolated peaks) are merged
    # into the neighbour separated by the SMALLER gap, i.e. the nearest in
    # baseline. The threshold is fixed from the initial blocks: merging only
    # grows integrals, so the loop terminates and the dominant block stays.
    def block_integral(b):
        return float(c[b[0]:b[1] + 1].sum())

    ref_max   = max((block_integral(b) for b in blocks), default=0.0)
    min_block = PART_MIN_BLOCK_FRAC * ref_max
    while len(blocks) > 1:
        integrals = [block_integral(b) for b in blocks]
        m = int(np.argmin(integrals))
        if integrals[m] >= min_block:
            break
        # nearest neighbour = the one separated by the smaller empty gap
        if m == 0:
            nbr = 1
        elif m == len(blocks) - 1:
            nbr = m - 1
        else:
            gap_left  = blocks[m][0]     - blocks[m - 1][1]
            gap_right = blocks[m + 1][0] - blocks[m][1]
            nbr = m - 1 if gap_left <= gap_right else m + 1
        a, b2 = min(m, nbr), max(m, nbr)
        blocks[a] = [blocks[a][0], blocks[b2][1]]   # span both (the gap is absorbed)
        del blocks[b2]

    # --- Baseline boundaries at the centre of the gaps between blocks -------
    boundaries = [data_min]
    for k in range(len(blocks) - 1):
        mid_bin = (blocks[k][1] + blocks[k + 1][0]) // 2   # inside the empty gap
        boundaries.append(h_base.GetXaxis().GetBinCenter(mid_bin + 1))
    boundaries.append(data_max)
    intervals = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    # --- One marker per partition (its tallest bin) -------------------------
    markers = []
    for s, e in blocks:
        local_max = s + int(np.argmax(c[s:e + 1]))
        markers.append(h_base.GetXaxis().GetBinCenter(local_max + 1))

    return intervals, markers, h_base


# ===========================================================================
# PER-PARTITION STABILIZATION
# ===========================================================================

class StabResult:
    """
    All per-baseline-partition stabilization products.

    Holds the linear-fit coefficients (slope, q_0) used to stabilize the
    partition, the calibrated stabilized amplitudes (a_stab_cal, merged across
    partitions for the combined spectrum) and every ROOT object needed to draw
    the partition's two debug rows.
    """
    def __init__(self, idx, blo, bhi):
        self.idx = idx
        self.blo = blo
        self.bhi = bhi
        self.sufficient      = False
        self.n_clean         = 0
        self.n_events        = 0       # events entering the stabilization
        self.slope           = 0.0
        self.q_0             = 0.0
        self.mean_amp_clean  = 0.0
        self.peak_x          = 0.0     # preliminary thallium-peak amplitude
        self.peak_sigma      = 0.0     # preliminary thallium-peak sigma
        self.a_stab_cal      = np.array([], dtype=np.float64)  # calibrated AFTER stabilization
        self.clean_amps      = np.array([], dtype=np.float64)  # clean thallium amplitudes BEFORE stabilization
        self.clean_bases     = np.array([], dtype=np.float64)  # baselines of the clean thallium events
        # ALL stab-window events (not only the clean subset) + matching baselines:
        # used for the BEFORE/AFTER peak fits (gaus+pol1) of the debug images.
        self.all_amps        = np.array([], dtype=np.float64)
        self.all_bases       = np.array([], dtype=np.float64)
        # Per-partition BEFORE (rescaled to energy) / AFTER (stabilized) peak fits
        # over ALL window events, gaus+pol1 on a shared axis.
        self.h_before        = None
        self.fit_before      = None
        self.h_after         = None
        self.fit_after       = None
        self.gaussian_counts = 0.0
        # Padded display ranges (x0, x1, y0, y1) for the amplitude-vs-baseline scatters
        self.view_clean      = None
        self.view_stab       = None
        # ROOT objects (drawing)
        self.h_heat_orig          = None   # pre-cleaning spectrum + preliminary fit
        self.fit_prelim           = None
        self.h_heat_clean         = None   # clean peak BEFORE stabilization
        self.fit_clean            = None
        self.g_heat_vs_base_clean = None   # amplitude vs baseline + linear fit
        self.f1                   = None   # FITTED line (red, always computed)
        self.f1_ext               = None   # EXTERNAL line (blue, when applied)
        self.h_heat_stab          = None   # peak AFTER stabilization (amplitude)
        self.fit_stab             = None
        self.g_heat_vs_base_stab  = None   # amplitude vs baseline after stab.
        self.h_heat_cal           = None   # calibrated & stabilized peak (energy)
        self.h_heat_cal_final     = None
        self.fit_cal              = None


def process_partition(amps_for_stab, bases_for_stab, ch_id, idx, blo, bhi,
                      manual_cuts=None, apply_heat_manual=False, peak_hint=None,
                      ext_line=None):
    """
    Run the full stabilization on ONE baseline partition.

    Same pipeline as before, but confined to the events of a single baseline
    interval: thallium-peak detection (FindThalliumPeak), outlier cleaning,
    Gaussian clean-peak fit, ROBUST pol1 fit of amplitude vs baseline and the
    per-event linear stabilization to TARGET_ENERGY. All products are stored in
    the returned StabResult.

    Parameters
    ----------
    amps_for_stab  : amplitudes of this partition (corr + LY + stab-window cuts).
    bases_for_stab : matching baselines.
    ch_id, idx     : channel id and partition index (object naming / titles).
    blo, bhi       : baseline interval bounds (titles only).
    manual_cuts    : optional manual-cut dict (heat window honoured only when
                     apply_heat_manual is True, i.e. a single partition).
    peak_hint      : optional (center, sigma) of the thallium peak found on the
                     COMBINED (all-partition) spectrum. When given, the peak
                     search of THIS partition is restricted to center +/-
                     max(PEAK_HINT_NSIGMA*sigma, PEAK_HINT_MIN_FRAC*heat_range),
                     avoiding noise locks in low-statistics partitions.

    Returns
    -------
    StabResult (res.sufficient is False when the partition has no events).
    """
    res = StabResult(idx, blo, bhi)
    res.n_events = len(amps_for_stab)
    tag = f"{ch_id}_p{idx}"

 # Carry the external line even for an empty partition, so it is still applied
    # to the output and is not overwritten by the nearest-partition fallback.
    if ext_line is not None:
        res.slope, res.q_0 = float(ext_line[0]), float(ext_line[1])

    if len(amps_for_stab) < 1:
        return res
    res.sufficient = True

    # Keep ALL window events (not only the clean subset): the BEFORE/AFTER peak
    # fits of the debug images use these, with a linear background.
    res.all_amps  = np.asarray(amps_for_stab, np.float64)
    res.all_bases = np.asarray(bases_for_stab, np.float64)

    # --- preliminary fit (locate the thallium peak in this partition) -------
    bins_heat, heat_min, heat_max = calcRobustLimitsAndBins(amps_for_stab.tolist())
    n_stab = len(amps_for_stab)

    res.h_heat_orig = ROOT.TH1F(
        f"h_heat_orig_{tag}",
        f"Ch {ch_id} P{idx} [{blo:.3f},{bhi:.3f}]: Spectrum after LY cut - Pre-cleaning;Amplitude;Counts",
        bins_heat, heat_min, heat_max)
    res.h_heat_orig.FillN(n_stab, amps_for_stab.astype(np.double),
                          np.ones(n_stab, np.double))

    # Restrict the peak search around the reference (richest) partition's peak.
    search_min = search_max = None
    if peak_hint is not None:
        hint_center, hint_sigma = peak_hint
        half_width = max(PEAK_HINT_NSIGMA * abs(hint_sigma),
                         PEAK_HINT_MIN_FRAC * (heat_max - heat_min))
        search_min = hint_center - half_width
        search_max = hint_center + half_width

    peak_x, peak_y = FindThalliumPeak(res.h_heat_orig, search_min, search_max)
    fit_window = (heat_max - heat_min) * 0.1
    res.fit_prelim = ROOT.TF1(f"fit_prelim_{tag}", "gaus",
                              peak_x - fit_window, peak_x + fit_window)
    res.fit_prelim.SetParameters(peak_y, peak_x, res.h_heat_orig.GetRMS() * 0.1)
    res.h_heat_orig.Fit(res.fit_prelim, "Q0 R L", "", peak_x - fit_window, peak_x + fit_window)

    mean_heat_prelim  = res.fit_prelim.GetParameter(1)
    sigma_heat_prelim = res.fit_prelim.GetParameter(2)
    if sigma_heat_prelim > fit_window or sigma_heat_prelim <= 0:
        mean_heat_prelim  = peak_x
        sigma_heat_prelim = res.h_heat_orig.GetRMS() * 0.1

    res.peak_x     = mean_heat_prelim
    res.peak_sigma = sigma_heat_prelim

    heat_cut_min = mean_heat_prelim - HEAT_CLEAN_NSIGMA * sigma_heat_prelim
    heat_cut_max = mean_heat_prelim + HEAT_CLEAN_NSIGMA * sigma_heat_prelim
    if manual_cuts and apply_heat_manual:
        if manual_cuts.get('heat_cut_min') is not None: heat_cut_min = manual_cuts['heat_cut_min']
        if manual_cuts.get('heat_cut_max') is not None: heat_cut_max = manual_cuts['heat_cut_max']

    # --- outlier cleaning ---------------------------------------------------
    clean_mask     = (amps_for_stab >= heat_cut_min) & (amps_for_stab <= heat_cut_max)
    clean_amps_np  = amps_for_stab[clean_mask]
    clean_bases_np = bases_for_stab[clean_mask]
    res.n_clean    = len(clean_amps_np)

    params_clean = GetCenteredBinning(clean_amps_np.tolist(),
                                      heat_min + (heat_max - heat_min) / 2.0)
    res.h_heat_clean = ROOT.TH1F(
        f"h_heat_clean_{tag}",
        f"Ch {ch_id} P{idx}: Thallium Peak BEFORE stabilization;Amplitude;Counts",
        params_clean.bins, params_clean.vis_min, params_clean.vis_max)
    res.h_heat_stab = ROOT.TH1F(
        f"h_heat_stab_{tag}",
        f"Ch {ch_id} P{idx}: Thallium Peak AFTER stabilization;Amplitude;Counts",
        params_clean.bins, params_clean.vis_min, params_clean.vis_max)

    if res.n_clean > 0:
        res.h_heat_clean.FillN(res.n_clean, clean_amps_np.astype(np.double),
                               np.ones(res.n_clean, np.double))

    res.g_heat_vs_base_clean = ROOT.TGraph(res.n_clean,
                                           clean_bases_np.astype(np.double),
                                           clean_amps_np.astype(np.double))
    res.g_heat_vs_base_clean.SetName(f"g_heat_vs_base_clean_{tag}")
    res.g_heat_vs_base_clean.SetTitle(
        f"Ch {ch_id} P{idx}: Amplitude vs Baseline + linear fit (before stab.);Baseline;Amplitude")

    res.fit_clean = ROOT.TF1(f"fit_clean_{tag}", "gaus",
                             params_clean.median - 5.0 * params_clean.robust_sigma,
                             params_clean.median + 5.0 * params_clean.robust_sigma)
    res.fit_clean.SetParameters(res.h_heat_clean.GetMaximum(),
                                params_clean.median, params_clean.robust_sigma)
    res.h_heat_clean.Fit(res.fit_clean, "Q0 R L")
    res.mean_amp_clean = res.fit_clean.GetParameter(1)

    # Padded display range for the BEFORE-stabilization scatter (set later).
    res.view_clean = padded_view(clean_bases_np, clean_amps_np)

    # Keep the clean (un-stabilized) amplitudes: they are merged across all
    # partitions and globally calibrated to build the BEFORE-stabilization peak.
    # The matching baselines let a fallback partition recompute a_stab_cal once it
    # inherits a neighbour's line (so it still feeds the combined AFTER spectrum).
    res.clean_amps  = clean_amps_np
    res.clean_bases = clean_bases_np

    # --- robust pol1 fit: ALWAYS computed (shown in red for comparison) -----
    fit_slope, fit_q0 = 0.0, res.mean_amp_clean
    if res.g_heat_vs_base_clean.GetN() > 3:
        res.f1 = ROOT.TF1(f"f1_{tag}", "pol1")
        res.g_heat_vs_base_clean.Fit(res.f1, "Q0 rob=0.90")
        fit_slope = res.f1.GetParameter(1)
        fit_q0    = res.f1.GetParameter(0)

    # --- line actually APPLIED ---------------------------------------------
    # External SLOPE-ONLY mode: reuse only the external slope (baseline-drift
    # correction) and RE-ANCHOR the intercept on THIS partition's own thallium
    # peak, so the absolute energy scale is set by the current data and the peak
    # lands on TARGET_ENERGY regardless of the calibration run's baseline.
    #   a_anchor = Gaussian-fit centre of the clean peak  (best peak estimator:
    #              ML-optimal for a Gaussian, immune to the residual tails/
    #              background that bias a raw mean or median);
    #   b_anchor = MEAN baseline of the same clean-peak events  (consistent with
    #              the linear model E[a] = q0 + slope*E[b], so mean pairs with
    #              mean; the median would bias q0 by slope*(median_b - mean_b)
    #              when the partition's baseline cluster is skewed).
    #   q0_loc   = a_anchor - slope_ext * b_anchor.
    # Partitions too poor to self-anchor keep slope == q_0 == 0 and inherit the
    # nearest anchored partition's line via the fallback in run_stabilization().
    if ext_line is not None:
        can_anchor = (clean_bases_np.size > 0
                      and np.isfinite(res.mean_amp_clean) and res.mean_amp_clean > 0)
        if can_anchor:
            slope_ext = float(ext_line[0])
            a_anchor  = res.mean_amp_clean
            b_anchor  = float(np.mean(clean_bases_np))
            res.slope = slope_ext
            res.q_0   = a_anchor - slope_ext * b_anchor
            res.f1_ext = ROOT.TF1(f"f1ext_{tag}", "pol1")
            res.f1_ext.SetParameters(res.q_0, res.slope)
        # else: leave slope == q_0 == 0 -> fallback inherits a neighbour's line.
    else:
        res.slope, res.q_0 = fit_slope, fit_q0

    # --- apply stabilization ------------------------------------------------
    expected_old = res.q_0 + res.slope * clean_bases_np
    with np.errstate(divide='ignore', invalid='ignore'):
        a_stab     = res.mean_amp_clean * clean_amps_np / expected_old
        a_stab_cal = TARGET_ENERGY      * clean_amps_np / expected_old

    fin_mask   = np.isfinite(a_stab) & np.isfinite(a_stab_cal)
    a_stab     = a_stab[fin_mask]
    a_stab_cal = a_stab_cal[fin_mask]
    bases_stab = clean_bases_np[fin_mask]
    res.a_stab_cal = a_stab_cal
    n_stab_fin = len(a_stab)

    res.view_stab = padded_view(bases_stab, a_stab)
    res.g_heat_vs_base_stab = ROOT.TGraph(n_stab_fin,
                                          bases_stab.astype(np.double),
                                          a_stab.astype(np.double))
    res.g_heat_vs_base_stab.SetName(f"g_heat_vs_base_stab_{tag}")
    res.g_heat_vs_base_stab.SetTitle(
        f"Ch {ch_id} P{idx}: Amplitude vs Baseline (after stab.);Baseline;Amplitude")

    if n_stab_fin > 0:
        res.h_heat_stab.FillN(n_stab_fin, a_stab.astype(np.double),
                              np.ones(n_stab_fin, np.double))

    res.fit_stab = ROOT.TF1(f"fit_stab_{tag}", "gaus",
                            res.mean_amp_clean - 5.0 * params_clean.robust_sigma,
                            res.mean_amp_clean + 5.0 * params_clean.robust_sigma)
    res.fit_stab.SetParameters(res.h_heat_stab.GetMaximum(),
                               res.mean_amp_clean, params_clean.robust_sigma)
    res.h_heat_stab.Fit(res.fit_stab, "Q0 R L")

    # --- calibrated histogram -----------------------------------------------
    params_cal = GetCenteredBinning(a_stab_cal.tolist(), TARGET_ENERGY)
    res.h_heat_cal = ROOT.TH1F(
        f"h_heat_cal_{tag}",
        f"Ch {ch_id} P{idx}: Calibrated & Stabilized Thallium Peak;Energy (keV);Counts",
        params_cal.bins, params_cal.vis_min, params_cal.vis_max)
    res.h_heat_cal.SetDirectory(0)
    if n_stab_fin > 0:
        res.h_heat_cal.FillN(n_stab_fin, a_stab_cal.astype(np.double),
                             np.ones(n_stab_fin, np.double))

    fit_min_cal = params_cal.median - 5.0 * params_cal.robust_sigma
    fit_max_cal = params_cal.median + 5.0 * params_cal.robust_sigma
    res.fit_cal = ROOT.TF1(f"fit_cal_{tag}", "gaus", fit_min_cal, fit_max_cal)
    res.fit_cal.SetParameters(res.h_heat_cal.GetMaximum(),
                              params_cal.median, params_cal.robust_sigma)
    res.h_heat_cal.Fit(res.fit_cal, "Q0 R L")

    res.h_heat_cal_final = res.h_heat_cal.Clone(f"h_heat_cal_final_{tag}")
    res.h_heat_cal_final.Reset()
    mask_cal_win = (a_stab_cal >= fit_min_cal) & (a_stab_cal <= fit_max_cal)
    n_cal_win    = int(mask_cal_win.sum())
    if n_cal_win > 0:
        res.h_heat_cal_final.FillN(n_cal_win, a_stab_cal[mask_cal_win].astype(np.double),
                                   np.ones(n_cal_win, np.double))

    if res.g_heat_vs_base_clean.GetN() > 3 and res.h_heat_cal.GetEntries() > 0:
        res.gaussian_counts = (res.fit_cal.GetParameter(0) * res.fit_cal.GetParameter(2)
                               * math.sqrt(2 * math.pi)
                               / res.h_heat_cal.GetXaxis().GetBinWidth(1))
    else:
        res.gaussian_counts = float(res.n_clean)

    return res


def _parse_channel_any(filename):
    """
    Channel parsed from a file name regardless of the run mode that produced it:
    the number after "ch" (mergedrun names) if present, otherwise the FIRST
    number (run names). External stabilized files may come from either mode, so
    their channel must NOT depend on the current ANALYSIS_MODE.
    """
    base = os.path.basename(filename)
    m = re.search(r"ch(\d+)", base)
    if m:
        return m.group(1)
    m = re.match(r"(\d+)_", base)
    return m.group(1) if m else None


def resolve_external_file(folder, ch_id):
    """
    Find, inside *folder*, the stabilized ROOT file whose channel matches
    *ch_id*. The channel is parsed mode-independently (after "ch" or first
    number), since the external files may have been produced in either mode and
    the other numbers in the name may differ. Returns the full path, or None.
    """
    if not folder or not os.path.isdir(folder):
        print(f"  [!] External-line folder not found: '{folder}'", file=sys.stderr)
        return None
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".root") and "stabilized" in fn and _parse_channel_any(fn) == str(ch_id):
            return os.path.join(folder, fn)
    return None


def load_external_stab_line(folder, ch_id):
    """
    Pick ONE stabilization line for channel *ch_id* from the external file found
    in *folder*: the line of the baseline interval with the MOST events
    (n_events). If only one line is stored, that one is used.

    Returns (slope, q0, baseline_low, baseline_high, n_events, path) or None
    when nothing usable is found (the caller then falls back to fitting).
    """
    path = resolve_external_file(folder, ch_id)
    if path is None:
        print(f"  [!] No external stabilized file for ch {ch_id} in "
              f"'{folder}'", file=sys.stderr)
        return None

    f = ROOT.TFile.Open(path, "READ")
    if not f or f.IsZombie():
        print(f"  [!] Cannot open external stab-line file: '{path}'", file=sys.stderr)
        return None

    t = f.Get(STAB_LINES_TREE_NAME)
    if not t or not hasattr(t, "GetEntries"):
        print(f"  [!] No '{STAB_LINES_TREE_NAME}' tree in '{path}'", file=sys.stderr)
        f.Close(); return None

    best = None   # (n_events, slope, q0, blo, bhi)
    try:
        for e in t:
            if str(int(e.channel)) != str(ch_id):
                continue
            ne = int(e.n_events)
            if best is None or ne > best[0]:
                best = (ne, float(e.slope), float(e.q0),
                        float(e.baseline_low), float(e.baseline_high))
    except Exception as ex:  # malformed tree / missing branch
        print(f"  [!] Failed reading external lines from '{path}': {ex}", file=sys.stderr)
        f.Close(); return None
    f.Close()

    if best is None:
        return None
    ne, slope, q0, blo, bhi = best
    return (slope, q0, blo, bhi, ne, path)


# ===========================================================================
# CORE ANALYSIS  (one file)
# ===========================================================================

def run_stabilization(
    filename,
    save_summary_jpeg=True, save_corr_jpeg=True, create_root_file=True,
    show_canvas=False, manual_cuts=None, output_dir=None,
    save_partition_jpeg=True
):
    """
    Full stabilization pipeline for a single ROOT file.

    Parameters
    ----------
    filename          : path of the input ROOT file.
    save_summary_jpeg : write the 4x3 overview JPEG.
    save_corr_jpeg    : write the correlation-analysis JPEG.
    create_root_file  : write the stabilized copy with the new TTree.
    show_canvas       : draw the canvases on screen (interactive mode); when
                        True the drawn ROOT objects are kept alive.
    manual_cuts       : optional dict overriding the automatic cuts
                        (chosen_ld, ly_cut_min/max, heat_cut_min/max).
    output_dir        : base folder for the output sub-folders; defaults to the
                        folder containing *filename*.
    save_partition_jpeg : write the baseline-partition debug JPEG (baseline
                        histogram + amplitude-vs-baseline scatter with the
                        partition boundaries drawn as vertical lines).

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
    # Channel parsed according to ANALYSIS_MODE (after "ch", or first number).
    ch_id = parse_channel_id(filename) or "0"

    # Results obtained with an EXTERNAL line are tagged so they never overwrite
    # the calibration outputs (debug folder, JPEGs and ROOT file).
    calib_suffix = "_line_for_calib" if USE_EXTERNAL_STAB_LINE else ""

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

    def _valid_tree(t):
        return bool(t) and hasattr(t, "GetEntries")

    # --- Main tree: heat-amplitude source -----------------------------------
    # For OPTIMUM_FILTER_CHANNELS -> 'optimalfilter'. Otherwise the default
    # 'corrected_amplitude', or 'stabilization_all' when the former is absent.
    # The "heat_amplitude" branch then resolves to the chosen main tree.
    use_optimum_filter = ch_id in {str(c) for c in OPTIMUM_FILTER_CHANNELS}

    main_tree_name, tree_main = None, None
    if use_optimum_filter:
        t = file.Get(AMP_TREE_OPTIMUM)
        if _valid_tree(t):
            main_tree_name, tree_main = AMP_TREE_OPTIMUM, t
            tree_baseline = file.Get("baseline")
            print (f">>> Ch {ch_id}: baseline from 'baseline.heat_baseline'.")
            
        else:
            print(f"  [!] Ch {ch_id}: '{AMP_TREE_OPTIMUM}' tree missing; using the default source.",
                  file=sys.stderr)
            use_optimum_filter = False
    if tree_main is None:
        t = file.Get(AMP_TREE_DEFAULT)
        if _valid_tree(t):
            main_tree_name, tree_main = AMP_TREE_DEFAULT, t
            tree_baseline = file.Get("corrected_amplitude")
            print (f">>> Ch {ch_id}: baseline from 'corrected_amplitude.heat_baseline'.")
        else:
            t = file.Get(AMP_TREE_FALLBACK)
            if _valid_tree(t):
                main_tree_name, tree_main = AMP_TREE_FALLBACK, t
                tree_baseline = file.Get("baseline")
                print (f">>> Ch {ch_id}: baseline from 'baseline.heat_baseline'.")


    if not _valid_tree(tree_main):
        print(f"  [!] No heat-amplitude tree "
              f"('{AMP_TREE_OPTIMUM}'/'{AMP_TREE_DEFAULT}'/'{AMP_TREE_FALLBACK}') "
              f"in {os.path.basename(filename)}", file=sys.stderr)
        file.Close(); return -1.0

    print(f">>> Ch {ch_id}: heat amplitude from '{main_tree_name}.heat_amplitude'.")

    tree_mod  = file.Get("module")
    tree_bad  = file.Get("badinterval")
    tree_trig = file.Get("numberoftriggers")
    tree_corr = file.Get("correlation_corr")
    tree_ly   = file.Get("LY")
    tree_baseline = file.Get("baseline")
    tree_time = file.Get("timestamp")   # heat_timefromstartrun (baseline-vs-time plot)
    # Optimum-filter tree, also used to compute the LY when the 'LY' tree is gone.
    tree_opt  = tree_main if main_tree_name == AMP_TREE_OPTIMUM else file.Get(AMP_TREE_OPTIMUM)

    tree_for_stabilization = tree_main

    print(f"{'='*60}\n")

    # --- Attach friend trees to the main tree -------------------------------
    # Only VALID trees are attached: a missing tree from file.Get() is a null
    # pointer (not Python None), so it must be filtered with _valid_tree. The
    # LY tree and the optimum-filter tree (LY ratio fallback) are included too.
    candidate_friends = [tree_mod, tree_bad, tree_trig, tree_corr, tree_cal,
                         tree_baseline, tree_ly, tree_opt, tree_time]
    already     = {tree_for_stabilization.GetName()}
    friend_list = []
    for t in candidate_friends:
        if _valid_tree(t) and t is not tree_main and t.GetName() not in already:
            friend_list.append(t)
            already.add(t.GetName())
    for t in friend_list:
        tree_for_stabilization.AddFriend(t)

    # Local aliases for the calibration windows (see PARAMETERS).
    cal_corr_min, cal_corr_max = CAL_CORR_MIN, CAL_CORR_MAX
    cal_ly_min,   cal_ly_max   = CAL_LY_MIN,   CAL_LY_MAX
    cal_stab_min, cal_stab_max = CAL_STAB_MIN, CAL_STAB_MAX

    # ======================================================================
    # OPTIONAL-BRANCH DETECTION
    # ======================================================================
    _ly_own_branches = ({b.GetName() for b in tree_ly.GetListOfBranches()}
                        if _valid_tree(tree_ly) else set())
    _opt_branches    = ({b.GetName() for b in tree_opt.GetListOfBranches()}
                        if _valid_tree(tree_opt) else set())
    _time_branches = ({b.GetName() for b in tree_time.GetListOfBranches()}
                      if _valid_tree(tree_time) else set())
    # Prefer 'time_cumulative'; fall back to 'heat_timefromstartrun' if absent.
    time_leaf = ("time_cumulative"       if "time_cumulative"       in _time_branches
                 else "heat_timefromstartrun" if "heat_timefromstartrun" in _time_branches
                 else None)
    has_time = time_leaf is not None
    has_heat_badinterval = bool(tree_for_stabilization.GetLeaf("heat_badinterval"))
    has_heat_issignal    = bool(tree_for_stabilization.GetLeaf("heat_issignal"))
    cal_tree_name        = tree_cal.GetName()
    ly_tree_name         = tree_ly.GetName() if _valid_tree(tree_ly) else None

    # Light-Yield source:
    #   "tree"  -> the 'LY' tree (LD1_LY / LD2_LY branches);
    #   "ratio" -> LDx_amplitude / heat_amplitude from the optimum-filter tree,
    #              when the 'LY' tree is absent;
    #   "off"   -> neither available; the LY cut is disabled.
    if "LD1_LY" in _ly_own_branches:
        ly_mode        = "tree"
        has_ld2_branch = "LD2_LY" in _ly_own_branches
    elif "LD1_amplitude" in _opt_branches and "heat_amplitude" in _opt_branches:
        ly_mode        = "ratio"
        has_ld2_branch = "LD2_amplitude" in _opt_branches
    else:
        ly_mode        = "off"
        has_ld2_branch = False
    apply_ly_cut = ly_mode != "off"

    if ly_mode == "tree":
        print(">>> Light Yield: from the 'LY' tree (LD1_LY/LD2_LY).")
    elif ly_mode == "ratio":
        print(f">>> Light Yield: 'LY' tree absent, computed as "
              f"{AMP_TREE_OPTIMUM}.LDx_amplitude / {AMP_TREE_OPTIMUM}.heat_amplitude.")
    else:
        print("  [!] No 'LY' tree and no optimum-filter LD amplitudes. LY cut DISABLED.")

    # ======================================================================
    # RDATAFRAME: C++-level filters + bulk read with AsNumpy
    # (replaces the per-event Python loop "for i in range(nEntries): ...")
    # ======================================================================
    print("Phase 1: Reading events via RDataFrame + NumPy...")

    rdf = ROOT.RDataFrame(tree_for_stabilization)

    # --- Quality filters (run at C++ speed) ---------------------------------
    rdf_f = rdf.Filter("heat_numberoftriggers == 1")
    if has_heat_badinterval:
        rdf_f = rdf_f.Filter("heat_badinterval == 0")
    if has_heat_issignal:
        rdf_f = rdf_f.Filter("heat_issignal != 0")

    rdf_f = rdf_f.Define("_cal_rough_", f"{cal_tree_name}.heat_amplitude")

    # --- Light Yield --------------------------------------------------------
    # "tree"  : tree-qualified access "LY.LD1_LY" (a bare friend leaf is not
    #           readable by TTreeReader; bare name only when LY is the main tree).
    # "ratio" : LDx_amplitude / heat_amplitude from the optimum-filter tree
    #           (qualified name, or bare when it is itself the main tree).
    if apply_ly_cut:
        if ly_mode == "tree":
            if ly_tree_name == tree_for_stabilization.GetName():
                _ld1_ly_expr, _ld2_ly_expr = "LD1_LY", "LD2_LY"
            else:
                _ld1_ly_expr = f"{ly_tree_name}.LD1_LY"
                _ld2_ly_expr = f"{ly_tree_name}.LD2_LY"
        else:  # ratio
            pfx = "" if main_tree_name == AMP_TREE_OPTIMUM else f"{AMP_TREE_OPTIMUM}."
            _ld1_ly_expr = f"{pfx}LD1_amplitude / {pfx}heat_amplitude"
            _ld2_ly_expr = f"{pfx}LD2_amplitude / {pfx}heat_amplitude"
        rdf_f = rdf_f.Define("_ld1_ly_", _ld1_ly_expr)
        if has_ld2_branch:
            rdf_f = rdf_f.Define("_ld2_ly_", _ld2_ly_expr)

    # --- Time from start of run (friend tree 'timestamp'), for the baseline-vs-
    #     time monitoring scatter. Qualified name: it lives on a friend tree.
    if has_time:
        time_expr = (time_leaf
                     if tree_time.GetName() == tree_for_stabilization.GetName()
                     else f"{tree_time.GetName()}.{time_leaf}")
        rdf_f = rdf_f.Define("_time_", time_expr)
        print(f">>> Baseline-vs-time scatter: using 'timestamp.{time_leaf}'.")

    # --- Columns to read ----------------------------------------------------
    cols = ["heat_amplitude", "_cal_rough_", "heat_correlation", "heat_baseline"]
    if has_time:
        cols.append("_time_")
    if apply_ly_cut:
        cols.append("_ld1_ly_")
        if has_ld2_branch:
            cols.append("_ld2_ly_")

    # --- Single bulk read ---------------------------------------------------
    np_data = rdf_f.AsNumpy(cols)

    # Friend trees may carry DIFFERENT entry counts (e.g. 'baseline' shorter than
    # the main tree, 'timestamp' longer): AsNumpy then returns columns of unequal
    # length. The first N rows are entry-aligned across trees, so truncate every
    # column to the shortest one to keep them consistent (the surplus tail rows
    # have no baseline/friend value and are dropped).
    N = min(len(np_data[c]) for c in cols)
    if any(len(np_data[c]) != N for c in cols):
        print(f"  [!] Friend trees have mismatched entry counts "
              f"({ {c: len(np_data[c]) for c in cols} }); truncating to {N}.")

    print(f"  Read {N} events after quality filters.")

    # --- Extract typed arrays (sliced to the common length N) ---------------
    ha        = np_data["heat_amplitude"][:N].astype(np.float64)
    cal_rough = np_data["_cal_rough_"][:N].astype(np.float64)
    corr      = np_data["heat_correlation"][:N].astype(np.float64)
    baseline  = np_data["heat_baseline"][:N].astype(np.float64)
    time_fsr  = np_data["_time_"][:N].astype(np.float64) if has_time else None

    # Light Yield read DIRECTLY from the LD1_LY / LD2_LY leaves.
    ld1_ly   = np_data["_ld1_ly_"][:N].astype(np.float64)       if apply_ly_cut     else np.zeros(N, np.float64)
    ld2_ly   = (np_data["_ld2_ly_"][:N].astype(np.float64)
                if (apply_ly_cut and has_ld2_branch) else np.zeros(N, np.float64))

    # --- Vectorised NaN/Inf removal -----------------------------------------
    valid = np.isfinite(ha) & np.isfinite(cal_rough) & np.isfinite(corr)
    ha        = ha[valid];   cal_rough = cal_rough[valid]
    corr      = corr[valid]; baseline = baseline[valid]
    ld1_ly    = ld1_ly[valid]; ld2_ly = ld2_ly[valid]
    if time_fsr is not None:
        time_fsr = time_fsr[valid]   # keep the time vector aligned with baseline
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
    # CONVERSION FACTOR  (vectorised)
    # ======================================================================
    mask_main = corr > corr_cut_dynamic
    mask_conv = mask_main & (cal_rough > cal_corr_min) & (cal_rough < cal_corr_max) & (cal_rough > 0)
    cnt_conv  = int(mask_conv.sum())

    if cnt_conv > 0:
        conv_raw = float(np.median(ha[mask_conv] / cal_rough[mask_conv]))
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
    # BASELINE PARTITIONS  (vectorised)
    # ======================================================================
    # After the correlation cut the amplitude-vs-baseline scatter splits into
    # clusters along the baseline axis. Find one interval per baseline peak; a
    # separate linear stabilization will later be applied to each partition.
    
    mask_baseline = mask_main & ( ha > 0) & (ha < 20000)
    base_main = baseline[mask_baseline]
    ha_main   = ha[mask_baseline]
    partition_intervals, partition_peaks, h_base_part = FindBaselinePartitions(
        base_main, ch_id, enabled=ENABLE_BASELINE_PARTITIONS)

    if not ENABLE_BASELINE_PARTITIONS:
        print(">>> Baseline partitioning DISABLED: single stabilization over the whole range.")

    # External SLOPE-ONLY line: the baseline intervals are still found as usual on
    # the current data; only the external SLOPE (most-populated interval in the
    # external file) is reused, while each partition re-anchors its own intercept
    # on its local Tl peak.
    ext_single = None
    if USE_EXTERNAL_STAB_LINE:
        chosen = load_external_stab_line(EXTERNAL_STAB_LINE_DIR, ch_id)
        if chosen is not None:
            slope_ext, q0_ext, blo_ext, bhi_ext, ne_ext, src_path = chosen
            ext_single = (slope_ext, q0_ext)   # q0_ext kept for reference; not applied
            print(f">>> Ch {ch_id}: EXTERNAL SLOPE {slope_ext:+.4g} from "
                  f"'{os.path.basename(src_path)}' (baseline [{blo_ext:.4f}, {bhi_ext:.4f}], "
                  f"{ne_ext} events, the most populated); intercept re-anchored "
                  f"per partition on the local Tl peak.")
        else:
            print(f"  [!] No external line for ch {ch_id}; falling back to fitted lines.")

    print(f">>> Found {len(partition_intervals)} baseline partition(s) "
          f"(block markers at: {', '.join(f'{p:.4f}' for p in partition_peaks) or 'none'}):")
    for k, (blo, bhi) in enumerate(partition_intervals):
        print(f"      Partition {k}: baseline in [{blo:.4f}, {bhi:.4f}]")

    # Decimated scatter for the debug plot (display only; analysis uses all data).
    g_base_part = make_scatter_graph(base_main, ha_main)
    g_base_part.SetName(f"g_base_part_{ch_id}")
    g_base_part.SetTitle(
        f"Ch {ch_id}: Amplitude vs Baseline with partitions;Baseline;Amplitude")

    # Baseline-vs-time monitoring scatter (same events as the baseline scatter).
    g_base_time = None
    if has_time:
        time_main   = time_fsr[mask_baseline]
        g_base_time = make_scatter_graph(time_main, base_main)
        g_base_time.SetName(f"g_base_time_{ch_id}")
        g_base_time.SetTitle(
            f"Ch {ch_id}: Baseline vs Time;{time_leaf};Baseline")

    # ======================================================================
    # LIGHT-YIELD ANALYSIS  (vectorised) -- LY read directly from LD1_LY/LD2_LY
    # ======================================================================
    if apply_ly_cut:
        mask_ly_range = mask_main & (ha > cal_ly_min * conv_factor) & (ha < cal_ly_max * conv_factor)

        ly1_all = ld1_ly
        ly2_all = ld2_ly

        mask_ld1_ly = mask_ly_range & np.isfinite(ly1_all)
        mask_ld2_ly = mask_ly_range & np.isfinite(ly2_all) if has_ld2_branch else np.zeros(N, bool)

        vals_ly1_np = ly1_all[mask_ld1_ly];  heat_ly1_np = ha[mask_ld1_ly]
        vals_ly2_np = ly2_all[mask_ld2_ly];  heat_ly2_np = ha[mask_ld2_ly]
        # CALIBRATION-RUN mode frames the histogram down to negative LY so the
        # rare alpha peak (which can be partially negative) is in view.
        _ly_limits = _ly_limits_run if ANALYSIS_MODE == "calibrationrun" else calcRobustLimitsAndBins
        bins_ly1, ly1_min, ly1_max = _ly_limits(vals_ly1_np.tolist())
        bins_ly2, ly2_min, ly2_max = _ly_limits(vals_ly2_np.tolist())

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

        # CALIBRATION-RUN mode: use the thallium-dominant peak finder.
        _analyze_ly = AnalyzeLightYieldRun if ANALYSIS_MODE == "calibrationrun" else AnalyzeLightYield
        res1 = _analyze_ly(h_ly1, f"LD1_{ch_id}") if h_ly1.GetEntries() > 0 else LYResult()
        res2 = _analyze_ly(h_ly2, f"LD2_{ch_id}") if h_ly2.GetEntries() > 0 else LYResult()

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

    # ======================================================================
    # PER-PARTITION STABILIZATION
    # ======================================================================
    # The thallium-peak detection, outlier cleaning, robust linear baseline fit
    # and the stabilization itself are run SEPARATELY on each baseline
    # partition (the amplitude-vs-baseline slope differs between the clusters).
    mask_stab_window = ((ha > cal_stab_min * conv_factor)
                        & (ha < cal_stab_max * conv_factor))

    # Assign every event to a partition from its baseline. The internal valley
    # edges tile the baseline axis; searchsorted clamps events outside the
    # tiled range onto the end partitions, so coverage is complete.
    internal_edges = np.array([iv[1] for iv in partition_intervals[:-1]],
                              dtype=np.float64)
    if internal_edges.size > 0:
        part_of_event = np.searchsorted(internal_edges, baseline, side="right")
    else:
        part_of_event = np.zeros(N, dtype=np.int64)

    # Thallium-peak search hint from the COMBINED spectrum (all partitions
    # merged): more robust than any single partition, and shared by all.
    amps_all_stab           = amp_for_analysis[mask_ly_pass & mask_stab_window]
    hint_center, hint_sigma = estimate_thallium_peak(amps_all_stab, f"{ch_id}_comb")
    peak_hint = (hint_center, hint_sigma) if hint_center != 0.0 else None
    if peak_hint is not None:
        print(f">>> Combined-spectrum Tl peak hint at amplitude {hint_center:.1f} "
              f"(sigma {hint_sigma:.1f}); used for all partitions.")

    part_results      = []
    apply_heat_manual = (len(partition_intervals) == 1)
    for idx, (blo, bhi) in enumerate(partition_intervals):
        mask_part = mask_ly_pass & mask_stab_window & (part_of_event == idx)
        amps_p  = amp_for_analysis[mask_part]
        bases_p = baseline[mask_part]
        # External slope (locally re-anchored intercept) or full per-partition fit.
        src = "external slope, local anchor" if ext_single is not None else "fitted line"
        print(f"  Partition {idx} baseline [{blo:.4f}, {bhi:.4f}]: "
              f"{len(amps_p)} events for stabilization ({src})")
        res_p = process_partition(amps_p, bases_p, ch_id, idx, blo, bhi,
                                  manual_cuts=manual_cuts,
                                  apply_heat_manual=apply_heat_manual,
                                  peak_hint=peak_hint, ext_line=ext_single)
        part_results.append(res_p)

    sufficient_events = any(r.sufficient for r in part_results)

    # Fallback (q_0, slope) for partitions WITHOUT a line (no fit and no external
    # line: slope == q_0 == 0): copy it from the nearest partition that has one.
    suff_idx = [i for i, r in enumerate(part_results)
                if not (r.q_0 == 0.0 and r.slope == 0.0)]
    if suff_idx:
        for i, r in enumerate(part_results):
            if r.q_0 == 0.0 and r.slope == 0.0:
                j = min(suff_idx, key=lambda s: abs(s - i))
                r.slope, r.q_0 = part_results[j].slope, part_results[j].q_0

    # ======================================================================
    # PER-PARTITION BEFORE/AFTER PEAK FITS  (all window events, gaus+pol1)
    # ======================================================================
    # For each partition, fit the Tl peak over ALL its stabilization-window events
    # (res.all_amps, not only the clean subset) with a Gaussian over a linear
    # background, on a shared before/after axis:
    #   BEFORE : window events RESCALED to energy on the partition's own Tl peak;
    #   AFTER  : window events STABILIZED per-event (already energy, NOT rescaled).
    for r in part_results:
        if r.all_amps.size < 1:
            continue
        # BEFORE reference = the partition's own clean-peak Gaussian centre (found
        # with the shared combined-peak hint in process_partition): robust also on
        # low-statistics partitions, where re-searching the raw window could latch
        # onto the continuum instead of the Tl peak.
        mu_ref = r.mean_amp_clean if (r.mean_amp_clean and r.mean_amp_clean > 0) \
                 else float(np.median(r.all_amps))
        before_e = TARGET_ENERGY * r.all_amps / mu_ref
        after_e = None
        if not (r.q_0 == 0.0 and r.slope == 0.0):
            with np.errstate(divide='ignore', invalid='ignore'):
                ae = TARGET_ENERGY * r.all_amps / (r.q_0 + r.slope * r.all_bases)
            ae = ae[np.isfinite(ae)]
            if ae.size > 0:
                after_e = ae
        # Reliable peak width (energy) from the partition's clean-peak Gaussian fit
        # (found with the shared hint): sizes a SMALL window matched to the peak,
        # robust where a low-statistics width estimate would be noisy.
        sig_hint = None
        if r.fit_clean is not None:
            s_amp = abs(r.fit_clean.GetParameter(2))
            if s_amp > 0 and mu_ref > 0:
                sig_hint = s_amp * TARGET_ENERGY / mu_ref
        r.h_before, r.fit_before, r.h_after, r.fit_after = shared_before_after_fits(
            before_e, after_e, TARGET_ENERGY, f"p_{ch_id}_{r.idx}", sig_hint=sig_hint)
        if r.h_before is not None:
            r.h_before.SetTitle(
                f"Ch {ch_id} P{r.idx}: Thallium BEFORE stab. (rescaled);Energy (keV);Counts")
        if r.h_after is not None:
            r.h_after.SetTitle(
                f"Ch {ch_id} P{r.idx}: Thallium AFTER stab.;Energy (keV);Counts")

    # ======================================================================
    # COMBINED BEFORE/AFTER SPECTRA  (all partitions merged, all window events)
    # ======================================================================
    # Same treatment as the alpha combined image and as the per-partition fits:
    #   BEFORE : ALL window events merged, rescaled to energy on the combined Tl
    #            peak (the width then includes the inter-partition spread);
    #   AFTER  : ALL window events, stabilized per-event (already energy).
    raw_parts = [r.all_amps for r in part_results if r.all_amps.size > 0]
    all_raw   = np.concatenate(raw_parts) if raw_parts else np.array([], np.float64)

    after_parts = []
    for r in part_results:
        if r.all_amps.size > 0 and not (r.q_0 == 0.0 and r.slope == 0.0):
            with np.errstate(divide='ignore', invalid='ignore'):
                cal = TARGET_ENERGY * r.all_amps / (r.q_0 + r.slope * r.all_bases)
            after_parts.append(cal[np.isfinite(cal)])
    all_after = np.concatenate(after_parts) if after_parts else np.array([], np.float64)

    all_before = np.array([], np.float64)
    if all_raw.size > 0:
        _, lo0, hi0 = calcRobustLimitsAndBins(all_raw.tolist())
        _, f0 = fit_peak_optimized(all_raw, lo0, hi0, f"before_raw_{ch_id}")
        mean_raw = f0.GetParameter(1) if (f0 is not None and f0.GetParameter(1) > 0) else 0.0
        if mean_raw == 0.0:
            mean_raw = float(np.median(all_raw))
        all_before = TARGET_ENERGY * all_raw / mean_raw

    h_before_comb, fit_before_comb, h_heat_cal_comb, fit_cal_comb = shared_before_after_fits(
        all_before if all_before.size > 0 else None,
        all_after  if all_after.size  > 0 else None,
        TARGET_ENERGY, f"comb_{ch_id}")
    if h_before_comb is not None:
        h_before_comb.SetTitle(
            f"Ch {ch_id}: Thallium Peak BEFORE stabilization (rescaled);Energy (keV);Counts")
    if h_heat_cal_comb is not None:
        h_heat_cal_comb.SetTitle(
            f"Ch {ch_id}: Thallium Peak AFTER stabilization (all partitions);Energy (keV);Counts")

    if fit_cal_comb is not None and h_heat_cal_comb is not None:
        gaussian_counts = (fit_cal_comb.GetParameter(0) * fit_cal_comb.GetParameter(2)
                           * math.sqrt(2 * math.pi)
                           / h_heat_cal_comb.GetXaxis().GetBinWidth(1))
    else:
        gaussian_counts = 0.0
        print("  [!] Warning: no event survived the cuts in any partition.")

    # ======================================================================
    # WRITE STABILIZED .ROOT FILE  (per-partition linear stabilization)
    # ======================================================================
    if create_root_file and sufficient_events:
        print("\nCreating output file by cloning the original file...")
        pure_filename = os.path.basename(filename)
        base_filename = pure_filename.rsplit('.', 1)[0] if '.' in pure_filename else pure_filename

        out_dir = os.path.join(output_dir, STAB_ROOT_DIR_NAME)
        os.makedirs(out_dir, exist_ok=True)

        out_filename     = os.path.join(out_dir, f"{base_filename}_thallium_stabilized{calib_suffix}.root")
        new_tree_name    = "stabilized_heater_thallium"
        new_tree_title   = "Heater + Thallium Stabilized"

        ROOT.gSystem.CopyFile(filename, out_filename, ROOT.kTRUE)
        file_out = ROOT.TFile(out_filename, "UPDATE")

        # Read the SAME per-event amplitude & baseline as the analysis: the heat
        # amplitude from the chosen main tree (corrected_amplitude /
        # stabilization_all / optimalfilter) and heat_baseline via the baseline
        # friend (its bare name resolves to the main tree when that one owns it).
        tree_source_out = file_out.Get(main_tree_name)
        tree_base_out   = file_out.Get("baseline")
        if tree_base_out and hasattr(tree_base_out, "GetEntries"):
            tree_source_out.AddFriend(tree_base_out)
        out_data = ROOT.RDataFrame(tree_source_out).AsNumpy(["heat_amplitude", "heat_baseline"])
        raw_amps_out  = out_data["heat_amplitude"].astype(np.float64)
        raw_bases_out = out_data["heat_baseline"].astype(np.float64)

        # Each event is stabilized with the (q_0, slope) of the partition its
        # baseline falls into (same valley edges used in the analysis).
        if internal_edges.size > 0:
            idx_out = np.searchsorted(internal_edges, raw_bases_out, side="right")
        else:
            idx_out = np.zeros(len(raw_bases_out), dtype=np.int64)
        slopes_arr = np.array([r.slope for r in part_results], dtype=np.float64)
        q0_arr     = np.array([r.q_0   for r in part_results], dtype=np.float64)

        with np.errstate(divide='ignore', invalid='ignore'):
            denom_out       = q0_arr[idx_out] + slopes_arr[idx_out] * raw_bases_out
            stabilized_amps = TARGET_ENERGY * raw_amps_out / denom_out

        new_tree           = ROOT.TTree(new_tree_name, new_tree_title)
        final_heat_amp_arr = array('d', [0.0])
        new_tree.Branch("heat_amplitude", final_heat_amp_arr, "heat_amplitude/D")
        for amp_val in stabilized_amps:
            final_heat_amp_arr[0] = float(amp_val)
            new_tree.Fill()

        new_tree.Write()
        global_tree = file_out.Get("global")
        if global_tree:
            global_tree.AddFriend(new_tree_name)
            global_tree.Write("", ROOT.TObject.kOverwrite)

        # --- Save the per-partition stabilization lines -----------------------
        if SAVE_STAB_LINES:
            ch_int = int(ch_id) if str(ch_id).isdigit() else -1
            lines_tree = ROOT.TTree(STAB_LINES_TREE_NAME, "Per-partition stabilization lines")
            _l_ch  = array('i', [0]); _l_p   = array('i', [0]); _l_ne = array('i', [0])
            _l_blo = array('d', [0.0]); _l_bhi = array('d', [0.0])
            _l_sl  = array('d', [0.0]); _l_q0  = array('d', [0.0])
            lines_tree.Branch("channel",       _l_ch,  "channel/I")
            lines_tree.Branch("partition",     _l_p,   "partition/I")
            lines_tree.Branch("n_events",      _l_ne,  "n_events/I")
            lines_tree.Branch("baseline_low",  _l_blo, "baseline_low/D")
            lines_tree.Branch("baseline_high", _l_bhi, "baseline_high/D")
            lines_tree.Branch("slope",         _l_sl,  "slope/D")
            lines_tree.Branch("q0",            _l_q0,  "q0/D")
            for idx, (blo, bhi) in enumerate(partition_intervals):
                r = part_results[idx]
                _l_ch[0]  = ch_int
                _l_p[0]   = idx
                _l_ne[0]  = int(r.n_events)
                _l_blo[0] = float(blo) if np.isfinite(blo) else 0.0
                _l_bhi[0] = float(bhi) if np.isfinite(bhi) else 0.0
                _l_sl[0]  = float(r.slope)
                _l_q0[0]  = float(r.q_0)
                lines_tree.Fill()
            lines_tree.Write()
            print(f">>> Saved {len(partition_intervals)} stabilization line(s) "
                  f"to '{STAB_LINES_TREE_NAME}' tree.")

        file_out.Close()
        print(f">>> Successfully saved '{new_tree_name}' tree to {out_filename}")

    # ======================================================================
    # CANVASES  (saved to JPEG, or shown on screen in interactive mode)
    # ======================================================================
    global_lines, canvases = [], []

    # All debug JPEGs of this channel go into a per-channel sub-folder inside
    # the debug directory: <SUMMARY_DIR>/ch<N>/ (created lazily on first save).
    debug_ch_dir = os.path.join(output_dir, SUMMARY_DIR_NAME, f"ch{ch_id}{calib_suffix}")

    def drawLines(ymax, conv, color, style, min_val, max_val):
        """Draw the two vertical range lines (min, max) scaled by *conv*."""
        l_min = ROOT.TLine(min_val * conv, 0, min_val * conv, ymax)
        l_max = ROOT.TLine(max_val * conv, 0, max_val * conv, ymax)
        for l, c, s in [(l_min, color, style), (l_max, color, style)]:
            l.SetLineColor(c); l.SetLineWidth(2); l.SetLineStyle(s); l.Draw("same")
        global_lines.extend([l_min, l_max])

    # ------------------------------------------------------- GLOBAL OVERVIEW (A)
    # First two rows: correlation + spectra (row 1) and light yield (row 2).
    # These are global (one per file); they do NOT depend on the partition.
    make_summary = show_canvas or save_summary_jpeg
    if make_summary and save_summary_jpeg:
        os.makedirs(debug_ch_dir, exist_ok=True)

    if make_summary:
        c_glob = ROOT.TCanvas(f"c_glob_{ch_id}",
                              f"Global Overview Ch {ch_id}", 500 * 3, 400 * 2)
        c_glob.Divide(3, 2)

        # pad 1 : correlation distribution with the cut line
        c_glob.cd(1); ROOT.gPad.SetGrid()
        h_corr.SetLineColor(ROOT.kBlack); h_corr.SetFillColorAlpha(ROOT.kBlack, 0.3)
        h_corr.Draw(); c_glob.Update()
        lcr = ROOT.TLine(corr_cut_dynamic, ROOT.gPad.GetUymin(),
                         corr_cut_dynamic, ROOT.gPad.GetUymax())
        lcr.SetLineColor(ROOT.kRed); lcr.SetLineWidth(2); lcr.SetLineStyle(2)
        lcr.Draw("same"); global_lines.append(lcr)

        # pad 2 : Calibration Rough (after the correlation cut)
        c_glob.cd(2); ROOT.gPad.SetGrid()
        h_cal_rough.SetLineColor(ROOT.kBlack); h_cal_rough.SetFillColorAlpha(ROOT.kGray, 0.5)
        h_cal_rough.Draw(); c_glob.Update()
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
        leg_rough.AddEntry(d_stab, "Stab Range", "l")
        leg_rough.Draw("same")
        global_lines.extend([d_ly, d_corr, d_stab, leg_rough])

        # pad 3 : Heater Stabilized (raw spectrum after the correlation cut)
        c_glob.cd(3); ROOT.gPad.SetGrid()
        h_raw.SetLineColor(ROOT.kBlack); h_raw.SetFillColorAlpha(ROOT.kOrange+1, 0.5)
        h_raw.Draw(); c_glob.Update()
        drawLines(ROOT.gPad.GetUymax(), conv_raw, ROOT.kBlue,    2, cal_ly_min,   cal_ly_max)
        drawLines(ROOT.gPad.GetUymax(), conv_raw, ROOT.kRed,     1, cal_corr_min, cal_corr_max)
        drawLines(ROOT.gPad.GetUymax(), conv_raw, ROOT.kGreen+2, 3, cal_stab_min, cal_stab_max)

        # ---------- ROW 2 : light yield + After Corr + LY Cut spectrum ----------
        if apply_ly_cut:
            c_glob.cd(4); ROOT.gPad.SetGrid()
            if h_ly1.GetEntries() > 0:
                h_ly1.SetStats(0); h_ly1.SetLineColor(ROOT.kRed)
                h_ly1.SetFillColorAlpha(ROOT.kRed, 0.3); h_ly1.Draw()
                if res1.fit_Tl:    res1.fit_Tl.Draw("same")
                if res1.fit_alpha: res1.fit_alpha.Draw("same")
                c_glob.Update()
                if res1.fit_Tl:
                    ll1 = ROOT.TLine(res1.cut_min, 0, res1.cut_min, ROOT.gPad.GetUymax())
                    lr1 = ROOT.TLine(res1.cut_max, 0, res1.cut_max, ROOT.gPad.GetUymax())
                    for l in (ll1, lr1):
                        l.SetLineColor(ROOT.kBlue); l.SetLineWidth(2)
                        l.SetLineStyle(2); l.Draw("same")
                    pt1 = CreateLYBox(res1, "LD1"); pt1.Draw()
                    global_lines.extend([ll1, lr1, pt1])

            c_glob.cd(5); ROOT.gPad.SetGrid()
            if h_ly2.GetEntries() > 0:
                h_ly2.SetStats(0); h_ly2.SetLineColor(ROOT.kRed)
                h_ly2.SetFillColorAlpha(ROOT.kRed, 0.3); h_ly2.Draw()
                if res2.fit_Tl:    res2.fit_Tl.Draw("same")
                if res2.fit_alpha: res2.fit_alpha.Draw("same")
                c_glob.Update()
                if res2.fit_Tl:
                    ll2 = ROOT.TLine(res2.cut_min, 0, res2.cut_min, ROOT.gPad.GetUymax())
                    lr2 = ROOT.TLine(res2.cut_max, 0, res2.cut_max, ROOT.gPad.GetUymax())
                    for l in (ll2, lr2):
                        l.SetLineColor(ROOT.kBlue); l.SetLineWidth(2)
                        l.SetLineStyle(2); l.Draw("same")
                    pt2 = CreateLYBox(res2, "LD2"); pt2.Draw()
                    global_lines.extend([ll2, lr2, pt2])

        c_glob.cd(6); ROOT.gPad.SetGrid()
        h2_full.Draw(); c_glob.Update()
        drawLines(ROOT.gPad.GetUymax(), conv_factor, ROOT.kBlue,    2, cal_ly_min,   cal_ly_max)
        drawLines(ROOT.gPad.GetUymax(), conv_factor, ROOT.kRed,     1, cal_corr_min, cal_corr_max)
        drawLines(ROOT.gPad.GetUymax(), conv_factor, ROOT.kGreen+2, 3, cal_stab_min, cal_stab_max)

        c_glob.Update()
        if save_summary_jpeg:
            out_jpg = os.path.join(debug_ch_dir, f"ch{ch_id}_global_overview{calib_suffix}.jpg")
            save_canvas_jpeg(c_glob, out_jpg)
        canvases.append(c_glob)

        # ------------------------------------ PER-PARTITION STABILIZATION (B)
        # Last two rows of the legacy overview, replicated PER baseline
        # partition. Each partition is a 2x3 block; blocks are stacked
        # vertically up to 3, then a new column starts.
        #   block row 0 : pre-cleaning spectrum | clean peak (+FWHM) | amp-vs-base + lin.fit
        #   block row 1 : calibrated peak (+res) | stabilized peak    | amp-vs-base after stab
        P            = len(part_results)
        n_block_rows = min(P, 3)
        n_block_cols = math.ceil(P / 3)
        n_pad_cols   = 3 * n_block_cols
        n_pad_rows   = 2 * n_block_rows
        c_title = (f"Per-partition Stabilization Ch {ch_id}"
                   + (" - EXTERNAL-SLOPE LINE (local anchor)" if USE_EXTERNAL_STAB_LINE else ""))
        c_parts = ROOT.TCanvas(f"c_parts_{ch_id}", c_title,
                               500 * n_pad_cols, 400 * n_pad_rows)
        c_parts.Divide(n_pad_cols, n_pad_rows)

        # Blue note added to the stabilized/calibrated boxes when an external
        # (calibration-run) line is applied.
        calib_note = "slope da run di calib., q0 ancorato localmente" if USE_EXTERNAL_STAB_LINE else None

        for r in part_results:
            block_col = r.idx // 3
            block_row = r.idx % 3
            base_col  = block_col * 3
            base_row  = block_row * 2

            def padno(sub_row, sub_col):
                return (base_row + sub_row) * n_pad_cols + (base_col + sub_col) + 1

            # (0,0) pre-cleaning spectrum + preliminary fit
            c_parts.cd(padno(0, 0)); ROOT.gPad.SetGrid()
            if r.sufficient and r.h_heat_orig is not None:
                r.h_heat_orig.SetLineColor(ROOT.kBlue); r.h_heat_orig.SetStats(0)
                r.h_heat_orig.Draw()
                if r.fit_prelim:
                    r.fit_prelim.SetLineColor(ROOT.kRed); r.fit_prelim.SetLineWidth(2)
                    r.fit_prelim.Draw("same")

            # (0,1) thallium peak BEFORE stabilization (rescaled) + gaus+pol1 fit
            #       over ALL window events, on the shared before/after axis.
            c_parts.cd(padno(0, 1)); ROOT.gPad.SetGrid()
            if r.sufficient and r.h_before is not None:
                r.h_before.SetStats(0); r.h_before.SetLineColor(ROOT.kBlack)
                r.h_before.SetFillColorAlpha(ROOT.kGray, 0.5); r.h_before.Draw()
                if r.fit_before:
                    r.fit_before.SetLineColor(ROOT.kBlue); r.fit_before.SetLineWidth(2)
                    r.fit_before.Draw("same")
                box = CreateFitBox(r.fit_before, f"P{r.idx} BEFORE stab.", ROOT.kGray + 2)
                box.Draw(); global_lines.append(box)

            # (0,2) amplitude vs baseline + linear fit (de-zoomed, with the line)
            c_parts.cd(padno(0, 2)); ROOT.gPad.SetGrid()
            if r.sufficient and r.g_heat_vs_base_clean is not None:
                gc = r.g_heat_vs_base_clean
                gc.SetMarkerStyle(20); gc.SetMarkerSize(0.5); gc.SetMarkerColor(ROOT.kMagenta)
                if r.view_clean is not None:
                    x0, x1, y0, y1 = r.view_clean
                    gc.GetXaxis().SetLimits(x0, x1)
                    gc.SetMinimum(y0); gc.SetMaximum(y1)
                gc.Draw("AP")
                if r.f1:
                    if r.view_clean is not None:
                        r.f1.SetRange(r.view_clean[0], r.view_clean[1])
                    r.f1.SetLineColor(ROOT.kRed); r.f1.SetLineWidth(2); r.f1.Draw("same")
                    lbox = CreateLineBox(r.f1, f"P{r.idx} FIT LINE", ROOT.kRed)
                    lbox.Draw(); global_lines.append(lbox)
                if r.f1_ext:
                    # Line actually applied (drawn in blue): external SLOPE with the
                    # intercept re-anchored on this partition's own Tl peak.
                    if r.view_clean is not None:
                        r.f1_ext.SetRange(r.view_clean[0], r.view_clean[1])
                    r.f1_ext.SetLineColor(ROOT.kBlue); r.f1_ext.SetLineWidth(2)
                    r.f1_ext.SetLineStyle(2); r.f1_ext.Draw("same")
                    ebox = CreateLineBox(r.f1_ext, f"P{r.idx} EXT-SLOPE LINE", ROOT.kBlue,
                                         x1=0.14, y1=0.58, x2=0.62, y2=0.72)
                    ebox.Draw(); global_lines.append(ebox)

            # (1,0) thallium peak AFTER stabilization (stabilized, NOT rescaled) +
            #       gaus+pol1 fit over ALL window events, on the shared axis.
            c_parts.cd(padno(1, 0)); ROOT.gPad.SetGrid()
            if r.sufficient and r.h_after is not None:
                r.h_after.SetStats(0); r.h_after.SetLineColor(ROOT.kBlack)
                r.h_after.SetFillColorAlpha(ROOT.kRed + 1, 0.5); r.h_after.Draw()
                if r.fit_after:
                    r.fit_after.SetLineColor(ROOT.kBlue); r.fit_after.SetLineWidth(2)
                    r.fit_after.Draw("same")
                box = CreateFitBox(r.fit_after, f"P{r.idx} AFTER stab.", ROOT.kRed + 1,
                                   note=calib_note)
                box.Draw(); global_lines.append(box)

            # (1,1) stabilized peak (amplitude) + fit + box
            c_parts.cd(padno(1, 1)); ROOT.gPad.SetGrid()
            if r.sufficient and r.h_heat_stab is not None:
                r.h_heat_stab.SetStats(0); r.h_heat_stab.SetLineColor(ROOT.kGreen + 2)
                r.h_heat_stab.Draw()
                if r.fit_stab:
                    r.fit_stab.SetLineColor(ROOT.kRed); r.fit_stab.SetLineWidth(2); r.fit_stab.Draw("same")
                box = CreateFitBox(r.fit_stab, f"P{r.idx} STABILIZED", ROOT.kGreen + 2,
                                   note=calib_note)
                box.Draw(); global_lines.append(box)

            # (1,2) amplitude vs baseline AFTER stabilization (de-zoomed)
            c_parts.cd(padno(1, 2)); ROOT.gPad.SetGrid()
            if r.sufficient and r.g_heat_vs_base_stab is not None:
                gs = r.g_heat_vs_base_stab
                gs.SetMarkerStyle(20); gs.SetMarkerSize(0.5); gs.SetMarkerColor(ROOT.kGreen + 2)
                if r.view_stab is not None:
                    x0, x1, y0, y1 = r.view_stab
                    gs.GetXaxis().SetLimits(x0, x1)
                    gs.SetMinimum(y0); gs.SetMaximum(y1)
                gs.Draw("AP")

        c_parts.Update()
        if save_summary_jpeg:
            out_jpg = os.path.join(debug_ch_dir, f"ch{ch_id}_partitions_stabilization{calib_suffix}.jpg")
            save_canvas_jpeg(c_parts, out_jpg)
        canvases.append(c_parts)

        # ------------------------- COMBINED THALLIUM PEAK: BEFORE vs AFTER (C)
        # Left  : partition peaks overlapped by a CONSTANT per-partition rescale
        #         (no baseline correction) -> resolution BEFORE stabilization.
        # Right : final stabilized peak (per-event baseline correction).
        c_comb = ROOT.TCanvas(f"c_comb_{ch_id}",
                              f"Combined Thallium Peak Ch {ch_id}", 1400, 600)
        c_comb.Divide(2, 1)

        def draw_combined(pad, h, fit, header, hcol):
            c_comb.cd(pad); ROOT.gPad.SetGrid()
            if h is None:
                return
            h.SetStats(0); h.SetLineColor(ROOT.kBlack)
            h.SetFillColorAlpha(hcol, 0.5); h.Draw()
            if fit:
                fit.SetLineColor(ROOT.kBlue); fit.SetLineWidth(2); fit.Draw("same")
            box = CreateFitBox(fit, header, hcol)
            box.Draw(); global_lines.append(box)

        draw_combined(1, h_before_comb, fit_before_comb, "BEFORE stabilization", ROOT.kGray + 2)
        draw_combined(2, h_heat_cal_comb, fit_cal_comb, "AFTER stabilization", ROOT.kRed + 1)

        c_comb.Update()
        if save_summary_jpeg:
            out_jpg = os.path.join(debug_ch_dir, f"ch{ch_id}_combined_thallium{calib_suffix}.jpg")
            save_canvas_jpeg(c_comb, out_jpg)
        canvases.append(c_comb)

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
            out_jpg = os.path.join(corr_dir, f"ch{ch_id}_correlation_cut{calib_suffix}.jpg")
            save_canvas_jpeg(c_corr, out_jpg)
        canvases.append(c_corr)

    # ---------------------------------------------------- BASELINE PARTITIONS
    make_part = show_canvas or save_partition_jpeg
    if make_part:
        n_part_pads = 3 if g_base_time is not None else 2
        c_part = ROOT.TCanvas(f"c_part_{ch_id}",
                              f"Baseline Partitions Ch {ch_id}",
                              1200, 450 * n_part_pads)
        c_part.Divide(1, n_part_pads)

        # All partition boundaries (valleys + outer bounds), de-duplicated.
        boundary_vals = sorted({b for iv in partition_intervals for b in iv
                                if np.isfinite(b)})

        # pad 1 : baseline histogram + peak markers + partition boundaries
        c_part.cd(1); ROOT.gPad.SetGrid()
        if h_base_part is not None:
            h_base_part.SetLineColor(ROOT.kBlack)
            h_base_part.SetFillColorAlpha(ROOT.kAzure + 1, 0.4)
            h_base_part.Draw(); c_part.Update()
            ymax_h = ROOT.gPad.GetUymax()
            for bx in boundary_vals:
                l = ROOT.TLine(bx, 0, bx, ymax_h)
                l.SetLineColor(ROOT.kRed); l.SetLineWidth(2); l.SetLineStyle(2)
                l.Draw("same"); global_lines.append(l)
            for px in partition_peaks:
                py = h_base_part.GetBinContent(h_base_part.GetXaxis().FindBin(px))
                m = ROOT.TMarker(px, py, 23)
                m.SetMarkerColor(ROOT.kGreen + 2); m.SetMarkerSize(1.6)
                m.Draw("same"); global_lines.append(m)

        # pad 2 : amplitude-vs-baseline scatter + partition boundaries
        c_part.cd(2); ROOT.gPad.SetGrid()
        g_base_part.SetMarkerStyle(20); g_base_part.SetMarkerSize(0.4)
        g_base_part.SetMarkerColor(ROOT.kBlue); g_base_part.Draw("AP"); c_part.Update()
        ymin_s, ymax_s = ROOT.gPad.GetUymin(), ROOT.gPad.GetUymax()
        for bx in boundary_vals:
            l = ROOT.TLine(bx, ymin_s, bx, ymax_s)
            l.SetLineColor(ROOT.kRed); l.SetLineWidth(2); l.SetLineStyle(2)
            l.Draw("same"); global_lines.append(l)

        # pad 3 : baseline-vs-time scatter + partition boundaries (horizontal here)
        if g_base_time is not None:
            c_part.cd(3); ROOT.gPad.SetGrid()
            g_base_time.SetMarkerStyle(20); g_base_time.SetMarkerSize(0.4)
            g_base_time.SetMarkerColor(ROOT.kBlue); g_base_time.Draw("AP"); c_part.Update()
            xmin_t, xmax_t = ROOT.gPad.GetUxmin(), ROOT.gPad.GetUxmax()
            for by in boundary_vals:
                l = ROOT.TLine(xmin_t, by, xmax_t, by)
                l.SetLineColor(ROOT.kRed); l.SetLineWidth(2); l.SetLineStyle(2)
                l.Draw("same"); global_lines.append(l)

        c_part.Update()
        if save_partition_jpeg:
            os.makedirs(debug_ch_dir, exist_ok=True)
            out_jpg = os.path.join(debug_ch_dir, f"ch{ch_id}_baseline_partitions{calib_suffix}.jpg")
            save_canvas_jpeg(c_part, out_jpg)
        canvases.append(c_part)

    # --- MEMORY PROTECTION: only when canvases are shown interactively ------
    # In batch (JPEG only) there is no need to keep them alive: the SaveAs is
    # already done. In interactive mode, instead, EVERY drawn object (hists,
    # graphs, fits, lines, boxes) must stay referenced; otherwise the Python
    # garbage collector frees them and the histograms vanish at the first
    # window repaint (e.g. a click).
    if show_canvas:
        global GLOBAL_KEEPALIVE
        GLOBAL_KEEPALIVE.extend(canvases)
        GLOBAL_KEEPALIVE.extend(global_lines)

        keep = [h_cal_rough, h_raw, h_corr, g_corr_vs_heat, h2_full,
                g_base_part, g_base_time, h_base_part,
                h_heat_cal_comb, fit_cal_comb, h_before_comb, fit_before_comb]
        if apply_ly_cut:
            keep.extend([h_ly1, h_ly2, g_ly1_vs_heat, g_ly2_vs_heat,
                         res1.fit_Tl, res1.fit_alpha, res2.fit_Tl, res2.fit_alpha])
        for r in part_results:
            keep.extend([r.h_heat_orig, r.fit_prelim, r.h_heat_clean, r.fit_clean,
                         r.g_heat_vs_base_clean, r.f1, r.f1_ext, r.h_heat_stab, r.fit_stab,
                         r.g_heat_vs_base_stab, r.h_heat_cal, r.h_heat_cal_final,
                         r.fit_cal, r.h_before, r.fit_before, r.h_after, r.fit_after])
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
        description="Analyse the .root files in BASE_DIR for the channels in "
                    "CHANNELS_TO_PROCESS (or the channels given on the command line).")
    parser.add_argument("channels", nargs="*", default=[],
                        help="Channels to analyse (e.g. 24 51). If omitted, "
                             "CHANNELS_TO_PROCESS is used.")
    args = parser.parse_args()

    # All run settings (BASE_DIR, CHANNELS_TO_PROCESS, the SAVE_*/CREATE_*/GUI
    # switches and the amplitude-source lists) live in the RUN CONFIGURATION
    # block at the top of this file. The command line, when channels are passed,
    # OVERRIDES CHANNELS_TO_PROCESS.
    if args.channels:
        target_channels = {str(c) for c in args.channels}
    else:
        target_channels = {str(c) for c in CHANNELS_TO_PROCESS}

    # GUI is active only when the flag is on AND exactly one channel is selected.
    gui_active = GUI_MANUAL_CUTS and len(target_channels) == 1

    # Folder to scan (depends on ANALYSIS_MODE).
    scan_dir = resolve_scan_dir()
    if ANALYSIS_MODE in ("run", "calibrationrun"):
        _tag = "CALIBRATION RUN" if ANALYSIS_MODE == "calibrationrun" else "RUN"
        print(f">>> Mode: {_tag} {int(RUN_NUMBER):06d}  (folder: {scan_dir})")
    else:
        print(f">>> Mode: mergedrun  (folder: {scan_dir})")

    if not os.path.isdir(scan_dir):
        print(f"[!] Error: folder not found: {scan_dir}")
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
        # Warm up tkinter BEFORE any TCanvas (needed on macOS: Tk registers the
        # NSApplication first, which ROOT then reuses without a Cocoa clash).
        try:
            _tk_warmup = tk.Tk()
            _tk_warmup.withdraw()
            _tk_warmup.update()
            _tk_warmup.destroy()
        except tk.TclError:
            pass

    # Collect the valid .root files (skip already-stabilized outputs).
    root_files = sorted(f for f in os.listdir(scan_dir)
                        if f.endswith(".root") and "stabilized" not in f)

    if not root_files:
        print(f"[!] No .root file found in {scan_dir}")
        sys.exit(1)

    any_success = False
    processed   = 0

    for fname in root_files:
        # Channel parsed according to ANALYSIS_MODE (after "ch", or first number).
        ch_id = parse_channel_id(fname)

        # Filter by the selected channels (empty set => take every file).
        if target_channels and (ch_id is None or ch_id not in target_channels):
            continue

        full_path = os.path.join(scan_dir, fname)
        print(f"\n>>> File: {fname}  (channel {ch_id})")

        try:
            if gui_active:
                # --------------------------------------------------------------
                # INTERACTIVE GUI MODE  (only reachable with a single channel)
                # Loop: show the canvases on screen, let the user tweak the
                # manual cuts and choose recalc / accept (save JPEG + ROOT) / quit.
                # --------------------------------------------------------------
                manual_cuts_dict = {}
                counts = -1.0
                while True:
                    counts = run_stabilization(
                        filename=full_path,
                        save_summary_jpeg=False, save_corr_jpeg=False,
                        create_root_file=False,
                        show_canvas=True,
                        manual_cuts=manual_cuts_dict if manual_cuts_dict else None,
                        output_dir=scan_dir,
                        save_partition_jpeg=False,
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
                        print("\n>>> Parameters accepted! Saving JPEG and stabilized ROOT...")
                        # Temporary batch: writing the JPEGs must not pop windows.
                        ROOT.gROOT.SetBatch(True)
                        counts = run_stabilization(
                            filename=full_path,
                            save_summary_jpeg=SAVE_SUMMARY_JPEG,
                            save_corr_jpeg=SAVE_CORR_JPEG,
                            create_root_file=CREATE_ROOT_FILE,
                            show_canvas=False,
                            manual_cuts=manual_cuts_dict if manual_cuts_dict else None,
                            output_dir=scan_dir,
                            save_partition_jpeg=SAVE_PARTITION_JPEG,
                        )
                        ROOT.gROOT.SetBatch(False)
                        break
            else:
                # --------------------------------------------------------------
                # BATCH MODE  (whole folder, empty list, or more than one channel)
                # --------------------------------------------------------------
                counts = run_stabilization(
                    filename=full_path,
                    save_summary_jpeg=SAVE_SUMMARY_JPEG,
                    save_corr_jpeg=SAVE_CORR_JPEG,
                    create_root_file=CREATE_ROOT_FILE,
                    show_canvas=False,
                    output_dir=scan_dir,
                    save_partition_jpeg=SAVE_PARTITION_JPEG,
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
        if SAVE_SUMMARY_JPEG or SAVE_PARTITION_JPEG:
            print(f"  Debug JPEGs       -> {os.path.join(scan_dir, SUMMARY_DIR_NAME)}/ch<N>/")
        if SAVE_CORR_JPEG:
            print(f"  Correlation JPEGs -> {os.path.join(scan_dir, CORR_DIR_NAME)}")
        if CREATE_ROOT_FILE:
            print(f"  Stabilized ROOT   -> {os.path.join(scan_dir, STAB_ROOT_DIR_NAME)}")
        print("=" * 50 + "\n")
    else:
        print("\n>>> No channel processed successfully. <<<\n")