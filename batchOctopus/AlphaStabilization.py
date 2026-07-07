#!/usr/bin/env python3
"""
AlphaStabilization.py  (batch analysis + optional interactive GUI)
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

  5. Select alpha-peak events surviving correlation + LY cuts, clean
     outliers around the peak, and perform a ROBUST pol1 fit of amplitude vs
     baseline (ROOT "ROB=0.90" estimator).

  6. Apply the per-event linear stabilization to the alpha reference energy:

         a_stab_cal = TARGET_ENERGY * amp / (q_0 + slope * baseline)

     Build the calibrated spectrum and measure its resolution (FWHM) from a
     Gaussian fit.

  7. [optional] Write a full copy of the input file to a dedicated folder and
     append the TTree "stabilized_heater_alpha" (one calibrated amplitude
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
  AlphaStabilizationDebug/ch<N>_stabilization_overview.jpg  (4x3 overview)
  CorrelationCut/ch<N>_correlation_cut.jpg                     (correlation cut)
  Stabilized_Output/<stem>_alpha_stabilized.root           (copy + new TTree)
  All output folders are created (if missing) inside BASE_DIR.

Execution
---------
  python3 AlphaStabilization.py                 # uses CHANNELS_TO_PROCESS
  python3 AlphaStabilization.py 24 51           # overrides with channels 24, 51
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
# The alpha LY-peak selection and the alpha-doublet fit are used in every mode.
ANALYSIS_MODE = "mergedrun"     # "mergedrun" or "run"

# mergedrun mode: folder containing the input .root files.
BASE_DIR = "/data/users/azanelli/octopus_work/CROSS/MergedRuns/CorrectedAmp"

# run mode: CROSS folder holding the RUNxxxxxx sub-folders, and the run number.
CROSS_DIR  = "/data/users/azanelli/octopus_work/CROSS"
RUN_NUMBER = 92                 # e.g. 96 -> folder RUN000096, sub-folder Coincidence

# --- Channels to analyse (number after "ch" in the file name) ---------------
#   []          -> process ALL files in BASE_DIR (batch).
#   [N]         -> single channel; GUI if GUI_MANUAL_CUTS is True.
#   [N, M, ...] -> several channels, always batch.
# Command-line channels (if given) OVERRIDE this list.
CHANNELS_TO_PROCESS = [25, 26, 27, 28, 29, 30, 55, 56, 57, 58, 59, 60] #

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
STAB_ROOT_DIR_NAME = os.path.join("..", "AlphaStabilizedAmp")            # stabilized ROOT files

SUMMARY_DIR_NAME   = os.path.join("..", "AlphaStabilizedAmp/AlphaStabilizationDebug")  # 4x3 overview JPEGs
CORR_DIR_NAME      = os.path.join("..", "AlphaStabilizedAmp/CorrelationCut")               # correlation-cut JPEGs
# All per-channel debug JPEGs (baseline partitions, global overview, per-partition
# stabilization, before/after comparison) go to <SUMMARY_DIR>/ch<N>/.

# --- Plotting ---------------------------------------------------------------
# Max number of markers drawn in any overview scatter (TGraph). Rasterising a
# scatter is ~linear in the marker count, so tens of thousands of points make
# the JPEG SaveAs extremely slow in batch mode. The decimation is for DISPLAY
# ONLY -- the analysis and all fits always use the full dataset.
MAX_SCATTER_POINTS = 4000

# --- Calibration-energy window (rough-calibration units) --------------------
# ALPHA region: the source shows TWO close peaks, the alpha-particle line (lower,
# dominant) and the alpha+nuclear-recoil line (higher), ~100 keV apart. On the
# test file (ch26) they sit at rough ~5835 (alpha) and ~5945 (alpha+recoil).
# ONE single analysis window is used for EVERYTHING (conversion factor, light-
# yield study and stabilization; no separate corr/LY/stab ranges). It is placed
# DYNAMICALLY: both doublet peaks are located with _two_peaks (prominence-ranked,
# the same function later used for the doublet fits) in the wide search region
# [CAL_SEARCH_MIN, CAL_SEARCH_MAX], then a CAL_WIN_WIDTH-wide window is centered
# on the MIDPOINT of the two peak positions. Searching for the two peaks --
# instead of the single most prominent one -- avoids the ambiguity of the
# "dominant" peak, which is sometimes the alpha line and sometimes the
# alpha+recoil line. Falls back to a single found peak, then to the center of
# the search region.
CAL_SEARCH_MIN, CAL_SEARCH_MAX = 5200.0, 6200.0  # doublet search region (rough)
CAL_WIN_WIDTH = 400.0           # width of the single analysis window (rough units),
                                # centered on the doublet midpoint: wide enough for
                                # both peaks (~110 keV apart) plus some continuum,
                                # tight enough to exclude other lines
SPEC_DISP_MIN, SPEC_DISP_MAX = 5400.0, 6300.0 # display range of the rough/raw spectra

# --- Alpha reference --------------------------------------------------------
# Which peak of the doublet to stabilize on:
#   True  -> the alpha+recoil (UPPER) peak  [full Q-value line]
#   False -> the alpha-particle (LOWER) peak
STABILIZE_ON_RECOIL = True
# TRUE energy (keV) of the peak selected above -- it fixes the absolute scale.
# Placeholders for 210-Po: alpha-particle line 5304.33, full Q (alpha+recoil)
# 5407.45. CONFIRM the isotope/value.
TARGET_ENERGY = 5407.45   # keV, alpha+recoil (Q-value) line  (CONFIRM)

# --- Thallium cross-check (resolution gain from the ALPHA stabilization) -----
# A SEPARATE, read-only diagnostic (it does NOT alter the stabilization or the
# output ROOT). It replicates the ThalliumStabilization pipeline up to the LY cut
# on the 208-Tl gamma line, then measures the Tl-line resolution BEFORE the alpha
# stabilization (corrected_amplitude, rescaled so the Tl peak sits on its nominal
# energy) and AFTER it (alpha-stabilized amplitude, already in energy). All fits
# use the same optimized-binning Gaussian machinery as the alpha-peak fits.
TL_EVAL_ENABLE   = True
TL_TARGET_ENERGY = 2614.511                        # keV, 208-Tl line nominal energy
# Same rough-calibration windows as ThalliumStabilization.py:
TL_CAL_CORR_MIN, TL_CAL_CORR_MAX = 2400.0, 2700.0  # conversion-factor window (rough)
TL_CAL_LY_MIN,   TL_CAL_LY_MAX   = 2350.0, 2700.0  # light-yield-study window (rough)
TL_CAL_STAB_MIN, TL_CAL_STAB_MAX = 2400.0, 2700.0  # Tl-peak fit window (rough)
# Display range of the Tl spectra (same as ThalliumStabilization.py): kept NARROW
# so the heater peak (well above the Tl line) stays out and the 208-Tl line is the
# dominant peak found inside.
TL_SPEC_DISP_MIN, TL_SPEC_DISP_MAX = 2300.0, 2800.0

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
PART_GAP_HEIGHT_FRAC = 0.05   # a bin belongs to a GAP when its height is below this fraction
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

def _ly_limits_alpha(vals):
    """
    Core-zoomed LY histogram (n_bins, min, max) for ALPHA selection.

    Alphas are heavily quenched, so their light yield is a NARROW spike at LY ~ 0;
    the gamma/beta contamination leaking into the alpha energy window is a sparse
    HIGH-LY tail. Frame the histogram tightly around the alpha core (median +/- K
    robust sigma) so the spike is well resolved for fitting, leaving the tail in
    the overflow. The acceptance cut is applied on the raw LY values afterwards,
    so events in the tail are still rejected.
    """
    if len(vals) == 0:
        return 80, -0.05, 0.15
    arr = np.asarray(vals, dtype=np.float64)
    med = float(np.median(arr))
    rs  = (np.percentile(arr, 75) - np.percentile(arr, 25)) / 1.349
    if rs <= 0:
        rs = (np.percentile(arr, 84) - np.percentile(arr, 16)) / 2.0
    if rs <= 0:
        rs = abs(med) * 0.5 or 1e-4
    K = 10.0
    return 80, float(med - K * rs), float(med + K * rs)


def AnalyzeLightYield(h_ly, name_ext):
    """
    ALPHA-selection LY analysis.

    Alphas cluster at the LOWEST light yield (heavy quenching, LY ~ 0); any gamma/
    beta contamination in the alpha energy window sits at HIGHER LY. So the alpha
    population is the dominant LOW-LY peak: select it and reject the high-LY tail.

    res.fit_Tl holds the selected alpha-peak Gaussian (so the existing drawing/box
    code shows it); res.fit_alpha is left unset; res.df_value is the peak
    resolution |mu|/sigma, used to pick the better light detector (the cleaner,
    tighter alpha core wins). Returns an LYResult.
    """
    res = LYResult()
    print("=" * 50)
    print(f"--- Alpha LY Analysis {h_ly.GetTitle()} ---")

    xMin = h_ly.GetXaxis().GetXmin()
    xMax = h_ly.GetXaxis().GetXmax()
    hist_range = xMax - xMin

    # Alpha peak = dominant LOW-LY peak. With the core-zoomed binning it is the
    # global maximum; refine with the leftmost (lowest-LY) significant TSpectrum
    # peak when several are found.
    peak_x = h_ly.GetXaxis().GetBinCenter(h_ly.GetMaximumBin())
    spec   = ROOT.TSpectrum(10)
    nP     = spec.Search(h_ly, 2, "goff", 0.05)
    if nP > 0:
        xp    = spec.GetPositionX()
        int_w = hist_range * 0.04
        cand  = []
        for i in range(nP):
            xv = xp[i]
            if not math.isfinite(xv): continue
            b0 = h_ly.GetXaxis().FindBin(xv - int_w)
            b1 = h_ly.GetXaxis().FindBin(xv + int_w)
            cand.append((xv, h_ly.Integral(b0, b1)))
        if cand:
            max_area = max(c[1] for c in cand)
            majors   = [c for c in cand if c[1] >= 0.10 * max_area]
            majors.sort(key=lambda c: c[0])           # ascending LY
            peak_x = majors[0][0]                      # leftmost major = alpha

    # --- Gaussian fit of the alpha peak -------------------------------------
    sig0 = h_ly.GetRMS() or hist_range * 0.05
    res.fit_Tl = ROOT.TF1(f"fit_Tl_{name_ext}", "gaus",
                          peak_x - 4.0 * sig0, peak_x + 4.0 * sig0)
    res.fit_Tl.SetLineColor(ROOT.kBlue)
    res.fit_Tl.SetParameters(h_ly.GetBinContent(h_ly.GetXaxis().FindBin(peak_x)),
                             peak_x, sig0 * 0.7)
    h_ly.Fit(res.fit_Tl, "Q0 R")

    mu = res.fit_Tl.GetParameter(1)
    sg = abs(res.fit_Tl.GetParameter(2))

    # Fallback on the histogram moments if the fit is unusable.
    if sg <= 0 or not (xMin <= mu <= xMax):
        mu, sg = h_ly.GetMean(), max(h_ly.GetRMS(), hist_range * 0.05)

    # --- Acceptance window: keep the alpha core, reject the high-LY tail -----
    res.cut_min  = mu - LY_N_SIGMA * sg
    res.cut_max  = mu + LY_N_SIGMA * sg
    # LD-selection metric: resolution of the alpha core (tighter -> higher).
    res.df_value = (abs(mu) / sg) if sg > 0 else -1.0

    print(f"  -> Alpha LY peak mu={mu:.3e}, sigma={sg:.3e}; "
          f"cut [{res.cut_min:.3e}, {res.cut_max:.3e}] (resolution {res.df_value:.2f})")
    return res


def CreateLYBox(res, title_prefix):
    """Build an NDC TPaveText summarising the LY thallium/alpha fit results."""
    pt = ROOT.TPaveText(0.60, 0.55, 0.93, 0.85, "NDC")
    pt.SetFillColor(ROOT.kWhite); pt.SetBorderSize(1)
    pt.SetTextAlign(12); pt.SetTextFont(42); pt.SetTextSize(0.028)
    pt.AddText("")   # top padding: keep text off the box border
    if res.df_value > 0:
        t = pt.AddText(f"{title_prefix} resolution |#mu|/#sigma = {res.df_value:.2f}")
        t.SetTextColor(ROOT.kBlack); t.SetTextFont(62); pt.AddText("")

    t_tl = pt.AddText("ALPHA LY FIT")
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
                 note=None, note_color=ROOT.kBlue, text_size=0.035):
    """
    NDC TPaveText with a Gaussian fit's results: chi2/ndf, mu, sigma, FWHM and
    the percentage resolution (FWHM/mu*100). An optional *note* line is added in
    *note_color* (used to flag that the calibration-run line was applied).
    Returns the TPaveText (a placeholder text is shown when *fit* is None).
    """
    pt = ROOT.TPaveText(x1, y1, x2, y2, "NDC")
    pt.SetFillColor(ROOT.kWhite); pt.SetBorderSize(1)
    pt.SetTextAlign(12); pt.SetTextFont(42); pt.SetTextSize(text_size)

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


def CreateDoubletBox(d, header, header_color=ROOT.kBlack,
                     x1=0.13, y1=0.50, x2=0.52, y2=0.88):
    """
    NDC TPaveText summarising an AlphaDoublet fit: chi2/ndf and, for BOTH peaks
    (alpha-particle and alpha+recoil), mu, sigma, FWHM and percentage resolution
    (FWHM/mu*100). *d* is the AlphaDoublet returned by fit_alpha_doublet.
    """
    pt = ROOT.TPaveText(x1, y1, x2, y2, "NDC")
    pt.SetFillColor(ROOT.kWhite); pt.SetBorderSize(1)
    pt.SetTextAlign(12); pt.SetTextFont(42); pt.SetTextSize(0.030)

    t_h = pt.AddText(header); t_h.SetTextColor(header_color); t_h.SetTextFont(62)
    if d is None or d.fit is None:
        pt.AddText("(fit non disponibile)")
        return pt

    ndf = d.fit.GetNDF()
    if ndf > 0:
        pt.AddText(f"#chi^{{2}}/ndf = {d.fit.GetChisquare():.1f}/{ndf} "
                   f"(P:{d.fit.GetProb():.2f})")

    def peak_lines(name, mu, sigma, col):
        sigma = abs(sigma)
        fwhm  = SIGMA_TO_FWHM * sigma
        t = pt.AddText(name); t.SetTextColor(col); t.SetTextFont(62)
        pt.AddText(f"  #mu = {mu:.2f}   #sigma = {sigma:.2f}")
        res = f"   ({100.0 * fwhm / abs(mu):.2f} %)" if abs(mu) > 1e-12 else ""
        pt.AddText(f"  FWHM = {fwhm:.2f}{res}")

    peak_lines("ALPHA peak", d.mu_a, d.sig_a, ROOT.kBlue + 1)
    if d.mu_r is not None:
        peak_lines("ALPHA+RECOIL peak", d.mu_r, d.sig_r, ROOT.kGreen + 3)
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


def find_dominant_peak(h_heat_orig, search_min=None, search_max=None):
    """
    Locate the dominant peak by scanning a sliding integration window and
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


class AlphaDoublet:
    """Result of fit_alpha_doublet: the optimized histogram, the fit, and the two
    peak positions/widths. The ALPHA-PARTICLE (lower) peak is the stabilization
    reference; the upper peak is the alpha+nuclear-recoil line."""
    def __init__(self):
        self.hist = None;  self.fit = None;  self.ok = False
        self.mu_a = 0.0;   self.sig_a = 0.0          # alpha-particle peak (lower)
        self.mu_r = None;  self.sig_r = 0.0          # alpha+recoil peak (upper)


def _two_peaks(h, search_min=None, search_max=None):
    """
    The two genuine peaks of a histogram (alpha + alpha-recoil), sorted by x
    (lower first), returned as (x, height).

    Ranks candidates by topographic PROMINENCE -- the height of the (lightly
    smoothed) peak above the local background on both sides -- NOT by integrated
    area. Area favours the broad low-energy continuum; prominence keeps true,
    structured peaks (which rise above their surroundings) and ignores flat
    background fluctuations. The second peak is accepted only if it is clearly
    significant, otherwise a single peak is returned.
    """
    nb = h.GetNbinsX()
    ax = h.GetXaxis()
    c  = np.array([h.GetBinContent(i) for i in range(1, nb + 1)], dtype=np.float64)
    # light smoothing so single-bin spikes are not mistaken for peaks
    ker = np.array([1., 2., 3., 2., 1.]); ker /= ker.sum()
    cs  = np.convolve(c, ker, mode="same")

    s   = ROOT.TSpectrum(15)
    npk = s.Search(h, 2, "goff", 0.03)
    xp  = s.GetPositionX()
    W   = max(3, int(nb * 0.04))          # background window ~ a couple of sigma
    cands = []
    for i in range(npk):
        x = xp[i]
        if not math.isfinite(x): continue
        if search_min is not None and x < search_min: continue
        if search_max is not None and x > search_max: continue
        b = ax.FindBin(x) - 1
        if b < 0 or b >= nb: continue
        # snap to the local (smoothed) maximum within +/-2 bins
        lo = max(0, b - 2); hi = min(nb, b + 3)
        b  = lo + int(np.argmax(cs[lo:hi]))
        height = cs[b]
        lbg = np.min(cs[max(0, b - 2 * W):max(1, b - W)]) if b - W > 0 else 0.0
        rbg = np.min(cs[min(nb - 1, b + W):min(nb, b + 2 * W)]) if b + W < nb else 0.0
        prom = height - 0.5 * (lbg + rbg)
        cands.append((ax.GetBinCenter(b + 1), prom, height))

    cands.sort(key=lambda t: t[1], reverse=True)     # by prominence
    keep, mind = [], (ax.GetXmax() - ax.GetXmin()) * 0.05
    for t in cands:
        if t[1] <= 0: continue
        if not any(abs(t[0] - q[0]) < mind for q in keep):
            keep.append(t)
        if len(keep) == 2: break
    # accept the 2nd peak only when clearly significant (else single peak)
    if len(keep) == 2 and keep[1][1] < max(3.0, 0.10 * keep[0][1]):
        keep = keep[:1]
    keep.sort(key=lambda t: t[0])
    return [(t[0], t[2]) for t in keep]


def fit_alpha_doublet(amps, win_lo, win_hi, tag, search_min=None, search_max=None):
    """
    Fit the ALPHA DOUBLET (alpha-particle line + alpha+recoil line) with a double
    Gaussian over a linear background, with binning optimized to the peak width.

    Strategy (mirrors the thallium peak-finding, extended to two peaks):
      1. coarse histogram -> locate the two prominent peaks (TSpectrum, by area);
      2. preliminary single-Gaussian fit of each peak -> seed widths;
      3. rebuild the histogram with bin width ~ sigma/4 (optimized binning);
      4. joint final fit  gaus(0)+gaus(3)+pol1(6)  seeded from the preliminaries.

    The LOWER peak is the alpha-particle line (the stabilization reference).
    Falls back to a single Gaussian (mu_r=None) when only one peak is found.
    Returns an AlphaDoublet. *amps* is the raw-amplitude NumPy array of the events.
    """
    res = AlphaDoublet()
    amps = np.asarray(amps, np.float64)
    amps = amps[np.isfinite(amps) & (amps >= win_lo) & (amps <= win_hi)]
    if amps.size < 1:
        return res

    def build(nb):
        h = ROOT.TH1F(f"h_dbl_{tag}_{nb}", "", nb, win_lo, win_hi)
        h.SetDirectory(0)
        h.FillN(amps.size, amps.astype(np.double), np.ones(amps.size, np.double))
        return h

    # 1. coarse pass + 2. preliminary single fits
    h0    = build(80)
    peaks = _two_peaks(h0, search_min, search_max)

    def prelim(mu):
        sg0 = (win_hi - win_lo) * 0.02
        g = ROOT.TF1(f"g_{tag}", "gaus", mu - 3 * sg0, mu + 3 * sg0)
        g.SetParameters(h0.GetBinContent(h0.GetXaxis().FindBin(mu)), mu, sg0)
        h0.Fit(g, "Q0 R")
        return g.GetParameter(0), g.GetParameter(1), abs(g.GetParameter(2))

    if len(peaks) >= 2:
        A1, m1, s1 = prelim(peaks[0][0])
        A2, m2, s2 = prelim(peaks[1][0])
        sig_ref = max(min(s1, s2), (win_hi - win_lo) * 0.004)
        nb = int(np.clip(round((win_hi - win_lo) / (sig_ref / 4.0)), 40, 300))
        h = build(nb); res.hist = h
        lo, hi = m1 - 3.5 * s1, m2 + 3.5 * s2
        ff = ROOT.TF1(tag, "gaus(0)+gaus(3)+pol1(6)", lo, hi)
        ff.SetParameters(A1, m1, s1, A2, m2, s2,
                         h.GetBinContent(h.GetXaxis().FindBin(lo)), 0.0)
        ff.SetParLimits(2, s1 * 0.3, s1 * 3.0)
        ff.SetParLimits(5, s2 * 0.3, s2 * 3.0)
        h.Fit(ff, "Q0 R")
        res.fit = ff
        res.mu_a, res.sig_a = ff.GetParameter(1), abs(ff.GetParameter(2))
        res.mu_r, res.sig_r = ff.GetParameter(4), abs(ff.GetParameter(5))
        res.ok = True
    else:
        # Single-peak fallback: dominant peak only.
        peak_x, peak_y = find_dominant_peak(h0, search_min, search_max)
        A1, m1, s1 = prelim(peak_x)
        sig_ref = max(s1, (win_hi - win_lo) * 0.004)
        nb = int(np.clip(round((win_hi - win_lo) / (sig_ref / 4.0)), 40, 300))
        h = build(nb); res.hist = h
        ff = ROOT.TF1(tag, "gaus", m1 - 4 * s1, m1 + 4 * s1)
        ff.SetParameters(A1, m1, s1)
        h.Fit(ff, "Q0 R")
        res.fit = ff
        res.mu_a, res.sig_a = ff.GetParameter(1), abs(ff.GetParameter(2))
        res.ok = True

    if res.sig_a <= 0:
        res.mu_a, res.sig_a = float(np.median(amps)), float(np.std(amps)) or 1.0
    return res


def doublet_ref(d):
    """
    (mu, sigma, amplitude) of the doublet peak used as the STABILIZATION reference:
    the alpha+recoil (upper) peak when STABILIZE_ON_RECOIL and it exists, otherwise
    the alpha-particle (lower) peak. The amplitude is the Gaussian height of that
    component (par0 for alpha, par3 for recoil in gaus(0)+gaus(3)+pol1(6)).
    """
    if STABILIZE_ON_RECOIL and d.mu_r is not None and d.sig_r > 0:
        amp = d.fit.GetParameter(3) if d.fit is not None else 0.0
        return d.mu_r, d.sig_r, amp
    amp = d.fit.GetParameter(0) if d.fit is not None else 0.0
    return d.mu_a, d.sig_a, amp


def estimate_alpha_peak(amps, name):
    """
    Robust (center, sigma) of the ALPHA-PARTICLE (lower) peak in an amplitude
    sample, via the alpha-doublet fit. Used on the COMBINED (all-partition)
    spectrum to seed the per-partition search. Returns (0.0, 0.0) when empty.
    """
    amps = np.asarray(amps, np.float64)
    amps = amps[np.isfinite(amps)]
    if amps.size < 1:
        return 0.0, 0.0
    _, lo, hi = calcRobustLimitsAndBins(amps.tolist())
    d = fit_alpha_doublet(amps, lo, hi, f"alphaest_{name}")
    if not d.ok or d.sig_a <= 0:
        return float(np.median(amps)), float(np.std(amps)) or 1.0
    mu, sig, _ = doublet_ref(d)
    return mu, sig


# ===========================================================================
# THALLIUM CROSS-CHECK  (resolution gain from the ALPHA stabilization)
# ===========================================================================
# This is a SEPARATE, read-only diagnostic. It reproduces the ThalliumStabilization
# pipeline up to the LY cut on the 208-Tl gamma line and measures the Tl-line
# resolution BEFORE and AFTER the alpha stabilization. It never changes the alpha
# stabilization or the output ROOT file.

def AnalyzeLightYieldThallium(h_ly, name_ext):
    """
    THALLIUM-selection LY analysis (ported from ThalliumStabilization.py).

    Gamma events (the 208-Tl line) sit at the HIGH light yield; alphas, when
    present, form a lower-LY companion. TSpectrum finds the peaks, ranked by area;
    the two largest well-separated positive peaks are kept, the Tl one being the
    HIGHER-LY of the two. A Gaussian fit gives the Tl mean/sigma and the acceptance
    window [mu - LY_N_SIGMA*sig, mu + LY_N_SIGMA*sig] (lower bound moved to the
    alpha/Tl valley when an alpha peak exists). Returns an LYResult; df_value is the
    alpha-vs-Tl discrimination factor, used to pick the better light detector.
    """
    res = LYResult()
    xMin = h_ly.GetXaxis().GetXmin(); xMax = h_ly.GetXaxis().GetXmax()
    hist_range = xMax - xMin
    fit_window = hist_range * 0.12

    spec   = ROOT.TSpectrum(10)
    nPeaks = spec.Search(h_ly, 2, "goff", 0.02)
    peakTl_X, peakAlpha_X = -999.0, -999.0

    if nPeaks > 0:
        xpeaks_buf = spec.GetPositionX()
        peaks_with_area, int_window = [], hist_range * 0.04
        for i in range(nPeaks):
            x_val = xpeaks_buf[i]
            if x_val <= 0.0: continue
            b0 = h_ly.GetXaxis().FindBin(x_val - int_window)
            b1 = h_ly.GetXaxis().FindBin(x_val + int_window)
            peaks_with_area.append((x_val, h_ly.Integral(b0, b1)))
        if not peaks_with_area:
            peakTl_X = h_ly.GetXaxis().GetBinCenter(h_ly.GetMaximumBin())
        else:
            peaks_with_area.sort(key=lambda p: p[1], reverse=True)
            valid_peaks, min_dist = [], hist_range * 0.08
            for p in peaks_with_area:
                if not any(abs(p[0] - vp[0]) < min_dist for vp in valid_peaks):
                    valid_peaks.append(p)
                if len(valid_peaks) == 2: break
            valid_peaks.sort(key=lambda p: p[0])          # ascending LY
            if len(valid_peaks) == 2:
                peakAlpha_X = valid_peaks[0][0]           # lower LY = alpha
                peakTl_X    = valid_peaks[1][0]           # higher LY = thallium
            else:
                peakTl_X = valid_peaks[0][0]
    else:
        peakTl_X = h_ly.GetXaxis().GetBinCenter(h_ly.GetMaximumBin())

    # --- Thallium Gaussian fit ----------------------------------------------
    fit_Tl_min, fit_Tl_max = peakTl_X - fit_window, peakTl_X + fit_window
    if peakAlpha_X != -999.0:
        dist = peakTl_X - peakAlpha_X
        if dist < fit_window * 1.5:
            fit_Tl_min = peakTl_X - dist * 0.4
    res.fit_Tl = ROOT.TF1(f"fit_Tl_{name_ext}", "gaus", fit_Tl_min, fit_Tl_max)
    res.fit_Tl.SetLineColor(ROOT.kBlue)
    res.fit_Tl.SetParameters(h_ly.GetBinContent(h_ly.GetXaxis().FindBin(peakTl_X)),
                             peakTl_X, (fit_Tl_max - fit_Tl_min) / 4.0)
    h_ly.Fit(res.fit_Tl, "Q0 R L")
    mean_Tl, sigma_Tl = res.fit_Tl.GetParameter(1), abs(res.fit_Tl.GetParameter(2))
    mean_alpha, sigma_alpha = -999.0, 0.0

    # --- Alpha Gaussian fit + discrimination factor -------------------------
    if peakAlpha_X != -999.0:
        fit_alpha_min, fit_alpha_max = peakAlpha_X - fit_window, peakAlpha_X + fit_window
        dist = peakTl_X - peakAlpha_X
        if dist < fit_window * 1.5:
            fit_alpha_max = peakAlpha_X + dist * 0.4
        res.fit_alpha = ROOT.TF1(f"fit_alpha_{name_ext}", "gaus", fit_alpha_min, fit_alpha_max)
        res.fit_alpha.SetLineColor(ROOT.kGreen + 2)
        res.fit_alpha.SetParameters(h_ly.GetBinContent(h_ly.GetXaxis().FindBin(peakAlpha_X)),
                                    peakAlpha_X, (fit_alpha_max - fit_alpha_min) / 4.0)
        h_ly.Fit(res.fit_alpha, "Q0+ R L")
        mean_alpha, sigma_alpha = res.fit_alpha.GetParameter(1), abs(res.fit_alpha.GetParameter(2))
        sigma_tot = math.sqrt(sigma_Tl**2 + sigma_alpha**2)
        res.df_value = abs(mean_Tl - mean_alpha) / sigma_tot if sigma_tot > 0 else -1.0
    else:
        res.df_value = (abs(mean_Tl) / sigma_Tl) if sigma_Tl > 0 else -1.0

    # --- Acceptance window --------------------------------------------------
    res.cut_max      = mean_Tl + LY_N_SIGMA * sigma_Tl
    standard_cut_min = mean_Tl - LY_N_SIGMA * sigma_Tl
    if peakAlpha_X != -999.0 and res.fit_alpha is not None and (sigma_alpha + sigma_Tl) > 0:
        valley_point = (mean_alpha * sigma_Tl + mean_Tl * sigma_alpha) / (sigma_alpha + sigma_Tl)
        valley_point = min(valley_point, mean_Tl - 1.5 * sigma_Tl)
        res.cut_min  = max(valley_point, standard_cut_min)
    else:
        res.cut_min = standard_cut_min
    return res


def fit_peak_optimized(amps, win_lo, win_hi, tag, center_hint=None):
    """
    Single-peak Gaussian fit with binning optimized to the peak width, using the
    SAME machinery as the alpha-doublet fitter (coarse histogram -> dominant peak
    -> preliminary Gaussian -> rebuild at bin width ~ sigma/4 -> final Gaussian).
    *center_hint*, when given, restricts the peak search to +/-15% of the range
    around it. Returns (hist, fit) or (None, None) for a too-small sample.
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
    peak_x, _ = find_dominant_peak(h0, s_min, s_max)

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
    Robust sigma of the peak in *energies* (already rescaled so the peak sits on
    *center*), via the optimized single-peak fitter. Returns the sigma or None.
    Used to size the SHARED before/after axis of the Tl comparison.
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
    FINAL Tl-peak fit for the before/after comparison.

    Histograms *energies* on the GIVEN shared axis (lo, hi, nb bins) -- so the
    before and after spectra use the SAME range and binning -- and fits the Tl peak
    with a Gaussian over a LINEAR background (gaus(0)+pol1(3)), the same peak +
    continuum model as the alpha-peak fits. Seeded at *center* with width
    *sig_seed*. The Gaussian parameters stay at indices 0,1,2, so CreateFitBox
    reads mu/sigma directly. Returns (hist, fit) or (None, None).
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
    ff.SetParLimits(2, seed * 0.2, seed * 3.0)
    # "L": Poisson maximum-likelihood fit -- the correct method for a low-statistics
    # histogram, natively accounting for empty bins (no need to weight them by hand).
    h.Fit(ff, "Q0 R L")
    return h, ff


def evaluate_thallium_resolution(ch_id, ha, cal_rough, baseline, ld1_ly, ld2_ly,
                                 mask_main, apply_ly_cut, part_of_event, part_results,
                                 h_corr, corr_cut_dynamic, alpha_target_energy,
                                 debug_ch_dir, calib_suffix, save_jpeg):
    """
    Measure the 208-Tl line resolution BEFORE and AFTER the alpha stabilization.

    Pipeline (mirrors ThalliumStabilization.py up to the LY cut):
      1. conversion factor from the Tl rough window;
      2. thallium-style LY analysis -> clean Tl gamma events;
      3. BEFORE: corrected_amplitude of the Tl gammas -> find & fit the Tl peak
         (alpha-style optimized Gaussian), rescale so the peak lands on
         TL_TARGET_ENERGY, refit -> resolution + FWHM;
      4. AFTER: the SAME events, alpha-stabilized (already in energy) -> fit.

    Writes two debug JPEGs (Tl global overview + before/after comparison, each box
    reporting resolution and FWHM). Returns a results dict (or None if it bailed).
    """
    keep = []   # keep ROOT objects alive until the SaveAs calls are done

    # (1) conversion factor (raw/rough) in the Tl window ---------------------
    m_conv = (mask_main & (cal_rough > TL_CAL_CORR_MIN) & (cal_rough < TL_CAL_CORR_MAX)
              & (cal_rough > 0))
    conv_tl = float(np.median(ha[m_conv] / cal_rough[m_conv])) if int(m_conv.sum()) > 0 else 1.0

    # (2) thallium-style LY cut ----------------------------------------------
    h_ly1 = h_ly2 = None
    res1 = res2 = None
    chosen_ld, ly_lo, ly_hi = 1, None, None
    if apply_ly_cut:
        m_ly = mask_main & (cal_rough > TL_CAL_LY_MIN) & (cal_rough < TL_CAL_LY_MAX)
        def _ly_hist(ly_arr, name):
            vals = ly_arr[m_ly & np.isfinite(ly_arr)]
            if vals.size < 20:
                return None, None
            nb, lo, hi = calcRobustLimitsAndBins(vals.tolist())
            h = ROOT.TH1F(name, f"Ch {ch_id}: {name};Light Yield;Counts", nb, lo, hi)
            h.SetDirectory(0)
            h.FillN(vals.size, vals.astype(np.double), np.ones(vals.size, np.double))
            return h, AnalyzeLightYieldThallium(h, name)
        h_ly1, res1 = _ly_hist(ld1_ly, f"Tl_LD1_{ch_id}")
        h_ly2, res2 = _ly_hist(ld2_ly, f"Tl_LD2_{ch_id}")
        df1 = res1.df_value if res1 else -1.0
        df2 = res2.df_value if res2 else -1.0
        chosen_ld  = 2 if df2 > df1 else 1
        chosen_res = res2 if chosen_ld == 2 else res1
        if chosen_res is not None and chosen_res.cut_max > chosen_res.cut_min:
            ly_lo, ly_hi = chosen_res.cut_min, chosen_res.cut_max
        print(f">>> Tl cross-check: LY cut LD{chosen_ld} "
              f"[{ly_lo if ly_lo is not None else float('nan'):.4f}, "
              f"{ly_hi if ly_hi is not None else float('nan'):.4f}]")

    ly_sel = ld2_ly if chosen_ld == 2 else ld1_ly
    if ly_lo is not None:
        mask_gamma = mask_main & np.isfinite(ly_sel) & (ly_sel >= ly_lo) & (ly_sel <= ly_hi)
    else:
        mask_gamma = mask_main

    # Tl gamma events inside the Tl stabilization window (raw amplitude) ------
    stab_lo_raw, stab_hi_raw = TL_CAL_STAB_MIN * conv_tl, TL_CAL_STAB_MAX * conv_tl
    mask_tl = mask_gamma & (ha > stab_lo_raw) & (ha < stab_hi_raw)
    n_tl = int(mask_tl.sum())
    if n_tl < 20:
        print(f"  [!] Ch {ch_id}: too few Tl gamma events ({n_tl}); Tl cross-check skipped.")
        return None
    amps_tl = ha[mask_tl]
    print(f">>> Tl cross-check: {n_tl} Tl gamma events in raw window "
          f"[{stab_lo_raw:.0f}, {stab_hi_raw:.0f}] (conv {conv_tl:.4f}).")

    # Both the BEFORE and the AFTER spectra are rescaled to the NOMINAL Tl energy
    # before the final fit: a single alpha-point calibration through the origin is
    # not enough (the energy-amplitude relation is non-linear), so the Tl line does
    # NOT land exactly on 2614.5 keV in either spectrum. We therefore locate the Tl
    # peak, rescale so it sits on TL_TARGET_ENERGY, and only then measure its width.
    def _rescale_to_nominal(vals, tag):
        """Find the Tl peak in *vals*, return vals rescaled so the peak -> nominal."""
        _, lo, hi = calcRobustLimitsAndBins(vals.tolist())
        _, f0 = fit_peak_optimized(vals, lo, hi, tag)
        mu0 = f0.GetParameter(1) if f0 is not None else 0.0
        if not (mu0 and mu0 > 0):
            mu0 = float(np.median(vals))
        return TL_TARGET_ENERGY * vals / mu0

    # (3) BEFORE: corrected_amplitude -> rescale to nominal energy --------------
    before_e = _rescale_to_nominal(amps_tl, f"tlb0_{ch_id}")

    # (4) AFTER: SAME events, alpha-stabilized amplitude -> rescale to nominal ---
    #     (same procedure as BEFORE: the single alpha-point calibration is not
    #     enough, the energy-amplitude relation being non-linear).
    slopes = np.array([r.slope for r in part_results], dtype=np.float64)
    q0s    = np.array([r.q_0   for r in part_results], dtype=np.float64)
    pe     = part_of_event[mask_tl]
    with np.errstate(divide='ignore', invalid='ignore'):
        denom       = q0s[pe] + slopes[pe] * baseline[mask_tl]
        after_stab  = alpha_target_energy * ha[mask_tl] / denom
    after_stab = after_stab[np.isfinite(after_stab)]
    after_e = _rescale_to_nominal(after_stab, f"tla0_{ch_id}") if after_stab.size >= 5 else None

    # SHARED axis (equal range + binning) so the before/after peaks are directly
    # comparable: window = nominal +/- 5*sigma_ref, bin width ~ sigma_ref/3, where
    # sigma_ref is the BROADER of the two peaks (so both fit inside).
    sig_b = estimate_peak_sigma(before_e, TL_TARGET_ENERGY, f"tlb_{ch_id}")
    sig_a = estimate_peak_sigma(after_e,  TL_TARGET_ENERGY, f"tla_{ch_id}") if after_e is not None else None
    sig_ref = max([s for s in (sig_b, sig_a) if s] or [15.0])
    W       = 3.5 * sig_ref                                  # smaller window
    tl_lo, tl_hi = TL_TARGET_ENERGY - W, TL_TARGET_ENERGY + W
    tl_nb   = int(np.clip(round((tl_hi - tl_lo) / max(sig_ref / 6.0, 1e-9)), 12, 200))  # smaller bins

    h_before, fit_before = fit_thallium_peak(before_e, TL_TARGET_ENERGY, tl_lo, tl_hi,
                                             tl_nb, sig_b or sig_ref, f"tlbefore_{ch_id}")
    h_after = fit_after = None
    if after_e is not None:
        h_after, fit_after = fit_thallium_peak(after_e, TL_TARGET_ENERGY, tl_lo, tl_hi,
                                               tl_nb, sig_a or sig_ref, f"tlafter_{ch_id}")

    def _summary(fit, label):
        if fit is None or abs(fit.GetParameter(1)) < 1e-9:
            return None
        mu, sg = fit.GetParameter(1), abs(fit.GetParameter(2))
        fwhm   = SIGMA_TO_FWHM * sg
        pct    = 100.0 * fwhm / abs(mu)
        print(f">>> Tl {label}: mu={mu:.1f} keV, FWHM={fwhm:.2f} keV ({pct:.2f} %)")
        return dict(mu=mu, sigma=sg, fwhm=fwhm, res_pct=pct)

    r_before = _summary(fit_before, "BEFORE alpha stab.")
    r_after  = _summary(fit_after,  "AFTER  alpha stab.")

    if not save_jpeg:
        return dict(before=r_before, after=r_after, conv_tl=conv_tl, n_tl=n_tl)

    os.makedirs(debug_ch_dir, exist_ok=True)

    # ---- IMAGE 1: thallium global overview (6 pads) ------------------------
    c1 = ROOT.TCanvas(f"c_tl_glob_{ch_id}", f"Tl Overview Ch {ch_id}", 500 * 3, 400 * 2)
    c1.Divide(3, 2)

    # pad 1: correlation distribution + cut
    c1.cd(1); ROOT.gPad.SetGrid()
    h_corr.Draw(); c1.Update()
    l = ROOT.TLine(corr_cut_dynamic, ROOT.gPad.GetUymin(), corr_cut_dynamic, ROOT.gPad.GetUymax())
    l.SetLineColor(ROOT.kRed); l.SetLineWidth(2); l.SetLineStyle(2); l.Draw("same"); keep.append(l)

    # pad 2: rough spectrum in the Tl region + the three windows. The display is
    # kept NARROW (TL_SPEC_DISP) so the heater peak (well above the Tl line) stays
    # out and the 208-Tl line is the dominant peak shown.
    c1.cd(2); ROOT.gPad.SetGrid()
    rlo, rhi = TL_SPEC_DISP_MIN, TL_SPEC_DISP_MAX
    m2 = mask_main & (cal_rough > rlo) & (cal_rough < rhi)
    hR = ROOT.TH1F(f"h_tlrough_{ch_id}",
                   f"Ch {ch_id}: Tl region (rough) after corr cut;Rough amplitude;Counts",
                   80, rlo, rhi); hR.SetDirectory(0)
    vR = cal_rough[m2]
    if vR.size: hR.FillN(vR.size, vR.astype(np.double), np.ones(vR.size, np.double))
    hR.SetLineColor(ROOT.kBlack); hR.SetFillColorAlpha(ROOT.kGray, 0.5); hR.Draw(); c1.Update()
    for a, b, col, st in [(TL_CAL_LY_MIN, TL_CAL_LY_MAX, ROOT.kBlue, 2),
                          (TL_CAL_CORR_MIN, TL_CAL_CORR_MAX, ROOT.kRed, 1),
                          (TL_CAL_STAB_MIN, TL_CAL_STAB_MAX, ROOT.kGreen + 2, 3)]:
        for xv in (a, b):
            ln = ROOT.TLine(xv, 0, xv, ROOT.gPad.GetUymax())
            ln.SetLineColor(col); ln.SetLineWidth(2); ln.SetLineStyle(st); ln.Draw("same"); keep.append(ln)
    keep.append(hR)

    # pad 3: raw spectrum near Tl (after corr cut)
    c1.cd(3); ROOT.gPad.SetGrid()
    hraw = ROOT.TH1F(f"h_tlraw_{ch_id}",
                     f"Ch {ch_id}: Tl region (raw) after corr cut;Raw amplitude;Counts",
                     80, rlo * conv_tl, rhi * conv_tl); hraw.SetDirectory(0)
    vRaw = ha[m2]
    if vRaw.size: hraw.FillN(vRaw.size, vRaw.astype(np.double), np.ones(vRaw.size, np.double))
    hraw.SetLineColor(ROOT.kBlack); hraw.SetFillColorAlpha(ROOT.kOrange + 1, 0.5); hraw.Draw()
    keep.append(hraw)

    # pads 4,5: LD1 / LD2 light yield + Tl fit + cut lines
    for pad, hh, rr, tag in [(4, h_ly1, res1, "LD1"), (5, h_ly2, res2, "LD2")]:
        c1.cd(pad); ROOT.gPad.SetGrid()
        if hh is not None and hh.GetEntries() > 0:
            hh.SetStats(0); hh.SetLineColor(ROOT.kRed); hh.SetFillColorAlpha(ROOT.kRed, 0.3); hh.Draw()
            if rr and rr.fit_Tl:    rr.fit_Tl.Draw("same")
            if rr and rr.fit_alpha: rr.fit_alpha.Draw("same")
            c1.Update()
            if rr and rr.cut_max > rr.cut_min:
                for xv in (rr.cut_min, rr.cut_max):
                    lc = ROOT.TLine(xv, 0, xv, ROOT.gPad.GetUymax())
                    lc.SetLineColor(ROOT.kBlue); lc.SetLineWidth(2); lc.SetLineStyle(2)
                    lc.Draw("same"); keep.append(lc)
            box = CreateLYBox(rr, tag) if (rr and rr.fit_Tl) else None
            if box: box.Draw(); keep.append(box)
            keep.append(hh)

    # pad 6: Tl raw spectrum after correlation + LY cut (selected gammas)
    c1.cd(6); ROOT.gPad.SetGrid()
    hsel = ROOT.TH1F(f"h_tlsel_{ch_id}",
                     f"Ch {ch_id}: Tl raw after corr + LY cut;Raw amplitude;Counts",
                     80, stab_lo_raw, stab_hi_raw); hsel.SetDirectory(0)
    hsel.FillN(amps_tl.size, amps_tl.astype(np.double), np.ones(amps_tl.size, np.double))
    hsel.SetLineColor(ROOT.kBlack); hsel.SetFillColorAlpha(ROOT.kGreen + 1, 0.5); hsel.Draw()
    keep.append(hsel)

    c1.Update()
    save_canvas_jpeg(c1, os.path.join(debug_ch_dir,
                     f"ch{ch_id}_thallium_overview{calib_suffix}.jpg"))

    # ---- IMAGE 2: before/after comparison (2 pads) -------------------------
    c2 = ROOT.TCanvas(f"c_tl_cmp_{ch_id}", f"Tl before/after Ch {ch_id}", 1400, 600)
    c2.Divide(2, 1)

    c2.cd(1); ROOT.gPad.SetGrid()
    if h_before is not None:
        h_before.SetStats(0)
        h_before.SetTitle(f"Ch {ch_id}: Tl BEFORE alpha stab. (rescaled);Energy (keV);Counts")
        h_before.SetLineColor(ROOT.kBlack); h_before.SetFillColorAlpha(ROOT.kGray, 0.5); h_before.Draw()
        if fit_before: fit_before.SetLineColor(ROOT.kBlue); fit_before.Draw("same")
        box_b = CreateFitBox(fit_before, "Tl BEFORE alpha stab.", ROOT.kGray + 2,
                             x1=0.14, y1=0.68, x2=0.42, y2=0.88, text_size=0.026)
        box_b.Draw(); keep.append(box_b); keep.append(h_before)

    c2.cd(2); ROOT.gPad.SetGrid()
    if h_after is not None:
        h_after.SetStats(0)
        h_after.SetTitle(f"Ch {ch_id}: Tl AFTER alpha stab.;Energy (keV);Counts")
        h_after.SetLineColor(ROOT.kBlack); h_after.SetFillColorAlpha(ROOT.kRed, 0.4); h_after.Draw()
        if fit_after: fit_after.SetLineColor(ROOT.kBlue); fit_after.Draw("same")
        box_a = CreateFitBox(fit_after, "Tl AFTER alpha stab.", ROOT.kRed + 1,
                             x1=0.14, y1=0.68, x2=0.42, y2=0.88, text_size=0.026)
        box_a.Draw(); keep.append(box_a); keep.append(h_after)

    c2.Update()
    save_canvas_jpeg(c2, os.path.join(debug_ch_dir,
                     f"ch{ch_id}_thallium_before_after{calib_suffix}.jpg"))

    return dict(before=r_before, after=r_after, conv_tl=conv_tl, n_tl=n_tl)


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


def robust_line(bases, amps, mad_trim=5.0, max_pairs_points=700):
    """
    Robust linear fit  amplitude = q0 + slope * baseline  that represents the BULK
    of the points and is immune to high-leverage BASELINE outliers (an isolated
    point far from the baseline cluster drags an ordinary or residual-trimmed fit
    toward itself precisely because of its leverage).

    Steps: (1) drop points whose baseline is more than *mad_trim* MADs from the
    baseline median (leverage outliers) -- for the FIT only, the events stay in
    the spectrum; (2) Theil-Sen estimate: slope = median of all pairwise slopes,
    q0 = median(a - slope*b). Parameter-free, ~29% breakdown.
    Returns (q0, slope, trimmed_mask); trimmed_mask is over the ORIGINAL input.
    """
    b0 = np.asarray(bases, np.float64); a0 = np.asarray(amps, np.float64)
    trimmed = np.zeros(b0.size, dtype=bool)     # True = leverage outlier (not fitted)
    ok = np.isfinite(b0) & np.isfinite(a0)
    b, a = b0[ok], a0[ok]
    if b.size < 2:
        return (float(np.median(a)) if a.size else 0.0), 0.0, trimmed
    bmed = np.median(b); mad = np.median(np.abs(b - bmed)) * 1.4826
    keep = np.ones(b.size, dtype=bool)
    if mad > 0:
        cand = np.abs(b - bmed) <= mad_trim * mad
        if cand.sum() >= max(4, int(0.5 * b.size)):
            keep = cand
    # record the trimmed leverage outliers on the ORIGINAL input (for the scatter)
    idx_ok = np.flatnonzero(ok)
    trimmed[idx_ok[~keep]] = True
    b, a = b[keep], a[keep]
    # subsample when large: the pairwise-slope set is O(n^2)
    if b.size > max_pairs_points:
        idx = np.random.default_rng(12345).choice(b.size, max_pairs_points, replace=False)
        bs, as_ = b[idx], a[idx]
    else:
        bs, as_ = b, a
    i, j = np.triu_indices(bs.size, 1)
    db = bs[j] - bs[i]; m = np.abs(db) > 0
    if not m.any():
        return float(np.median(a)), 0.0, trimmed
    slope = float(np.median((as_[j] - as_[i])[m] / db[m]))
    q0    = float(np.median(a - slope * b))
    return q0, slope, trimmed


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
        self.mu_recoil       = None    # alpha+recoil (upper) peak position, if found
        self.peak_x          = 0.0     # preliminary thallium-peak amplitude
        self.peak_sigma      = 0.0     # preliminary thallium-peak sigma
        self.a_stab_cal      = np.array([], dtype=np.float64)  # calibrated AFTER stabilization
        self.clean_amps      = np.array([], dtype=np.float64)  # clean alpha amplitudes BEFORE stabilization
        self.clean_bases     = np.array([], dtype=np.float64)  # baselines of the clean alpha events
        self.all_amps        = np.array([], dtype=np.float64)  # ALL stab-window amps (full doublet, raw)
        self.all_bases       = np.array([], dtype=np.float64)  # matching baselines (for the stabilized doublet)
        self.doublet_before  = None    # per-partition AlphaDoublet BEFORE stab (rescaled to energy)
        self.doublet_after   = None    # per-partition AlphaDoublet AFTER stab (stabilized energy)
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
        self.g_heat_vs_base_trim  = None   # leverage outliers excluded from the line
        self.fit_brange           = None   # (min,max) baseline of the points used in the line
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
    interval: thallium-peak detection (find_dominant_peak), outlier cleaning,
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

    # --- alpha-doublet fit (locate the alpha-particle peak in this partition) --
    bins_heat, heat_min, heat_max = calcRobustLimitsAndBins(amps_for_stab.tolist())
    n_stab = len(amps_for_stab)

    # Double-Gaussian (+ linear background) fit of the alpha doublet with binning
    # optimized to the peak width. h_heat_orig is the optimized histogram; the
    # ALPHA-PARTICLE (lower) peak is the stabilization reference. The peak search
    # is NOT restricted to the hint window: the alpha+recoil peak sits ABOVE the
    # alpha line and a tight window would clip it (the prominence finder already
    # rejects background fluctuations). The stab window bounds the doublet region.
    doublet = fit_alpha_doublet(amps_for_stab, heat_min, heat_max, f"fit_prelim_{tag}")
    res.h_heat_orig = doublet.hist
    res.h_heat_orig.SetTitle(
        f"Ch {ch_id} P{idx} [{blo:.3f},{bhi:.3f}]: Alpha doublet after LY cut - Pre-cleaning;Amplitude;Counts")
    res.fit_prelim = doublet.fit

    mean_heat_prelim, sigma_heat_prelim, _ = doublet_ref(doublet)   # reference peak
    res.mu_recoil     = doublet.mu_r          # alpha+recoil peak (None if single)
    if sigma_heat_prelim <= 0:
        mean_heat_prelim  = float(np.median(amps_for_stab))
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
        f"Ch {ch_id} P{idx}: Alpha Peak BEFORE stabilization;Amplitude;Counts",
        params_clean.bins, params_clean.vis_min, params_clean.vis_max)
    res.h_heat_stab = ROOT.TH1F(
        f"h_heat_stab_{tag}",
        f"Ch {ch_id} P{idx}: Alpha Peak AFTER stabilization;Amplitude;Counts",
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

    # Keep ALL stab-window events (full doublet: alpha + recoil) and baselines,
    # so the combined BEFORE/AFTER image can fit BOTH peaks (clean_amps holds only
    # the alpha-only core used for the stabilization reference).
    res.all_amps  = amps_for_stab
    res.all_bases = bases_for_stab

    # --- robust pol1 fit: ALWAYS computed (shown in red for comparison) -----
    fit_slope, fit_q0 = 0.0, res.mean_amp_clean
    if res.g_heat_vs_base_clean.GetN() > 3:
        # Robust (Theil-Sen + leverage-outlier trim) line: represents the bulk of
        # the points and is not dragged by a single baseline outlier outside the
        # cluster (unlike a residual-trimmed rob= fit). The leverage outliers are
        # excluded from the LINE ONLY (they are still stabilized as normal events);
        # they are drawn in a distinct colour on the scatter.
        fit_q0, fit_slope, trim = robust_line(clean_bases_np, clean_amps_np)
        res.f1 = ROOT.TF1(f"f1_{tag}", "pol1")
        res.f1.SetParameters(fit_q0, fit_slope)
        kept = ~trim
        if kept.any():
            res.fit_brange = (float(clean_bases_np[kept].min()),
                              float(clean_bases_np[kept].max()))
        n_trim = int(trim.sum())
        if n_trim > 0:
            res.g_heat_vs_base_trim = ROOT.TGraph(
                n_trim, clean_bases_np[trim].astype(np.double),
                clean_amps_np[trim].astype(np.double))
            res.g_heat_vs_base_trim.SetName(f"g_heat_vs_base_trim_{tag}")

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
        f"Ch {ch_id} P{idx}: Calibrated & Stabilized Alpha Peak;Energy (keV);Counts",
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

    # The single analysis window (conversion factor + LY + stabilization) is set
    # dynamically after the correlation cut -- see the ANALYSIS WINDOW block.

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
    # ANALYSIS WINDOW  (single window: conversion factor + LY + stabilization)
    # ======================================================================
    # Locate BOTH doublet peaks (alpha-particle + alpha+recoil) on the ROUGH
    # spectrum after the correlation cut, searching the wide region
    # [CAL_SEARCH_MIN, CAL_SEARCH_MAX] with _two_peaks (prominence-ranked, the
    # same function later used for the doublet fits), then center the single
    # CAL_WIN_WIDTH-wide window on the MIDPOINT of the two peak positions.
    mask_main = corr > corr_cut_dynamic
    win_center = 0.5 * (CAL_SEARCH_MIN + CAL_SEARCH_MAX)     # last-resort fallback
    rough_search = cal_rough[mask_main & (cal_rough > CAL_SEARCH_MIN)
                             & (cal_rough < CAL_SEARCH_MAX)]
    if rough_search.size > 20:
        nb_s = int(np.clip((CAL_SEARCH_MAX - CAL_SEARCH_MIN) / 5.0, 100, 800))
        h_s = ROOT.TH1F(f"h_calwin_{ch_id}", "", nb_s, CAL_SEARCH_MIN, CAL_SEARCH_MAX)
        h_s.SetDirectory(0)
        h_s.FillN(rough_search.size, rough_search.astype(np.double),
                  np.ones(rough_search.size, np.double))
        peaks_win = _two_peaks(h_s)                  # both doublet peaks, sorted by x
        if len(peaks_win) >= 2:
            p_lo, p_hi = peaks_win[0][0], peaks_win[1][0]
            win_center = 0.5 * (p_lo + p_hi)         # midpoint of the doublet
            print(f">>> Analysis window centered on the doublet midpoint "
                  f"{win_center:.0f} rough (peaks {p_lo:.0f}, {p_hi:.0f}).")
        elif len(peaks_win) == 1:
            win_center = peaks_win[0][0]
            print(f">>> Only one peak found; analysis window centered on it "
                  f"({win_center:.0f} rough).")
        else:
            win_center, _ = find_dominant_peak(h_s)  # no TSpectrum peak -> tallest bin
            print(f">>> No TSpectrum peak; analysis window centered on the dominant "
                  f"bin ({win_center:.0f} rough).")
    else:
        print(f"  [!] Too few events in the search region; analysis window centered "
              f"on the search-region midpoint ({win_center:.0f} rough).")
    cal_win_min = win_center - 0.5 * CAL_WIN_WIDTH
    cal_win_max = win_center + 0.5 * CAL_WIN_WIDTH
    print(f">>> Analysis window (corr + LY + stab): "
          f"[{cal_win_min:.0f}, {cal_win_max:.0f}] rough.")

    # ======================================================================
    # CONVERSION FACTOR  (vectorised)
    # ======================================================================
    mask_conv = mask_main & (cal_rough > cal_win_min) & (cal_rough < cal_win_max) & (cal_rough > 0)
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
                             80, SPEC_DISP_MIN, SPEC_DISP_MAX)
    if n_main > 0:
        h_cal_rough.FillN(n_main, cal_rough[mask_main].astype(np.double), _w_main)

    h_raw = ROOT.TH1F(f"h_raw_{ch_id}",
                       f"Ch {ch_id}: Heater Stabilized after Correlation Cut; Amplitude; Counts",
                       80, SPEC_DISP_MIN * conv_raw, SPEC_DISP_MAX * conv_raw)
    if n_main > 0:
        h_raw.FillN(n_main, ha[mask_main].astype(np.double), _w_main)

    h2_full = ROOT.TH1F(f"h2_full_{ch_id}",
                         f"Ch {ch_id}: Heater Stabilized after Correlation and LY Cut; Amplitude; Counts",
                         80, SPEC_DISP_MIN * conv_factor, SPEC_DISP_MAX * conv_factor)
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
        mask_ly_range = mask_main & (ha > cal_win_min * conv_factor) & (ha < cal_win_max * conv_factor)

        ly1_all = ld1_ly
        ly2_all = ld2_ly

        mask_ld1_ly = mask_ly_range & np.isfinite(ly1_all)
        mask_ld2_ly = mask_ly_range & np.isfinite(ly2_all) if has_ld2_branch else np.zeros(N, bool)

        vals_ly1_np = ly1_all[mask_ld1_ly];  heat_ly1_np = ha[mask_ld1_ly]
        vals_ly2_np = ly2_all[mask_ld2_ly];  heat_ly2_np = ha[mask_ld2_ly]
        # ALPHA mode: zoom the LY histogram to the (narrow, low-LY) alpha core.
        bins_ly1, ly1_min, ly1_max = _ly_limits_alpha(vals_ly1_np.tolist())
        bins_ly2, ly2_min, ly2_max = _ly_limits_alpha(vals_ly2_np.tolist())

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

        # ALPHA mode: select the low-LY alpha peak on each light detector.
        res1 = AnalyzeLightYield(h_ly1, f"LD1_{ch_id}") if h_ly1.GetEntries() > 0 else LYResult()
        res2 = AnalyzeLightYield(h_ly2, f"LD2_{ch_id}") if h_ly2.GetEntries() > 0 else LYResult()

        # Pick the LD whose alpha core is best resolved (higher |mu|/sigma).
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
    # The alpha-peak detection, outlier cleaning, robust linear baseline fit
    # and the stabilization itself are run SEPARATELY on each baseline
    # partition (the amplitude-vs-baseline slope differs between the clusters).
    # The stabilization window IS the single analysis window (found dynamically
    # from the doublet midpoint -- see the ANALYSIS WINDOW block), converted to
    # raw-amplitude units.
    stab_lo = cal_win_min * conv_factor
    stab_hi = cal_win_max * conv_factor
    mask_stab_window = (ha > stab_lo) & (ha < stab_hi)

    # Assign every event to a partition from its baseline. The internal valley
    # edges tile the baseline axis; searchsorted clamps events outside the
    # tiled range onto the end partitions, so coverage is complete.
    internal_edges = np.array([iv[1] for iv in partition_intervals[:-1]],
                              dtype=np.float64)
    if internal_edges.size > 0:
        part_of_event = np.searchsorted(internal_edges, baseline, side="right")
    else:
        part_of_event = np.zeros(N, dtype=np.int64)

    # Alpha-peak search hint from the COMBINED spectrum (all partitions merged):
    # more robust than any single partition, and shared by all.
    amps_all_stab           = amp_for_analysis[mask_ly_pass & mask_stab_window]
    hint_center, hint_sigma = estimate_alpha_peak(amps_all_stab, f"{ch_id}_comb")
    peak_hint = (hint_center, hint_sigma) if hint_center != 0.0 else None
    if peak_hint is not None:
        print(f">>> Combined-spectrum alpha peak hint at amplitude {hint_center:.1f} "
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

    # Per-partition alpha-doublet fits, BEFORE (rescaled to energy on the
    # partition's own alpha mean) and AFTER (per-event stabilized), so the
    # per-partition figure shows BOTH peaks with the same double-Gaussian fit and
    # FWHM as the combined image.
    for r in part_results:
        if r.all_amps.size < 1:
            continue
        _, lo, hi = calcRobustLimitsAndBins(r.all_amps.tolist())
        d_raw = fit_alpha_doublet(r.all_amps, lo, hi, f"pdbraw_{ch_id}_{r.idx}")
        mu_ref = doublet_ref(d_raw)[0] if d_raw.ok else 0.0
        if not (mu_ref and mu_ref > 0):
            mu_ref = float(np.median(r.all_amps))
        before_e = TARGET_ENERGY * r.all_amps / mu_ref
        _, lo, hi = calcRobustLimitsAndBins(before_e.tolist())
        r.doublet_before = fit_alpha_doublet(before_e, lo, hi, f"pdbbef_{ch_id}_{r.idx}")
        r.doublet_before.hist.SetTitle(
            f"Ch {ch_id} P{r.idx}: Alpha doublet BEFORE stab. (rescaled);Energy (keV);Counts")
        if not (r.q_0 == 0.0 and r.slope == 0.0):
            with np.errstate(divide='ignore', invalid='ignore'):
                after_e = TARGET_ENERGY * r.all_amps / (r.q_0 + r.slope * r.all_bases)
            after_e = after_e[np.isfinite(after_e)]
            if after_e.size > 0:
                _, lo, hi = calcRobustLimitsAndBins(after_e.tolist())
                r.doublet_after = fit_alpha_doublet(after_e, lo, hi, f"pdbaft_{ch_id}_{r.idx}")
                r.doublet_after.hist.SetTitle(
                    f"Ch {ch_id} P{r.idx}: Alpha doublet AFTER stab.;Energy (keV);Counts")

    # ======================================================================
    # COMBINED DOUBLET SPECTRA  (all partitions merged) -- alpha + recoil peaks
    # ======================================================================
    # Both the BEFORE and the AFTER spectra keep the FULL doublet (alpha + recoil,
    # res.all_amps), so a double-Gaussian fit (the same optimized-binning fitter
    # used per partition) shows the resolution of BOTH peaks.
    #
    # BEFORE: merge the raw amplitudes of every partition, fit the doublet, then
    #   rescale by the ALPHA-peak mean so that peak lands on TARGET_ENERGY, and
    #   refit the doublet on the rescaled (energy) spectrum. The width then
    #   includes the inter-partition spread -> reference resolution.
    # AFTER: per-event stabilized amplitudes of the full doublet (already energy).
    raw_parts   = [r.all_amps for r in part_results if r.all_amps.size > 0]
    all_raw     = np.concatenate(raw_parts) if raw_parts else np.array([], np.float64)

    after_parts = []
    for r in part_results:
        if r.all_amps.size > 0 and not (r.q_0 == 0.0 and r.slope == 0.0):
            denom = r.q_0 + r.slope * r.all_bases
            with np.errstate(divide='ignore', invalid='ignore'):
                cal = TARGET_ENERGY * r.all_amps / denom
            after_parts.append(cal[np.isfinite(cal)])
    all_after = np.concatenate(after_parts) if after_parts else np.array([], np.float64)

    # BEFORE: doublet fit on the raw combined spectrum, rescale on the alpha mean,
    # refit the doublet on the rescaled (energy) spectrum.
    doublet_before = None
    if all_raw.size > 0:
        _, r_lo, r_hi = calcRobustLimitsAndBins(all_raw.tolist())
        d_raw   = fit_alpha_doublet(all_raw, r_lo, r_hi, f"before_raw_{ch_id}")
        mu_ref_raw = doublet_ref(d_raw)[0] if d_raw.ok else 0.0
        if not (mu_ref_raw and mu_ref_raw > 0):
            mu_ref_raw = float(np.median(all_raw))
        all_before = TARGET_ENERGY * all_raw / mu_ref_raw
        _, b_lo, b_hi = calcRobustLimitsAndBins(all_before.tolist())
        doublet_before = fit_alpha_doublet(all_before, b_lo, b_hi, f"before_comb_{ch_id}")
        doublet_before.hist.SetTitle(
            f"Ch {ch_id}: Alpha doublet BEFORE stabilization (rescaled);Energy (keV);Counts")

    # AFTER: doublet fit on the per-event stabilized spectrum.
    doublet_after = None
    if all_after.size > 0:
        _, a_lo, a_hi = calcRobustLimitsAndBins(all_after.tolist())
        doublet_after = fit_alpha_doublet(all_after, a_lo, a_hi, f"after_comb_{ch_id}")
        doublet_after.hist.SetTitle(
            f"Ch {ch_id}: Alpha doublet AFTER stabilization (all partitions);Energy (keV);Counts")

    # Counts under the (stabilized) alpha peak, from its Gaussian component.
    if doublet_after is not None and doublet_after.fit is not None:
        _mu_r, _sg_r, _amp_r = doublet_ref(doublet_after)
        bw = doublet_after.hist.GetXaxis().GetBinWidth(1)
        gaussian_counts = (_amp_r * _sg_r * math.sqrt(2 * math.pi) / bw) if _sg_r > 0 else 0.0
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

        out_filename     = os.path.join(out_dir, f"{base_filename}_alpha_stabilized{calib_suffix}.root")
        new_tree_name    = "stabilized_heater_alpha"
        new_tree_title   = "Heater + Alpha Stabilized"

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
        drawLines(ROOT.gPad.GetUymax(), 1.0, ROOT.kGreen+2, 3, cal_win_min, cal_win_max)
        leg_rough = ROOT.TLegend(0.15, 0.75, 0.45, 0.85)
        leg_rough.SetBorderSize(1); leg_rough.SetFillColor(0)
        d_win = ROOT.TLine(); d_win.SetLineColor(ROOT.kGreen+2); d_win.SetLineWidth(3); d_win.SetLineStyle(3)
        leg_rough.AddEntry(d_win, "Analysis window (corr+LY+stab)", "l")
        leg_rough.Draw("same")
        global_lines.extend([d_win, leg_rough])

        # pad 3 : Heater Stabilized (raw spectrum after the correlation cut)
        c_glob.cd(3); ROOT.gPad.SetGrid()
        h_raw.SetLineColor(ROOT.kBlack); h_raw.SetFillColorAlpha(ROOT.kOrange+1, 0.5)
        h_raw.Draw(); c_glob.Update()
        drawLines(ROOT.gPad.GetUymax(), conv_raw, ROOT.kGreen+2, 3, cal_win_min, cal_win_max)

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
        drawLines(ROOT.gPad.GetUymax(), conv_factor, ROOT.kGreen+2, 3, cal_win_min, cal_win_max)

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

            # (0,1) doublet BEFORE stabilization (rescaled to energy) + fit + box
            c_parts.cd(padno(0, 1)); ROOT.gPad.SetGrid()
            if r.sufficient and r.doublet_before is not None and r.doublet_before.hist is not None:
                db = r.doublet_before
                db.hist.SetStats(0); db.hist.SetLineColor(ROOT.kBlack)
                db.hist.SetFillColorAlpha(ROOT.kMagenta, 0.4); db.hist.Draw()
                if db.fit is not None:
                    db.fit.SetLineColor(ROOT.kMagenta+2); db.fit.SetLineWidth(2)
                    db.fit.SetNpx(600); db.fit.Draw("same")
                box = CreateDoubletBox(db, f"P{r.idx} BEFORE", ROOT.kMagenta+2)
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
                # leverage outliers excluded from the linear fit: drawn in orange
                # (they ARE still stabilized as normal events).
                if r.g_heat_vs_base_trim is not None:
                    gt = r.g_heat_vs_base_trim
                    gt.SetMarkerStyle(29); gt.SetMarkerSize(1.8); gt.SetMarkerColor(ROOT.kOrange + 7)
                    gt.Draw("P same")
                    tnote = ROOT.TPaveText(0.14, 0.14, 0.66, 0.20, "NDC")
                    tnote.SetFillColor(ROOT.kWhite); tnote.SetBorderSize(1)
                    tnote.SetTextAlign(12); tnote.SetTextFont(42); tnote.SetTextSize(0.035)
                    tt = tnote.AddText("orange = excluded from the line")
                    tt.SetTextColor(ROOT.kOrange + 7)
                    tnote.Draw(); global_lines.append(tnote)
                if r.f1:
                    # draw the line ONLY over the baseline range of the points used
                    # in the fit, so it is clear the orange outliers are not fitted.
                    lr = r.fit_brange if r.fit_brange is not None else r.view_clean[:2]
                    r.f1.SetRange(lr[0], lr[1])
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

            # (1,0) calibrated & stabilized doublet (energy) + fit + resolution box
            c_parts.cd(padno(1, 0)); ROOT.gPad.SetGrid()
            if r.sufficient and r.doublet_after is not None and r.doublet_after.hist is not None:
                da = r.doublet_after
                da.hist.SetLineColor(ROOT.kBlack); da.hist.SetLineWidth(1)
                da.hist.SetFillColorAlpha(ROOT.kRed + 1, 0.4); da.hist.SetStats(0); da.hist.Draw()
                if da.fit is not None:
                    da.fit.SetLineColor(ROOT.kBlue); da.fit.SetLineWidth(2)
                    da.fit.SetNpx(600); da.fit.Draw("same")
                box = CreateDoubletBox(da, f"P{r.idx} AFTER (stab.)", ROOT.kRed + 1)
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

        # ------------------- COMBINED ALPHA DOUBLET: BEFORE vs AFTER (C)
        # Left  : combined raw spectrum rescaled so the alpha peak sits on the
        #         nominal energy -> double-Gaussian fit, resolution BEFORE stab.
        # Right : per-event stabilized doublet (already in energy) -> double
        #         Gaussian fit. Both boxes report mu/sigma/FWHM for the two peaks.
        c_comb = ROOT.TCanvas(f"c_comb_{ch_id}",
                              f"Combined Alpha Doublet Ch {ch_id}", 1400, 600)
        c_comb.Divide(2, 1)

        def draw_doublet(pad, d, header, hcol):
            c_comb.cd(pad); ROOT.gPad.SetGrid()
            if d is None or d.hist is None:
                return
            d.hist.SetLineColor(ROOT.kBlack); d.hist.SetLineWidth(1)
            d.hist.SetFillColorAlpha(hcol, 0.35); d.hist.SetStats(0); d.hist.Draw()
            if d.fit is not None:
                d.fit.SetLineColor(ROOT.kBlue); d.fit.SetLineWidth(2)
                d.fit.SetNpx(600); d.fit.Draw("same")
            box = CreateDoubletBox(d, header, hcol)
            box.Draw(); global_lines.append(box)

        draw_doublet(1, doublet_before, "BEFORE stabilization", ROOT.kGray + 2)
        draw_doublet(2, doublet_after,  "AFTER stabilization",  ROOT.kRed + 1)

        c_comb.Update()
        if save_summary_jpeg:
            out_jpg = os.path.join(debug_ch_dir, f"ch{ch_id}_combined_alpha{calib_suffix}.jpg")
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

    # ==================================================================
    # THALLIUM CROSS-CHECK (resolution gain from the ALPHA stabilization)
    # ==================================================================
    # Separate, read-only diagnostic: measure the 208-Tl line resolution BEFORE
    # (corrected_amplitude, rescaled to the nominal Tl energy) and AFTER (alpha-
    # stabilized amplitude) the alpha stabilization. Wrapped in try/except so a Tl
    # cross-check failure can never break the main alpha output.
    if TL_EVAL_ENABLE:
        try:
            evaluate_thallium_resolution(
                ch_id, ha, cal_rough, baseline, ld1_ly, ld2_ly,
                mask_main, apply_ly_cut, part_of_event, part_results,
                h_corr, corr_cut_dynamic, TARGET_ENERGY,
                debug_ch_dir, calib_suffix, save_summary_jpeg)
        except Exception as e:
            print(f"  [!] Ch {ch_id}: thallium cross-check failed: {e}", file=sys.stderr)

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
                g_base_part, g_base_time, h_base_part]
        for _d in (doublet_before, doublet_after):
            if _d is not None:
                keep.extend([_d.hist, _d.fit])
        if apply_ly_cut:
            keep.extend([h_ly1, h_ly2, g_ly1_vs_heat, g_ly2_vs_heat,
                         res1.fit_Tl, res1.fit_alpha, res2.fit_Tl, res2.fit_alpha])
        for r in part_results:
            keep.extend([r.h_heat_orig, r.fit_prelim, r.h_heat_clean, r.fit_clean,
                         r.g_heat_vs_base_clean, r.g_heat_vs_base_trim, r.f1, r.f1_ext,
                         r.h_heat_stab, r.fit_stab,
                         r.g_heat_vs_base_stab, r.h_heat_cal, r.h_heat_cal_final,
                         r.fit_cal])
            for _d in (r.doublet_before, r.doublet_after):
                if _d is not None:
                    keep.extend([_d.hist, _d.fit])
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