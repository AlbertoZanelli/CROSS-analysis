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
import csv
import re
from array import array
from datetime import datetime

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
BASE_DIR = "/data/users/azanelli/octopus_work/CROSS/MergedRuns/CorrectedAmp"  # e.g. .../MergedRuns/CorrectedAmps
BASE_DIR_PERSONAL = "/Users/albertozanelli/Desktop/Tesi_Erasmus/CROSS-analysis/CROSS/MergedRuns/CorrectedAmp"

# run mode: CROSS folder holding the RUNxxxxxx sub-folders, and the run number.
CROSS_DIR  = "/Users/albertozanelli/Desktop/Tesi_Erasmus/CROSS-analysis/CROSS"
RUN_NUMBER = 92                 # e.g. 96 -> folder RUN000096, sub-folder Coincidence

# --- Channels to analyse (number after "ch" in the file name) ---------------
#   []          -> process ALL files in BASE_DIR (batch).
#   [N]         -> single channel; GUI if GUI_MANUAL_CUTS is True.
#   [N, M, ...] -> several channels, always batch.
# Command-line channels (if given) OVERRIDE this list.
CHANNELS_TO_PROCESS = [19, 20, 21, 22, 23, 24, 49, 50, 51, 52, 53, 54, 85, 86, 87, 88, 89, 90, 61, 62, 63, 64, 65, 66, 25, 26, 27, 28, 29, 30, 55, 56, 57, 58, 59, 60]

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
ALPHA_PARTICLE_ENERGY = 5304.33   # keV, alpha-particle line (the doublet's lower peak)
# The two lines are a FIXED distance apart, so the doublet's separation is a
# known FRACTION of its position -- about 1.9 % -- whatever the units of the
# spectrum. The window search uses it to tell the doublet from two unrelated
# structures: on ch57 the most prominent pair was a continuum fluctuation at
# rough 5222 and the real alpha peak at 5768, ten per cent apart, and the window
# ended up between them. The rough calibration is only approximate, hence a
# generous band: separations within [1/tol, tol] times the expected one pass.
DOUBLET_SEP_REL = (TARGET_ENERGY - ALPHA_PARTICLE_ENERGY) / TARGET_ENERGY
DOUBLET_SEP_TOL = 2.0
# Bin width of the doublet histograms, as sigma / this. Lower = WIDER bins. A
# low-statistics partition fitted on sigma/4 bins has a handful of counts per
# bin, the two peaks come out ragged and the joint fit can end up describing the
# lower line twice -- on ch57 partition 0 the stabilization then anchored on the
# alpha peak instead of the alpha+recoil one. Wider bins average that noise away.
DOUBLET_BIN_DIV = 2.0
# Bin counts of the exploratory pass, tried IN ORDER until the doublet is
# resolved. Fewer bins = WIDER bins = more counts per bin, so a weak partner line
# rises above the noise instead of being rejected as a fluctuation. On ch65 the
# stabilized spectrum yields only one peak at 60 bins and the fit used to
# collapse to a single Gaussian (sigma 52, a meaningless 2.28 %).
DOUBLET_COARSE_BINS = (60, 40, 30, 20, 15)

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

# --- Thallium chain comparison (combined-thallium canvas) --------------------
# The cross-check is drawn with the SAME machinery as ThalliumStabilization.py:
# one COLUMN per amplitude (the corrected amplitude and the alpha-stabilized
# one) and three ROWS -- full spectrum, peak + fit in native units, the same
# peak rescaled to the Tl energy. Window, binning and width bounds all descend
# from ONE measurement of the line where it is unambiguous: the corrected
# amplitude of the gamma events selected by the correlation + LY cuts. That is
# the role the per-partition thallium peaks play in ThalliumStabilization.py --
# here the partitions are anchored to the ALPHA line, so they cannot serve.
TL_CHAIN_DISP_MIN, TL_CHAIN_DISP_MAX = 2300.0, 2800.0  # row 1, in rough units
TL_CHAIN_FULL_BINS   = 80     # bins of the full-range spectra (row 1)
TL_CHAIN_MAIN_KEY    = "corrected"   # the variable the line is measured on
TL_CHAIN_SEARCH_FRAC = 0.04   # half-width of the band the peak is looked for in
TL_CHAIN_WIN_NSIGMA  = 6      # fallback half-window, in sigma
TL_CHAIN_MIN_BINS    = 15
TL_CHAIN_MAX_BINS    = 200
TL_PEAK_MEAN_MAX_SHIFT = 2.0  # how far the fitted mean may move, in sigma
# Defaults of the per-channel settings (see TL_CHAIN_CSV_PATH), in the order of
# the chain variables: corrected amplitude, alpha-stabilized amplitude.
TL_CHAIN_PEAK_NSIGMA = 6.0    # window half-width = this many measured sigmas
TL_CHAIN_WIN_SCALE   = [1.0, 1.0]
TL_CHAIN_BIN_DIV     = [4.0, 4.0]
TL_CHAIN_SIG_SCALE   = [1.0, 1.0]
TL_CHAIN_SIG_LO      = 0.3    # width bound, as a fraction of the EXPECTED sigma
TL_CHAIN_SIG_HI      = 1.5
# Typical relative width (sigma/mu) of the thallium line, used when NO partition
# peak is available and the interval cannot be measured. Same constant, same role
# and same value as in ThalliumStabilization.py.
CHAIN_SIG_TYPICAL = 0.010
# Whether the LOCAL Tl peak of a partition is trusted is decided on how well it
# stands out of the continuum, NOT on the event count: a clear peak found on few
# events is still a peak, while many events with no visible line are not. Same
# constants and same rule as in ThalliumStabilization.py.
PEAK_SIGNIF_NSIGMA = 2.0
PEAK_SIGNIF_MIN    = 3.0
# Per-channel settings, on file. Kept SEPARATE from the thallium program's
# chain_settings.csv: the variables are different (there the whole heater chain,
# here the two amplitudes of the alpha stabilization), so one file per program
# keeps the hand-tuned rows of each safe.
TL_CHAIN_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "alpha_chain_settings.csv")
TL_USE_CHAIN_CSV  = True
# The corrected amplitude is the SAME variable the thallium program fits, so the
# panels are set up from THAT program's file instead of being tuned twice: the
# columns ending in "_corrected" of chain_settings.csv win over the local ones,
# for BOTH panels (corrected and alpha-stabilized).
# Set to None to keep everything local.
TL_SHARED_CHAIN_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "chain_settings.csv")
# ...and, going one step further, the WINDOW itself: the thallium program writes
# the interval it fitted each panel in (win_frac / res_exp) into its results
# table, so the corrected-amplitude panel here can be fitted in exactly the same
# window as there. The "before" of the alpha comparison is then not merely
# similar to the thallium program's corrected panel -- it is the same
# measurement, and the two tables can be put on one figure. Falls back to the
# local measurement when the file or the row is missing.
TL_REUSE_THALLIUM_WINDOW = True
TL_THALLIUM_RES_CSV = os.path.join("..", "ThalliumStabilizedAmp",
                                   "thallium_resolutions.csv")
# Results table: same columns as the thallium program's, so the same plotting
# program reads it (PlotThalliumResolutions.py --csv <this file>).
TL_SAVE_RES_CSV   = True
TL_RES_CSV_NAME   = "alpha_thallium_resolutions.csv"
# Results of the ALPHA line itself (the alpha+recoil peak the analysis reports),
# before and after the stabilization: same long format and same columns as the
# tables above, so the same plotting program reads it.
SAVE_ALPHA_RES_CSV = True
ALPHA_RES_CSV_NAME = "alpha_resolutions.csv"

# --- Correlation cut --------------------------------------------------------
# The percentile is taken over the correlations ABOVE the heater cluster, not
# above the fixed CORR_VALID_MIN: the pulser sits at very high, stable
# correlation, and a cut that does not clear it leaves its razor-thin amplitude
# peak in the spectrum -- on ch57 that peak is inside the alpha search region
# and the analysis window used to be built on it. Same recipe as
# ThalliumStabilization.py (AnalyzeHeaterCorrThreshold).
CORR_VALID_MIN      = 0.9995   # fallback lower bound, when there is no heater
CORR_CUT_PERCENTILE = 0.10    # dynamic cut at this percentile of valid corr


# --- Heater-correlation lower bound -----------------------------------------
HEATER_CORR_MIN   = 0.992   # heater-correlation histogram lower edge (search start)
HEATER_CORR_BINS  = 70      # bins of that histogram
HEATER_CORR_NSIGMA = 2.0    # lower bound = heater-peak mean + this many sigma

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
PART_WARM_MIN_COUNTS = 5      # second, LOW threshold (raw counts) used only to recover a
                              # population the relative one cannot see: a cluster far from the
                              # main peak is short in BIN HEIGHT but can hold a large share of
                              # the events. It becomes a partition only if its integral passes
                              # PART_MIN_BLOCK_FRAC, so strays never do
# Fewer clean events than this in a partition: no local line is fitted, the
# partition inherits the nearest fitted one (a Gaussian and a linear fit need a
# handful of points to mean anything, and zero points cannot even be histogrammed).
PART_MIN_CLEAN_EVENTS = 2

# --- Linear fit of amplitude vs baseline, inside a partition ----------------
# Two methods, one switch:
#   "theilsen" : robust_line() -- slope = median of all pairwise slopes. NO point
#                is excluded, and there is nothing to tune (breakdown ~29 % by
#                construction).
#   "rob"      : ROOT's LTS fit, Fit(f1, "Q0 rob=<LINE_FIT_ROB>"), the same one
#                ThalliumStabilization.py uses. LINE_FIT_ROB is the fraction of
#                points KEPT: 0.85 keeps 85 % and TRIMS the worst 15 % on the
#                residuals. Valid range 0.5-1.0 exclusive -- ROOT silently resets
#                anything >= 1.0 back to 0.5.
# Set LINE_FIT_METHOD = "theilsen" to go back to the previous behaviour.
LINE_FIT_METHOD = "theilsen"
LINE_FIT_ROB    = 0.85

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
    
    if not os.path.isdir(BASE_DIR):
        return BASE_DIR_PERSONAL
    
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


def ld_usable(vals, min_events=20):
    """
    Is this light detector usable for the LY cut?

    A detector that is not there still has its LD*_LY branch, filled with
    EXACTLY zero for every event (ch60: 254793 zeros in LD2). Fitted, that spike
    at the origin yields a splendid |mu|/sigma and wins the "best resolved"
    comparison, so the cut ends up being made on a channel that measured
    nothing. A detector is usable only if it has enough events with a NON-ZERO
    light yield.
    """
    v = np.asarray(vals, np.float64)
    return int(np.count_nonzero(np.isfinite(v) & (v != 0.0))) >= min_events


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

def CreateLYBoxThallium(res, title_prefix):
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

def fit_metrics(fit, hist=None):
    """
    Results of a Gaussian(+background) fit as a plain dict: position, width,
    FWHM, percentage resolution and their errors, the fit quality and -- when
    *hist* is given -- the counts under the Gaussian component.

    Single source of the numbers shown in the fit boxes AND written to the
    results CSV, so the two can never disagree. Returns None when *fit* is None.
    """
    if fit is None:
        return None

    mu     = fit.GetParameter(1)
    mu_e   = fit.GetParError(1)
    sigma  = abs(fit.GetParameter(2))
    sig_e  = fit.GetParError(2)
    fwhm   = SIGMA_TO_FWHM * sigma
    fwhm_e = SIGMA_TO_FWHM * sig_e

    # Percentage resolution and its error. R = 100*FWHM/mu, so the relative
    # errors of FWHM and mu add in quadrature (they are treated as
    # uncorrelated, which is the usual approximation for a Gaussian fit).
    res = res_e = float("nan")
    if abs(mu) > 1e-12:
        res   = 100.0 * fwhm / abs(mu)
        rel_f = (fwhm_e / fwhm) if fwhm > 0 else 0.0
        rel_m = mu_e / abs(mu)
        res_e = res * math.sqrt(rel_f * rel_f + rel_m * rel_m)

    ndf = fit.GetNDF()
    # Counts under the peak: the Gaussian integral in bin units.
    n_peak = float("nan")
    if hist is not None:
        bw = hist.GetXaxis().GetBinWidth(1)
        if bw > 0:
            n_peak = fit.GetParameter(0) * sigma * math.sqrt(2 * math.pi) / bw

    return dict(mu=mu, mu_err=mu_e, sigma=sigma, sigma_err=sig_e,
                fwhm=fwhm, fwhm_err=fwhm_e, res=res, res_err=res_e,
                chi2=fit.GetChisquare(), ndf=ndf,
                prob=(fit.GetProb() if ndf > 0 else float("nan")),
                n_peak=n_peak,
                n_hist=(hist.GetEntries() if hist is not None else float("nan")))


def doublet_metrics(d):
    """
    The reported peak of an AlphaDoublet as a plain dict, errors included: the
    alpha+recoil line when the fit resolved the doublet, the single fitted peak
    otherwise. Single source of the numbers in the fit boxes AND in the results
    CSV, so the two cannot disagree. None when there is no fit.
    """
    if d is None or d.fit is None:
        return None
    if d.mu_r is not None and d.sig_r > 0:
        name, mu, mu_e = "ALPHA+RECOIL peak", d.mu_r, d.mu_r_err
        sigma, sig_e   = abs(d.sig_r), d.sig_r_err
    else:
        name, mu, mu_e = "ALPHA peak (single)", d.mu_a, d.mu_a_err
        sigma, sig_e   = abs(d.sig_a), d.sig_a_err

    fwhm, fwhm_e = SIGMA_TO_FWHM * sigma, SIGMA_TO_FWHM * sig_e
    res = res_e = float("nan")
    if abs(mu) > 1e-12:
        res   = 100.0 * fwhm / abs(mu)
        rel_f = (fwhm_e / fwhm) if fwhm > 0 else 0.0
        rel_m = mu_e / abs(mu)
        res_e = res * math.sqrt(rel_f * rel_f + rel_m * rel_m)

    ndf = d.fit.GetNDF()
    return dict(name=name, mu=mu, mu_err=mu_e, sigma=sigma, sigma_err=sig_e,
                fwhm=fwhm, fwhm_err=fwhm_e, res=res, res_err=res_e,
                chi2=d.fit.GetChisquare(), ndf=ndf,
                prob=(d.fit.GetProb() if ndf > 0 else float("nan")))


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

    m = fit_metrics(fit)

    if m["ndf"] > 0:
        pt.AddText(f"#chi^{{2}}/ndf = {m['chi2']:.1f}/{m['ndf']} "
                   f"(P:{m['prob']:.2f})")
    pt.AddText(f"#mu = {m['mu']:.4f} #pm {m['mu_err']:.4f}")
    pt.AddText(f"#sigma = {m['sigma']:.5f} #pm {m['sigma_err']:.5f}")
    pt.AddText(f"FWHM = {m['fwhm']:.5f} #pm {m['fwhm_err']:.5f}")
    if math.isfinite(m["res"]):
        pt.AddText(f"Risoluzione = {m['res']:.2f} #pm {m['res_err']:.2f} %")
    return pt


def CreateDoubletBox(d, header, header_color=ROOT.kBlack,
                     x1=0.13, y1=0.50, x2=0.52, y2=0.88):
    """
    NDC TPaveText with the results of the peak the analysis REPORTS: the
    alpha+recoil line (the full Q-value line, the stabilization reference when
    STABILIZE_ON_RECOIL), with mu, sigma, FWHM and percentage resolution, each
    with its fit error. The alpha-particle line is fitted -- the doublet is
    fitted jointly, and having both peaks in the model is what keeps them from
    stealing each other's counts -- but it is not the quantity of interest, so
    it is not printed. When the fit found a single peak, that one is shown.
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

    m = doublet_metrics(d)
    if m is None:
        return pt
    t = pt.AddText(m["name"]); t.SetTextColor(ROOT.kGreen + 3); t.SetTextFont(62)
    pt.AddText(f"  #mu = {m['mu']:.2f} #pm {m['mu_err']:.2f}")
    pt.AddText(f"  #sigma = {m['sigma']:.2f} #pm {m['sigma_err']:.2f}")
    pt.AddText(f"  FWHM = {m['fwhm']:.2f} #pm {m['fwhm_err']:.2f}")
    if math.isfinite(m["res"]):
        pt.AddText(f"  Risoluzione = {m['res']:.2f} #pm {m['res_err']:.2f} %")
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


def AnalyzeHeaterCorrThreshold(heater_corrs, ch_id, fallback):
    """
    Correlation value 'just above the heater events'.

    This is used as the LOWER bound of the interval over which the dynamic
    correlation-cut percentile (CORR_CUT_PERCENTILE) is taken, replacing the
    fixed CORR_VALID_MIN. Heater (pulser) events cluster at very high, stable
    correlation; the returned value sits just above that cluster.

    Procedure (heater events only, taken RAW: heater flag == 1, corr>HEATER_CORR_MIN):
      - histogram the heater-event correlations;
      - locate the RIGHTMOST peak (TSpectrum, else the tallest bin);
      - Gaussian-fit it and return  mean + HEATER_CORR_NSIGMA*sigma;
      - if that exceeds 1.0, walk the peak's right tail down to the first bin
        below 5 % of the peak height and use that correlation instead.
    Any failure (<=10 heater events, degenerate fit, tail scan failing) returns
    *fallback* (CORR_VALID_MIN).

    Returns (threshold, h_corr_heater or None, fit or None). The histogram and
    fit are kept for the debug drawing.
    """
    raw = np.asarray(heater_corrs, dtype=np.float64)
    raw = raw[np.isfinite(raw) & (raw > HEATER_CORR_MIN)]
    if len(raw) <= 10:
        print(f">>> Ch {ch_id}: <=10 heater events with corr>{HEATER_CORR_MIN}; "
              f"heater threshold falls back to CORR_VALID_MIN={fallback:.6f}.")
        return fallback, None, None

    h = ROOT.TH1F(f"h_corr_heater_{ch_id}",
                  f"Ch {ch_id}: Heater Correlation; Correlation; Counts",
                  HEATER_CORR_BINS, HEATER_CORR_MIN, 1.00005)
    h.SetDirectory(0)
    h.FillN(len(raw), raw.astype(np.double), np.ones(len(raw), np.double))

    spec = ROOT.TSpectrum(5)
    nP   = spec.Search(h, 2, "goff", 0.2)
    if nP > 0:
        xp        = spec.GetPositionX()
        rightmost = max(xp[i] for i in range(nP))
    else:
        rightmost = h.GetXaxis().GetBinCenter(h.GetMaximumBin())
    print(f"  -> Ch {ch_id}: {nP} peak(s) in heater correlation; rightmost at {rightmost:.6f}.")

    fit_window = max(0.0004, h.GetRMS() * 1.5)
    fit = ROOT.TF1(f"fit_corr_heater_{ch_id}", "gaus",
                   rightmost - fit_window, rightmost + fit_window)
    fit.SetParameters(h.GetBinContent(h.GetXaxis().FindBin(rightmost)), rightmost, 0.0001)
    # Log-likelihood fit ("L"): the low-amplitude heater pedestal inflates the
    # histogram RMS (hence the fit window); a plain chi^2 fit then drifts off the
    # narrow rightmost peak onto the pedestal. The likelihood fit (as used for the
    # LY gaussians) locks onto the tall, narrow heater peak instead.
    h.Fit(fit, "Q0 R L")

    mean, sigma = fit.GetParameter(1), fit.GetParameter(2)
    thr = mean + HEATER_CORR_NSIGMA * sigma

    if thr > 1.0:
        # Gaussian tail overshoots 1.0: use the first low-count bin of the tail.
        max_bin    = h.GetMaximumBin()
        thresh_cnt = h.GetBinContent(max_bin) * 0.05
        thr = None
        for b in range(max_bin, h.GetNbinsX() + 1):
            if h.GetBinContent(b) <= thresh_cnt:
                thr = h.GetXaxis().GetBinCenter(b)
                break
        if thr is None or thr > 1.0:
            thr = fallback
            print(f"  [!] Ch {ch_id}: heater cut > 1.0 and tail scan failed; "
                  f"threshold falls back to {fallback:.6f}.")
        else:
            print(f"  [!] Ch {ch_id}: heater Gaussian cut > 1.0; using first "
                  f"low-count bin at {thr:.6f}.")
    else:
        print(f">>> Ch {ch_id}: heater correlation threshold (mean+{HEATER_CORR_NSIGMA:g}sigma) = "
              f"{thr:.6f} (mean={mean:.6f}, sigma={sigma:.6f}).")
    return thr, h, fit


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
        # Fit errors of the peak the analysis reports (see CreateDoubletBox).
        self.mu_a_err = 0.0; self.sig_a_err = 0.0
        self.mu_r_err = 0.0; self.sig_r_err = 0.0


def _peak_candidates(h, search_min=None, search_max=None):
    """
    Peak candidates of a histogram, ranked by topographic PROMINENCE -- the
    height of the (lightly smoothed) peak above the local background on both
    sides -- NOT by integrated area. Area favours the broad low-energy
    continuum; prominence keeps true, structured peaks and ignores flat
    background fluctuations. Returns [(x, prominence, height)], best first.
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
    return cands


def _two_peaks(h, search_min=None, search_max=None):
    """
    The two genuine peaks of a histogram (alpha + alpha-recoil), sorted by x
    (lower first), returned as (x, height).

    The second peak is accepted only if it is clearly significant, otherwise a
    single peak is returned.
    """
    ax = h.GetXaxis()
    keep, mind = [], (ax.GetXmax() - ax.GetXmin()) * 0.05
    for t in _peak_candidates(h, search_min, search_max):
        if t[1] <= 0: continue
        if not any(abs(t[0] - q[0]) < mind for q in keep):
            keep.append(t)
        if len(keep) == 2: break
    # accept the 2nd peak only when clearly significant (else single peak)
    if len(keep) == 2 and keep[1][1] < max(3.0, 0.10 * keep[0][1]):
        keep = keep[:1]
    keep.sort(key=lambda t: t[0])
    return [(t[0], t[2]) for t in keep]


def doublet_pair(h, search_min=None, search_max=None):
    """
    The two peaks of the alpha DOUBLET in *h*, as (x_low, x_high).

    ANCHORED ON THE MOST PROMINENT candidate: one of the two lines is always the
    dominant structure of the search region -- that is why the region is
    searched there. Its partner is the candidate whose separation is CLOSEST TO
    THE EXPECTED ONE (above or below it, within a factor DOUBLET_SEP_TOL): the
    distance between the two lines is a physical constant, so it identifies the
    partner far better than its prominence does. On ch57 the anchor (rough 5768)
    has two candidates in band, a continuum bump 2.86 % below and the real
    partner 2.02 % above; picking by prominence took the bump.

    Ranking pairs by their COMBINED prominence instead lets two mediocre bumps
    outscore the real line: on ch57 a pair of continuum fluctuations (5222 +
    5338, prominences 13.3 + 12.8) beat the true alpha peak with its weaker
    partner (5768 + 5848, 19.4 + 5.5) by half a unit, and the window was built
    500 units away from the doublet.

    Returns None when the anchor has no compatible partner (the caller then
    falls back to the most prominent single peak).
    """
    cands = [c for c in _peak_candidates(h, search_min, search_max) if c[1] > 0]
    if not cands:
        return None
    x0    = cands[0][0]                       # most prominent peak: the anchor
    lo_f  = DOUBLET_SEP_REL / DOUBLET_SEP_TOL
    hi_f  = DOUBLET_SEP_REL * DOUBLET_SEP_TOL
    best  = None
    for x, prom, _ in cands[1:]:
        x1, x2 = sorted((x0, x))
        if x2 <= 0:
            continue
        sep = (x2 - x1) / x2
        if not (lo_f <= sep <= hi_f):
            continue
        # closest to the expected separation; prominence only breaks ties
        score = (abs(sep - DOUBLET_SEP_REL), -prom)
        if best is None or score < best[0]:
            best = (score, x1, x2)
    return (best[1], best[2]) if best else None


def doublet_expected(h, mu_a, mu_r, search_min=None, search_max=None):
    """
    The doublet of a spectrum ALREADY IN ENERGY, where the two lines can only be
    at ALPHA_PARTICLE_ENERGY and TARGET_ENERGY: the candidate CLOSEST to each
    expected position, instead of the most prominent one and its partner.

    On a channel whose alpha-particle line is broad and degraded the prominence
    anchor lands inside that blob and finds a continuum structure at the right
    separation inside it, ignoring the real recoil peak: on ch65 the combined
    stabilized spectrum was fitted as a pair at 5245 + 5314 keV and reported the
    ALPHA+RECOIL line at 5313.8 (sigma 70, a meaningless 3.1 %), while the recoil
    peak sits at 5407 and every per-partition fit found it.

    Returns None when there are no candidates or both lines snap to the same one
    (the caller then falls back to the prominence search).
    """
    cands = [c for c in _peak_candidates(h, search_min, search_max) if c[1] > 0]
    if not cands:
        return None
    x_a = min(cands, key=lambda c: abs(c[0] - mu_a))[0]
    x_r = min(cands, key=lambda c: abs(c[0] - mu_r))[0]
    if x_a == x_r:
        return None
    return (min(x_a, x_r), max(x_a, x_r))

def fit_alpha_doublet(amps, win_lo, win_hi, tag, search_min=None, search_max=None,
                      expect=None):
    """
    Fit the ALPHA DOUBLET (alpha-particle line + alpha+recoil line) with a double
    Gaussian over a linear background, with binning optimized to the peak width.

    Strategy (mirrors the thallium peak-finding, extended to two peaks):
      1. coarse histogram -> the two peaks: nearest to *expect* when the two
         positions are known (a spectrum already in energy), else via
         doublet_pair (physical separation). Retried with progressively WIDER
         bins until resolved;
      2. preliminary single-Gaussian fit of each peak -> seed widths;
      3. rebuild the histogram with bin width ~ sigma/4 (optimized binning);
      4. joint final fit  gaus(0)+gaus(3)+pol1(6)  seeded from the preliminaries.

    The LOWER peak is the alpha-particle line. TWO Gaussians are ALWAYS fitted:
    where the finder cannot see the partner even at the widest binning, it is
    seeded at the physical separation.
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

    # 1. coarse pass: progressively WIDER bins until the doublet is resolved.
    #    The partner is identified by doublet_pair, i.e. by the PHYSICAL
    #    separation of the two lines, not by its prominence.
    #    *expect* = (mu_alpha, mu_recoil) on a spectrum already in energy: the
    #    positions are known, so the peaks are taken nearest to them.
    h0 = pair = None
    for nb0 in DOUBLET_COARSE_BINS:
        h0   = build(nb0)
        pair = (doublet_expected(h0, expect[0], expect[1], search_min, search_max)
                if expect is not None else None)
        if pair is None:
            pair = doublet_pair(h0, search_min, search_max)
        if pair is not None:
            break

    if pair is None:
        # Even the widest bins show one peak only. The doublet is ALWAYS fitted
        # with two Gaussians: the separation of the two lines is a physical
        # constant, so the partner's position is known even where the finder
        # cannot see it. It goes on the side holding more counts.
        peak_x, _ = find_dominant_peak(h0, search_min, search_max)
        ax     = h0.GetXaxis()
        up, dn = peak_x * (1.0 + DOUBLET_SEP_REL), peak_x * (1.0 - DOUBLET_SEP_REL)
        pair   = ((peak_x, up)
                  if h0.GetBinContent(ax.FindBin(up)) >= h0.GetBinContent(ax.FindBin(dn))
                  else (dn, peak_x))
        print(f"  [!] {tag}: doublet partner not found; seeded at the physical "
              f"separation ({100 * DOUBLET_SEP_REL:.2f} %).")

    # 2. preliminary single fits -> seed widths
    def prelim(mu):
        sg0 = (win_hi - win_lo) * 0.02
        g = ROOT.TF1(f"g_{tag}", "gaus", mu - 3 * sg0, mu + 3 * sg0)
        g.SetParameters(h0.GetBinContent(h0.GetXaxis().FindBin(mu)), mu, sg0)
        # Bounded like every other seed fit here: left free, a Gaussian on a WEAK
        # peak walks off it and balloons. On ch19 P0 the recoil seed at 8149 came
        # back as mu = 7653, sigma = 179 -- outside its own +/-34 fit range -- and
        # every bound derived from it was then centred on nothing.
        g.SetParLimits(1, mu - 3 * sg0, mu + 3 * sg0)
        g.SetParLimits(2, sg0 * 0.2, sg0 * 3.0)
        h0.Fit(g, "Q0 R")
        return g.GetParameter(0), g.GetParameter(1), abs(g.GetParameter(2))

    A1, m1, s1 = prelim(pair[0])
    A2, m2, s2 = prelim(pair[1])

    # 3. rebuild at bin width ~ sigma / DOUBLET_BIN_DIV  4. joint final fit
    sig_ref = max(min(s1, s2), (win_hi - win_lo) * 0.004)
    nb = int(np.clip(round((win_hi - win_lo) / (sig_ref / DOUBLET_BIN_DIV)), 30, 300))
    h = build(nb); res.hist = h
    lo, hi = m1 - 3.5 * s1, m2 + 3.5 * s2
    ff = ROOT.TF1(tag, "gaus(0)+gaus(3)+pol1(6)", lo, hi)
    ff.SetParameters(A1, m1, s1, A2, m2, s2,
                     h.GetBinContent(h.GetXaxis().FindBin(lo)), 0.0)
    ff.SetParLimits(2, s1 * 0.3, s1 * 3.0)
    ff.SetParLimits(5, s2 * 0.3, s2 * 3.0)
    # Each mean stays within HALF the doublet separation of the peak the search
    # found: the two Gaussians can then neither swap nor collapse onto the same
    # line. Left free, the weaker (recoil) component walks off its peak onto the
    # continuum -- on ch19 P0 the search found the pair at 7997 + 8149 and the
    # fit slid the second Gaussian down to 7849, so the partition was anchored
    # one whole separation away (the rescaled peak landed at 5604 keV instead of
    # 5407). The separation is a physical constant, so this bound costs nothing.
    half_sep = 0.5 * DOUBLET_SEP_REL * pair[1]
    ff.SetParLimits(1, pair[0] - half_sep, pair[0] + half_sep)
    ff.SetParLimits(4, pair[1] - half_sep, pair[1] + half_sep)
    h.Fit(ff, "Q0 R")
    res.fit = ff
    res.mu_a, res.sig_a = ff.GetParameter(1), abs(ff.GetParameter(2))
    res.mu_r, res.sig_r = ff.GetParameter(4), abs(ff.GetParameter(5))
    res.mu_a_err, res.sig_a_err = ff.GetParError(1), ff.GetParError(2)
    res.mu_r_err, res.sig_r_err = ff.GetParError(4), ff.GetParError(5)
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


def peak_significance(h, center, sigma):
    """
    How well the peak at *center* stands out of the local continuum, in standard
    deviations: the excess over the background, divided by its Poisson
    fluctuation. The peak region is center +/- PEAK_SIGNIF_NSIGMA*sigma and the
    background is taken from two side bands of the same width, one on each side.
    Returns 0.0 when it cannot be evaluated.
    """
    w = PEAK_SIGNIF_NSIGMA * abs(sigma)
    if not (w > 0):
        return 0.0
    ax = h.GetXaxis()
    band = lambda a, b: h.Integral(ax.FindBin(a), ax.FindBin(b))
    n_peak = band(center - w, center + w)
    n_side = band(center - 3 * w, center - w) + band(center + w, center + 3 * w)
    bkg    = n_side / 2.0                      # side bands are twice the peak width
    # Excess over ITS OWN fluctuation: sqrt(n_peak), not sqrt(background). With a
    # nearly empty background the latter saturates at its floor and turns any
    # handful of counts into a "significant" peak.
    return (n_peak - bkg) / math.sqrt(max(n_peak, 1.0))

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

    peak_x, peak_y = find_dominant_peak(h)
    fit_window = (hi - lo) * 0.1
    f = ROOT.TF1(f"f_peakest_{name}", "gaus", peak_x - fit_window, peak_x + fit_window)
    f.SetParameters(peak_y, peak_x, h.GetRMS() * 0.1)
    h.Fit(f, "Q0 R L", "", peak_x - fit_window, peak_x + fit_window)

    mean, sigma = f.GetParameter(1), f.GetParameter(2)
    if sigma > fit_window or sigma <= 0:
        mean, sigma = peak_x, h.GetRMS() * 0.1
    return mean, sigma

def chain_peak_interval(part_results, peak_nsigma=None):
    """
    Where the thallium peak is, and how wide, from the per-partition peaks the
    stabilization has already fitted. Returns (frac_lo, frac_hi, res_exp):

      frac            : half-width of the interval spanned by the partition peak
                        regions (mu +/- CHAIN_PEAK_NSIGMA sigma), as a FRACTION of
                        the mean peak position. Being a fraction, the same interval
                        applies to every amplitude of the chain: each panel only
                        needs its own peak position, no conversion factor. It is
                        SYMMETRIC by construction (the wider of the two sides): the
                        interval is built around the MEAN of the partition peaks,
                        but is applied around each panel's own peak, and on a
                        multi-modal spectrum that peak is the dominant mode, not the
                        centre of the structure -- an asymmetric interval would then
                        come out shifted, with the white margin on one side only.
      mu_corr         : the peak position, in main-amplitude units.
      res_exp         : the expected relative width (sigma/mu) of the COMBINED
                        peak -- the width INSIDE a partition and the SPREAD between
                        partition positions, added in quadrature. The spread is
                        what broadens the un-stabilized amplitudes, and is exactly
                        what the stabilization removes.

    Both come from the peak the stabilization actually works on, so they are far
    more reliable than anything measured over a window where the continuum
    dominates. Falls back to (0, 0, CHAIN_SIG_TYPICAL) with no partition peak.
    """
    peaks = [(r.mean_amp_clean, abs(r.fit_clean.GetParameter(2)), max(r.n_clean, 1))
             for r in part_results
             if r.sufficient and r.fit_clean is not None and r.mean_amp_clean > 0]
    if not peaks:
        return 0.0, 0.0, CHAIN_SIG_TYPICAL
    w_tot  = sum(n for _, _, n in peaks)
    mu_avg = sum(m * n for m, _, n in peaks) / w_tot
    if not (mu_avg > 0):
        return 0.0, 0.0, CHAIN_SIG_TYPICAL
    sig_in = sum(s * n for _, s, n in peaks) / w_tot
    spread = math.sqrt(sum(n * (m - mu_avg) ** 2 for m, _, n in peaks) / w_tot)
    nsig = CHAIN_PEAK_NSIGMA if peak_nsigma is None else peak_nsigma
    lo = min(m - nsig * s for m, s, _ in peaks)
    hi = max(m + nsig * s for m, s, _ in peaks)
    frac = max(1.0 - lo / mu_avg, hi / mu_avg - 1.0, 0.0)
    return frac, mu_avg, math.hypot(sig_in, spread) / mu_avg

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


def fit_thallium_peak(energies, center, lo, hi, nb, sig_seed, tag,
                      sigma_bounds=None):
    """
    FINAL thallium-peak fit of the cross-check (ported from
    ThalliumStabilization.py).

    Histograms *energies* on the GIVEN axis (lo, hi, nb bins) and fits the Tl
    peak with a Gaussian over a FLAT background (gaus(0)+pol0(3)) using a
    Poisson maximum-likelihood fit ("L", correct for low statistics, natively
    handling empty bins). Seeded at *center* with width *sig_seed*. The Gaussian
    parameters stay at indices 0,1,2, so CreateFitBox reads mu/sigma directly.

    The background is flat and not linear: over a window this narrow the
    continuum is nearly flat anyway, and a constant spends one parameter less on
    the few counts around the line. A slope, on top of that, is free to go
    NEGATIVE inside the window (its parameter is the intercept at x = 0, far
    outside the fit range, so it cannot be bounded in any physical way).

    Returns (hist, fit) or (None, None).
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
    seed  = max(sig_seed, 1.0)
    h_max = max(h.GetMaximum(), 1.0)
    ff = ROOT.TF1(f"fit_{tag}", "gaus(0)+pol0(3)", lo, hi)
    ff.SetParameter(0, h_max); ff.SetParameter(1, center); ff.SetParameter(2, seed)
    # Constrain the Gaussian so the minimiser cannot walk away from the peak:
    #   [0] amplitude  > 0      -- otherwise it can flip NEGATIVE and "fit" a dip;
    #   [1] mean       near the seeded centre -- the Tl line is known to sit there;
    #   [2] width      inside an ABSOLUTE band -- keeps it on the PEAK and lets the
    #                              constant take the continuum.
    ff.SetParLimits(0, 0.0, 10.0 * h_max)
    # The background level is a number of counts: it cannot be negative, and with
    # few counts the fit does drive it below zero. Started from the edge bins.
    _edge = 0.5 * (h.GetBinContent(1) + h.GetBinContent(nb))
    ff.SetParameter(3, max(_edge, 1e-3))
    ff.SetParLimits(3, 0.0, max(10.0 * h_max, 1.0))
    ff.SetParLimits(1, center - TL_PEAK_MEAN_MAX_SHIFT * seed,
                       center + TL_PEAK_MEAN_MAX_SHIFT * seed)
    if sigma_bounds is not None:
        # Explicit, ABSOLUTE width band: the plausible range for this line is
        # known, so the bound does not depend on how good the seed is -- with a
        # seed-derived bound an off seed pegs the width and every panel comes out
        # with the same, meaningless resolution.
        ff.SetParLimits(2, *sigma_bounds)
    else:
        ff.SetParLimits(2, seed * 0.2, seed * 3.0)
    # "L": Poisson maximum-likelihood fit -- the correct method for a low-statistics
    # histogram, natively accounting for empty bins (no need to weight them by hand).
    h.Fit(ff, "Q0 R L")
    return h, ff


# ---------------------------------------------------------------------------
# THALLIUM CHAIN: per-channel settings, window, results table
# ---------------------------------------------------------------------------

def rough_to_units(values, cal_rough, mask):
    """
    Scale factor k such that  values ~ k * cal_rough, as the median ratio over
    the *mask* events. It converts a window given in ROUGH-calibration units
    into the units of either chain variable, so both panels of the combined
    canvas cover the same physical range on their own scale.
    """
    v, r = values[mask], cal_rough[mask]
    good = np.isfinite(v) & np.isfinite(r) & (r > 0)
    return float(np.median(v[good] / r[good])) if good.any() else 1.0


class TlChainSettings:
    """Settings of the thallium chain comparison for ONE channel."""
    # The keys are the chain variables: the amplitude the alpha stabilization
    # works on, and the same amplitude after it.
    KEYS   = ("corrected", "alpha")
    FIELDS = (*(f"win_scale_{k}" for k in KEYS), *(f"bin_div_{k}" for k in KEYS),
              *(f"sig_scale_{k}" for k in KEYS),
              "peak_nsigma", "sig_lo", "sig_hi")

    def __init__(self, values=None):
        d = dict(zip(self.FIELDS, [
            *(list(TL_CHAIN_WIN_SCALE) + [1.0] * 2)[:2],
            *(list(TL_CHAIN_BIN_DIV)   + [4.0] * 2)[:2],
            *(list(TL_CHAIN_SIG_SCALE) + [1.0] * 2)[:2],
            TL_CHAIN_PEAK_NSIGMA, TL_CHAIN_SIG_LO, TL_CHAIN_SIG_HI]))
        if values:
            for k in self.FIELDS:
                if values.get(k) not in (None, ""):
                    d[k] = float(values[k])
        for k, v in d.items():
            setattr(self, k, v)

    def bin_div(self, key):
        """Bin width (as sigma / this) of the chain variable *key*."""
        return getattr(self, f"bin_div_{key}", 4.0)

    def win_scale(self, key):
        """Window widening of the chain variable *key*."""
        return getattr(self, f"win_scale_{key}", 1.0)

    def sig_scale(self, key):
        """Scale of the EXPECTED width for the chain variable *key*."""
        return getattr(self, f"sig_scale_{key}", 1.0)

    def as_row(self, ch_id):
        row = {"channel": ch_id}
        row.update({k: getattr(self, k) for k in self.FIELDS})
        return row


def _apply_shared_corrected(cfg, ch_id):
    """
    Set BOTH chain panels from the thallium program's corrected-amplitude tuning.

    That program fits the very same variable, so its per-channel tuning applies
    here unchanged: taking it from there means the panels are tuned ONCE, in one
    file, instead of drifting apart in two. The "_corrected" columns of
    chain_settings.csv are used for the corrected-amplitude panel AND for the
    alpha-stabilized one -- the two are the same line measured on the same
    events, before and after, so there is no reason to frame or bin them
    differently. Silently does nothing when the shared file or the channel's row
    is missing.
    """
    if not (TL_SHARED_CHAIN_CSV and os.path.exists(TL_SHARED_CHAIN_CSV)):
        return
    try:
        with open(TL_SHARED_CHAIN_CSV, newline="") as fh:
            row = next((r for r in csv.DictReader(fh)
                        if str(r.get("channel", "")).strip() == str(ch_id)), None)
    except OSError as e:
        print(f"  [!] Cannot read {TL_SHARED_CHAIN_CSV}: {e}", file=sys.stderr)
        return
    if row is None:
        return
    taken = []
    for f in ("win_scale", "bin_div", "sig_scale"):
        v = row.get(f"{f}_corrected")
        if v not in (None, ""):
            for key in TlChainSettings.KEYS:          # corrected AND alpha
                setattr(cfg, f"{f}_{key}", float(v))
            taken.append(f)
    for f in ("peak_nsigma", "sig_lo", "sig_hi"):     # channel-level, not per-variable
        v = row.get(f)
        if v not in (None, ""):
            setattr(cfg, f, float(v))
            taken.append(f)
    if taken:
        print(f">>> Tl chain: ch {ch_id} seeded from "
              f"{os.path.basename(TL_SHARED_CHAIN_CSV)} (corrected-amplitude "
              f"column, both panels): {', '.join(taken)}.")


def thallium_window(ch_id, res_csv_path):
    """
    The window the THALLIUM program fitted this channel's corrected-amplitude
    panel in, as (win_frac, res_exp, mu), read from its results table. Returns
    None when the table, the row, or the columns are not there -- the caller
    then measures the window itself, as before.
    """
    if not os.path.exists(res_csv_path):
        return None
    try:
        with open(res_csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                if (str(r.get("channel", "")).strip() != str(ch_id)
                        or r.get("variable") != "corrected"
                        or r.get("row") != "native"):
                    continue
                try:
                    frac = float(r.get("win_frac", "nan"))
                    rexp = float(r.get("res_exp", "nan"))
                    mu   = float(r.get("mu", "nan"))
                except (TypeError, ValueError):
                    return None
                if all(math.isfinite(x) and x > 0 for x in (frac, rexp, mu)):
                    return frac, rexp, mu
                return None
    except OSError:
        return None
    return None




def tl_chain_settings(ch_id):
    """
    Settings of the thallium chain for channel *ch_id*, from TL_CHAIN_CSV_PATH.

    Same self-maintaining file as the thallium program: a channel with no row is
    APPENDED with the values in use, existing rows are never rewritten (that is
    what keeps hand-tuned values safe), and a file written before a new column
    existed is upgraded in place with the program defaults.
    """
    rows, header, on_file = [], ["channel", *TlChainSettings.FIELDS], None
    if os.path.exists(TL_CHAIN_CSV_PATH):
        try:
            with open(TL_CHAIN_CSV_PATH, newline="") as fh:
                reader  = csv.DictReader(fh)
                rows    = list(reader)
                on_file = list(reader.fieldnames or [])
        except OSError as e:
            print(f"  [!] Cannot read {TL_CHAIN_CSV_PATH}: {e}", file=sys.stderr)

    if on_file is not None and any(f not in on_file for f in header):
        try:
            upgraded = [TlChainSettings(r).as_row(str(r.get("channel", "")).strip())
                        for r in rows]
            with open(TL_CHAIN_CSV_PATH, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
                w.writeheader(); w.writerows(upgraded)
            print(f">>> {os.path.basename(TL_CHAIN_CSV_PATH)} upgraded with the new "
                  f"columns ({', '.join(f for f in header if f not in on_file)}).")
        except OSError as e:
            print(f"  [!] Cannot upgrade {TL_CHAIN_CSV_PATH}: {e}", file=sys.stderr)

    mine = next((r for r in rows if str(r.get("channel", "")).strip() == str(ch_id)), None)

    # The channel's OWN row wins: once it is on file it is the tuning for this
    # channel, hand-edited or not, and nothing overwrites it.
    if mine is not None and TL_USE_CHAIN_CSV:
        cfg = TlChainSettings(mine)
        print(f">>> Tl chain settings for ch {ch_id} read from "
              f"{os.path.basename(TL_CHAIN_CSV_PATH)}.")
        return cfg

    # Channel not on file: seed EVERY parameter from the thallium program's
    # corrected-amplitude column, for both panels, and write the row so it can
    # be tuned from there on.
    cfg = TlChainSettings(None)
    _apply_shared_corrected(cfg, ch_id)
    if mine is not None:                  # row there, but the CSV is disabled
        return cfg

    try:                                  # channel not on file yet: append it
        with open(TL_CHAIN_CSV_PATH, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            if not rows and fh.tell() == 0:
                w.writeheader()
            w.writerow(cfg.as_row(ch_id))
        print(f">>> Tl chain settings of ch {ch_id} added to "
              f"{os.path.basename(TL_CHAIN_CSV_PATH)}.")
    except OSError as e:
        print(f"  [!] Cannot write {TL_CHAIN_CSV_PATH}: {e}", file=sys.stderr)
    return cfg


# Columns of the results CSV -- the SAME as ThalliumStabilization.py writes, so
# PlotThalliumResolutions.py reads this table too (pass it with --csv).
TL_RES_CSV_FIELDS = [
    "channel", "step", "variable", "label", "row", "background",
    "mu", "mu_err", "sigma", "sigma_err", "fwhm", "fwhm_err",
    "resolution_pct", "resolution_err_pct",
    "chi2", "ndf", "prob", "n_peak", "n_hist", "win_frac", "res_exp", "date",
]


def _tl_res_sort_key(r):
    """Order of the results CSV: channel, then chain position, row."""
    try:
        ch = int(str(r.get("channel", "")).strip())
    except (TypeError, ValueError):
        ch = 1 << 30
    try:
        step = int(str(r.get("step", "")).strip())
    except (TypeError, ValueError):
        step = 1 << 30
    return (ch, step, 0 if str(r.get("row", "")) == "native" else 1)


def write_tl_resolution_csv(path, ch_id, new_rows):
    """
    Write this channel's fit results to the results CSV at *path*.

    The file accumulates the whole detector: the rows of *ch_id* are REPLACED
    (a re-analysis must not leave the old numbers behind) and every other
    channel is kept as it is.
    """
    old = []
    if os.path.exists(path):
        try:
            with open(path, newline="") as fh:
                old = [r for r in csv.DictReader(fh)
                       if str(r.get("channel", "")).strip() != str(ch_id)]
        except OSError as e:
            print(f"  [!] Cannot read {path}: {e}", file=sys.stderr)

    rows = sorted(old + list(new_rows), key=_tl_res_sort_key)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=TL_RES_CSV_FIELDS,
                               restval="nan", extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f">>> {len(new_rows)} Tl fit result(s) of ch {ch_id} written to "
              f"{os.path.basename(path)} ({len(rows)} row(s) in total).")
    except OSError as e:
        print(f"  [!] Cannot write {path}: {e}", file=sys.stderr)


def write_alpha_resolution_csv(path, ch_id, doublet_before, doublet_after):
    """
    Write the reported alpha peak (see doublet_metrics) BEFORE and AFTER the
    stabilization, with the error on every value, the resolution included.
    Replaces this channel's rows and leaves the others alone.
    """
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def num(v, fmt="{:.6g}"):
        return fmt.format(v) if (v is not None and math.isfinite(v)) else "nan"

    new_rows = []
    for step, (variable, label, d) in enumerate(
            (("corrected", "Before alpha stabilization", doublet_before),
             ("alpha_line", "After alpha stabilization",  doublet_after))):
        m = doublet_metrics(d)
        if m is None:
            continue
        new_rows.append({
            "channel": ch_id, "step": step, "variable": variable, "label": label,
            "row": "energy", "background": "pol1",
            "mu": num(m["mu"]),       "mu_err": num(m["mu_err"]),
            "sigma": num(m["sigma"]), "sigma_err": num(m["sigma_err"]),
            "fwhm": num(m["fwhm"]),   "fwhm_err": num(m["fwhm_err"]),
            "resolution_pct": num(m["res"], "{:.4f}"),
            "resolution_err_pct": num(m["res_err"], "{:.4f}"),
            "chi2": num(m["chi2"], "{:.3f}"), "ndf": m["ndf"],
            "prob": num(m["prob"], "{:.4f}"),
            "n_peak": "nan", "n_hist": "nan",
            "win_frac": "nan", "res_exp": "nan",
            "date": stamp,
        })
    if not new_rows:
        return

    old = []
    if os.path.exists(path):
        try:
            with open(path, newline="") as fh:
                old = [r for r in csv.DictReader(fh)
                       if str(r.get("channel", "")).strip() != str(ch_id)]
        except OSError as e:
            print(f"  [!] Cannot read {path}: {e}", file=sys.stderr)

    rows = sorted(old + new_rows, key=_tl_res_sort_key)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=TL_RES_CSV_FIELDS,
                               restval="nan", extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f">>> Alpha-line results of ch {ch_id} written to "
              f"{os.path.basename(path)} ({len(rows)} row(s) in total).")
    except OSError as e:
        print(f"  [!] Cannot write {path}: {e}", file=sys.stderr)


def collect_tl_resolution_rows(ch_id, chain):
    """One CSV row per fitted panel: the two rows of the canvas, per variable."""
    stamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    panels = (("native", "h_zoom",   "fit_zoom"),
              ("energy", "h_energy", "fit_energy"))

    def num(v, fmt="{:.6g}"):
        return fmt.format(v) if (v is not None and math.isfinite(v)) else "nan"

    rows = []
    for p in chain:
        for row_name, hkey, fkey in panels:
            m = fit_metrics(p.get(fkey), p.get(hkey))
            if m is None:
                continue
            rows.append({
                "channel": ch_id, "step": p["idx"], "variable": p["key"],
                "label": p["label"], "row": row_name, "background": "pol0",
                "mu": num(m["mu"]),           "mu_err": num(m["mu_err"]),
                "sigma": num(m["sigma"]),     "sigma_err": num(m["sigma_err"]),
                "fwhm": num(m["fwhm"]),       "fwhm_err": num(m["fwhm_err"]),
                "resolution_pct": num(m["res"], "{:.4f}"),
                "resolution_err_pct": num(m["res_err"], "{:.4f}"),
                "chi2": num(m["chi2"], "{:.3f}"), "ndf": m["ndf"],
                "prob": num(m["prob"], "{:.4f}"),
                "n_peak": num(m["n_peak"], "{:.2f}"),
                "n_hist": num(m["n_hist"], "{:.0f}"),
                "win_frac": num(p["interval"][0], "{:.8f}"),
                "res_exp":  num(p["interval"][1], "{:.8f}"),
                "date": stamp,
            })
    return rows


def fit_peak_centred(values, mu, tag, interval, cfg, key):
    """
    Fit the thallium peak of *values* with the recipe of ThalliumStabilization.py:

      1. Take the peak position *mu* as given -- inside the selection window the
         continuum can be taller than the line and a peak finder locks onto the
         window edge, while the converted position is right to a fraction of a
         per cent.
      2. Frame it with the peak INTERVAL measured on the corrected amplitude
         (*interval* = (frac, res_exp)): the window is mu*(1 -/+ frac), symmetric
         so the peak keeps a white margin on BOTH sides. It is an ABSOLUTE
         interval, taken where the line is unambiguous, so it contains the line
         whatever the continuum does around it -- unlike a window built from the
         locally measured width, which on a strong continuum comes out several
         times too wide and lets the fit balloon over the background.
      3. Bin it on the EXPECTED width (res_exp*mu), so the binning does not
         depend on a local measurement either.
      4. gaus(0)+pol0(3) Poisson-likelihood fit, the width bounded by
         [sig_lo, sig_hi] x that expected width: that is what keeps the Gaussian
         on the peak instead of letting it spread onto the continuum.

    Returns (hist, fit, mu, sigma); (None, None, 0.0, 0.0) when it cannot fit.
    """
    if not (mu > 0):
        return None, None, 0.0, 0.0

    frac, res_exp = interval

    # On the MAIN amplitude the position is already exact: *mu* is the peak
    # measured on this very variable. Elsewhere it is converted, and the
    # conversion factor is a median over a window the continuum dominates, so it
    # can be a few per cent off -- too much for the fit to recover, its mean
    # being bounded. There the peak is looked for in a BAND around it.
    refine = (key != TL_CHAIN_MAIN_KEY)
    if refine:
        _lo_s, _hi_s = mu * (1.0 - TL_CHAIN_SEARCH_FRAC), mu * (1.0 + TL_CHAIN_SEARCH_FRAC)
        _, _seed = fit_peak_optimized(values, _lo_s, _hi_s, f"{tag}_seed")
        if _seed is not None and _lo_s < _seed.GetParameter(1) < _hi_s:
            mu = _seed.GetParameter(1)

    h = f = None
    # TWO PASSES on the refined variables: the first window is centred on the
    # converted position, the second on the peak that fit actually found. It is a
    # refinement, not a search: the mean stays within TL_PEAK_MEAN_MAX_SHIFT
    # sigmas of the centre, so the window cannot walk away onto the continuum.
    n_pass = 2 if refine else 1
    res_var = res_exp * cfg.sig_scale(key)
    for _pass in range(n_pass):
        sigma   = res_var * abs(mu)                # expected width in these units
        sig_min = cfg.sig_lo * sigma
        sig_max = cfg.sig_hi * sigma
        if not (sig_max > sig_min > 0):
            return None, None, 0.0, 0.0
        if frac > 0.0:
            lo, hi = mu * (1.0 - frac), mu * (1.0 + frac)
        else:
            lo, hi = mu - TL_CHAIN_WIN_NSIGMA * sigma, mu + TL_CHAIN_WIN_NSIGMA * sigma
        nb = int(np.clip(round((hi - lo) / (sigma / cfg.bin_div(key))),
                         TL_CHAIN_MIN_BINS, TL_CHAIN_MAX_BINS))
        h, f = fit_thallium_peak(values, mu, lo, hi, nb, sigma,
                                 tag if _pass == n_pass - 1 else f"{tag}_pre",
                                 sigma_bounds=(sig_min, sig_max))
        if f is None:
            return None, None, 0.0, 0.0
        if not (f.GetParameter(1) > 0):
            break
        mu = f.GetParameter(1)
    return h, f, f.GetParameter(1), abs(f.GetParameter(2))


def thallium_partition_peak(amps_p, ch_id, idx, peak_hint=None):
    """
    The thallium peak of ONE baseline partition, on the gamma events of that
    partition. Lifted verbatim from the corresponding section of
    process_partition() in ThalliumStabilization.py -- peak search restricted to
    the combined-spectrum hint, local position adopted only when it is
    significant over the continuum, preliminary Gaussian, cleaning at
    +/-HEAT_CLEAN_NSIGMA, final Gaussian on the clean peak.

    The stabilization itself is NOT run: here the partitions are stabilized on
    the ALPHA line, and this is only the measurement chain_peak_interval() needs.
    Returns a StabResult carrying the four fields it reads (sufficient,
    mean_amp_clean, fit_clean, n_clean).
    """
    res = StabResult(idx, 0.0, 0.0)
    amps_for_stab = np.asarray(amps_p, np.float64)
    res.n_events  = len(amps_for_stab)
    tag = f"tlpart_{ch_id}_p{idx}"
    if res.n_events < 1:
        return res
    res.sufficient = True

    bins_heat, heat_min, heat_max = calcRobustLimitsAndBins(amps_for_stab.tolist())
    n_stab = len(amps_for_stab)

    res.h_heat_orig = ROOT.TH1F(
        f"h_tlheat_orig_{tag}",
        f"Ch {ch_id} P{idx}: Tl spectrum after LY cut - Pre-cleaning;Amplitude;Counts",
        bins_heat, heat_min, heat_max)
    res.h_heat_orig.SetDirectory(0)
    res.h_heat_orig.FillN(n_stab, amps_for_stab.astype(np.double),
                          np.ones(n_stab, np.double))

    search_min = search_max = None
    if peak_hint is not None:
        hint_center, hint_sigma = peak_hint
        half_width = PEAK_HINT_NSIGMA * abs(hint_sigma)
        search_min, search_max = hint_center - half_width, hint_center + half_width

    peak_x, peak_y = find_dominant_peak(res.h_heat_orig, search_min, search_max)

    if peak_hint is not None:
        signif = peak_significance(res.h_heat_orig, peak_x, hint_sigma)
        if signif < PEAK_SIGNIF_MIN:
            peak_x = hint_center
            peak_y = res.h_heat_orig.GetBinContent(
                res.h_heat_orig.GetXaxis().FindBin(hint_center))
            print(f"  -> Ch {ch_id} Tl P{idx}: local peak not significant "
                  f"({signif:.1f} < {PEAK_SIGNIF_MIN} sigma, {n_stab} events); "
                  f"position taken from the combined spectrum ({hint_center:.1f}).")
        else:
            print(f"  -> Ch {ch_id} Tl P{idx}: local peak at {peak_x:.1f} "
                  f"({signif:.1f} sigma over the continuum, {n_stab} events).")

    fit_window = (heat_max - heat_min) * 0.1
    res.fit_prelim = ROOT.TF1(f"fit_tlprelim_{tag}", "gaus",
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

    clean_amps_np = amps_for_stab[(amps_for_stab >= heat_cut_min)
                                  & (amps_for_stab <= heat_cut_max)]
    res.n_clean   = len(clean_amps_np)

    params_clean = GetCenteredBinning(clean_amps_np.tolist(),
                                      heat_min + (heat_max - heat_min) / 2.0)
    res.h_heat_clean = ROOT.TH1F(
        f"h_tlheat_clean_{tag}",
        f"Ch {ch_id} P{idx}: Thallium Peak;Amplitude;Counts",
        params_clean.bins, params_clean.vis_min, params_clean.vis_max)
    res.h_heat_clean.SetDirectory(0)

    if res.n_clean < PART_MIN_CLEAN_EVENTS:
        print(f"  -> Ch {ch_id} Tl P{idx}: only {res.n_clean} event(s) left by the "
              f"cleaning window (< {PART_MIN_CLEAN_EVENTS}); no local Tl peak.")
        res.sufficient = False
        return res

    res.h_heat_clean.FillN(res.n_clean, clean_amps_np.astype(np.double),
                           np.ones(res.n_clean, np.double))
    res.fit_clean = ROOT.TF1(f"fit_tlclean_{tag}", "gaus",
                             params_clean.median - 5.0 * params_clean.robust_sigma,
                             params_clean.median + 5.0 * params_clean.robust_sigma)
    res.fit_clean.SetParameters(res.h_heat_clean.GetMaximum(),
                                params_clean.median, params_clean.robust_sigma)
    res.h_heat_clean.Fit(res.fit_clean, "Q0 R L")
    res.mean_amp_clean = res.fit_clean.GetParameter(1)
    return res

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
        # Same window as ThalliumStabilization.py: on the RAW amplitude scaled by
        # the conversion factor (mask_ly_range there), not on cal_rough directly.
        # The two pick slightly different events, and the LY sigma -- hence the
        # cut, hence the gamma selection -- came out different on every channel.
        m_ly = (mask_main & (ha > TL_CAL_LY_MIN * conv_tl)
                          & (ha < TL_CAL_LY_MAX * conv_tl))
        def _ly_hist(ly_arr, name):
            vals = ly_arr[m_ly & np.isfinite(ly_arr)]
            if vals.size < 1:
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
        # A detector that is not there gives LY = 0 for every event: excluded.
        ok1 = ld_usable(ld1_ly[m_ly]) if apply_ly_cut else False
        ok2 = ld_usable(ld2_ly[m_ly]) if apply_ly_cut else False
        if ok1 and ok2:
            chosen_ld = 2 if df2 > df1 else 1
        elif ok1 or ok2:
            chosen_ld = 1 if ok1 else 2
        else:
            chosen_ld = 1
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

    # ======================================================================
    # THALLIUM CHAIN  (combined-thallium canvas, as in ThalliumStabilization.py)
    # ======================================================================
    # One COLUMN per amplitude -- the corrected amplitude the alpha
    # stabilization works on, and the same amplitude after it -- and three ROWS:
    #   1. the Tl display range [TL_CHAIN_DISP_MIN, MAX] (rough units) converted
    #      into the variable's own units -- the spectrum as it is;
    #   2. the same data zoomed on the Tl peak, with the gaus+pol0 fit;
    #   3. the same peak rescaled to TL_TARGET_ENERGY, so the two FWHMs are
    #      directly comparable.
    cfg = tl_chain_settings(ch_id)

    # WHERE the line is, and HOW WIDE -- from the per-partition thallium peaks,
    # exactly as ThalliumStabilization.py derives them (chain_peak_interval).
    # The partitions are the SAME ones (FindBaselinePartitions is the same
    # search, byte for byte), so the peaks are fitted on the same event groups;
    # only the selection differs -- the gammas picked here. A single free fit
    # over the whole window is NOT used: on a channel with few gamma events it
    # settles on the continuum instead of the line (ch57: 62 events gave
    # sigma/mu = 2.72 % against the 0.47 % the thallium program measures).
    peak_hint = estimate_thallium_peak(amps_tl, f"{ch_id}_tlcomb")
    peak_hint = peak_hint if peak_hint[0] != 0.0 else None
    if peak_hint is not None:
        print(f">>> Combined-spectrum Tl peak hint at amplitude {peak_hint[0]:.1f} "
              f"(sigma {peak_hint[1]:.1f}); used for all partitions.")

    n_part   = (int(part_of_event.max()) + 1) if part_of_event.size else 1
    tl_parts = [thallium_partition_peak(ha[mask_tl & (part_of_event == i)],
                                        ch_id, i, peak_hint)
                for i in range(n_part)]
    tl_frac, mu_main, res_exp = chain_peak_interval(tl_parts, cfg.peak_nsigma)
    src = "measured here"
    if not (mu_main > 0 and res_exp > 0):
        print(f"  [!] Ch {ch_id}: the Tl line could not be measured on the "
              f"corrected amplitude; cross-check skipped.")
        return None

    # <output>/AlphaStabilizationDebug/ch<N> -> <output>/, next to the stabilized
    # ROOT files: where this program's results table goes, and the anchor the
    # thallium table's relative path is resolved against.
    res_dir = os.path.dirname(os.path.dirname(debug_ch_dir))

    # Prefer the window the THALLIUM program fitted this very variable in: the
    # corrected amplitude is the same quantity there, so reusing its interval
    # makes the two "before" numbers the same measurement instead of two similar
    # ones. win_frac on file already carries that program's win_scale_corrected,
    # so it is divided out and re-applied per variable below -- the alpha panel
    # then gets the same BASE window, widened by its own factor.
    if TL_REUSE_THALLIUM_WINDOW:
        _shared = thallium_window(ch_id, os.path.normpath(
            os.path.join(res_dir, TL_THALLIUM_RES_CSV)))
        if _shared is not None:
            _frac, _rexp, _mu = _shared
            _ws = cfg.win_scale("corrected") or 1.0
            tl_frac, res_exp, mu_main = _frac / _ws, _rexp, _mu
            src = "from the thallium results table"

    print(f">>> Tl chain: peak at {mu_main:.1f}; window "
          f"+/-{100 * tl_frac * cfg.win_scale('corrected'):.2f}%, expected width "
          f"{100 * res_exp:.2f}% ({src}).")

    # Per-event alpha-stabilized amplitude: each event corrected with the line of
    # its own baseline partition (the quantity the output ROOT stores). Computed
    # on the WHOLE sample, like the corrected amplitude, so both variables are
    # full-length arrays and can be masked the same way.
    slopes = np.array([r.slope for r in part_results], dtype=np.float64)
    q0s    = np.array([r.q_0   for r in part_results], dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        amp_alpha = alpha_target_energy * ha / (q0s[part_of_event]
                                                + slopes[part_of_event] * baseline)

    # (key, label, per-event values, x-axis title, colour)
    chain_defs = [
        ("corrected", "Corrected amplitude", ha,        "Amplitude (a.u.)", ROOT.kGray + 2),
        ("alpha",     "Alpha stabilized",    amp_alpha, "Energy (keV)",     ROOT.kAzure + 2),
    ]
    # Conversion factor of the MAIN amplitude. Measured on m_conv -- correlation
    # cut AND rough value inside the Tl window -- the same mask the conversion
    # factor of the thallium program is measured on. Requiring the ROUGH value to
    # be in the window matters: a merged run can carry runs whose rough
    # calibration is off by an order of magnitude, and those events alone move
    # the median enough to put the row-1 range on the wrong decade.
    k_main = rough_to_units(ha, cal_rough, m_conv)

    chain = []
    for i, (key, label, values, xtitle, colour) in enumerate(chain_defs):
        # The panels are drawn on the GAMMA sample (correlation + thallium LY
        # cut), not on the narrow window the peak was measured in: row 1 has to
        # show the continuum around the line, and rows 2 and 3 get their window
        # from the interval, not from a pre-selection.
        sel = values[mask_gamma]
        sel = sel[np.isfinite(sel)]
        if sel.size < 5:
            continue

        k      = rough_to_units(values, cal_rough, m_conv)
        lo, hi = TL_CHAIN_DISP_MIN * k, TL_CHAIN_DISP_MAX * k
        h_full = ROOT.TH1F(f"h_tlchain_full_{ch_id}_{i}",
                           f"Ch {ch_id}: {label};{xtitle};Counts",
                           TL_CHAIN_FULL_BINS, lo, hi)
        h_full.SetDirectory(0)
        h_full.FillN(sel.size, sel.astype(np.double), np.ones(sel.size, np.double))

        # Where the line sits in these units: the measured peak converted with
        # this variable's factor. Given to the fit, not searched for.
        mu_exp   = mu_main * k / k_main if k_main > 0 else 0.0
        interval = (tl_frac * cfg.win_scale(key), res_exp)
        h_zoom, fit_zoom, mu, _ = fit_peak_centred(
            sel, mu_exp, f"h_tlchain_zoom_{ch_id}_{i}", interval, cfg, key)
        if h_zoom is not None:
            h_zoom.SetTitle(f"Ch {ch_id}: {label} - peak;{xtitle};Counts")

        chain.append(dict(
            idx=i, key=key, label=label, colour=colour, h_full=h_full,
            interval=interval, h_zoom=h_zoom, fit_zoom=fit_zoom,
            energy=(TL_TARGET_ENERGY * sel / mu) if mu > 0 else None,
            ))

    # --- row 3: the same peaks, in energy ------------------------------------
    # Each sample rescaled so its peak sits on TL_TARGET_ENERGY, then fitted with
    # the SAME routine as row 2: the panel is row 2 on the energy scale, so the
    # two rows always agree and the two amplitudes can be compared directly.
    for p in chain:
        p["h_energy"], p["fit_energy"] = None, None
        if p["energy"] is None:
            continue
        p["h_energy"], p["fit_energy"], _, _ = fit_peak_centred(
            p["energy"], TL_TARGET_ENERGY, f"h_tlchain_energy_{ch_id}_{p['idx']}",
            p["interval"], cfg, p["key"])
        if p["h_energy"] is not None:
            p["h_energy"].SetTitle(
                f"Ch {ch_id}: {p['label']} - energy;Energy (keV);Counts")

    def _summary(p, label):
        m = fit_metrics(p.get("fit_energy"), p.get("h_energy")) if p else None
        if m is None or not math.isfinite(m["res"]):
            return None
        print(f">>> Tl {label}: mu={m['mu']:.1f} keV, FWHM={m['fwhm']:.2f} keV "
              f"({m['res']:.2f} +/- {m['res_err']:.2f} %)")
        return dict(mu=m["mu"], sigma=m["sigma"], fwhm=m["fwhm"],
                    res_pct=m["res"], res_err_pct=m["res_err"])

    p_before = next((p for p in chain if p["key"] == "corrected"), None)
    p_after  = next((p for p in chain if p["key"] == "alpha"), None)
    r_before = _summary(p_before, "BEFORE alpha stab.")
    r_after  = _summary(p_after,  "AFTER  alpha stab.")

    # --- results table --------------------------------------------------------
    if TL_SAVE_RES_CSV and chain:
        rows = collect_tl_resolution_rows(ch_id, chain)
        if rows:
            write_tl_resolution_csv(os.path.join(res_dir, TL_RES_CSV_NAME),
                                    ch_id, rows)

    if not save_jpeg:
        return dict(before=r_before, after=r_after, conv_tl=conv_tl, n_tl=n_tl)

    os.makedirs(debug_ch_dir, exist_ok=True)

    # ---- IMAGE: the combined-thallium canvas (3 rows x 2 columns) ----------
    n_col = len(chain)
    if n_col > 0:
        c_ch = ROOT.TCanvas(f"c_tlcomb_{ch_id}",
                            f"Combined Thallium Peak Ch {ch_id}",
                            600 * n_col, 1350)
        c_ch.Divide(n_col, 3)

        def _draw(pad, h, fit, header, colour):
            c_ch.cd(pad); ROOT.gPad.SetGrid()
            if h is None:
                return
            h.SetStats(0); h.SetLineColor(ROOT.kBlack)
            h.SetFillColorAlpha(colour, 0.5); h.Draw()
            if fit is None:                 # first row: spectrum only
                return
            fit.SetLineColor(ROOT.kBlue); fit.SetLineWidth(2); fit.Draw("same")
            box = CreateFitBox(fit, header, colour,
                               x1=0.13, y1=0.64, x2=0.40, y2=0.85, text_size=0.026)
            box.Draw(); keep.append(box)

        for col, p in enumerate(chain):
            _draw(col + 1,             p["h_full"],   None,            p["label"], p["colour"])
            _draw(n_col + col + 1,     p["h_zoom"],   p["fit_zoom"],   p["label"], p["colour"])
            _draw(2 * n_col + col + 1, p["h_energy"], p["fit_energy"],
                  f"{p['label']} (energy)", p["colour"])
            keep.extend([p["h_full"], p["h_zoom"], p["fit_zoom"],
                         p["h_energy"], p["fit_energy"]])

        c_ch.Update()
        save_canvas_jpeg(c_ch, os.path.join(
            debug_ch_dir, f"ch{ch_id}_combined_thallium{calib_suffix}.jpg"))

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
            box = CreateLYBoxThallium(rr, tag) if (rr and rr.fit_Tl) else None
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

    return dict(before=r_before, after=r_after, conv_tl=conv_tl, n_tl=n_tl)


# ===========================================================================
# BASELINE PARTITIONING
# ===========================================================================

def _bin_runs(mask, min_gap):
    """
    Runs of True bins in *mask*, as [start, end] pairs (0-based, inclusive),
    with runs separated by fewer than *min_gap* False bins merged together --
    the same "a separation must be at least min_gap bins wide" rule the block
    search uses, so a couple of empty bins inside a population do not split it.
    """
    runs = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if runs and (i - runs[-1][1] - 1) < min_gap:
                runs[-1][1] = j - 1
            else:
                runs.append([i, j - 1])
            i = j
        else:
            i += 1
    return runs

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

    # --- Recover populations too LOW for the relative threshold -------------
    # The threshold above is a fraction of the TALLEST bin, and that comparison
    # is scale-dependent: a narrow, very tall main peak (a quiet channel whose
    # baseline histogram spans a wide range because of a second level) makes 3 %
    # of the maximum taller than an ENTIRE secondary population. On ch86 a block
    # of ~30 % of the events, five units of baseline away, sat below it and was
    # swallowed by the outer partition.
    # What makes a population negligible is how many EVENTS it holds, not how
    # tall its bins are, so the histogram is scanned again with a threshold low
    # enough to see any real cluster, and a run that does not overlap a block
    # already found is kept when its integral passes the same PART_MIN_BLOCK_FRAC
    # test the blocks are judged by. Nothing already found can be lost here: the
    # step only ADDS blocks.
    warm = c > PART_WARM_MIN_COUNTS
    for w_start, w_end in _bin_runs(warm, PART_MIN_GAP_BINS):
        if any(w_start <= b[1] and b[0] <= w_end for b in blocks):
            continue                                  # already inside a block
        if float(c[w_start:w_end + 1].sum()) < PART_MIN_BLOCK_FRAC * max(
                float(c[b[0]:b[1] + 1].sum()) for b in blocks):
            continue                                  # negligible: a few strays
        blocks.append([w_start, w_end])
    blocks.sort()

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


def robust_line(bases, amps, max_pairs_points=700):
    """
    Linear fit  amplitude = q0 + slope * baseline  over ALL the clean points:
    Theil-Sen estimate, slope = median of all pairwise slopes, q0 = median(a -
    slope*b). Parameter-free.

    NO point is excluded: every clean event of the partition enters the line.
    Returns (q0, slope).
    """
    b0 = np.asarray(bases, np.float64); a0 = np.asarray(amps, np.float64)
    ok = np.isfinite(b0) & np.isfinite(a0)
    b, a = b0[ok], a0[ok]
    if b.size < 2:
        return (float(np.median(a)) if a.size else 0.0), 0.0
    # subsample when large: the pairwise-slope set is O(n^2)
    if b.size > max_pairs_points:
        idx = np.random.default_rng(12345).choice(b.size, max_pairs_points, replace=False)
        bs, as_ = b[idx], a[idx]
    else:
        bs, as_ = b, a
    i, j = np.triu_indices(bs.size, 1)
    db = bs[j] - bs[i]; m = np.abs(db) > 0
    if not m.any():
        return float(np.median(a)), 0.0
    slope = float(np.median((as_[j] - as_[i])[m] / db[m]))
    q0    = float(np.median(a - slope * b))
    return q0, slope


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

    # Keep ALL stab-window events (full doublet: alpha + recoil) and baselines,
    # so the combined BEFORE/AFTER image can fit BOTH peaks (clean_amps holds only
    # the alpha-only core used for the stabilization reference). Set HERE, before
    # anything can bail out, so a partition that gets no line of its own still
    # feeds the combined spectra.
    res.all_amps  = np.asarray(amps_for_stab, np.float64)
    res.all_bases = np.asarray(bases_for_stab, np.float64)

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

    # A partition whose cleaning window keeps (almost) nothing cannot be fitted:
    # give up on it here, leaving slope == q_0 == 0 so that run_stabilization makes
    # it inherit the nearest fitted partition's line. Going on would build a TGraph
    # with zero points, which the ROOT bindings reject (and crash on, on some
    # builds), and would fit an empty histogram.
    if res.n_clean < PART_MIN_CLEAN_EVENTS:
        print(f"  -> Ch {ch_id} P{idx}: only {res.n_clean} event(s) left by the "
              f"cleaning window (< {PART_MIN_CLEAN_EVENTS}); no local line, it will "
              f"inherit the nearest partition's.")
        res.sufficient = False
        return res

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
        if LINE_FIT_METHOD == "rob":
            # ROOT LTS: keeps LINE_FIT_ROB of the points, trims the rest. Which
            # ones it dropped is internal, so they cannot be marked on the plot.
            res.g_heat_vs_base_clean.Fit(res.f1, f"Q0 rob={LINE_FIT_ROB:.2f}")
            fit_q0, fit_slope = res.f1.GetParameter(0), res.f1.GetParameter(1)
        else:
            # Theil-Sen over ALL the clean points: no event is excluded.
            fit_q0, fit_slope = robust_line(clean_bases_np, clean_amps_np)
            res.f1.SetParameters(fit_q0, fit_slope)
        res.fit_brange = (float(clean_bases_np.min()), float(clean_bases_np.max()))

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
    tree_heater = file.Get("flagpropagator_heater")   # heater flag (pulser events)
    # Optimum-filter tree, also used to compute the LY when the 'LY' tree is gone.
    tree_opt  = tree_main if main_tree_name == AMP_TREE_OPTIMUM else file.Get(AMP_TREE_OPTIMUM)

    tree_for_stabilization = tree_main

    print(f"{'='*60}\n")

    # --- Attach friend trees to the main tree -------------------------------
    # Only VALID trees are attached: a missing tree from file.Get() is a null
    # pointer (not Python None), so it must be filtered with _valid_tree. The
    # LY tree and the optimum-filter tree (LY ratio fallback) are included too.
    candidate_friends = [tree_mod, tree_bad, tree_trig, tree_corr, tree_cal,
                         tree_baseline, tree_ly, tree_opt, tree_time, tree_heater]
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

    # --- Heater (pulser) events ---------------------------------------------
    # The pulser fires at a FIXED amplitude, so it makes a razor-thin peak that
    # can sit anywhere -- on ch57 it lands at rough 5315, inside the alpha search
    # region, where it is 85 % of the events. The window search then centres on
    # it, the stabilization anchors to it, and the two real alpha peaks (rough
    # ~5760 / ~5840 there) are never seen. Pulser events are not physics: they
    # are dropped from the whole analysis.
    # Leaf named "IsHeater" or "heat_IsHeater" depending on the production.
    heater_flag_leaf = next((lf for lf in ("IsHeater", "heat_IsHeater")
                             if tree_for_stabilization.GetLeaf(lf)), None)
    if heater_flag_leaf:
        rdf_f = rdf_f.Define("_is_heater_", heater_flag_leaf)

    # --- Columns to read ----------------------------------------------------
    cols = ["heat_amplitude", "_cal_rough_", "heat_correlation", "heat_baseline"]
    if heater_flag_leaf:
        cols.append("_is_heater_")
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
    is_heater = (np_data["_is_heater_"][:N].astype(np.int32) == 1
                 if heater_flag_leaf else np.zeros(N, bool))

    # Light Yield read DIRECTLY from the LD1_LY / LD2_LY leaves.
    ld1_ly   = np_data["_ld1_ly_"][:N].astype(np.float64)       if apply_ly_cut     else np.zeros(N, np.float64)
    ld2_ly   = (np_data["_ld2_ly_"][:N].astype(np.float64)
                if (apply_ly_cut and has_ld2_branch) else np.zeros(N, np.float64))

    # --- Vectorised NaN/Inf removal -----------------------------------------
    valid = np.isfinite(ha) & np.isfinite(cal_rough) & np.isfinite(corr)
    ha        = ha[valid];   cal_rough = cal_rough[valid]
    corr      = corr[valid]; baseline = baseline[valid]
    ld1_ly    = ld1_ly[valid]; ld2_ly = ld2_ly[valid]
    is_heater = is_heater[valid]
    if time_fsr is not None:
        time_fsr = time_fsr[valid]   # keep the time vector aligned with baseline
    N = len(ha)

    # ======================================================================
    # DYNAMIC CORRELATION CUT  (vectorised)
    # ======================================================================
    # Lower edge of the interval the percentile is taken over: the correlation
    # value just ABOVE the heater (pulser) cluster, measured on the heater events
    # themselves (read RAW: flag == 1, corr > HEATER_CORR_MIN, no quality cuts),
    # exactly as in ThalliumStabilization.py. Falls back to CORR_VALID_MIN when
    # there is no heater flag or the fit is unusable.
    if heater_flag_leaf:
        _hd = (ROOT.RDataFrame(tree_for_stabilization)
               .Filter(f"{heater_flag_leaf} == 1")
               .AsNumpy(["heat_correlation"]))
        heater_corrs = np.asarray(_hd["heat_correlation"], np.float64)
        corr_valid_min_eff, h_corr_heater, fit_corr_heater = AnalyzeHeaterCorrThreshold(
            heater_corrs, ch_id, CORR_VALID_MIN)
    else:
        corr_valid_min_eff, h_corr_heater, fit_corr_heater = CORR_VALID_MIN, None, None
        heater_corrs = np.empty(0, np.float64)
        print(f">>> Ch {ch_id}: no 'IsHeater'/'heat_IsHeater' flag; correlation "
              f"interval lower bound = CORR_VALID_MIN = {CORR_VALID_MIN:.6f}.")
    print(f">>> Correlation interval lower bound (above heater): {corr_valid_min_eff:.6f}")

    mask_corr_valid = corr > corr_valid_min_eff
    corr_above      = corr[mask_corr_valid]
    ha_above        = ha[mask_corr_valid]
    corr_sorted     = np.sort(corr_above)
    n_corr          = len(corr_sorted)

    corr_hist_min    = float(corr_sorted[int(n_corr * 0.01)]) if n_corr > 0 else corr_valid_min_eff
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
    # The correlation cut already sits above the heater cluster, so the pulser is
    # out by construction. The flag is still applied: it is exact, it costs
    # nothing, and it also removes the heater events whose correlation happens to
    # fluctuate above the cut.
    mask_main = (corr > corr_cut_dynamic) & ~is_heater
    if heater_flag_leaf:
        _n_h = int((is_heater & (corr > corr_cut_dynamic)).sum())
        print(f">>> Heater (pulser) events surviving the correlation cut and "
              f"removed by the flag: {_n_h}.")
    win_center = 0.5 * (CAL_SEARCH_MIN + CAL_SEARCH_MAX)     # last-resort fallback
    rough_search = cal_rough[mask_main & (cal_rough > CAL_SEARCH_MIN)
                             & (cal_rough < CAL_SEARCH_MAX)]
    if rough_search.size > 20:
        nb_s = int(np.clip((CAL_SEARCH_MAX - CAL_SEARCH_MIN) / 5.0, 100, 800))
        h_s = ROOT.TH1F(f"h_calwin_{ch_id}", "", nb_s, CAL_SEARCH_MIN, CAL_SEARCH_MAX)
        h_s.SetDirectory(0)
        h_s.FillN(rough_search.size, rough_search.astype(np.double),
                  np.ones(rough_search.size, np.double))
        pair = doublet_pair(h_s)     # the pair whose separation fits the lines
        peaks_win = _two_peaks(h_s)  # fallback: most prominent structures
        if pair is not None:
            p_lo, p_hi = pair
            win_center = 0.5 * (p_lo + p_hi)         # midpoint of the doublet
            print(f">>> Analysis window centered on the doublet midpoint "
                  f"{win_center:.0f} rough (peaks {p_lo:.0f}, {p_hi:.0f}; "
                  f"separation {100*(p_hi-p_lo)/p_hi:.2f}%, expected "
                  f"{100*DOUBLET_SEP_REL:.2f}%).")
        elif len(peaks_win) >= 1:
            win_center = peaks_win[0][0]
            print(f">>> No peak pair with the doublet separation; window centered "
                  f"on the most prominent peak ({win_center:.0f} rough).")
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

        # Pick the LD whose alpha core is best resolved (higher |mu|/sigma),
        # among the ones that actually measured light (see ld_usable).
        ok1, ok2 = ld_usable(vals_ly1_np), ld_usable(vals_ly2_np)
        if not (ok1 or ok2):
            print("  [!] Neither light detector has a non-zero light yield; "
                  "LD1 kept, the LY cut is meaningless on this channel.")
            chosen_ld = 1
        elif ok1 and ok2:
            chosen_ld = 2 if res2.df_value > res1.df_value else 1
        else:
            chosen_ld = 1 if ok1 else 2
            print(f"  [!] LD{2 if ok1 else 1} has no non-zero light yield "
                  f"(detector absent); LD{chosen_ld} used for the LY cut.")
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
        r.doublet_before = fit_alpha_doublet(before_e, lo, hi, f"pdbbef_{ch_id}_{r.idx}",
                                             expect=(ALPHA_PARTICLE_ENERGY, TARGET_ENERGY))
        r.doublet_before.hist.SetTitle(
            f"Ch {ch_id} P{r.idx}: Alpha doublet BEFORE stab. (rescaled);Energy (keV);Counts")
        if not (r.q_0 == 0.0 and r.slope == 0.0):
            with np.errstate(divide='ignore', invalid='ignore'):
                after_e = TARGET_ENERGY * r.all_amps / (r.q_0 + r.slope * r.all_bases)
            after_e = after_e[np.isfinite(after_e)]
            if after_e.size > 0:
                _, lo, hi = calcRobustLimitsAndBins(after_e.tolist())
                r.doublet_after = fit_alpha_doublet(after_e, lo, hi, f"pdbaft_{ch_id}_{r.idx}",
                                                    expect=(ALPHA_PARTICLE_ENERGY, TARGET_ENERGY))
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
        doublet_before = fit_alpha_doublet(all_before, b_lo, b_hi, f"before_comb_{ch_id}",
                                           expect=(ALPHA_PARTICLE_ENERGY, TARGET_ENERGY))
        doublet_before.hist.SetTitle(
            f"Ch {ch_id}: Alpha doublet BEFORE stabilization (rescaled);Energy (keV);Counts")

    # AFTER: doublet fit on the per-event stabilized spectrum.
    doublet_after = None
    if all_after.size > 0:
        _, a_lo, a_hi = calcRobustLimitsAndBins(all_after.tolist())
        doublet_after = fit_alpha_doublet(all_after, a_lo, a_hi, f"after_comb_{ch_id}",
                                          expect=(ALPHA_PARTICLE_ENERGY, TARGET_ENERGY))
        doublet_after.hist.SetTitle(
            f"Ch {ch_id}: Alpha doublet AFTER stabilization (all partitions);Energy (keV);Counts")

    if SAVE_ALPHA_RES_CSV:
        write_alpha_resolution_csv(
            os.path.join(output_dir, STAB_ROOT_DIR_NAME, ALPHA_RES_CSV_NAME),
            ch_id, doublet_before, doublet_after)

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
                if r.f1:
                    # draw the line over the baseline range of the fitted points
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
                         r.g_heat_vs_base_clean, r.f1, r.f1_ext,
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