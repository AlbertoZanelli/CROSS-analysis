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
# "calibrationrun": same files/paths/parsing as "run", but the LY cut uses the
#                   thallium-dominant peak finder (AnalyzeLightYieldRun), suited
#                   to calibration runs where Tl >> alpha and alpha can be at
#                   negative LY. "run" and "mergedrun" use the standard finder.
ANALYSIS_MODE = "mergedrun"     # "mergedrun", "run" or "calibrationrun"

# mergedrun mode: folder containing the input .root files.
BASE_DIR = "/Users/albertozanelli/Desktop/Tesi_Erasmus/CROSS-analysis/CROSS/MergedRuns/CorrectedAmp"  # e.g. .../MergedRuns/CorrectedAmps

# run mode: CROSS folder holding the RUNxxxxxx sub-folders, and the run number.
CROSS_DIR  = "/data/users/azanelli/octopus_work/CROSS-analysis/CROSS"
RUN_NUMBER = 96                 # e.g. 96 -> folder RUN000096, sub-folder Coincidence

# --- Channels to analyse (number after "ch" in the file name) ---------------
#   []          -> process ALL files in BASE_DIR (batch).
#   [N]         -> single channel; GUI if GUI_MANUAL_CUTS is True.
#   [N, M, ...] -> several channels, always batch.
# Command-line channels (if given) OVERRIDE this list.
CHANNELS_TO_PROCESS = [25, 26, 27, 28, 29, 30, 55, 56, 57, 58, 59, 60] 

# --- Output / GUI switches --------------------------------------------------
SAVE_SUMMARY_JPEG   = True    # per-channel debug JPEGs (global, partitions, before/after)
SAVE_CORR_JPEG      = True    # correlation-analysis JPEGs
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
# Columns of the combined-thallium canvas kept on those channels: they have no
# heater stabilization, and their main amplitude is the optimum-filter one, which
# the rough-calibration panel already shows (one calibration apart). What is left
# is the chain they actually follow: the amplitude, and the same amplitude after
# the thallium stabilization.
OPTIMUM_FILTER_CHAIN_KEYS = ("rough", "stabilized")


# ===========================================================================
# PARAMETERS  <- tune these
# ===========================================================================

# --- Output folder names (created if missing, inside BASE_DIR) --------------
STAB_ROOT_DIR_NAME = os.path.join("..", "ThalliumStabilizedAmp")            # stabilized ROOT files

SUMMARY_DIR_NAME   = os.path.join("..", "ThalliumStabilizedAmp/ThalliumStabilizationDebug")  # 4x3 overview JPEGs
CORR_DIR_NAME      = os.path.join("..", "ThalliumStabilizedAmp/CorrelationCut")               # correlation-cut JPEGs
# All per-channel debug JPEGs (baseline partitions, global overview, per-partition
# stabilization, before/after comparison) go to <SUMMARY_DIR>/ch<N>/.

# --- Results table (resolutions of the amplitude chain) ---------------------
# Every fit of the combined-thallium canvas (rows 2 and 3, both background
# models) is written to a CSV, one row per panel, so the resolutions of all the
# analysed channels end up in a single file that can be plotted afterwards
# (see PlotThalliumResolutions.py). The file accumulates over runs: re-analysing
# a channel REPLACES its rows, the other channels are left untouched.
SAVE_RES_CSV      = True
RES_CSV_DIR_NAME  = os.path.join("..", "ThalliumStabilizedAmp")   # folder of the CSV
RES_CSV_NAME      = "thallium_resolutions.csv"                    # (+ calib suffix)

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

# --- Before/after comparison histograms (per partition) ---------------------
# Shared before/after axis of the calibrated thallium peak. The window is
# center +/- BA_WIN_NSIGMA*sigma; the bin count is STATISTICS-AWARE so that
# low-population partitions do not end up with many empty bins (holes) that
# spoil the fit: aim for ~BA_TARGET_PER_BIN counts per bin (on the sparser of
# the before/after samples), never fewer than BA_MIN_BINS bins, and never a
# finer binning than the peak resolution supports (~sigma/BA_RES_BIN_DIV).
BA_WIN_NSIGMA     = 6   # half-window of the shared axis, in sigma
BA_TARGET_PER_BIN = 1.0   # target average counts per bin (statistics cap)
BA_RES_BIN_DIV    = 3.0   # resolution cap: finest bin width ~ sigma / this
BA_MIN_BINS       = 15    # never use fewer bins than this
BA_MAX_BINS       = 200   # never use more bins than this
# How far (in sigma) the fitted peak mean may move from the seeded centre. The
# Tl line position is known, so a tight leash keeps the fit off background bumps.
PEAK_MEAN_MAX_SHIFT = 2.0

# --- Amplitude-chain comparison (combined-thallium canvas) ------------------
# Range shown in the first row, in ROUGH-calibration units; it is converted into
# each variable's own units (rough_to_units), so every panel covers the same
# physical range. Same values as the global-overview spectra.
CHAIN_DISP_MIN, CHAIN_DISP_MAX = 2300.0, 2800.0
CHAIN_FULL_BINS   = 80     # bins of the full-range spectra (first row)
# Peak windows of rows 2 and 3, with the recipe of the original before/after fits:
# the width is MEASURED first (estimate_peak_sigma), then the window is
# +/-CHAIN_WIN_NSIGMA sigma with bin width ~ sigma/CHAIN_BIN_DIV[i], and the fit is
# allowed to move the width within [CHAIN_SIG_LO, CHAIN_SIG_HI] x the measured one.
# Peak windows of rows 2 and 3, with the fit recipe of ThalliumStabilization_old:
# the width is MEASURED on the sample (estimate_peak_sigma), the window is
# +/-CHAIN_WIN_NSIGMA sigma, the bin width ~ sigma/CHAIN_BIN_DIV[i] (resolution cap
# only, no statistics cap), and the fit lets the width move within
# [CHAIN_SIG_LO, CHAIN_SIG_HI] x the measured one.

CHAIN_WIN_NSIGMA = 6      # half-window, in sigma
# Bin width of the peak windows, ~ sigma / this, ONE ENTRY PER CHAIN VARIABLE in
# the order below (calibration rough, heater stabilized, corrected amplitude,
# thallium stabilized): a broad, poorly populated peak needs coarser bins than a
# narrow, rich one. Higher value = finer bins.
CHAIN_BIN_DIV    = [4, 4, 4, 4]
# Half-width of the band, as a fraction of the expected position, in which the
# peak is LOOKED FOR before the fit. The expected position (converted from the
# stabilization peak) can be off by a few per mille to a few per cent, because the
# conversion factors are medians over a window the continuum dominates; searching
# inside a band recovers the real peak, while the band keeps the finder away from
# the continuum that piles up at the edges of the stabilization window.
CHAIN_SEARCH_FRAC = 0.04
# Key of the MAIN amplitude in chain_defs: the variable the stabilization works
# on. Its peak position is known exactly (see fit_peak_centred), so it is the one
# variable whose window is never refined.
CHAIN_MAIN_KEY    = "corrected"
CHAIN_MIN_BINS   = 15
CHAIN_MAX_BINS   = 200

# The peak width is bounded by an ABSOLUTE band, as a fraction of the width the
# stabilization itself measured (see chain_expected_width): a band derived from the
# locally measured width cannot stop the fit from ballooning, because it is exactly
# that measurement which fails when the line sits on a strong continuum.
CHAIN_PEAK_NSIGMA = 6 # peak interval: partition peak +/- this many sigmas
CHAIN_PEAK_NSIGMA = 6.0   # peak interval: partition peak +/- this many sigmas
# Per-variable widening of that interval for rows 2 and 3, in the order of the
# chain variables: calibration rough, heater stabilized, corrected amplitude,
# thallium stabilized. The interval is one fraction for the whole channel, but the
# amplitudes BEFORE the stabilization carry a broader line (the drift between
# partitions is still in it), so they need a wider view to keep a white margin on
# both sides of the peak. The bin width does not change: a wider window simply
# shows more histogram. Set an entry to 1.0 to leave that variable as it is.
CHAIN_WIN_SCALE   = [1.0, 1.5, 1.5, 1.0]
# Per-variable scale of the EXPECTED WIDTH, same order. The width comes from the
# partition peaks of the stabilization, i.e. from the amplitude the stabilization
# works on: it is right for that variable and for the stabilized one, but the
# amplitudes BEFORE it can carry a much broader line (a rough calibration a few
# per cent off spreads the line over several times its own width). Since the
# expected width sets the binning, the sigma bounds and the leash on the mean,
# a line that broad cannot be fitted at all with the width of another variable:
# the fit hits the lower sigma bound and locks onto a single spike. This factor
# multiplies the expected width for ONE variable only, leaving the others alone.
# Per channel in the CSV (sig_scale_<key>); 1.0 = width as measured.
CHAIN_SIG_SCALE   = [1.0, 1.0, 1.0, 1.0]

# --- Per-channel settings of the chain comparison, on file ------------------
# The values above are the DEFAULTS. They can be tuned per channel in a CSV, one
# row per channel, so a channel that needs a wider view (or a different binning)
# does not force a change that affects all the others.
#
# The file maintains itself: any analysed channel missing from it is APPENDED with
# the values currently set in the program (so after one run the file lists every
# channel, ready to be edited by hand). Existing rows are NEVER rewritten -- that
# is what keeps hand-tuned values safe -- so there is no "save" flag to remember:
# to reset a channel to the program defaults, delete its row.
#
# USE_CHAIN_CSV only decides whether the file is READ: with False the program
# values are used for everything, and the file is still kept up to date.
CHAIN_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "chain_settings.csv")
USE_CHAIN_CSV  = True
CHAIN_SIG_LO     = 0.3    # width bound, as a fraction of the EXPECTED sigma
CHAIN_SIG_HI     = 1.5
# The chain fit window is the interval where the THALLIUM EVENTS sit, built from
# the per-partition peaks already measured for the stabilization: every partition
# contributes its own peak +/- TL_RANGE_NSIGMA sigmas, and the union is taken. It
# is what guarantees the window covers the WHOLE line on the pre-stabilization
# amplitudes, where the partitions are drifted apart and the peak is hard to frame.
# The interval is used as a FRACTION of the peak position, so it converts to any
# variable without relying on the absolute accuracy of a conversion factor, and it
# already carries TL_RANGE_NSIGMA sigmas of continuum on each side. TL_MIN_NSIGMA
# is only the fallback half-window used when no partition peak is available.
TL_RANGE_NSIGMA  = 5.0
TL_MIN_NSIGMA    = 4.0

# --- Baseline partitioning --------------------------------------------------
# The amplitude-vs-baseline scatter splits into several clusters along the
# baseline axis. Partitions separate ONLY blocks that are clearly DETACHED in
# the baseline histogram: a boundary is placed across a gap of consecutive
# EMPTY bins. Close peaks with counts between them stay in the same partition.
PART_N_BINS          = 200    # bins of the baseline histogram used for the search
PART_MIN_GAP_BINS    = 4     # a separation needs at least this many consecutive low bins
PART_GAP_HEIGHT_FRAC = 0.03   # a bin belongs to a GAP when its height is below this fraction
                              # of the tallest block: gaps are judged by RELATIVE height, not
                              # raw counts (a few stray counts in a deep valley still separate)
PART_MIN_BLOCK_FRAC  = 0.15   # blocks below this fraction of the MOST POPULATED block are
                              # absorbed into the nearest (in baseline) block: small isolated
                              # peaks join the closest partition instead of forming their own

# --- Thallium-peak search hint (multi-partition) ----------------------------
# The thallium peak is first located on the COMBINED spectrum (all partitions
# merged): that position is reused as a SEARCH HINT for every partition.
#
# The search half-width SHRINKS WITH THE STATISTICS of the partition: with few
# events the local shape is noise, so the combined-spectrum POSITION is far more
# trustworthy than whatever looks most peak-like locally. Below
# PEAK_HINT_TRUST_NEVENTS the hint position is taken outright (no search at all);
# from there up to PEAK_HINT_FULL_NEVENTS the half-width grows linearly from
# PEAK_HINT_MIN_NSIGMA to PEAK_HINT_NSIGMA (in units of the hint sigma).
PEAK_HINT_NSIGMA        = 8.0   # search half-width, in hint sigmas
# Whether the LOCAL peak of a partition is trusted is decided on how well it stands
# out of the continuum, NOT on the event count: a clear peak found on few events is
# still a peak, while many events with no visible line are not. The peak region is
# +/-PEAK_SIGNIF_NSIGMA hint sigmas, the background comes from the side bands, and
# the local position is adopted only above PEAK_SIGNIF_MIN standard deviations.
PEAK_SIGNIF_NSIGMA      = 2.0
PEAK_SIGNIF_MIN         = 3.0
# Fewer clean events than this in a partition: no local line is fitted, the
# partition inherits the nearest fitted one (a Gaussian and a linear fit need a
# handful of points to mean anything, and zero points cannot even be histogrammed).
PART_MIN_CLEAN_EVENTS   = 5

# --- PREVIOUS VERSION (kept for easy rollback) ------------------------------
# The half-width was the LARGER of a sigma-based and a range-based estimate, so
# it was generous by construction and independent of the partition statistics.
# On ch58 P0 (37 events) it opened to +/-160 around a hint at 6897, letting the
# finder lock onto a noise structure at ~7033 instead of the true peak at ~6900.
# To restore: re-enable PEAK_HINT_MIN_FRAC below and the commented block in
# process_partition().
# PEAK_HINT_NSIGMA   = 8.0    # search half-width in units of the combined-peak sigma
# PEAK_HINT_MIN_FRAC = 0.20   # ...but at least this fraction of the partition heat range


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


def CreateFitBox(fit, header, header_color=ROOT.kBlack,
                 x1=0.13, y1=0.60, x2=0.45, y2=0.80,
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


# --- Heater-correlation lower bound -----------------------------------------
HEATER_CORR_MIN   = 0.992   # heater-correlation histogram lower edge (search start)
HEATER_CORR_BINS  = 70      # bins of that histogram
HEATER_CORR_NSIGMA = 2.0    # lower bound = heater-peak mean + this many sigma


def AnalyzeHeaterCorrThreshold(heater_corrs, ch_id, fallback):
    """
    Correlation value 'just above the heater events'.

    This is used as the LOWER bound of the interval over which the dynamic
    correlation-cut percentile (CORR_CUT_PERCENTILE) is taken, replacing the
    fixed CORR_VALID_MIN. Heater (pulser) events cluster at very high, stable
    correlation; the returned value sits just above that cluster.

    Procedure (heater events only, taken RAW: IsHeater==1, corr>HEATER_CORR_MIN):
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


def fit_thallium_peak(energies, center, lo, hi, nb, sig_seed, tag, free_sigma=False,
                      sigma_bounds=None, background="pol1"):
    """
    FINAL thallium-peak fit for the before/after comparison.

    Histograms *energies* on the GIVEN shared axis (lo, hi, nb bins) -- so before
    and after use the SAME range and binning -- and fits the Tl peak with a Gaussian
    over a *background* polynomial (gaus(0)+pol1(3) by default; "pol0" for a FLAT
    background) using a Poisson maximum-likelihood
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
    seed  = max(sig_seed, 1.0)
    h_max = max(h.GetMaximum(), 1.0)
    ff = ROOT.TF1(f"fit_{tag}", f"gaus(0)+{background}(3)", lo, hi)
    ff.SetParameter(0, h_max); ff.SetParameter(1, center); ff.SetParameter(2, seed)
    for _ip in range(3, ff.GetNpar()):          # background parameters
        ff.SetParameter(_ip, 0.0)
    # Constrain the Gaussian so the minimiser cannot walk away from the peak:
    #   [0] amplitude  > 0      -- otherwise it can flip NEGATIVE and "fit" a dip
    #                              in the continuum instead of the peak;
    #   [1] mean       near the seeded centre -- the Tl line is known to sit there,
    #                              so it must not latch onto a background bump;
    #   [2] width      near the (reliable) seed -- keeps it on the PEAK and lets
    #                              pol1 take the continuum, instead of ballooning
    #                              into a broad blob that swallows the background.
    # Without the first two limits the fit is unstable across ROOT versions (it
    # converges on ROOT 6.36 but diverges to an inverted Gaussian on 6.40).
    ff.SetParLimits(0, 0.0, 10.0 * h_max)
    if background == "pol0":
        # A FLAT background is a level of counts: it cannot be negative, and with
        # few counts the fit does drive it below zero (it buys a slightly better
        # likelihood under the peak). The constant is the only background parameter
        # here, so the bound is direct. It is started from the mean of the two edge
        # bins rather than from 0, which would sit exactly on the limit.
        _edge = 0.5 * (h.GetBinContent(1) + h.GetBinContent(nb))
        ff.SetParameter(3, max(_edge, 1e-3))
        ff.SetParLimits(3, 0.0, max(10.0 * h_max, 1.0))
    ff.SetParLimits(1, center - PEAK_MEAN_MAX_SHIFT * seed,
                       center + PEAK_MEAN_MAX_SHIFT * seed)
    if sigma_bounds is not None:
        # Explicit, ABSOLUTE width band (used by the chain fits): the plausible
        # range for this line is known, so the bound does not depend on how good
        # the seed is -- with a seed-derived bound an off seed pegs the width and
        # every panel comes out with the same, meaningless resolution.
        ff.SetParLimits(2, *sigma_bounds)
    elif free_sigma:
        # Width left completely FREE: *sig_seed* only starts the search. Used
        # where several samples share one seed (the chain comparison), because a
        # seed-derived bound would pin them all to the same value and destroy the
        # very comparison the panels are meant to show.
        pass
    else:
        # --- seed-derived bound (kept for the per-partition plots) ------------
        # Lower bound well above zero: a Poisson-likelihood fit otherwise
        # maximises the likelihood by collapsing the Gaussian into a needle on
        # the single tallest bin (sigma pegged at the bound, error > value).
        ff.SetParLimits(2, seed * 0.6, seed * 2.0)
    h.Fit(ff, "Q0 R L")
    return h, ff


def shared_before_after_fits(before_e, after_e, center, tag, sig_hint=None,
                             stats_aware=True):
    """
    Fit a BEFORE and an AFTER energy sample of the thallium peak on a SHARED axis
    (equal range + binning) with gaus(0)+pol1(3) Poisson-likelihood fits. The axis
    is window = center +/- BA_WIN_NSIGMA*sigma_ref with bin width ~ sigma_ref/
    BA_RES_BIN_DIV. *after_e* may be None.

    *sig_hint*: a RELIABLE peak width (energy). When given (e.g. the partition's
    clean-peak sigma), it sets BOTH the window and the seed, giving a SMALL window
    tightly matched to the peak -- avoiding the case where a noisy low-statistics
    width estimate inflates the window and lets the background swallow the peak.
    When None (combined spectrum), the window uses the BROADER of the two measured
    peaks and the seed the narrower one.

    *stats_aware*: when True (per-partition), the bin count is additionally capped
    so sparse partitions keep ~BA_TARGET_PER_BIN counts/bin (no empty-bin holes).
    When False (combined spectrum, see combined_before_after_fits) only the
    resolution cap applies, keeping the fine binning the narrow AFTER peak needs.
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
    W  = BA_WIN_NSIGMA * sig_win
    lo, hi = center - W, center + W
    # Resolution cap: do not bin finer than the peak width supports.
    nb = round((hi - lo) / max(sig_win / BA_RES_BIN_DIV, 1e-9))
    if stats_aware:
        # Statistics cap: keep ~BA_TARGET_PER_BIN counts/bin on the SPARSER sample
        # (before/after share one axis), so low-population partitions avoid holes.
        counts_in_win = [int(np.sum(np.isfinite(x) & (x >= lo) & (x <= hi)))
                         for x in (before_e, after_e) if x is not None]
        n_min   = min(counts_in_win) if counts_in_win else 0
        nb_stat = max(1, round(n_min / BA_TARGET_PER_BIN))
        nb = min(nb, nb_stat)
    nb = int(np.clip(nb, BA_MIN_BINS, BA_MAX_BINS))
    h_b = f_b = h_a = f_a = None
    if before_e is not None:
        h_b, f_b = fit_thallium_peak(before_e, center, lo, hi, nb, sig_seed, f"{tag}_before")
    if after_e is not None:
        h_a, f_a = fit_thallium_peak(after_e, center, lo, hi, nb, sig_seed, f"{tag}_after")
    return h_b, f_b, h_a, f_a


def combined_before_after_fits(before_e, after_e, center, tag, sig_hint=None):
    """
    Before/after fits for the COMBINED thallium peak (all partitions merged).

    Dedicated variant of shared_before_after_fits with the ORIGINAL binning:
    resolution-only (bin width ~ sigma_ref/BA_RES_BIN_DIV), WITHOUT the
    statistics cap. The combined AFTER peak is narrow while its window is set by
    the broad BEFORE spread; the statistics cap would then widen the bins and
    smear the peak onto a couple of bins. Keeping the fine resolution binning
    preserves the peak shape.
    Returns (h_before, fit_before, h_after, fit_after) (fit_* may be None).
    """
    return shared_before_after_fits(before_e, after_e, center, tag,
                                    sig_hint=sig_hint, stats_aware=False)


def rough_to_units(values, cal_rough, mask):
    """
    Scale factor k such that  values ~ k * cal_rough, as the median ratio over the
    *mask* events. It converts a window given in ROUGH-calibration units into the
    units of any chain variable, so every panel of the combined canvas covers the
    same physical range on its own scale:
      - rough calibration itself      -> k = 1;
      - corrected amplitude           -> k = the usual raw/rough conversion factor;
      - thallium-stabilized amplitude -> k = keV per rough unit, i.e. about
        TARGET_ENERGY / (Tl peak position in rough units), so k ~ 1 and the window
        stays ~[CHAIN_DISP_MIN, CHAIN_DISP_MAX] keV.
    """
    v, r = values[mask], cal_rough[mask]
    good = np.isfinite(v) & np.isfinite(r) & (r > 0)
    return float(np.median(v[good] / r[good])) if good.any() else 1.0


class ChainSettings:
    """Settings of the amplitude-chain comparison for ONE channel (see CHAIN_CSV_PATH)."""
    # CSV columns, after "channel". The four win_scale entries are keyed by chain
    # variable, not by position, so they stay attached to the right variable even
    # when a column is not plotted (e.g. the heater one on an optimum-filter channel).
    KEYS   = ("rough", "heater", "corrected", "stabilized")
    FIELDS = (*(f"win_scale_{k}" for k in KEYS), *(f"bin_div_{k}" for k in KEYS),
              *(f"sig_scale_{k}" for k in KEYS),
              "peak_nsigma", "sig_lo", "sig_hi")

    def __init__(self, values=None):
        d = dict(zip(self.FIELDS, [
            *(list(CHAIN_WIN_SCALE) + [1.0] * 4)[:4],
            *(list(CHAIN_BIN_DIV)   + [4.0] * 4)[:4],
            *(list(CHAIN_SIG_SCALE) + [1.0] * 4)[:4],
            CHAIN_PEAK_NSIGMA, CHAIN_SIG_LO, CHAIN_SIG_HI]))
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
        """Window widening of the chain variable *key* ('rough', 'heater', ...)."""
        return getattr(self, f"win_scale_{key}", 1.0)

    def sig_scale(self, key):
        """Scale of the EXPECTED width for the chain variable *key*."""
        return getattr(self, f"sig_scale_{key}", 1.0)

    def as_row(self, ch_id):
        row = {"channel": ch_id}
        row.update({k: getattr(self, k) for k in self.FIELDS})
        return row


def chain_settings(ch_id):
    """
    Settings of the chain comparison for channel *ch_id*.

    Read from CHAIN_CSV_PATH when USE_CHAIN_CSV and the channel has a row there;
    otherwise the program defaults. Either way, a channel with no row is appended
    to the file with the values in use, so the CSV ends up listing every analysed
    channel without a separate "save" step. Existing rows are left untouched.
    """
    rows, header, on_file = [], ["channel", *ChainSettings.FIELDS], None
    if os.path.exists(CHAIN_CSV_PATH):
        try:
            with open(CHAIN_CSV_PATH, newline="") as fh:
                reader  = csv.DictReader(fh)
                rows    = list(reader)
                on_file = list(reader.fieldnames or [])
        except OSError as e:
            print(f"  [!] Cannot read {CHAIN_CSV_PATH}: {e}", file=sys.stderr)

    # A new setting adds a COLUMN: an older file is upgraded in place, the missing
    # cells filled with the program defaults (ChainSettings does that on read).
    # Without this the appended rows would be written against a header that does
    # not describe them, and every value after the missing column would shift.
    if on_file is not None and any(f not in on_file for f in header):
        try:
            upgraded = [ChainSettings(r).as_row(str(r.get("channel", "")).strip())
                        for r in rows]
            with open(CHAIN_CSV_PATH, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
                w.writeheader()
                w.writerows(upgraded)
            print(f">>> {os.path.basename(CHAIN_CSV_PATH)} upgraded with the new "
                  f"columns ({', '.join(f for f in header if f not in on_file)}).")
        except OSError as e:
            print(f"  [!] Cannot upgrade {CHAIN_CSV_PATH}: {e}", file=sys.stderr)

    mine = next((r for r in rows if str(r.get("channel", "")).strip() == str(ch_id)), None)
    cfg  = ChainSettings(mine if (mine and USE_CHAIN_CSV) else None)
    if mine is not None:
        if USE_CHAIN_CSV:
            print(f">>> Chain settings for ch {ch_id} read from "
                  f"{os.path.basename(CHAIN_CSV_PATH)}.")
        return cfg

    try:                                  # channel not on file yet: append it
        with open(CHAIN_CSV_PATH, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            if not rows and fh.tell() == 0:
                w.writeheader()
            w.writerow(cfg.as_row(ch_id))
        print(f">>> Chain settings of ch {ch_id} added to "
              f"{os.path.basename(CHAIN_CSV_PATH)} (program values).")
    except OSError as e:
        print(f"  [!] Cannot write {CHAIN_CSV_PATH}: {e}", file=sys.stderr)
    return cfg


# Columns of the results CSV (long format: ONE ROW PER FITTED PANEL, so a new
# amplitude or a new background model only adds rows, never columns).
#   step/variable/label : position, key and name of the amplitude in the chain
#   row                 : "native" = row 2 of the canvas, in the variable's own
#                         units; "energy" = row 3, rescaled so the peak sits on
#                         TARGET_ENERGY (the one to use to compare channels)
#   background          : "pol1" (combined_thallium) or "pol0" (…_flatbkg)
#   n_peak              : counts under the Gaussian; n_hist: entries in the window
#
# Canonical order of the chain, used for "step": it must NOT depend on the
# column the variable ends up in, because on the OPTIMUM_FILTER_CHANNELS the
# heater column is missing and every later variable would shift by one.
CHAIN_STEP_ORDER = ("rough", "heater", "corrected", "stabilized")

RES_CSV_FIELDS = [
    "channel", "step", "variable", "label", "row", "background",
    "mu", "mu_err", "sigma", "sigma_err", "fwhm", "fwhm_err",
    "resolution_pct", "resolution_err_pct",
    "chi2", "ndf", "prob", "n_peak", "n_hist", "date",
]


def _res_csv_sort_key(r):
    """Order of the results CSV: channel, then chain position, row, background."""
    try:
        ch = int(str(r.get("channel", "")).strip())
    except (TypeError, ValueError):
        ch = 1 << 30
    try:
        step = int(str(r.get("step", "")).strip())
    except (TypeError, ValueError):
        step = 1 << 30
    return (ch, step,
            0 if str(r.get("row", "")) == "native" else 1,
            str(r.get("background", "")))


def write_resolution_csv(path, ch_id, new_rows):
    """
    Write this channel's fit results to the results CSV at *path*.

    The file accumulates the whole detector: the rows of *ch_id* are REPLACED
    (a re-analysis must not leave the old numbers behind) and every other
    channel is kept as it is. Rows are rewritten sorted, so the file stays
    readable when channels are analysed out of order.
    """
    old = []
    if os.path.exists(path):
        try:
            with open(path, newline="") as fh:
                old = [r for r in csv.DictReader(fh)
                       if str(r.get("channel", "")).strip() != str(ch_id)]
        except OSError as e:
            print(f"  [!] Cannot read {path}: {e}", file=sys.stderr)

    rows = sorted(old + list(new_rows), key=_res_csv_sort_key)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=RES_CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f">>> {len(new_rows)} fit result(s) of ch {ch_id} written to "
              f"{os.path.basename(path)} ({len(rows)} row(s) in total).")
    except OSError as e:
        print(f"  [!] Cannot write {path}: {e}", file=sys.stderr)


def collect_resolution_rows(ch_id, chain):
    """
    One CSV row per fitted panel of the combined-thallium canvas: the two rows
    of the canvas (native units and energy) times the two background models.
    Panels whose fit did not converge are skipped, not written as zeros.
    """
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    panels = (("native", "pol1", "h_zoom",     "fit_zoom"),
              ("native", "pol0", "h_zoom_c",   "fit_zoom_c"),
              ("energy", "pol1", "h_energy",   "fit_energy"),
              ("energy", "pol0", "h_energy_c", "fit_energy_c"))

    def num(v, fmt="{:.6g}"):
        return fmt.format(v) if (v is not None and math.isfinite(v)) else "nan"

    rows = []
    for p in chain:
        for row_name, bkg, hkey, fkey in panels:
            m = fit_metrics(p.get(fkey), p.get(hkey))
            if m is None:
                continue
            step = (CHAIN_STEP_ORDER.index(p["key"])
                    if p["key"] in CHAIN_STEP_ORDER else p["idx"])
            rows.append({
                "channel": ch_id, "step": step, "variable": p["key"],
                "label": p["label"], "row": row_name, "background": bkg,
                "mu": num(m["mu"]),           "mu_err": num(m["mu_err"]),
                "sigma": num(m["sigma"]),     "sigma_err": num(m["sigma_err"]),
                "fwhm": num(m["fwhm"]),       "fwhm_err": num(m["fwhm_err"]),
                "resolution_pct": num(m["res"], "{:.4f}"),
                "resolution_err_pct": num(m["res_err"], "{:.4f}"),
                "chi2": num(m["chi2"], "{:.3f}"), "ndf": m["ndf"],
                "prob": num(m["prob"], "{:.4f}"),
                "n_peak": num(m["n_peak"], "{:.2f}"),
                "n_hist": num(m["n_hist"], "{:.0f}"),
                "date": stamp,
            })
    return rows


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


def fit_peak_centred(values, mu, tag, interval, cfg, key, background="pol1"):
    """
    Fit the thallium peak of *values* with the recipe of ThalliumStabilization_old:

      1. Take the peak position *mu* as given: the position the stabilization
         measured, converted to these units with the factors of the global
         overview. It is NOT searched for here -- inside the stabilization window
         the continuum can be taller than the line (on the rough and heater
         amplitudes it often is) and a peak finder locks onto the window edge,
         while the converted position is right to a fraction of a per cent.
      2. Frame it with the peak INTERVAL of the stabilization (*interval* =
         (frac, res_exp) from chain_peak_interval): the window is mu*(1 -/+ frac),
         symmetric so the peak keeps a white margin on BOTH sides. It is an
         ABSOLUTE interval, taken from
         where the stabilization found the thallium events, so it contains the line
         whatever the continuum does around it -- unlike a window built from the
         locally measured width, which on a strong continuum comes out several
         times too wide and lets the fit balloon over the background.
      3. Bin it on the EXPECTED width (res_exp*mu), so the binning does not depend
         on a local measurement either. The bin width is the one set for this
         chain variable (*key*, see ChainSettings).
      4. gaus(0)+*background*(3) Poisson-likelihood fit, the width bounded by
         [CHAIN_SIG_LO, CHAIN_SIG_HI] x that expected width: that is what keeps the
         Gaussian on the peak instead of letting it spread onto the continuum.

    Returns (hist, fit, mu, sigma); (None, None, 0.0, 0.0) when it cannot fit.
    """
    if not (mu > 0):
        return None, None, 0.0, 0.0

    frac, res_exp = interval

    # On the MAIN amplitude the position is already exact: *mu* is the peak the
    # stabilization measured on this very variable (its conversion factor is
    # k_main, so the conversion is the identity). The window given is therefore
    # used as it is -- no search, no recentring. Looking for the peak again can
    # only move it off: on some channels the finder drifts onto a neighbouring
    # structure a couple of per cent away.
    refine = (key != CHAIN_MAIN_KEY)
    if refine:
        # Elsewhere the position is converted, and the conversion factors are
        # medians over a window the continuum dominates, so it can be a few per
        # cent off -- too much for the fit to recover, its mean being bounded. The
        # peak is looked for in a BAND around it: wide enough to contain the line
        # even then, narrow enough to keep the finder away from the continuum that
        # piles up at the edges of the stabilization window.
        _lo_s, _hi_s = mu * (1.0 - CHAIN_SEARCH_FRAC), mu * (1.0 + CHAIN_SEARCH_FRAC)
        _, _seed = fit_peak_optimized(values, _lo_s, _hi_s, f"{tag}_seed")
        if _seed is not None and _lo_s < _seed.GetParameter(1) < _hi_s:
            mu = _seed.GetParameter(1)

    h = f = None
    # TWO PASSES. The first window is centred on the position given (converted from
    # the peak the stabilization measured); the second is re-centred on the peak
    # that fit actually found, so the framing ends up on the line even when the
    # conversion is off by a fraction of a per cent. It is a refinement, not a
    # search: fit_thallium_peak keeps the mean within PEAK_MEAN_MAX_SHIFT sigmas of
    # the centre, so the window cannot walk away onto the continuum.
    n_pass = 2 if refine else 1
    # Expected width for THIS variable: the one the stabilization measured, times
    # the per-variable factor (see CHAIN_SIG_SCALE). It sets the binning, the
    # bounds on the fitted width and the leash on the mean, so on a variable
    # whose line is genuinely broader it has to be scaled up or nothing fits.
    res_var = res_exp * cfg.sig_scale(key)
    for _pass in range(n_pass):
        sigma   = res_var * abs(mu)                # expected width in these units
        sig_min = cfg.sig_lo * sigma
        sig_max = cfg.sig_hi * sigma
        if not (sig_max > sig_min > 0):
            return None, None, 0.0, 0.0
        if frac > 0.0:
            lo, hi = mu * (1.0 - frac), mu * (1.0 + frac)
        else:                                      # no partition peak available
            lo, hi = mu - CHAIN_WIN_NSIGMA * sigma, mu + CHAIN_WIN_NSIGMA * sigma
        nb = int(np.clip(round((hi - lo) / (sigma / cfg.bin_div(key))),
                         CHAIN_MIN_BINS, CHAIN_MAX_BINS))
        # The last pass keeps *tag*, so the objects that are drawn carry the
        # expected name; the first one is only used to locate the peak.
        h, f = fit_thallium_peak(values, mu, lo, hi, nb, sigma,
                                 tag if _pass == n_pass - 1 else f"{tag}_pre",
                                 sigma_bounds=(sig_min, sig_max), background=background)
        if f is None:
            return None, None, 0.0, 0.0
        if not (f.GetParameter(1) > 0):
            break
        mu = f.GetParameter(1)
    return h, f, f.GetParameter(1), abs(f.GetParameter(2))


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

    # Look for the peak around the combined-spectrum position, then decide whether
    # to TRUST the local one: it is adopted only if it really stands out of the
    # continuum (peak_significance). Otherwise the combined position is used, which
    # is measured on far more events. Judging by significance rather than by event
    # count keeps a clear peak found on few events, and rejects a noise bump found
    # on many.
    search_min = search_max = None
    if peak_hint is not None:
        hint_center, hint_sigma = peak_hint
        half_width = PEAK_HINT_NSIGMA * abs(hint_sigma)
        search_min, search_max = hint_center - half_width, hint_center + half_width

    peak_x, peak_y = FindThalliumPeak(res.h_heat_orig, search_min, search_max)

    if peak_hint is not None:
        signif = peak_significance(res.h_heat_orig, peak_x, hint_sigma)
        if signif < PEAK_SIGNIF_MIN:
            peak_x = hint_center
            peak_y = res.h_heat_orig.GetBinContent(
                res.h_heat_orig.GetXaxis().FindBin(hint_center))
            print(f"  -> Ch {ch_id} P{idx}: local peak not significant "
                  f"({signif:.1f} < {PEAK_SIGNIF_MIN} sigma, {n_stab} events); "
                  f"position taken from the combined spectrum ({hint_center:.1f}).")
        else:
            print(f"  -> Ch {ch_id} P{idx}: local peak at {peak_x:.1f} "
                  f"({signif:.1f} sigma over the continuum, {n_stab} events).")

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
    tree_time = file.Get("timestamp")   # heat_timefromstartrun (baseline-vs-time plot)
    tree_heater = file.Get("flagpropagator_heater")   # IsHeater flag (pulser events)
    # Optimum-filter tree, also used to compute the LY when the 'LY' tree is gone.
    tree_opt  = tree_main if main_tree_name == AMP_TREE_OPTIMUM else file.Get(AMP_TREE_OPTIMUM)
    # Heater-stabilized amplitude tree: one more step of the processing chain,
    # compared side by side in the combined-thallium canvas.
    tree_stab = tree_main if main_tree_name == AMP_TREE_FALLBACK else file.Get(AMP_TREE_FALLBACK)

    tree_for_stabilization = tree_main

    print(f"{'='*60}\n")

    # --- Attach friend trees to the main tree -------------------------------
    # Only VALID trees are attached: a missing tree from file.Get() is a null
    # pointer (not Python None), so it must be filtered with _valid_tree. The
    # LY tree and the optimum-filter tree (LY ratio fallback) are included too.
    candidate_friends = [tree_mod, tree_bad, tree_trig, tree_corr, tree_cal,
                         tree_baseline, tree_ly, tree_opt, tree_time, tree_heater,
                         tree_stab]
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
    has_heater_flag      = bool(tree_for_stabilization.GetLeaf("IsHeater"))
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

    # --- Earlier steps of the amplitude chain (combined-thallium comparison) --
    # The same thallium peak is also measured on the optimum-filter amplitude and
    # on the heater-stabilized amplitude, so the canvas can show how the
    # resolution evolves along the processing chain.
    has_amp_stab = _valid_tree(tree_stab)
    if has_amp_stab:
        _pfx = "" if main_tree_name == AMP_TREE_FALLBACK else f"{AMP_TREE_FALLBACK}."
        rdf_f = rdf_f.Define("_amp_stab_", f"{_pfx}heat_amplitude")

    # --- Columns to read ----------------------------------------------------
    cols = ["heat_amplitude", "_cal_rough_", "heat_correlation", "heat_baseline"]
    if has_time:
        cols.append("_time_")
    if apply_ly_cut:
        cols.append("_ld1_ly_")
        if has_ld2_branch:
            cols.append("_ld2_ly_")
    if has_amp_stab:
        cols.append("_amp_stab_")

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

    # Earlier amplitude-chain steps (kept aligned with the other arrays).
    amp_stab = np_data["_amp_stab_"][:N].astype(np.float64) if has_amp_stab else None

    # --- Vectorised NaN/Inf removal -----------------------------------------
    valid = np.isfinite(ha) & np.isfinite(cal_rough) & np.isfinite(corr)
    ha        = ha[valid];   cal_rough = cal_rough[valid]
    corr      = corr[valid]; baseline = baseline[valid]
    ld1_ly    = ld1_ly[valid]; ld2_ly = ld2_ly[valid]
    if amp_stab is not None: amp_stab = amp_stab[valid]
    if time_fsr is not None:
        time_fsr = time_fsr[valid]   # keep the time vector aligned with baseline
    N = len(ha)

    # ======================================================================
    # HEATER-CORRELATION LOWER BOUND
    # ======================================================================
    # Lower edge of the interval over which the correlation-cut percentile is
    # taken. Instead of the fixed CORR_VALID_MIN, use the correlation value just
    # ABOVE the heater (pulser) events (mean + 5*sigma of their correlation
    # peak). Heater events are read RAW here (IsHeater==1, corr>HEATER_CORR_MIN,
    # NO quality filters), independent of the main-event selection. Falls back to
    # CORR_VALID_MIN when the heater flag is absent or the fit is unreliable.
    if has_heater_flag:
        _hd = (ROOT.RDataFrame(tree_for_stabilization)
               .Filter("IsHeater == 1")
               .AsNumpy(["heat_correlation", "heat_amplitude"]))
        heater_corrs = np.asarray(_hd["heat_correlation"], np.float64)
        heater_amps  = np.asarray(_hd["heat_amplitude"],   np.float64)
        corr_valid_min_eff, h_corr_heater, fit_corr_heater = AnalyzeHeaterCorrThreshold(
            heater_corrs, ch_id, CORR_VALID_MIN)
    else:
        corr_valid_min_eff, h_corr_heater, fit_corr_heater = CORR_VALID_MIN, None, None
        heater_corrs = np.empty(0, np.float64)
        heater_amps  = np.empty(0, np.float64)
        print(f">>> Ch {ch_id}: no 'IsHeater' flag; correlation interval lower bound "
              f"= CORR_VALID_MIN = {CORR_VALID_MIN:.6f}.")

    # ======================================================================
    # DYNAMIC CORRELATION CUT  (vectorised)
    # ======================================================================
    mask_corr_valid = corr > corr_valid_min_eff
    corr_above      = corr[mask_corr_valid]
    ha_above        = ha[mask_corr_valid]
    corr_sorted     = np.sort(corr_above)
    n_corr          = len(corr_sorted)

    corr_hist_min    = float(corr_sorted[int(n_corr * 0.01)]) if n_corr > 0 else corr_valid_min_eff
    corr_cut_dynamic = (float(corr_sorted[int(n_corr * CORR_CUT_PERCENTILE)])
                        if n_corr > 0 else 0.9995)

    print(f">>> Correlation interval lower bound (above heater): {corr_valid_min_eff:.6f}")
    print(f">>> Correlation Cut ({int(CORR_CUT_PERCENTILE*100)}th percentile): {corr_cut_dynamic:.6f}")

    # Scatter (decimated for display) and distribution of the correlation.
    g_corr_vs_heat = make_scatter_graph(ha_above, corr_above)
    g_corr_vs_heat.SetName(f"g_corr_vs_heat_{ch_id}")
    g_corr_vs_heat.SetTitle(
        f"Ch {ch_id}: Correlation vs Heat Amplitude; Heat Amplitude; Correlation")

    # Diagnostic scatter for the correlation plot's left panel: ALL quality
    # events (blue) with the heater events overlaid (red). It lets one check by
    # eye that the lower-bound line sits just ABOVE the dense heater block.
    g_phys_disp   = make_scatter_graph(ha, corr)
    g_heater_disp = make_scatter_graph(heater_amps, heater_corrs)
    # Display y-range: from a bit below the heater block up to 1, so both the
    # heater cluster and the lower-bound line are visible.
    _hblock = heater_corrs[np.isfinite(heater_corrs) & (heater_corrs > HEATER_CORR_MIN)]
    corr_disp_lo = ((min(corr_valid_min_eff, float(np.median(_hblock))) - 0.003)
                    if _hblock.size else corr_valid_min_eff - 0.003)
    # Fixed x-range (amplitude has far outliers that would flatten the scatter).
    corr_disp_xlo, corr_disp_xhi = 0.0, 12000.0

    # Histogram spans the whole percentile interval: from the heater lower bound
    # (corr_valid_min_eff) up to 1, so the vertical cut line can be checked.
    h_corr = ROOT.TH1F(f"h_corr_{ch_id}",
                        f"Ch {ch_id}: Correlation Distribution; Correlation; Counts",
                        100, corr_valid_min_eff, 1.00005)
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

    # NOTE: a per-variable conversion factor (median of amp/cal_rough) used to be
    # computed here to express the stabilization window in the optimum-filter
    # units. It is no longer needed: the chain comparison selects the events ONCE
    # on the corrected amplitude and reads the same events through every variable,
    # so no window conversion is involved.

    # ======================================================================
    # SPECTRAL HISTOGRAMS  (bulk FillN)
    # ======================================================================
    n_main  = int(mask_main.sum())
    _w_main = np.ones(n_main, np.double)

    h_cal_rough = ROOT.TH1F(f"h_cal_rough_{ch_id}",
                             f"Ch {ch_id}: Calibration Rough after Correlation Cut; Amplitude; Counts",
                             CHAIN_FULL_BINS, CHAIN_DISP_MIN, CHAIN_DISP_MAX)
    if n_main > 0:
        h_cal_rough.FillN(n_main, cal_rough[mask_main].astype(np.double), _w_main)

    h_raw = ROOT.TH1F(f"h_raw_{ch_id}",
                       f"Ch {ch_id}: Heater Stabilized after Correlation Cut; Amplitude; Counts",
                       CHAIN_FULL_BINS, CHAIN_DISP_MIN * conv_raw, CHAIN_DISP_MAX * conv_raw)
    if n_main > 0:
        h_raw.FillN(n_main, ha[mask_main].astype(np.double), _w_main)

    h2_full = ROOT.TH1F(f"h2_full_{ch_id}",
                         f"Ch {ch_id}: Heater Stabilized after Correlation and LY Cut; Amplitude; Counts",
                         CHAIN_FULL_BINS, CHAIN_DISP_MIN * conv_factor, CHAIN_DISP_MAX * conv_factor)
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
    # AMPLITUDE-CHAIN COMPARISON  (combined-thallium canvas)
    # ======================================================================
    # One COLUMN per amplitude variable of the processing chain, three ROWS:
    #   1. the stabilization range [CHAIN_DISP_MIN, CHAIN_DISP_MAX] (rough units)
    #      converted into the variable's own units -- the spectrum as it is;
    #   2. the same data zoomed on the thallium peak, with the gaus+pol1 fit;
    #   3. the same peak rescaled to energy (peak -> TARGET_ENERGY) on ONE shared
    #      axis, so the FWHMs of all the variables are directly comparable.
    # Events: correlation + LY cut, for every variable. The window is then
    # converted per variable (rough_to_units), so none of them is cut on another
    # variable's scale -- which on a merged run would mix gain-misaligned runs.

    # Per-event thallium-stabilized amplitude (the quantity the output ROOT
    # stores): every event corrected with the line of its own baseline partition.
    slopes = np.array([r.slope for r in part_results], np.float64)
    q0s    = np.array([r.q_0   for r in part_results], np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        amp_tl_stab = TARGET_ENERGY * ha / (q0s[part_of_event]
                                            + slopes[part_of_event] * baseline)

    # (label, per-event values, x-axis title, colour). The main-tree amplitude is
    # 'corrected_amplitude' except on the OPTIMUM_FILTER_CHANNELS.
    main_label = ("Corrected amplitude" if main_tree_name == AMP_TREE_DEFAULT
                  else main_tree_name)
    # (key, label, values, x-axis title, colour). The key ties the variable to its
    # per-channel settings (ChainSettings), independently of the column position.
    # On an OPTIMUM_FILTER_CHANNELS channel only the rough calibration and the
    # thallium-stabilized amplitude are kept (OPTIMUM_FILTER_CHAIN_KEYS): the two
    # heater columns are steps of a chain that channel does not follow, and the
    # main-amplitude column would repeat the rough one -- it IS the optimum-filter
    # amplitude, the same quantity the rough panel shows one calibration apart.
    chain_defs = [
        ("rough",      "Calibration rough",   cal_rough,   "Rough amplitude",  ROOT.kAzure + 1),
        ("heater",     "Heater stabilized",   amp_stab,    "Amplitude (a.u.)", ROOT.kGreen + 2),
        ("corrected",  main_label,            ha,          "Amplitude (a.u.)", ROOT.kGray + 2),
        ("stabilized", "Thallium stabilized", amp_tl_stab, "Energy (keV)",     ROOT.kRed + 1),
    ]
    if use_optimum_filter:
        chain_defs = [c for c in chain_defs if c[0] in OPTIMUM_FILTER_CHAIN_KEYS]

    # Thallium peak position in ROUGH units, measured where the line dominates
    # (mask_conv = correlation cut + Tl window). Converted with each variable's
    # factor it says where to look for the line in that variable -- necessary
    # because over the full display range the continuum dominates the spectrum.
    mu_rough, _ = estimate_thallium_peak(cal_rough[mask_conv], f"{ch_id}_rough")
    if not (mu_rough > 0):
        mu_rough = TARGET_ENERGY

    # Width expected for the thallium peak of this channel, from the partition
    # peaks the stabilization already measured: it bounds every panel's fit.
    cfg = chain_settings(ch_id)
    tl_frac, tl_mu_main, tl_res = chain_peak_interval(part_results, cfg.peak_nsigma)
    tl_interval = (tl_frac, tl_res)
    # Conversion factor of the MAIN amplitude: the same rough -> amplitude factor
    # the global overview uses. Every variable's factor is expressed relative to
    # it, so the peak position measured by the stabilization carries over.
    k_main = rough_to_units(ha, cal_rough, mask_conv)
    print(f">>> Chain fit window: +/-{100*tl_interval[0]:.2f}% around the peak, "
          f"expected width {100*tl_interval[1]:.2f}% (both from the partition peaks).")

    # --- rows 1 and 2: each variable in its OWN units ------------------------
    chain = []
    for i, (key, label, values, xtitle, colour) in enumerate(chain_defs):
        if values is None:
            continue
        # mask_conv (correlation cut + Tl window) is the same sample the raw/rough
        # conversion factor is measured on, so the corrected-amplitude panel spans
        # exactly the range of the global-overview spectrum.
        k      = rough_to_units(values, cal_rough, mask_conv)
        lo, hi = CHAIN_DISP_MIN * k, CHAIN_DISP_MAX * k
        sel    = values[mask_ly_pass]
        sel    = sel[np.isfinite(sel)]

        h_full = ROOT.TH1F(f"h_chain_full_{ch_id}_{i}",
                           f"Ch {ch_id}: {label};{xtitle};Counts",
                           CHAIN_FULL_BINS, lo, hi)
        h_full.SetDirectory(0)
        if sel.size > 0:
            h_full.FillN(sel.size, sel.astype(np.double), np.ones(sel.size, np.double))

        # Rows 2 and 3: the peak is measured on the stabilization-window events
        # (where the line dominates) and framed with the thallium interval.
        # Where the line sits in these units: the stabilization peak converted
        # with this variable's factor. Given to the fit, not searched for.
        mu_exp = tl_mu_main * k / k_main if (tl_mu_main > 0 and k_main > 0) else 0.0
        # Widen the shared peak interval for this variable (see CHAIN_WIN_SCALE).
        interval = (tl_interval[0] * cfg.win_scale(key), tl_interval[1])
        h_zoom, fit_zoom, mu, sigma = fit_peak_centred(
            sel, mu_exp, f"h_chain_zoom_{ch_id}_{i}", interval, cfg, key)
        if h_zoom is not None:
            h_zoom.SetTitle(f"Ch {ch_id}: {label} - peak;{xtitle};Counts")
        # Same fit with a FLAT background, drawn in a parallel canvas: over a window
        # this narrow the continuum is nearly flat, and a constant has one parameter
        # less to spend on the few counts around the peak.
        h_zoom_c, fit_zoom_c, _, _ = fit_peak_centred(
            sel, mu_exp, f"h_chain_zoomc_{ch_id}_{i}", interval, cfg, key,
            background="pol0")
        if h_zoom_c is not None:
            h_zoom_c.SetTitle(f"Ch {ch_id}: {label} - peak;{xtitle};Counts")

        chain.append(dict(
            idx=i, key=key, label=label, colour=colour, h_full=h_full, interval=interval,
            h_zoom=h_zoom, fit_zoom=fit_zoom,
            h_zoom_c=h_zoom_c, fit_zoom_c=fit_zoom_c,
            # the same sample rescaled so its peak sits on TARGET_ENERGY (row 3)
            energy=(TARGET_ENERGY * sel / mu)      if mu > 0 else None,
            ))

    # --- row 3: the same peaks, in energy ------------------------------------
    # Each sample rescaled so its peak sits on TARGET_ENERGY, then fitted with the
    # SAME routine as row 2: the panel is row 2 on the energy scale, so the two
    # rows always agree and the resolutions of the different amplitudes can be
    # compared directly from the boxes (all the peaks now sit at the same energy).
    for p in chain:
        p["h_energy"], p["fit_energy"] = None, None
        p["h_energy_c"], p["fit_energy_c"] = None, None
        if p["energy"] is None:
            continue
        p["h_energy"], p["fit_energy"], _, _ = fit_peak_centred(
            p["energy"], TARGET_ENERGY, f"h_chain_energy_{ch_id}_{p['idx']}",
            p["interval"], cfg, p["key"])
        p["h_energy_c"], p["fit_energy_c"], _, _ = fit_peak_centred(
            p["energy"], TARGET_ENERGY, f"h_chain_energyc_{ch_id}_{p['idx']}",
            p["interval"], cfg, p["key"], background="pol0")
        for _h in (p["h_energy"], p["h_energy_c"]):
            if _h is not None:
                _h.SetTitle(f"Ch {ch_id}: {p['label']} - energy;Energy (keV);Counts")

    # Counts under the stabilized thallium peak, from its Gaussian component.
    stab_panel = next((p for p in chain if p["label"] == "Thallium stabilized"), None)
    if stab_panel is not None and stab_panel["fit_energy"] is not None:
        f_st = stab_panel["fit_energy"]
        gaussian_counts = (f_st.GetParameter(0) * f_st.GetParameter(2)
                           * math.sqrt(2 * math.pi)
                           / stab_panel["h_energy"].GetXaxis().GetBinWidth(1))
    else:
        gaussian_counts = 0.0
        print("  [!] Warning: no event survived the cuts in any partition.")

    # --- Results table: the fitted resolutions of the chain, on file ---------
    # Written on the FINAL run only: in the interactive GUI the canvases are
    # rebuilt at every parameter change, and those previews must not end up in
    # the results file (the accepted run goes through show_canvas=False).
    if SAVE_RES_CSV and not show_canvas:
        res_rows = collect_resolution_rows(ch_id, chain)
        if res_rows:
            write_resolution_csv(
                os.path.join(output_dir, RES_CSV_DIR_NAME,
                             f"{os.path.splitext(RES_CSV_NAME)[0]}{calib_suffix}.csv"),
                ch_id, res_rows)

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

        # ----------------------------- COMBINED THALLIUM: AMPLITUDE CHAIN (C)
        # One COLUMN per amplitude variable of the chain, three ROWS:
        #   1. full stabilization range, in the variable's own units (no fit);
        #   2. the same data zoomed on the Tl peak, with the fit;
        #   3. the peak rescaled to energy, so the resolutions of the different
        #      amplitudes can be compared directly.
        # The canvas is produced TWICE, with the two background models of the peak
        # fit -- linear and flat -- so they can be compared side by side.
        n_col = len(chain)

        def build_chain_canvas(suffix, keys, title_note):
            """Draw the 3 x n_col chain canvas using the given set of fit keys."""
            kz, kfz, ke, kfe = keys
            c = ROOT.TCanvas(f"c_comb{suffix}_{ch_id}",
                             f"Combined Thallium Peak Ch {ch_id}{title_note}",
                             600 * n_col, 1350)
            c.Divide(n_col, 3)

            def draw(pad, h, fit, header, colour):
                c.cd(pad); ROOT.gPad.SetGrid()
                if h is None:
                    return
                h.SetStats(0); h.SetLineColor(ROOT.kBlack)
                h.SetFillColorAlpha(colour, 0.5); h.Draw()
                if fit is None:                 # first row: spectrum only
                    return
                fit.SetLineColor(ROOT.kBlue); fit.SetLineWidth(2); fit.Draw("same")
                # Narrow and tall: the peak sits in the middle of the pad, so a wide
                # box would cover it.
                box = CreateFitBox(fit, header, colour,
                                   x1=0.13, y1=0.64, x2=0.40, y2=0.85, text_size=0.026)
                box.Draw(); global_lines.append(box)

            for col, p in enumerate(chain):
                draw(col + 1,             p["h_full"], None,      p["label"], p["colour"])
                draw(n_col + col + 1,     p[kz],       p[kfz],    p["label"], p["colour"])
                draw(2 * n_col + col + 1, p[ke],       p[kfe],
                     f"{p['label']} (energy)", p["colour"])

            c.Update()
            if save_summary_jpeg:
                save_canvas_jpeg(c, os.path.join(
                    debug_ch_dir, f"ch{ch_id}_combined_thallium{suffix}{calib_suffix}.jpg"))
            canvases.append(c)

        build_chain_canvas("", ("h_zoom", "fit_zoom", "h_energy", "fit_energy"),
                           "")
        build_chain_canvas("_flatbkg",
                           ("h_zoom_c", "fit_zoom_c", "h_energy_c", "fit_energy_c"),
                           " (flat background)")

    # ----------------------------------------------------------- CORRELATION
    make_corr = show_canvas or save_corr_jpeg
    if make_corr:
        c_corr = ROOT.TCanvas(f"c_corr_{ch_id}", f"Correlation Analysis Ch {ch_id}", 1800, 600)
        c_corr.Divide(3, 1)

        # -- Left: correlation vs heat amplitude (physics blue + heater red) -----
        # Horizontal line at corr_valid_min_eff: the correlation value just ABOVE
        # the heater cluster (the minimum of the percentile interval). The heater
        # events (red) let one verify the line sits just above their block.
        c_corr.cd(1); ROOT.gPad.SetGrid()
        frame_corr = ROOT.gPad.DrawFrame(
            corr_disp_xlo, corr_disp_lo, corr_disp_xhi, 1.00005,
            f"Ch {ch_id}: Correlation vs Heat Amplitude; Heat Amplitude; Correlation")
        global_lines.append(frame_corr)
        g_phys_disp.SetMarkerStyle(20); g_phys_disp.SetMarkerSize(0.4)
        g_phys_disp.SetMarkerColor(ROOT.kBlue); g_phys_disp.Draw("P same")
        if g_heater_disp.GetN() > 0:
            g_heater_disp.SetMarkerStyle(20); g_heater_disp.SetMarkerSize(0.4)
            g_heater_disp.SetMarkerColor(ROOT.kRed); g_heater_disp.Draw("P same")
        c_corr.Update()
        lc1 = ROOT.TLine(ROOT.gPad.GetUxmin(), corr_valid_min_eff,
                          ROOT.gPad.GetUxmax(), corr_valid_min_eff)
        lc1.SetLineColor(ROOT.kAzure + 1); lc1.SetLineWidth(2); lc1.SetLineStyle(2)
        lc1.Draw("same"); global_lines.append(lc1)
        leg1 = ROOT.TLegend(0.30, 0.13, 0.88, 0.30)
        leg1.SetTextSize(0.03); leg1.SetFillColorAlpha(ROOT.kWhite, 0.7)
        leg1.AddEntry(g_phys_disp,   "physics events", "p")
        leg1.AddEntry(g_heater_disp, "heater events",  "p")
        leg1.AddEntry(lc1, f"min (above heater) = {corr_valid_min_eff:.6f}", "l")
        leg1.Draw("same"); global_lines.append(leg1)

        # -- Middle: heater correlation histogram + fit + threshold line ---------
        # The heater-only correlation distribution, the Gaussian fit of its
        # rightmost peak, and a vertical line at corr_valid_min_eff
        # (= mean + HEATER_CORR_NSIGMA*sigma, the lower bound derived from the
        # fit). The x-range matches the first scatter's y-range so the two panels
        # line up. Log-y so the low pedestal shows.
        c_corr.cd(2); ROOT.gPad.SetGrid(); ROOT.gPad.SetLogy()
        if h_corr_heater is not None:
            h_corr_heater.SetLineColor(ROOT.kBlack)
            h_corr_heater.SetFillColorAlpha(ROOT.kRed, 0.3)
            h_corr_heater.GetXaxis().SetRangeUser(
                max(corr_disp_lo, HEATER_CORR_MIN), 1.00005)
            h_corr_heater.Draw()
            if fit_corr_heater is not None:
                fit_corr_heater.SetLineColor(ROOT.kBlue); fit_corr_heater.SetLineWidth(2)
                fit_corr_heater.Draw("same")
            c_corr.Update()
            # Log-y pad: gPad y-limits are log10; convert back for the TLine.
            y_lo2, y_hi2 = 10 ** ROOT.gPad.GetUymin(), 10 ** ROOT.gPad.GetUymax()
            lc2 = ROOT.TLine(corr_valid_min_eff, y_lo2, corr_valid_min_eff, y_hi2)
            lc2.SetLineColor(ROOT.kAzure + 1); lc2.SetLineWidth(2); lc2.SetLineStyle(2)
            lc2.Draw("same"); global_lines.append(lc2)
            leg2 = ROOT.TLegend(0.13, 0.75, 0.70, 0.88)
            leg2.SetTextSize(0.03); leg2.SetFillColorAlpha(ROOT.kWhite, 0.7)
            if fit_corr_heater is not None:
                leg2.AddEntry(fit_corr_heater, "heater peak fit", "l")
            leg2.AddEntry(lc2, f"min (mean+{HEATER_CORR_NSIGMA:g}#sigma) = "
                               f"{corr_valid_min_eff:.6f}", "l")
            leg2.Draw("same"); global_lines.append(leg2)

        # -- Right: correlation distribution [min, 1], with the actual cut line --
        # Histogram runs from corr_valid_min_eff to 1; vertical line at the real
        # cut corr_cut_dynamic (the CORR_CUT_PERCENTILE-th percentile).
        c_corr.cd(3); ROOT.gPad.SetGrid()
        h_corr.SetLineColor(ROOT.kBlack); h_corr.SetFillColorAlpha(ROOT.kBlack, 0.3)
        h_corr.Draw(); c_corr.Update()
        lc3 = ROOT.TLine(corr_cut_dynamic, ROOT.gPad.GetUymin(),
                          corr_cut_dynamic, ROOT.gPad.GetUymax())
        lc3.SetLineColor(ROOT.kRed); lc3.SetLineWidth(2); lc3.SetLineStyle(2)
        lc3.Draw("same"); global_lines.append(lc3)
        leg3 = ROOT.TLegend(0.13, 0.78, 0.62, 0.88)
        leg3.SetTextSize(0.03); leg3.SetFillColorAlpha(ROOT.kWhite, 0.7)
        leg3.AddEntry(lc3, f"cut ({int(CORR_CUT_PERCENTILE*100)}th pct) = {corr_cut_dynamic:.6f}", "l")
        leg3.Draw("same"); global_lines.append(leg3)

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
                g_phys_disp, g_heater_disp, h_corr_heater, fit_corr_heater,
                g_base_part, g_base_time, h_base_part]
        for p in chain:
            keep.extend([p["h_full"], p["h_zoom"], p["fit_zoom"],
                         p["h_energy"], p["fit_energy"],
                         p["h_zoom_c"], p["fit_zoom_c"],
                         p["h_energy_c"], p["fit_energy_c"]])
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