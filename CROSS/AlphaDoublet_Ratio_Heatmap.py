#!/usr/bin/env python3
"""
AlphaDoublet_InteractiveFit.py
==============================
Interactive single-channel tool to fit the alpha / (alpha + recoil) doublet
in the "stabilization_all" amplitude spectrum and store the amplitude(area)
ratio.

Pipeline overview
-----------------
  1. Pick ONE channel (command line, e.g. `... 50`, or the CHANNEL constant)
     and locate its ROOT file inside BASE_DIR ("ch<N>" in the file name).

  2. Bulk-read the "stabilization_all" amplitude (branch heat_amplitude) with
     RDataFrame + AsNumpy, applying QUALITY filters only (single trigger, good
     interval, signal flag) -- NO correlation / NO light-yield cuts -- plus the
     high-energy cut  calibration_rough > CAL_ROUGH_MIN.

  3. Show the amplitude spectrum in an interactive GUI (matplotlib embedded in
     tkinter). The user selects, by dragging on the plot, the fit range of each
     of the two peaks (Peak 1 = left, Peak 2 = right). The four limits also
     appear in editable text boxes.

  4. "Fit" performs:
        - two INDEPENDENT preliminary Gaussian fits, one on each selected range;
        - a FINAL double-Gaussian fit over [left limit of Peak 1, right limit of
          Peak 2], seeded with the preliminary results.
     Results shown: chi2/ndf and p-value of the final fit, the two Gaussian
     areas, and their ratio.

  5. "Save" writes the plot as JPEG (FIT_DIR_NAME) and appends/updates one CSV
     row with the channel, the two chosen ranges, the areas, the ratio and the
     p-value. Re-running an already-analysed channel pre-loads the saved ranges
     into the GUI; editing + saving overwrites that channel's row and image.

Backend
-------
  ROOT is used in BATCH mode for file reading only (no ROOT canvas). The GUI
  and the fits use tkinter + matplotlib + scipy, so there is no ROOT/Tk Cocoa
  clash on macOS.

Input
-----
  ROOT file in BASE_DIR for the requested channel, containing the trees:
    stabilization_all (branch heat_amplitude),
    calibration_rough (or calibration_all), numberoftriggers, badinterval,
    module, baseline.

Output
------
  AlphaDoubletFits/ch<N>_alpha_doublet.jpg
  AlphaRatio_<TAG>.csv   (one row per channel)

Execution
---------
  python3 AlphaDoublet_InteractiveFit.py 50      # analyse channel 50
  python3 AlphaDoublet_InteractiveFit.py         # uses the CHANNEL constant
"""

# ===========================================================================
# IMPORTS
# ===========================================================================

# -- standard library --------------------------------------------------------
import os
import sys
import csv
import math
import re
import argparse

# -- third-party -------------------------------------------------------------
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist

import matplotlib
matplotlib.use("TkAgg")                    # interactive backend embedded in Tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector

import tkinter as tk
from tkinter import ttk, messagebox

# -- domain-specific ---------------------------------------------------------
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(True)                  # file reading only, no ROOT graphics


# ===========================================================================
# PARAMETERS  <- tune these
# ===========================================================================

# --- Folder holding the .root files -----------------------------------------
BASE_DIR = "/data/users/azanelli/octopus_work/CROSS/RUN000207/Coincidence"

# --- Default channel (overridden by the command-line argument) --------------
CHANNEL = 50

# --- Selection --------------------------------------------------------------
CAL_ROUGH_MIN = 4000.0     # keep only events with calibration_rough > this
MIN_EVENTS    = 30         # minimum surviving events to build the spectrum

# --- Spectrum ---------------------------------------------------------------
SPECTRUM_NBINS = 250       # bins of the stabilization-amplitude spectrum

# --- Ratio definition -------------------------------------------------------
# True  -> ratio = area_left / area_right  (alpha / (alpha + recoil))
RATIO_LEFT_OVER_RIGHT = True

# --- Output -----------------------------------------------------------------
TAG           = "Merged206_209"        # label used in the output file names
FIT_DIR_NAME  = "AlphaDoubletFits"     # folder for the per-channel JPEGs

# --- Gaussian -> FWHM constant (kept for completeness) ----------------------
SIGMA_TO_FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))

# CSV column order.
CSV_FIELDS = ["channel", "p1_min", "p1_max", "p2_min", "p2_max",
              "area_left", "area_right", "ratio", "pvalue"]


# ===========================================================================
# FIT MODEL
# ===========================================================================

def gauss(x, A, mu, sigma):
    """Single Gaussian: A * exp(-0.5*((x-mu)/sigma)^2)."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def two_gauss(x, A1, m1, s1, A2, m2, s2):
    """Sum of two Gaussians."""
    return gauss(x, A1, m1, s1) + gauss(x, A2, m2, s2)


def gauss_area_counts(A, sigma, bin_width):
    """
    Area of a Gaussian peak expressed in counts:
        integral(A*exp(...)) / bin_width = A * sigma * sqrt(2pi) / bin_width
    """
    return A * abs(sigma) * math.sqrt(2.0 * math.pi) / bin_width


# ===========================================================================
# ROOT I/O  (read one channel's amplitude array)
# ===========================================================================

def find_file_for_channel(base_dir, channel):
    """Return the first .root file in *base_dir* whose 'ch<N>' equals channel."""
    for f in sorted(os.listdir(base_dir)):
        if not f.endswith(".root") or "stabilized" in f:
            continue
        m = re.search(r"ch(\d+)", f)
        if m and int(m.group(1)) == int(channel):
            return os.path.join(base_dir, f)
    return None


def read_channel_amplitudes(filename):
    """
    Read the 'stabilization_all' amplitude after quality + calibration cuts.

    Quality filters: heat_numberoftriggers == 1, heat_badinterval == 0,
    heat_issignal != 0 (the last two only if the leaf exists). High-energy cut:
    calibration_rough > CAL_ROUGH_MIN. NO correlation / NO LY cuts.

    Returns a NumPy array of surviving amplitudes (empty array on failure).
    """
    file = ROOT.TFile.Open(filename, "READ")
    if not file or file.IsZombie():
        print("  [!] Cannot open file.", file=sys.stderr)
        return np.array([])

    tree_main = file.Get("stabilization_all")
    if not tree_main or not hasattr(tree_main, "GetEntries") or tree_main.GetEntries() == 0:
        print("  [!] Missing/empty 'stabilization_all' tree.", file=sys.stderr)
        file.Close(); return np.array([])

    tree_cal = file.Get("calibration_rough") or file.Get("calibration_all")
    if not tree_cal or not hasattr(tree_cal, "GetEntries"):
        print("  [!] Missing calibration tree.", file=sys.stderr)
        file.Close(); return np.array([])

    for name in ["module", "badinterval", "numberoftriggers", "baseline"]:
        t = file.Get(name)
        if t is not None:
            tree_main.AddFriend(t)
    tree_main.AddFriend(tree_cal)

    cal_tree_name        = tree_cal.GetName()
    has_heat_badinterval = bool(tree_main.GetLeaf("heat_badinterval"))
    has_heat_issignal    = bool(tree_main.GetLeaf("heat_issignal"))

    rdf   = ROOT.RDataFrame(tree_main)
    rdf_f = rdf.Filter("heat_numberoftriggers == 1")
    if has_heat_badinterval:
        rdf_f = rdf_f.Filter("heat_badinterval == 0")
    if has_heat_issignal:
        rdf_f = rdf_f.Filter("heat_issignal != 0")

    # Rough calibration lives in a friend tree -> tree-qualified access.
    rdf_f = rdf_f.Define("_cal_rough_", f"{cal_tree_name}.heat_amplitude")

    data = rdf_f.AsNumpy(["heat_amplitude", "_cal_rough_"])
    amp  = data["heat_amplitude"].astype(np.float64)
    cal  = data["_cal_rough_"].astype(np.float64)
    file.Close()

    mask = np.isfinite(amp) & np.isfinite(cal) & (cal > CAL_ROUGH_MIN)
    return amp[mask]


# ===========================================================================
# CSV PERSISTENCE  (one row per channel, keyed by channel)
# ===========================================================================

def csv_path():
    return os.path.join(BASE_DIR, f"AlphaRatio_{TAG}.csv")


def load_csv_row(channel):
    """Return the saved row dict for *channel*, or None if absent."""
    path = csv_path()
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                if int(float(row["channel"])) == int(channel):
                    return row
            except (KeyError, ValueError):
                continue
    return None


def save_csv_row(row):
    """Insert or overwrite the row for row['channel'], keeping the file sorted."""
    path = csv_path()
    rows = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    rows[int(float(r["channel"]))] = r
                except (KeyError, ValueError):
                    continue
    rows[int(row["channel"])] = row
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for ch in sorted(rows):
            w.writerow({k: rows[ch].get(k, "") for k in CSV_FIELDS})
    print(f">>> CSV updated: {path}")


# ===========================================================================
# INTERACTIVE GUI
# ===========================================================================

class DoubletFitGUI:
    """
    tkinter + matplotlib GUI to choose the two preliminary fit ranges, run the
    preliminary + final fits, display the results, and save plot + CSV row.
    """

    def __init__(self, channel, amps):
        self.channel = int(channel)
        self.amps    = amps

        # --- Histogram (built once) -----------------------------------------
        lo, hi = self._spectrum_range(amps)
        self.counts, edges = np.histogram(amps, bins=SPECTRUM_NBINS, range=(lo, hi))
        self.centers   = 0.5 * (edges[:-1] + edges[1:])
        self.bin_width = edges[1] - edges[0]

        # --- State ----------------------------------------------------------
        self.active_peak = 1                 # which peak the drag selects (1/2)
        self.ranges = {1: [None, None], 2: [None, None]}   # [min, max] per peak
        self.last_result = None              # dict with the fit outputs
        self.span_patches = {1: None, 2: None}

        # --- Build window ---------------------------------------------------
        self.root = tk.Tk()
        self.root.title(f"Alpha doublet fit - Ch {self.channel}")
        self._build_ui()

        # --- Pre-load saved ranges (if the channel is already in the CSV) ---
        saved = load_csv_row(self.channel)
        if saved:
            try:
                self._set_entries(float(saved["p1_min"]), float(saved["p1_max"]),
                                  float(saved["p2_min"]), float(saved["p2_max"]))
                self._read_ranges_from_entries()
                self._draw_base()
                self.do_fit()       # show the saved fit immediately
                self.status.set("Loaded saved ranges from CSV. Adjust and re-fit if needed.")
            except (ValueError, KeyError):
                self._draw_base()
        else:
            self._draw_base()

    # ----------------------------------------------------------------------
    @staticmethod
    def _spectrum_range(vals):
        lo = float(np.percentile(vals, 0.5))
        hi = float(np.percentile(vals, 99.5))
        if hi <= lo:
            lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            hi = lo + 1.0
        pad = (hi - lo) * 0.10
        return lo - pad, hi + pad

    # ----------------------------------------------------------------------
    def _build_ui(self):
        # Figure on the left.
        self.fig = Figure(figsize=(8, 5.5), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

        # Drag-to-select range (applies to the active peak).
        self.span = SpanSelector(
            self.ax, self._on_span_select, "horizontal", useblit=True,
            props=dict(alpha=0.25, facecolor="tab:blue"), interactive=False)

        # Controls on the right.
        panel = ttk.Frame(self.root, padding=10)
        panel.pack(side="right", fill="y")

        ttk.Label(panel, text=f"Channel {self.channel}",
                  font=("Arial", 13, "bold")).pack(anchor="w", pady=(0, 8))

        # Which peak the mouse drag sets.
        tgt = ttk.LabelFrame(panel, text="Drag on the plot sets:")
        tgt.pack(fill="x", pady=4)
        self.peak_var = tk.IntVar(value=1)
        ttk.Radiobutton(tgt, text="Peak 1 (left)",  variable=self.peak_var,
                        value=1, command=self._on_peak_change).pack(anchor="w")
        ttk.Radiobutton(tgt, text="Peak 2 (right)", variable=self.peak_var,
                        value=2, command=self._on_peak_change).pack(anchor="w")

        # Editable range boxes.
        rng = ttk.LabelFrame(panel, text="Fit ranges (editable)")
        rng.pack(fill="x", pady=6)
        self.entries = {}
        for key, label in [("p1_min", "Peak 1 min"), ("p1_max", "Peak 1 max"),
                           ("p2_min", "Peak 2 min"), ("p2_max", "Peak 2 max")]:
            row = ttk.Frame(rng); row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=11).pack(side="left")
            e = ttk.Entry(row, width=12); e.pack(side="right")
            self.entries[key] = e

        ttk.Button(panel, text="Fit", command=self.do_fit).pack(fill="x", pady=(10, 4))

        # Results.
        res = ttk.LabelFrame(panel, text="Fit results")
        res.pack(fill="x", pady=6)
        self.results_text = tk.Text(res, width=34, height=9, state="disabled",
                                    font=("Courier", 10))
        self.results_text.pack(fill="x", padx=4, pady=4)

        ttk.Button(panel, text="Salva (JPEG + CSV)", command=self.on_save).pack(fill="x", pady=4)
        ttk.Button(panel, text="Esci", command=self.root.destroy).pack(fill="x", pady=(4, 0))

        self.status = tk.StringVar(value="Drag on the plot to set Peak 1 / Peak 2 ranges, then Fit.")
        ttk.Label(panel, textvariable=self.status, wraplength=240,
                  foreground="gray25").pack(anchor="w", pady=(10, 0))

    # ----------------------------------------------------------------------
    def _on_peak_change(self):
        self.active_peak = self.peak_var.get()
        color = "tab:blue" if self.active_peak == 1 else "tab:green"
        self.span.set_props(facecolor=color)

    def _on_span_select(self, xmin, xmax):
        """Store the dragged range into the active peak and redraw."""
        if xmax - xmin <= 0:
            return
        self.ranges[self.active_peak] = [float(xmin), float(xmax)]
        keymin = "p1_min" if self.active_peak == 1 else "p2_min"
        keymax = "p1_max" if self.active_peak == 1 else "p2_max"
        self._set_entry(keymin, xmin)
        self._set_entry(keymax, xmax)
        self._draw_base()

    # ----------------------------------------------------------------------
    def _set_entry(self, key, value):
        self.entries[key].delete(0, tk.END)
        self.entries[key].insert(0, f"{value:.2f}")

    def _set_entries(self, p1lo, p1hi, p2lo, p2hi):
        self._set_entry("p1_min", p1lo); self._set_entry("p1_max", p1hi)
        self._set_entry("p2_min", p2lo); self._set_entry("p2_max", p2hi)

    def _read_ranges_from_entries(self):
        """Read the four limits from the boxes into self.ranges (sorted L/R)."""
        try:
            p1 = sorted([float(self.entries["p1_min"].get()),
                         float(self.entries["p1_max"].get())])
            p2 = sorted([float(self.entries["p2_min"].get()),
                         float(self.entries["p2_max"].get())])
        except ValueError:
            return False
        # Enforce Peak 1 = left, Peak 2 = right by their centres.
        if 0.5 * (p1[0] + p1[1]) > 0.5 * (p2[0] + p2[1]):
            p1, p2 = p2, p1
            self._set_entries(p1[0], p1[1], p2[0], p2[1])
        self.ranges[1] = p1
        self.ranges[2] = p2
        return True

    # ----------------------------------------------------------------------
    def _draw_base(self, draw_fit=False):
        """Redraw the histogram and the two selected-range shades."""
        self.ax.clear()
        self.ax.step(self.centers, self.counts, where="mid", color="black", lw=1.0)
        self.ax.fill_between(self.centers, self.counts, step="mid",
                             color="0.8", alpha=0.6)
        self.ax.set_xlabel("stabilization_all amplitude")
        self.ax.set_ylabel("Counts")
        self.ax.set_title(f"Ch {self.channel} - alpha doublet "
                          f"(cal_rough > {CAL_ROUGH_MIN:.0f})")

        for pk, col in [(1, "tab:blue"), (2, "tab:green")]:
            lo, hi = self.ranges[pk]
            if lo is not None and hi is not None:
                self.ax.axvspan(lo, hi, color=col, alpha=0.15)
                self.ax.axvline(lo, color=col, ls=":", lw=1)
                self.ax.axvline(hi, color=col, ls=":", lw=1)

        if draw_fit and self.last_result is not None:
            self._overlay_fit()

        self.ax.grid(alpha=0.3)
        self.canvas.draw_idle()

    def _overlay_fit(self):
        r = self.last_result
        xfit = np.linspace(r["fit_lo"], r["fit_hi"], 600)
        self.ax.plot(xfit, two_gauss(xfit, *r["popt"]), color="red", lw=2,
                     label="double-Gaussian fit")
        self.ax.plot(xfit, gauss(xfit, r["popt"][0], r["popt"][1], r["popt"][2]),
                     color="tab:blue", ls="--", lw=1.5, label="peak 1 (alpha)")
        self.ax.plot(xfit, gauss(xfit, r["popt"][3], r["popt"][4], r["popt"][5]),
                     color="tab:green", ls="--", lw=1.5, label="peak 2 (alpha+recoil)")
        # Compact result annotation on the plot.
        txt = (f"p-value = {r['pvalue']:.3f}\n"
               f"area L = {r['area_left']:.0f}\n"
               f"area R = {r['area_right']:.0f}\n"
               f"ratio = {r['ratio']:.3f}")
        self.ax.text(0.97, 0.97, txt, transform=self.ax.transAxes, ha="right",
                     va="top", fontsize=9, family="monospace",
                     bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9))
        self.ax.legend(loc="upper left", fontsize=8)

    # ----------------------------------------------------------------------
    def _fit_one_gauss(self, lo, hi):
        """Independent Gaussian fit on the bins inside [lo, hi]. Returns popt."""
        sel = (self.centers >= lo) & (self.centers <= hi)
        x, y = self.centers[sel], self.counts[sel].astype(float)
        if len(x) < 4:
            raise RuntimeError("too few bins in the selected range")
        err = np.sqrt(np.where(y > 0, y, 1.0))
        A0  = max(y.max(), 1.0)
        mu0 = x[int(np.argmax(y))]
        s0  = max((hi - lo) / 4.0, self.bin_width)
        popt, _ = curve_fit(gauss, x, y, p0=[A0, mu0, s0],
                            sigma=err, absolute_sigma=True, maxfev=20000)
        return popt

    def do_fit(self):
        """Two preliminary fits, then the final double-Gaussian fit."""
        if not self._read_ranges_from_entries():
            messagebox.showerror("Error", "Invalid range values.")
            return
        (p1lo, p1hi) = self.ranges[1]
        (p2lo, p2hi) = self.ranges[2]
        if None in (p1lo, p1hi, p2lo, p2hi):
            messagebox.showwarning("Ranges", "Select both peak ranges first.")
            return

        try:
            # --- preliminary independent fits ---
            g1 = self._fit_one_gauss(p1lo, p1hi)
            g2 = self._fit_one_gauss(p2lo, p2hi)

            # --- final fit on [left limit of peak 1, right limit of peak 2] ---
            fit_lo, fit_hi = p1lo, p2hi
            sel = (self.centers >= fit_lo) & (self.centers <= fit_hi)
            x, y = self.centers[sel], self.counts[sel].astype(float)
            err = np.sqrt(np.where(y > 0, y, 1.0))
            p0  = [g1[0], g1[1], abs(g1[2]), g2[0], g2[1], abs(g2[2])]
            popt, _ = curve_fit(two_gauss, x, y, p0=p0,
                                sigma=err, absolute_sigma=True, maxfev=40000)
        except Exception as e:
            messagebox.showerror("Fit failed", str(e))
            self.status.set(f"Fit failed: {e}")
            return

        # Re-order the two components by mean (left / right).
        comp = sorted([(popt[0], popt[1], abs(popt[2])),
                       (popt[3], popt[4], abs(popt[5]))], key=lambda c: c[1])
        (AL, muL, sL), (AR, muR, sR) = comp
        popt_ordered = [AL, muL, sL, AR, muR, sR]

        # chi2 / p-value of the final fit.
        model = two_gauss(x, *popt)
        chi2  = float(np.sum(((y - model) / err) ** 2))
        ndf   = max(len(x) - 6, 1)
        pval  = float(chi2_dist.sf(chi2, ndf))

        # Areas (counts) and ratio.
        area_L = gauss_area_counts(AL, sL, self.bin_width)
        area_R = gauss_area_counts(AR, sR, self.bin_width)
        ratio  = (area_L / area_R if RATIO_LEFT_OVER_RIGHT else area_R / area_L) \
                 if (area_R != 0 and area_L != 0) else float("nan")

        self.last_result = dict(
            popt=popt_ordered, fit_lo=fit_lo, fit_hi=fit_hi,
            chi2=chi2, ndf=ndf, pvalue=pval,
            area_left=area_L, area_right=area_R, ratio=ratio,
            muL=muL, muR=muR, sL=sL, sR=sR,
        )

        self._draw_base(draw_fit=True)
        self._show_results()
        self.status.set("Fit done. Adjust ranges and re-fit, or Save.")

    # ----------------------------------------------------------------------
    def _show_results(self):
        r = self.last_result
        ratio_lbl = "areaL/areaR" if RATIO_LEFT_OVER_RIGHT else "areaR/areaL"
        lines = [
            f"chi2/ndf = {r['chi2']:.1f}/{r['ndf']}",
            f"p-value  = {r['pvalue']:.4f}",
            "",
            f"PEAK 1 (left)",
            f"  mu={r['muL']:.2f} sigma={r['sL']:.2f}",
            f"  area={r['area_left']:.1f}",
            f"PEAK 2 (right)",
            f"  mu={r['muR']:.2f} sigma={r['sR']:.2f}",
            f"  area={r['area_right']:.1f}",
            "",
            f"{ratio_lbl} = {r['ratio']:.4f}",
        ]
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "\n".join(lines))
        self.results_text.config(state="disabled")

    # ----------------------------------------------------------------------
    def on_save(self):
        """Save the JPEG and write/overwrite this channel's CSV row."""
        if self.last_result is None:
            messagebox.showwarning("Save", "Run a fit before saving.")
            return
        r = self.last_result

        # --- JPEG ---
        fit_dir = os.path.join(BASE_DIR, FIT_DIR_NAME)
        os.makedirs(fit_dir, exist_ok=True)
        jpg = os.path.join(fit_dir, f"ch{self.channel}_alpha_doublet.jpg")
        self.fig.savefig(jpg, dpi=200, bbox_inches="tight")
        print(f">>> Saved JPEG: {jpg}")

        # --- CSV row (chosen ranges + areas + ratio + p-value) ---
        row = {
            "channel":   self.channel,
            "p1_min":    f"{self.ranges[1][0]:.4f}",
            "p1_max":    f"{self.ranges[1][1]:.4f}",
            "p2_min":    f"{self.ranges[2][0]:.4f}",
            "p2_max":    f"{self.ranges[2][1]:.4f}",
            "area_left": f"{r['area_left']:.4f}",
            "area_right": f"{r['area_right']:.4f}",
            "ratio":     f"{r['ratio']:.6f}",
            "pvalue":    f"{r['pvalue']:.6f}",
        }
        save_csv_row(row)
        self.status.set(f"Saved: {os.path.basename(jpg)} + CSV row.")
        messagebox.showinfo("Saved", f"JPEG and CSV row saved for channel {self.channel}.")

    # ----------------------------------------------------------------------
    def run(self):
        self.root.mainloop()


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Interactive alpha-doublet fit for one channel.")
    parser.add_argument("channel", nargs="?", type=int, default=CHANNEL,
                        help="Channel number to analyse (default: CHANNEL constant).")
    args = parser.parse_args()
    channel = args.channel

    if not os.path.isdir(BASE_DIR):
        print(f"[!] Error: folder not found: {BASE_DIR}")
        sys.exit(1)

    filename = find_file_for_channel(BASE_DIR, channel)
    if filename is None:
        print(f"[!] No .root file for channel {channel} in {BASE_DIR}")
        sys.exit(1)

    print(f">>> Channel {channel}: {os.path.basename(filename)}")
    amps = read_channel_amplitudes(filename)
    if len(amps) < MIN_EVENTS:
        print(f"[!] Too few events ({len(amps)} < {MIN_EVENTS}). Aborting.")
        sys.exit(1)
    print(f"  {len(amps)} events after quality + (cal_rough > {CAL_ROUGH_MIN:.0f}).")

    DoubletFitGUI(channel, amps).run()


if __name__ == "__main__":
    main()