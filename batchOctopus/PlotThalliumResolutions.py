#!/usr/bin/env python3
"""
===============================================================================
 PlotThalliumResolutions.py -- resolution of the 208-Tl line, channel by channel
===============================================================================

 Reads the results table written by ThalliumStabilization.py
 (thallium_resolutions.csv, one row per fitted panel of the combined-thallium
 canvas) and draws TWO figures:

   1. RESOLUTION -- the percentage resolution of every step of the analysis,
      with its error, as a function of the channel number. All the steps of a
      channel sit on the SAME vertical line: with a full detector on the x axis
      there is no room to dodge them sideways, and the vertical alignment is
      what lets the eye compare the steps within a channel and the same step
      across channels.

   2. SIGNIFICANCE -- how significant the resolution change produced by the
      thallium stabilization is, channel by channel:

          z = (R_heater2 - R_Tl) / sqrt(sigma_heater2^2 + sigma_Tl^2)

      positive when the resolution IMPROVES. The normalised difference of two
      fitted quantities follows a Student t, not a Gaussian, so the "not
      significant" band is drawn at the two-sided 95 % t quantile computed
      channel by channel with the effective number of degrees of freedom
      (Welch-Satterthwaite on the two fits' ndf).

      CAVEAT: the two resolutions are measured on the SAME events, so their
      errors are strongly correlated and the sum in quadrature OVERESTIMATES
      the error of the difference. z is therefore conservative -- a significant
      improvement stays significant, a marginal one may be more significant
      than it looks here.

 Both figures are read from the fits with a FLAT background (pol0) on the
 energy-rescaled peak: over a window this narrow the continuum is nearly flat,
 and a constant spends one parameter less on the few counts around the line.

 Panels whose fit gave no error (low statistics: the fit converges but the error
 matrix does not) are drawn as HOLLOW markers without a bar, never silently
 dropped and never with a fake zero error.

 Usage
 -----
     conda activate pyrootAlbi          # any environment with matplotlib
     python PlotThalliumResolutions.py                    # defaults below
     python PlotThalliumResolutions.py --channels 26 27 57 58
     python PlotThalliumResolutions.py --csv /path/to/thallium_resolutions.csv

 The figures are written next to the CSV as .pdf (vector, for the thesis) and
 .png (300 dpi, for slides), unless --out says otherwise.
===============================================================================
"""

import os
import sys
import csv
import math
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # file output only, no window needed
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default results table: the one ThalliumStabilization.py fills when run on the
# local test data. Give --csv on the cluster (or anywhere else).
DEFAULT_CSV = os.path.normpath(os.path.join(
    SCRIPT_DIR, "..", "CROSS", "MergedRuns", "ThalliumStabilizedAmp",
    "thallium_resolutions.csv"))

TARGET_ENERGY = 2614.511      # keV, the line the energy row is rescaled to

# Steps of the analysis, in order, with the name shown in the legend, the colour
# and the marker. The keys are the ones ThalliumStabilization.py writes in the
# "variable" column of the results table.
STEPS = [
    # key            legend label                          colour     marker
    ("rough",      "Optimum filter amplitude",           "#3F7FBF", "o"),
    ("heater",     "After first heater stabilization",   "#2E9E5B", "s"),
    ("corrected",  "After second heater stabilization",  "#8C8C8C", "^"),
    ("stabilized", "After thallium stabilization",       "#C43D3D", "D"),
]
STEP_LABEL = {k: lab for k, lab, _, _ in STEPS}

# The two steps compared in the significance figure: before and after the
# thallium stabilization.
Z_BEFORE, Z_AFTER = "corrected", "stabilized"


# ===========================================================================
# DATA
# ===========================================================================

def to_float(s):
    """CSV cell -> float; anything unparsable (including 'nan') becomes nan."""
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def read_rows(path, row_kind, background):
    """
    Rows of the results CSV for one canvas row and one background model.

    Returns {step: {channel: {res, err, ndf, n_peak}}} and the sorted channel
    list. Only entries with a finite resolution are kept: a panel with no fit
    has nothing to say, and plotting it at zero would be a lie.
    """
    if not os.path.exists(path):
        sys.exit(f"[!] Results table not found: {path}\n"
                 f"    Run ThalliumStabilization.py first, or pass --csv.")

    data, channels = {}, set()
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("row") != row_kind or r.get("background") != background:
                continue
            try:
                ch = int(str(r["channel"]).strip())
            except (KeyError, TypeError, ValueError):
                continue
            res = to_float(r.get("resolution_pct"))
            if not math.isfinite(res):
                continue
            data.setdefault(r.get("variable"), {})[ch] = dict(
                res=res, err=to_float(r.get("resolution_err_pct")),
                ndf=to_float(r.get("ndf")), n_peak=to_float(r.get("n_peak")))
            channels.add(ch)

    if not channels:
        sys.exit(f"[!] No row with row='{row_kind}' and background='{background}' "
                 f"in {path}")
    return data, sorted(channels)


def student_interval(nu, cl=0.95):
    """
    Two-sided *cl* quantile of a Student t with *nu* degrees of freedom: the
    |z| above which the change is significant. Falls back to the Gaussian value
    when scipy is not installed or nu is unusable (the t tends to the Gaussian
    for large nu anyway).
    """
    if not (math.isfinite(nu) and nu > 0):
        return 1.959963985                       # Gaussian 95 %, two-sided
    try:
        from scipy import stats
        return float(stats.t.ppf(0.5 + cl / 2.0, nu))
    except ImportError:
        return 1.959963985


def student_pvalue(z, nu):
    """Two-sided p-value of *z* under a Student t with *nu* d.o.f."""
    if not math.isfinite(z):
        return float("nan")
    try:
        from scipy import stats
        if math.isfinite(nu) and nu > 0:
            return float(2.0 * stats.t.sf(abs(z), nu))
        return float(2.0 * stats.norm.sf(abs(z)))
    except ImportError:
        return math.erfc(abs(z) / math.sqrt(2.0))


def significance(before, after):
    """
    Significance of the resolution change between two fits of the same channel.

        z    = (R_before - R_after) / sqrt(err_before^2 + err_after^2)   (z > 0
               means the resolution got BETTER: the FWHM/mu went down)
        nu   = effective d.o.f. of that difference (Welch-Satterthwaite over the
               two fits' ndf), used for the Student-t quantiles
        p    = two-sided p-value of z under that t

    Returns (z, nu, p), all nan when either fit has no usable error.
    """
    nan = float("nan")
    if before is None or after is None:
        return nan, nan, nan
    eb, ea = before["err"], after["err"]
    if not (math.isfinite(eb) and math.isfinite(ea) and (eb > 0 or ea > 0)):
        return nan, nan, nan

    var = eb * eb + ea * ea
    if var <= 0:
        return nan, nan, nan
    z = (before["res"] - after["res"]) / math.sqrt(var)

    # Welch-Satterthwaite: the difference of two estimates with different
    # precisions and different ndf behaves like a t with this many d.o.f.
    nb, na = before["ndf"], after["ndf"]
    den = 0.0
    if math.isfinite(nb) and nb > 0:
        den += eb ** 4 / nb
    if math.isfinite(na) and na > 0:
        den += ea ** 4 / na
    nu = (var ** 2 / den) if den > 0 else float("nan")

    return z, nu, student_pvalue(z, nu)


# ===========================================================================
# COMMON STYLE
# ===========================================================================

def new_axes(n_ch, height=5.6):
    """A figure sized for *n_ch* channels, with the shared look of both plots."""
    width = min(max(6.5 + 0.55 * n_ch, 8.0), 22.0)
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")
    # Alternate channel slots with a very light band: it groups what belongs to
    # one channel without adding a single line to the foreground.
    for i in range(n_ch):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="#000000", alpha=0.035, lw=0,
                       zorder=0)
    return fig, ax


def finish_axes(ax, channels, xlabel="Channel"):
    """Ticks, grid and spines, once the data are drawn."""
    n_ch = len(channels)
    ax.set_xticks(np.arange(n_ch))
    ax.set_xticklabels([str(c) for c in channels], fontsize=10.5,
                       rotation=(90 if n_ch > 24 else (45 if n_ch > 14 else 0)))
    ax.set_xlim(-0.5, n_ch - 0.5)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=8)
    ax.grid(axis="y", ls=":", lw=0.8, color="#8A8A8A", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#666666")
    ax.tick_params(axis="both", which="both", color="#666666", labelsize=10)


def add_subtitle(ax, *lines):
    """Grey caption between the title and the frame; one call, one or two lines."""
    ax.text(0.0, 1.015, "\n".join(lines), transform=ax.transAxes,
            fontsize=10.5, color="#5A5A5A", va="bottom", linespacing=1.5)


def save(fig, base, quiet=False):
    base = os.path.splitext(base)[0]
    for ext, kw in ((".pdf", {}), (".png", dict(dpi=300))):
        fig.savefig(base + ext, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), **kw)
        if not quiet:
            print(f">>> Saved {base + ext}")
    plt.close(fig)


# ===========================================================================
# FIGURE 1 -- RESOLUTION vs CHANNEL
# ===========================================================================

def make_resolution_figure(data, channels, args):
    steps = [s for s in STEPS if s[0] in args.steps and s[0] in data]
    if not steps:
        sys.exit(f"[!] None of the requested steps is in the table: "
                 f"{', '.join(args.steps)}")

    fig, ax = new_axes(len(channels))
    x = np.arange(len(channels), dtype=float)
    n_ch = len(channels)
    m_big = 8.0 if n_ch <= 10 else 6.0
    m_std = 6.0 if n_ch <= 10 else 4.5

    handles, any_missing_err = [], False

    for i, (key, label, colour, marker) in enumerate(steps):
        series = data.get(key, {})
        last   = (key == steps[-1][0])       # end of the chain: draw it on top

        xs, ys, es, xs_ne, ys_ne = [], [], [], [], []
        for j, ch in enumerate(channels):
            if ch not in series:
                continue
            rec = series[ch]
            if math.isfinite(rec["err"]) and rec["err"] > 0:
                xs.append(x[j]); ys.append(rec["res"]); es.append(rec["err"])
            else:                            # fit error matrix failed
                xs_ne.append(x[j]); ys_ne.append(rec["res"])

        size   = m_big if last else m_std
        # All the steps of a channel share one vertical line, so the bars are
        # kept thin and slightly transparent: they cross each other by design.
        alpha  = 1.0 if last else 0.8
        zorder = 6 if last else 4

        if xs:
            ax.errorbar(xs, ys, yerr=es, fmt=marker, ms=size, mfc=colour,
                        mec="white" if last else colour,
                        mew=1.1 if last else 0.6, ecolor=colour,
                        elinewidth=1.4 if last else 1.0,
                        capsize=3.0 if last else 2.2,
                        capthick=1.2 if last else 0.9,
                        ls="none", alpha=alpha, zorder=zorder)
        if xs_ne:
            any_missing_err = True
            ax.plot(xs_ne, ys_ne, marker, ms=size, mfc="none", mec=colour,
                    mew=1.4, ls="none", alpha=alpha, zorder=zorder)

        handles.append(Line2D([], [], color=colour, marker=marker, ms=size,
                              mfc=colour, mec=colour, ls="none", label=label))

    if any_missing_err:
        handles.append(Line2D([], [], color="#555555", marker="o", ms=m_std,
                              mfc="none", mec="#555555", mew=1.4, ls="none",
                              label="fit error not available"))

    ax.set_ylabel("Resolution  FWHM/$\\mu$  [%]", fontsize=12, labelpad=8)
    if args.logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    finish_axes(ax, channels)

    # Right axis: the same numbers as an absolute FWHM. Only meaningful on the
    # energy row, where every peak sits at TARGET_ENERGY by construction.
    if args.row == "energy" and not args.logy:
        ax_r = ax.secondary_yaxis(
            "right", functions=(lambda p: p * TARGET_ENERGY / 100.0,
                                lambda k: k * 100.0 / TARGET_ENERGY))
        ax_r.set_ylabel(f"FWHM at {TARGET_ENERGY:.1f} keV  [keV]",
                        fontsize=12, labelpad=10)
        ax_r.tick_params(labelsize=10, color="#666666")

    ax.set_title("Thallium peak resolution at the various steps of the analysis",
                 fontsize=14.5, fontweight="bold", pad=26)
    add_subtitle(ax, f"Gaussian + {'flat' if args.background == 'pol0' else 'linear'}"
                     f" background fit on the "
                     f"{'energy-rescaled' if args.row == 'energy' else 'native-units'}"
                     f" peak  |  {len(channels)} channels")

    leg = ax.legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, -0.14), ncol=2 if len(handles) > 3 else 1,
                    frameon=True, fontsize=10, borderpad=0.7,
                    columnspacing=1.8, handletextpad=0.7)
    leg.get_frame().set_edgecolor("#CCCCCC")
    leg.get_frame().set_facecolor("white")

    fig.tight_layout()
    return fig


# ===========================================================================
# FIGURE 2 -- SIGNIFICANCE OF THE THALLIUM STABILIZATION
# ===========================================================================

def make_significance_figure(data, channels, args, results):
    """
    z of every channel, with the Student-t "not significant" band. *results* is
    the list of (channel, z, nu, p) already computed by compute_significance.
    """
    fig, ax = new_axes(len(channels), height=5.2)

    good = [(i, z, nu, p) for i, (ch, z, nu, p) in enumerate(results)
            if math.isfinite(z)]
    if not good:
        sys.exit(f"[!] No channel has both '{Z_BEFORE}' and '{Z_AFTER}' with a "
                 f"usable error: nothing to compare.")

    col_up, col_dn = "#2E9E5B", "#C43D3D"      # improvement / worsening

    # Per-channel acceptance band: the t quantile depends on the d.o.f. of that
    # channel's two fits, so the band is a staircase, not a pair of lines.
    t_crit = [student_interval(nu, args.cl) for _, _, nu, _ in good]
    xs     = np.array([i for i, _, _, _ in good], float)
    edges  = np.concatenate([xs - 0.5, [xs[-1] + 0.5]])
    band   = np.concatenate([t_crit, [t_crit[-1]]])
    ax.fill_between(edges, -band, band, step="post", color="#9AA0A6",
                    alpha=0.16, lw=0, zorder=1)
    ax.step(edges, band, where="post", color="#9AA0A6", lw=1.0, alpha=0.8,
            zorder=2)
    ax.step(edges, -np.array(band), where="post", color="#9AA0A6", lw=1.0,
            alpha=0.8, zorder=2)

    ax.axhline(0.0, color="#444444", lw=1.2, zorder=3)

    zs = np.array([z for _, z, _, _ in good])
    for i, z, nu, p in good:
        colour = col_up if z >= 0 else col_dn
        ax.plot([i], [z], "o", ms=8.5, mfc=colour, mec="white", mew=1.2,
                zorder=6)
        # Stem down to zero: it turns a cloud of dots into a per-channel bar of
        # evidence, which is what a significance plot is read for.
        ax.plot([i, i], [0.0, z], "-", color=colour, lw=1.6, alpha=0.55,
                zorder=4)

    ax.set_ylabel("Significance  $z$", fontsize=12, labelpad=8)
    lim = max(3.0, 1.15 * float(np.abs(zs).max()), 1.15 * max(t_crit))
    ax.set_ylim(-lim, lim)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    finish_axes(ax, channels)

    # Which way is up: stated on the plot, not left to the caption.
    ax.text(0.99, 0.95, "$\\uparrow$ improvement", transform=ax.transAxes,
            ha="right", va="top", fontsize=10.5, color=col_up, fontweight="bold")
    ax.text(0.99, 0.05, "$\\downarrow$ worsening", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10.5, color=col_dn,
            fontweight="bold")

    ax.set_title("Statistical significance of the resolution change produced by "
                 "the thallium stabilization",
                 fontsize=13.5, fontweight="bold", pad=54)
    nus = [nu for _, _, nu, _ in good if math.isfinite(nu)]
    nu_note = f"$\\nu\\approx${np.median(nus):.0f}" if nus else "Gaussian limit"
    add_subtitle(ax,
                 f"{STEP_LABEL[Z_BEFORE]}  $\\rightarrow$  {STEP_LABEL[Z_AFTER]}"
                 f"  |  {len(good)} channels",
                 "$z=(R_\\mathrm{before}-R_\\mathrm{after})/"
                 "\\sqrt{\\sigma_\\mathrm{before}^2+\\sigma_\\mathrm{after}^2}$"
                 "  |  errors added in quadrature")

    handles = [
        Line2D([], [], color=col_up, marker="o", ms=8, mfc=col_up, mec=col_up,
               ls="none", label="resolution improved  ($z>0$)"),
        Line2D([], [], color=col_dn, marker="o", ms=8, mfc=col_dn, mec=col_dn,
               ls="none", label="resolution worsened  ($z<0$)"),
        Patch(facecolor="#9AA0A6", alpha=0.3,
              label=f"not significant at {100*args.cl:.0f} % "
                    f"(Student $t$, {nu_note})"),
    ]
    leg = ax.legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True,
                    fontsize=10, borderpad=0.7, columnspacing=1.8,
                    handletextpad=0.7)
    leg.get_frame().set_edgecolor("#CCCCCC")
    leg.get_frame().set_facecolor("white")

    fig.tight_layout()
    return fig


def compute_significance(data, channels):
    """[(channel, z, nu, p)] for every channel, in the order given."""
    before, after = data.get(Z_BEFORE, {}), data.get(Z_AFTER, {})
    return [(ch, *significance(before.get(ch), after.get(ch))) for ch in channels]


# ===========================================================================
# REPORT
# ===========================================================================

def print_tables(data, channels, args, results):
    """The same numbers as text, ready to be copied into the thesis."""
    keys = [s[0] for s in STEPS if s[0] in args.steps and s[0] in data]

    head = f"{'ch':>5} | " + " | ".join(f"{k:^18}" for k in keys)
    print(f"\nResolution FWHM/mu [%]  ({args.row} row, {args.background} background)")
    print(head)
    print("-" * len(head))
    for ch in channels:
        cells = []
        for k in keys:
            rec = data.get(k, {}).get(ch)
            if rec is None:
                cells.append(" " * 18)
            elif math.isfinite(rec["err"]):
                cells.append(f"{rec['res']:7.3f} +/- {rec['err']:5.3f}")
            else:
                cells.append(f"{rec['res']:7.3f} +/-   n/a")
        print(f"{ch:>5} | " + " | ".join(f"{c:^18}" for c in cells))

    print(f"\nSignificance of {STEP_LABEL[Z_BEFORE]} -> {STEP_LABEL[Z_AFTER]}")
    print(f"{'ch':>5} | {'z':>7} | {'nu_eff':>7} | {'p (2-sided)':>12} | verdict")
    print("-" * 58)
    for ch, z, nu, p in results:
        if not math.isfinite(z):
            print(f"{ch:>5} |     n/a |     n/a |          n/a | no usable error")
            continue
        t_c = student_interval(nu, args.cl)
        if abs(z) < t_c:
            verdict = "not significant"
        else:
            verdict = "improvement" if z > 0 else "WORSENING"
        print(f"{ch:>5} | {z:7.2f} | {nu:7.1f} | {p:12.2e} | {verdict}")
    print()


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Resolution of the 208-Tl line, and the significance of the "
                    "change produced by the thallium stabilization, from the "
                    "table written by ThalliumStabilization.py.")
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help=f"results table (default: {DEFAULT_CSV})")
    ap.add_argument("--row", choices=["energy", "native"], default="energy",
                    help="canvas row to read (default: energy, the rescaled "
                         "peak, comparable across steps and channels)")
    ap.add_argument("--background", choices=["pol0", "pol1"], default="pol0",
                    help="background model of the fit (default: pol0, flat)")
    ap.add_argument("--steps", nargs="+", default=[s[0] for s in STEPS],
                    choices=[s[0] for s in STEPS],
                    help="analysis steps to draw in the resolution figure")
    ap.add_argument("--channels", nargs="+", type=int, default=None,
                    help="only these channels (default: all in the table)")
    ap.add_argument("--cl", type=float, default=0.95,
                    help="confidence level of the significance band (0.95)")
    ap.add_argument("--logy", action="store_true",
                    help="logarithmic y axis on the resolution figure")
    ap.add_argument("--no-significance", dest="significance",
                    action="store_false",
                    help="only draw the resolution figure")
    ap.add_argument("--out", default=None,
                    help="base name of the output files (default: next to the "
                         "CSV, one base per figure)")
    args = ap.parse_args()

    data, channels = read_rows(args.csv, args.row, args.background)
    if args.channels:
        channels = [c for c in channels if c in set(args.channels)]
        if not channels:
            sys.exit("[!] None of the requested channels is in the table.")

    out_dir = os.path.dirname(os.path.abspath(args.csv))
    tag     = f"{args.row}_{args.background}"
    base    = os.path.splitext(args.out)[0] if args.out else None

    results = compute_significance(data, channels)
    print_tables(data, channels, args, results)

    save(make_resolution_figure(data, channels, args),
         (base + "_resolution") if base else
         os.path.join(out_dir, f"thallium_resolutions_{tag}"))

    if args.significance:
        save(make_significance_figure(data, channels, args, results),
             (base + "_significance") if base else
             os.path.join(out_dir, f"thallium_significance_{tag}"))


if __name__ == "__main__":
    main()
