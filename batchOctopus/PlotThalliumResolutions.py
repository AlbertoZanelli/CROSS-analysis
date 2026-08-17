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

          z = (R_before - R_after) / sqrt(sigma_before^2 + sigma_after^2)

      positive when the resolution IMPROVES. "before" is the last step the
      channel has before the thallium stabilization: normally the second heater
      stabilization, but the optimum-filter channels have no heater chain and
      are compared against their own amplitude (drawn as hollow markers). The normalised difference of two
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

 The figures are written next to the CSV as .png (300 dpi), unless --out says
 otherwise.
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

# Channels to leave out of BOTH figures -- e.g. a channel whose fits are known
# to be meaningless. Put the channel numbers here, as integers:
#     EXCLUDE_CHANNELS = [29, 55]
# They are dropped whatever else is asked for (--channels included), and the
# ones actually dropped are printed when the program runs.
EXCLUDE_CHANNELS = []

# Steps of the analysis, in order, with the name shown in the legend, the colour
# and the marker. The keys are the ones ThalliumStabilization.py writes in the
# "variable" column of the results table.
# Saturated colours, all dark enough to read on white at a glance and to survive
# a figure printed at half size: with a whole detector on the x axis the markers
# are small, and a pale colour (the old grey of the second heater step) simply
# disappears. Distinct hues rather than shades of one, so the four steps are told
# apart by colour alone, before the marker shape is even visible.
STEPS = [
    # key            legend label                          colour     marker
    ("rough",      "Optimum filter amplitude",           "#1A5FA8", "o"),
    ("heater",     "After first heater stabilization",   "#12855F", "s"),
    ("corrected",  "After second heater stabilization",  "#D9720B", "^"),
    ("stabilized", "After thallium stabilization",       "#C1272D", "D"),
]
STEP_LABEL = {k: lab for k, lab, _, _ in STEPS}

# The two steps compared in the significance figure: before and after the
# thallium stabilization. The "before" is the LAST step preceding it that the
# channel actually has -- a channel with no heater stabilization has no
# 'corrected' column at all (its amplitude is the optimum-filter one, which the
# rough panel already carries), and there the step before the thallium
# stabilization is 'rough'. Those points are drawn hollow: the comparison is
# still before/after, but not against the same step as the other channels.
Z_BEFORE_KEYS = ("corrected", "rough")
Z_BEFORE, Z_AFTER = Z_BEFORE_KEYS[0], "stabilized"


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

    data, channels, seen = {}, set(), {}
    n_dup, conflicted = 0, False
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            raw_ch = str(r.get("channel", "")).strip()
            # A table merged by git can carry conflict markers: they parse as
            # rows with a nonsense channel, and BOTH sides of the merge are then
            # in the file. Say so instead of quietly plotting one of the two.
            if raw_ch.startswith(("<<<<<<<", "=======", ">>>>>>>")):
                conflicted = True
                continue
            if r.get("row") != row_kind or r.get("background") != background:
                continue
            try:
                ch = int(raw_ch)
            except (TypeError, ValueError):
                continue
            res = to_float(r.get("resolution_pct"))
            if not math.isfinite(res):
                continue
            # Same panel written twice (a re-analysis merged in, or the two sides
            # of a conflict): keep the most recent fit, so the figure does not
            # depend on the order of the lines.
            key, stamp = (r.get("variable"), ch), str(r.get("date", ""))
            if key in seen:
                n_dup += 1
                if stamp < seen[key]:
                    continue
            seen[key] = stamp
            data.setdefault(r.get("variable"), {})[ch] = dict(
                res=res, err=to_float(r.get("resolution_err_pct")),
                ndf=to_float(r.get("ndf")), n_peak=to_float(r.get("n_peak")))
            channels.add(ch)

    if conflicted:
        print(f"[!] {os.path.basename(path)} contains git conflict markers: it "
              f"holds BOTH sides of a merge. Resolve it before trusting the "
              f"figures.", file=sys.stderr)
    if n_dup:
        print(f"[!] {n_dup} duplicate panel(s) in {os.path.basename(path)}; the "
              f"most recent 'date' was used for each.", file=sys.stderr)

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
    """
    A figure sized for *n_ch* channels, with the shared look of both plots.

    All the steps of a channel share one x position, so the width only has to
    give each CHANNEL room (~0.3 in), not each point: a whole detector fits in a
    figure that still reads when dropped on a thesis page, instead of a metres-
    wide strip that has to be shrunk until the labels vanish.
    """
    width = min(max(4.5 + 0.30 * n_ch, 8.0), 15.0)
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    # Alternate channel slots: with 30+ columns this is what keeps the eye on the
    # right channel, so it has to be visible -- the old 3.5 % grey was not.
    for i in range(n_ch):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="#4A5568", alpha=0.07, lw=0,
                       zorder=0)
    return fig, ax


def finish_axes(ax, channels, xlabel="Channel"):
    """Ticks, grid and spines, once the data are drawn."""
    n_ch = len(channels)
    labels = [str(c) for c in channels]
    # Channel numbers are two digits: they fit horizontally even for a full
    # detector, and horizontal labels are read at a glance. Rotate only when
    # they would actually collide.
    rot = 90 if (n_ch > 45 or max(len(l) for l in labels) > 3 and n_ch > 20) else 0
    ax.set_xticks(np.arange(n_ch))
    ax.set_xticklabels(labels, fontsize=10.5 if n_ch <= 20 else 9.5, rotation=rot)
    ax.set_xlim(-0.5, n_ch - 0.5)
    ax.set_xlabel(xlabel, fontsize=12.5, labelpad=8)
    ax.grid(axis="y", ls="-", lw=0.7, color="#B8BCC4", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#3A3A3A")
        ax.spines[side].set_linewidth(1.1)
    ax.tick_params(axis="both", which="both", color="#3A3A3A", labelsize=10.5)


def add_subtitle(ax, *lines):
    """Grey caption between the title and the frame; one call, one or two lines."""
    ax.text(0.0, 1.015, "\n".join(lines), transform=ax.transAxes,
            fontsize=10.5, color="#5A5A5A", va="bottom", linespacing=1.5)


def save(fig, base, quiet=False):
    out = os.path.splitext(base)[0] + ".png"
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=300)
    if not quiet:
        print(f">>> Saved {out}")
    plt.close(fig)


# ===========================================================================
# FIGURE 1 -- RESOLUTION vs CHANNEL
# ===========================================================================

def make_resolution_figure(data, channels, args):
    steps = [s for s in STEPS if s[0] in args.steps and s[0] in data]
    if not steps:
        sys.exit(f"[!] None of the requested steps is in the table: "
                 f"{', '.join(args.steps)}")

    n_ch = len(channels)
    # A wide axis needs a little more height, or the frame turns into a strip.
    fig, ax = new_axes(n_ch, height=5.6 if n_ch <= 24 else 6.2)
    x = np.arange(n_ch, dtype=float)
    # Markers stay readable down to a full detector: they never overlap
    # horizontally (one x per channel), so only the vertical crowding matters.
    m_big = 8.0 if n_ch <= 10 else (7.0 if n_ch <= 24 else 6.0)
    m_std = 6.5 if n_ch <= 10 else (5.5 if n_ch <= 24 else 5.0)

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
        zorder = 6 if last else 4
        # A white rim around every marker: the steps of a channel sit on ONE
        # vertical line and routinely overlap, and without the rim they merge
        # into a single blob at detector scale. No transparency -- a washed-out
        # colour is exactly what made the old figure need zooming.
        if xs:
            ax.errorbar(xs, ys, yerr=es, fmt=marker, ms=size, mfc=colour,
                        mec="white", mew=1.0 if last else 0.8, ecolor=colour,
                        elinewidth=1.5 if last else 1.2,
                        capsize=2.5 if n_ch <= 24 else 0.0, capthick=1.1,
                        ls="none", zorder=zorder)
        if xs_ne:
            any_missing_err = True
            ax.plot(xs_ne, ys_ne, marker, ms=size, mfc="white", mec=colour,
                    mew=1.6, ls="none", zorder=zorder)

        handles.append(Line2D([], [], color=colour, marker=marker, ms=size + 1,
                              mfc=colour, mec="white", mew=0.8, ls="none",
                              label=label))

    if any_missing_err:
        handles.append(Line2D([], [], color="#444444", marker="o", ms=m_std + 1,
                              mfc="white", mec="#444444", mew=1.6, ls="none",
                              label="fit error not available"))

    ax.set_ylabel("Resolution  FWHM/$\\mu$  [%]", fontsize=12.5, labelpad=8)
    if args.logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0, top=(args.ymax if args.ymax else None))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    finish_axes(ax, channels)

    # A point pushed off the top by --ymax is marked at the edge, never dropped
    # without trace: an off-scale channel is exactly the one worth noticing.
    if args.ymax:
        for key, _, colour, marker in steps:
            for j, ch in enumerate(channels):
                rec = data.get(key, {}).get(ch)
                if rec is not None and rec["res"] > args.ymax:
                    # CARETUP, not the filled triangle: that one is already the
                    # marker of the second heater step.
                    ax.plot([x[j]], [args.ymax], marker=6, ms=9, color=colour,
                            clip_on=False, zorder=8)

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

    # One row of entries when the figure is wide enough for it: the legend is
    # read once, and a two-row block under a 30-channel axis wastes the space
    # the channels need.
    ncol = 4 if fig.get_figwidth() >= 12 else (2 if len(handles) > 3 else 1)
    leg  = ax.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, -0.13), ncol=ncol,
                     frameon=True, fontsize=11, borderpad=0.7,
                     columnspacing=1.6, handletextpad=0.6)
    leg.get_frame().set_edgecolor("#AAAAAA")
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
    fig, ax = new_axes(len(channels),
                       height=5.2 if len(channels) <= 24 else 5.8)

    good = [(i, z, nu, p, bk) for i, (ch, z, nu, p, bk) in enumerate(results)
            if math.isfinite(z)]
    if not good:
        sys.exit(f"[!] No channel has both a step before the thallium "
                 f"stabilization and '{Z_AFTER}' with a usable error: nothing "
                 f"to compare.")

    col_up, col_dn = "#12855F", "#C1272D"      # improvement / worsening
    n_ch  = len(channels)
    m_sig = 8.5 if n_ch <= 10 else (7.5 if n_ch <= 24 else 6.5)

    # Per-channel acceptance band: the t quantile depends on the d.o.f. of that
    # channel's two fits, so the band is a staircase, not a pair of lines.
    t_crit = [student_interval(nu, args.cl) for _, _, nu, _, _ in good]
    xs     = np.array([i for i, _, _, _, _ in good], float)
    edges  = np.concatenate([xs - 0.5, [xs[-1] + 0.5]])
    band   = np.concatenate([t_crit, [t_crit[-1]]])
    ax.fill_between(edges, -band, band, step="post", color="#7E8794",
                    alpha=0.22, lw=0, zorder=1)
    ax.step(edges, band, where="post", color="#5C6470", lw=1.2, zorder=2)
    ax.step(edges, -np.array(band), where="post", color="#5C6470", lw=1.2,
            zorder=2)

    ax.axhline(0.0, color="#2A2A2A", lw=1.4, zorder=3)

    zs       = np.array([z for _, z, _, _, _ in good])
    fallback = sorted({bk for _, _, _, _, bk in good if bk != Z_BEFORE})
    for i, z, nu, p, bk in good:
        colour = col_up if z >= 0 else col_dn
        # Hollow when the "before" is not the usual step: the point is still a
        # before/after of the thallium stabilization, but of a shorter chain.
        if bk == Z_BEFORE:
            ax.plot([i], [z], "o", ms=m_sig, mfc=colour, mec="white", mew=1.0,
                    zorder=6)
        else:
            ax.plot([i], [z], "o", ms=m_sig, mfc="white", mec=colour, mew=2.0,
                    zorder=6)
        # Stem down to zero: it turns a cloud of dots into a per-channel bar of
        # evidence, which is what a significance plot is read for. Solid, not
        # faded: at detector scale the stems ARE the shape of the figure.
        ax.plot([i, i], [0.0, z], "-", color=colour,
                lw=2.2 if n_ch <= 24 else 1.8, alpha=0.85, zorder=4)

    ax.set_ylabel("Significance  $z$", fontsize=12.5, labelpad=8)
    # Limits follow the data instead of being mirrored around zero: when almost
    # every channel improves, a symmetric range spends half the figure on an
    # empty lower half and squashes the points that carry the message. The
    # acceptance band is always kept fully visible.
    ax.set_ylim(min(-1.35 * max(t_crit), 1.15 * float(zs.min())),
                max( 1.35 * max(t_crit), 1.15 * float(zs.max())))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    finish_axes(ax, channels)

    # Which way is up: stated on the plot, not left to the caption.
    # On a white patch: the lower one sits on the acceptance band.
    _bbox = dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2.0)
    ax.text(0.99, 0.955, "$\\uparrow$ improvement", transform=ax.transAxes,
            ha="right", va="top", fontsize=11, color=col_up, fontweight="bold",
            bbox=_bbox, zorder=7)
    ax.text(0.99, 0.045, "$\\downarrow$ worsening", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11, color=col_dn,
            fontweight="bold", bbox=_bbox, zorder=7)

    ax.set_title("Statistical significance of the resolution change produced by "
                 "the thallium stabilization",
                 fontsize=13.5, fontweight="bold", pad=54)
    nus = [nu for _, _, nu, _, _ in good if math.isfinite(nu)]
    nu_note = f"$\\nu\\approx${np.median(nus):.0f}" if nus else "Gaussian limit"
    add_subtitle(ax,
                 f"{STEP_LABEL[Z_BEFORE]}  $\\rightarrow$  {STEP_LABEL[Z_AFTER]}"
                 f"  |  {len(good)} channels",
                 "$z=(R_\\mathrm{before}-R_\\mathrm{after})/"
                 "\\sqrt{\\sigma_\\mathrm{before}^2+\\sigma_\\mathrm{after}^2}$"
                 "  |  errors added in quadrature")

    handles = [
        Line2D([], [], color=col_up, marker="o", ms=9, mfc=col_up, mec="white",
               mew=0.8, ls="none", label="resolution improved  ($z>0$)"),
        Line2D([], [], color=col_dn, marker="o", ms=9, mfc=col_dn, mec="white",
               mew=0.8, ls="none", label="resolution worsened  ($z<0$)"),
        Patch(facecolor="#9AA0A6", alpha=0.3,
              label=f"not significant at {100*args.cl:.0f} % "
                    f"(Student $t$, {nu_note})"),
    ]
    for bk in fallback:
        handles.append(Line2D([], [], color="#555555", marker="o", ms=8,
                              mfc="white", mec="#555555", mew=2.0, ls="none",
                              label=f"compared against {STEP_LABEL[bk].lower()}"))

    # An extra entry pushes the legend onto a second row: drop it further so it
    # does not sit on the x-axis label.
    leg = ax.legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, -0.14 - 0.07 * (len(handles) > 3)),
                    ncol=3, frameon=True,
                    fontsize=10, borderpad=0.7, columnspacing=1.8,
                    handletextpad=0.7)
    leg.get_frame().set_edgecolor("#CCCCCC")
    leg.get_frame().set_facecolor("white")

    fig.tight_layout()
    return fig


def compute_significance(data, channels):
    """
    [(channel, z, nu, p, before_key)] for every channel, in the order given.
    *before_key* is the step the comparison actually used (see Z_BEFORE_KEYS).
    """
    after = data.get(Z_AFTER, {})
    out   = []
    for ch in channels:
        for key in Z_BEFORE_KEYS:
            rec = data.get(key, {}).get(ch)
            if rec is not None:
                out.append((ch, *significance(rec, after.get(ch)), key))
                break
        else:
            out.append((ch, float("nan"), float("nan"), float("nan"), None))
    return out


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

    print(f"\nSignificance of the thallium stabilization (-> {STEP_LABEL[Z_AFTER]})")
    print(f"{'ch':>5} | {'before':^12} | {'z':>7} | {'nu_eff':>7} | "
          f"{'p (2-sided)':>12} | verdict")
    print("-" * 74)
    for ch, z, nu, p, bk in results:
        name = bk if bk else "-"
        if not math.isfinite(z):
            print(f"{ch:>5} | {name:^12} |     n/a |     n/a |          n/a | "
                  f"no usable error")
            continue
        t_c = student_interval(nu, args.cl)
        if abs(z) < t_c:
            verdict = "not significant"
        else:
            verdict = "improvement" if z > 0 else "WORSENING"
        print(f"{ch:>5} | {name:^12} | {z:7.2f} | {nu:7.1f} | {p:12.2e} | {verdict}")
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
    ap.add_argument("--ymax", type=float, default=None,
                    help="upper limit of the resolution axis, in %%. A single "
                         "bad channel otherwise squashes the whole detector "
                         "into the bottom of the frame; points above the limit "
                         "are marked with a triangle at the top edge")
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

    # Channels left out of BOTH figures (see EXCLUDE_CHANNELS). Applied last, so
    # it holds whatever else was asked for, and reported: a channel missing from
    # a thesis figure must never be missing silently.
    if EXCLUDE_CHANNELS:
        dropped  = [c for c in channels if c in set(EXCLUDE_CHANNELS)]
        channels = [c for c in channels if c not in set(EXCLUDE_CHANNELS)]
        if dropped:
            print(">>> Excluded channel(s): " + ", ".join(str(c) for c in dropped))
        if not channels:
            sys.exit("[!] EXCLUDE_CHANNELS leaves no channel to plot.")

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
