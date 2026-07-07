"""
baseline_jump_detector.py  (v7 - baseline vs amplitude fit)
===========================================================================
Same as v6 but the linear fit is done on  baseline (x)  vs  amplitude (y)
instead of time (x) vs amplitude (y).

Time is still used to:
  - detect segment boundaries (rolling median over ordered events)
  - sort events

Within each segment a robust linear fit  amplitude = p0 + slope * baseline
is performed.  The correction factor at event i is then:

    corrected_amp_i = amp_i * NOMINAL / (p0 + slope * baseline_i)

where NOMINAL is the median heater amplitude over the whole run.
"""

import sys, os, time, re
import numpy as np
from scipy.ndimage import median_filter

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.Math.MinimizerOptions.SetDefaultMinimizer("Minuit2")

# ===========================================================================
# PARAMETERS  <- tune these
# ===========================================================================

DIRECTORY           = "/data/users/azanelli/octopus_work/CROSS/MergedRuns/"
DIRECTORYOUT        = "/data/users/azanelli/octopus_work/CROSS/MergedRuns/CorrectedAmp/"
SKIP_CHANNELS = [25, 59]

# Rolling median (used on amplitude-vs-time for segment detection only)
WINDOW_EVENTS       = 500

# Segment detection
NOMINAL             = 10000.0
DEVIATION_THRESHOLD = 0.3
MIN_SEGMENT_EVENTS  = 500

# Robust linear fit
FIT_THIN            = 1
ROB_FRACTION        = 0.70

# Fit quality guards  (segments failing these are marked fit_ok=False
# and fall back to a flat correction at the segment median amplitude)
MIN_FIT_EVENTS      = 40    # minimum heater events in a segment after thinning
MIN_BL_RANGE        = 0.001  # minimum baseline spread (bl_hi - bl_lo);
                             # below this the fit is numerically meaningless
MAX_SLOPE_ABS       = 5000   # |slope| [amplitude / baseline_unit] above this
                             # the fit is considered diverged and discarded

# Debug plot
DEBUG_PLOT          = True
PLOT_THIN           = 5
DEBUG_DIR           = "/data/users/azanelli/octopus_work/CROSS/MergedRuns/CorrectedAmp/PlotFit/"

MIN_BL_RANGE        = 0.001

# Optional data-range cuts applied before the fit (None = no cut)
FIT_BL_MIN          = -9.0  
FIT_BL_MAX          = 9.0   
FIT_AMP_MIN         = 9000  
FIT_AMP_MAX         = 11000  

RMS = 0

# ===========================================================================
# HELPERS
# ===========================================================================

def rolling_median(arr, win):
    return median_filter(arr.astype(np.float64), size=win, mode="nearest")


def tgraph(x_arr, y_arr):
    x = np.ascontiguousarray(x_arr, dtype=np.float64)
    y = np.ascontiguousarray(y_arr, dtype=np.float64)
    return ROOT.TGraph(len(x), x, y)


def tline(x1, y1, x2, y2, color, width=2, style=1):
    l = ROOT.TLine(float(x1), float(y1), float(x2), float(y2))
    l.SetLineColor(color); l.SetLineWidth(width); l.SetLineStyle(style)
    return l


SEG_COLORS = [
    ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kMagenta + 1,
    ROOT.kCyan + 2, ROOT.kOrange + 2, ROOT.kGreen + 2,
    ROOT.kViolet + 2, ROOT.kTeal + 2,
]

# ===========================================================================
# DATA LOADING  (pulser / heater events only, for segment detection + fit)
# ===========================================================================

def load_data(filepath):
    """Returns (times, baselines, amplitude) sorted by time, after quality cuts."""

    global NOMINAL, RMS

    f = ROOT.TFile.Open(filepath, "READ")
    if not f or f.IsZombie():
        print(f"  [ERR] Cannot open {filepath}")
        return None, None, None

    required = (
        "baseline", "timestamp", "module",
        "badinterval", "numberoftriggers", "flagpropagator_heater",
        "stabilization_all", "optimumfilter_all"
    )
    trees = {}
    for name in required:
        t = f.Get(name)
        if not t:
            print(f"  [ERR] Tree '{name}' not found in {filepath}")
            f.Close()
            return None, None, None
        trees[name] = t

    n = trees["baseline"].GetEntries()
    t_bl, t_ts     = trees["baseline"],         trees["timestamp"]
    t_mod, t_bi    = trees["module"],           trees["badinterval"]
    t_ntrg, t_flag = trees["numberoftriggers"], trees["flagpropagator_heater"]
    t_stabAmp      = trees["stabilization_all"]

    amplitudes, timestamps, baselines = [], [], []

    for i in range(n):
        t_bl.GetEntry(i);  t_ts.GetEntry(i);    t_mod.GetEntry(i)
        t_bi.GetEntry(i);  t_ntrg.GetEntry(i);  t_flag.GetEntry(i)
        t_stabAmp.GetEntry(i)

        if int(t_bi.heat_badinterval)        != 0: continue
        if int(t_ntrg.heat_numberoftriggers) != 1: continue

        try:
            if int(t_mod.issignal)      != 1: continue
        except Exception:
            if int(t_mod.heat_issignal) != 1: continue

        try:
            if int(t_flag.IsHeater)      != 1: continue
        except Exception:
            if int(t_flag.heat_IsHeater) != 1: continue

        amp  = float(t_stabAmp.heat_amplitude)
        ts   = float(t_ts.time_cumulative)
        bl   = float(t_bl.heat_baseline)        # <-- baseline value
        if not (np.isfinite(amp) and np.isfinite(ts) and np.isfinite(bl)):
            continue

        amplitudes.append(amp)
        timestamps.append(ts)
        baselines.append(bl)

    f.Close()

    NOMINAL = np.median(amplitudes)
    RMS     = np.sqrt(np.median((amplitudes - NOMINAL)**2))

    n_ok = len(amplitudes)
    if n_ok < WINDOW_EVENTS * 2:
        print(f"  [WARN] Only {n_ok} events after filtering -- skipping.")
        return None, None, None

    times = np.array(timestamps, dtype=np.float64)
    amps  = np.array(amplitudes, dtype=np.float64)
    bls   = np.array(baselines,  dtype=np.float64)
    idx   = np.argsort(times)
    return times[idx], bls[idx], amps[idx]


# ===========================================================================
# SEGMENT DETECTION  (still uses time-ordered amplitude, unchanged)
# ===========================================================================

def find_segments(times, amplitude, filename):
    rm        = rolling_median(amplitude, WINDOW_EVENTS)
    deviation = np.abs((rm - NOMINAL))

    padded = np.concatenate([[False], deviation > DEVIATION_THRESHOLD, [False]])
    diff   = np.diff(padded.astype(np.int8))
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0]

    boundary_indices = set()
    for s, e in zip(starts, ends):
        if e - s >= MIN_SEGMENT_EVENTS:
            boundary_indices.add(s)
            boundary_indices.add(e)

    n         = len(times)
    split_pts = sorted([0] + list(boundary_indices) + [n])
    split_pts = [split_pts[0]] + [
        split_pts[i] for i in range(1, len(split_pts))
        if split_pts[i] != split_pts[i - 1]
    ]

    all_segs = []
    for k in range(len(split_pts) - 1):
        s = split_pts[k]
        e = split_pts[k + 1]
        if e - s < 2:
            continue
        all_segs.append({
            "seg_idx"  : k + 1,
            "i_start"  : s,
            "i_end"    : e,
            "t_start_s": float(times[s]),
            "t_end_s"  : float(times[min(e, n - 1)]),
        })

    internal    = sorted(boundary_indices)
    breakpoints = [
        (float(times[min(idx, n - 1)]), f"B{k + 1}")
        for k, idx in enumerate(internal)
    ]

    return all_segs, rm, deviation, breakpoints


# ===========================================================================
# ROBUST LINEAR FIT PER SEGMENT  (baseline on x, amplitude on y)
# ===========================================================================

def fit_segments(baselines, amplitude, all_segs):
    """
    For each segment fits:  amplitude = intercept + slope * baseline
    using ROOT's robust fitter (ROB=fraction).
    """
    fit_funcs = []
    rob_opt   = f"ROB={ROB_FRACTION:.2f}"

    for seg in all_segs:
        s = seg["i_start"]
        e = seg["i_end"]

        bl_seg = baselines[s:e:FIT_THIN]
        a_seg  = amplitude[s:e:FIT_THIN]

        # --- optional range cuts before fit ----------------------------------
        mask = np.ones(len(bl_seg), dtype=bool)
        if FIT_BL_MIN  is not None: mask &= (bl_seg >= FIT_BL_MIN)
        if FIT_BL_MAX  is not None: mask &= (bl_seg <= FIT_BL_MAX)
        if FIT_AMP_MIN is not None: mask &= (a_seg  >= FIT_AMP_MIN)
        if FIT_AMP_MAX is not None: mask &= (a_seg  <= FIT_AMP_MAX)
        n_cut = int(np.sum(~mask))
        if n_cut:
            print(f"    [range cut] Seg {seg['seg_idx']:2d}: "
                  f"removed {n_cut}/{len(bl_seg)} points "
                  f"(BL=[{FIT_BL_MIN},{FIT_BL_MAX}]  "
                  f"AMP=[{FIT_AMP_MIN},{FIT_AMP_MAX}])")
        bl_seg = bl_seg[mask]
        a_seg  = a_seg[mask]

        n_pts    = len(bl_seg)
        bl_range = float(bl_seg.max() - bl_seg.min()) if n_pts > 1 else 0.0

        # --- pre-fit guards --------------------------------------------------
        fail_reason = None
        if n_pts < MIN_FIT_EVENTS:
            fail_reason = f"too few points ({n_pts} < {MIN_FIT_EVENTS})"
        elif bl_range < MIN_BL_RANGE:
            fail_reason = (f"baseline range too narrow "
                           f"({bl_range:.6f} < {MIN_BL_RANGE})")

        if fail_reason:
            med_amp = float(np.median(a_seg))
            print(f"    [SKIP fit] Seg {seg['seg_idx']:2d}: {fail_reason} "
                  f"-> flat correction at median={med_amp:.2f}")
            seg.update({"slope": 1.0, "intercept": 0.0, # flat correction: no change from nominal
                        "bl_centre": float(np.median(bl_seg)),
                        "bl_lo": float(bl_seg.min()),
                        "bl_hi": float(bl_seg.max()),
                        "chi2": None, "ndf": None,
                        "fit_ok": False, "skip_reason": fail_reason})
            fit_funcs.append(None)
            continue

        bl_centre  = float(np.median(bl_seg))
        bl_shifted = bl_seg - bl_centre

        bl_lo = float(bl_shifted.min()) - 1.0
        bl_hi = float(bl_shifted.max()) + 1.0

        g     = tgraph(bl_shifted, a_seg)
        fname = f"f_seg_{seg['seg_idx']}"
        f1    = ROOT.TF1(fname, "pol1", bl_lo, bl_hi)

        g.Fit(f1, f"SQN {rob_opt}")

        slope     = f1.GetParameter(1)
        intercept = f1.GetParameter(0)   # value at bl_centre

        # --- post-fit guard: slope sanity check ------------------------------
        if abs(slope) > MAX_SLOPE_ABS:
            med_amp = float(np.median(a_seg))
            fail_reason = (f"slope diverged (|{slope:.1f}| > {MAX_SLOPE_ABS})")
            print(f"    [BAD fit]  Seg {seg['seg_idx']:2d}: {fail_reason} "
                  f"-> flat correction at median={med_amp:.2f}")
            seg.update({"slope": 0.0, "intercept": med_amp,
                        "bl_centre": bl_centre,
                        "bl_lo": float(bl_seg.min()),
                        "bl_hi": float(bl_seg.max()),
                        "chi2": f1.GetChisquare(), "ndf": f1.GetNDF(),
                        "fit_ok": False, "skip_reason": fail_reason})
            fit_funcs.append(None)
            continue

        seg.update({
            "slope"    : slope,
            "intercept": intercept,        # = predicted amplitude at bl_centre
            "bl_centre": bl_centre,
            "bl_lo"    : float(bl_seg.min()),
            "bl_hi"    : float(bl_seg.max()),
            "chi2"     : f1.GetChisquare(),
            "ndf"      : f1.GetNDF(),
            "fit_ok"   : True,
        })
        fit_funcs.append((f1, bl_centre))

    return fit_funcs


# ===========================================================================
# BUILD FAST SEGMENT LOOKUP  (keyed on time boundaries, corrects via baseline)
# ===========================================================================

def build_seg_lookup(all_segs):
    """
    Include every segment that has a valid intercept, whether the linear
    fit succeeded (fit_ok=True) or fell back to a flat median (fit_ok=False
    but intercept is set).  Segments with intercept=None are excluded.
    """
    params = []
    for seg in all_segs:
        if seg.get("intercept") is None:
            continue
        params.append((
            seg["t_start_s"],
            seg["t_end_s"],
            seg.get("slope", 0.0),
            seg["intercept"],
            seg["bl_centre"],
            seg["seg_idx"],
        ))
    params.sort(key=lambda x: x[0])
    return params


def get_correction(ts, bl, seg_params_list):
    """
    Returns (predicted_amplitude, seg_id) for event at time ts with
    baseline value bl, using the segment whose time window contains ts.
    predicted_amplitude = intercept + slope * (bl - bl_centre)
    """
    for t0, t1, slope, intercept, bl_centre, seg_id in seg_params_list:
        if t0 <= ts <= t1:
            return intercept + slope * (bl - bl_centre), seg_id

    # Fallback: closest segment centre
    if seg_params_list:
        best = min(seg_params_list,
                   key=lambda s: abs(ts - 0.5 * (s[0] + s[1])))
        t0, t1, slope, intercept, bl_centre, seg_id = best
        return intercept + slope * (bl - bl_centre), seg_id

    return None, -1


# ===========================================================================
# DETECT issignal BRANCH NAME
# ===========================================================================

def detect_issignal_branch(filepath):
    f = ROOT.TFile.Open(filepath, "READ")
    t = f.Get("module")
    t.GetEntry(0)
    try:
        _ = int(t.issignal)
        name = "issignal"
    except Exception:
        name = "heat_issignal"
    f.Close()
    return name


# ===========================================================================
# WRITE OUTPUT FILE
# ===========================================================================

def write_corrected_tree(filepath, all_segs, issignal_branch):
    seg_params_list = build_seg_lookup(all_segs)
    signal_cut      = f"{issignal_branch} == 1"

    stem     = os.path.splitext(os.path.basename(filepath))[0]
    outdir   = DIRECTORYOUT or os.path.dirname(filepath)
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{stem}_corr.root")

    print(f"  Output    : {out_path}", flush=True)
    print(f"  Signal cut: '{signal_cut}'", flush=True)

    f_in = ROOT.TFile.Open(filepath, "READ")
    if not f_in or f_in.IsZombie():
        print(f"  [ERR] Cannot open {filepath}")
        return

    f_out = ROOT.TFile.Open(out_path, "RECREATE")
    if not f_out or f_out.IsZombie():
        print(f"  [ERR] Cannot create {out_path}")
        f_in.Close()
        return

    t_module = f_in.Get("module")
    if not t_module:
        print("  [ERR] 'module' tree not found")
        f_out.Close(); f_in.Close()
        return

    t_module.Draw(">>elist", signal_cut, "entrylist")
    elist = ROOT.gDirectory.Get("elist")
    if not elist or elist.GetN() == 0:
        print(f"  [ERR] Entry list is empty for cut '{signal_cut}'")
        f_out.Close(); f_in.Close()
        return

    n_signal = elist.GetN()
    print(f"  Entry list: {n_signal} signal entries selected", flush=True)

    tree_names = []
    for key in f_in.GetListOfKeys():
        obj_name  = key.GetName()
        classname = key.GetClassName()

        if classname.startswith("TTree") or classname.startswith("TNtuple"):
            tree_in = f_in.Get(obj_name)
            tree_in.SetEntryList(elist)
            f_out.cd()
            tree_out_copy = tree_in.CopyTree("")
            tree_in.SetEntryList(ROOT.nullptr)
            if not tree_out_copy:
                print(f"  [WARN] CopyTree returned null for '{obj_name}' -- skipping")
                continue
            tree_out_copy.Write("", ROOT.TObject.kOverwrite)
            n_out = tree_out_copy.GetEntries()
            print(f"    {obj_name:40s}  {tree_in.GetEntries():7d} -> {n_out:7d} entries")
            tree_names.append(obj_name)
        else:
            obj = f_in.Get(obj_name)
            if obj:
                f_out.cd()
                obj.Write(obj_name, ROOT.TObject.kOverwrite)

    # ---- build corrected_amplitude tree ------------------------------------
    # Now needs baseline from the filtered baseline tree as well
    t_stab_out = f_out.Get("stabilization_all")
    t_ts_out   = f_out.Get("timestamp")
    t_bl_out   = f_out.Get("baseline")          # <-- needed for baseline value

    if not t_stab_out or not t_ts_out or not t_bl_out:
        print("  [ERR] filtered stabilization_all, timestamp, or baseline "
              "missing -- skipping corrected_amplitude")
        f_out.Close(); f_in.Close()
        return

    n_sig = min(t_stab_out.GetEntries(),
                t_ts_out.GetEntries(),
                t_bl_out.GetEntries())

    import array as _array
    f_out.cd()
    corr_tree  = ROOT.TTree("corrected_amplitude",
                            "per-event corrected amplitude (signal only)")
    corr_val   = _array.array("d", [0.0])
    ts_val     = _array.array("d", [0.0])
    bl_val     = _array.array("d", [0.0])
    seg_id_val = _array.array("i", [0])

    corr_tree.Branch("heat_amplitude",  corr_val,   "heat_amplitude/D")
    corr_tree.Branch("time_cumulative", ts_val,     "time_cumulative/D")
    corr_tree.Branch("heat_baseline",   bl_val,     "heat_baseline/D")
    corr_tree.Branch("segment_id",      seg_id_val, "segment_id/I")

    n_zero = 0
    for i in range(n_sig):
        t_stab_out.GetEntry(i)
        t_ts_out.GetEntry(i)
        t_bl_out.GetEntry(i)

        amp = float(t_stab_out.heat_amplitude)
        ts  = float(t_ts_out.time_cumulative)
        bl  = float(t_bl_out.heat_baseline)

        if not (np.isfinite(amp) and np.isfinite(ts) and np.isfinite(bl)):
            corr   = 0.0
            seg_id = -1
            n_zero += 1
        else:
            denom, seg_id = get_correction(ts, bl, seg_params_list)
            if denom is None or abs(denom) < 1e-9:
                corr   = 0.0
                seg_id = -1
                n_zero += 1
            else:
                corr = amp * NOMINAL / denom

        corr_val[0]   = corr
        ts_val[0]     = ts
        bl_val[0]     = bl
        seg_id_val[0] = seg_id
        corr_tree.Fill()

    print(f"    {'corrected_amplitude':40s}  {n_sig:7d} entries  "
          f"({n_zero} set to zero)")

    # ---- update global tree friend -----------------------------------------
    globalTree = f_out.Get("global")
    if not globalTree:
        print("  [INFO] Creating empty 'global' tree")
        globalTree = ROOT.TTree("global", "Empty global tree")
        f_out.cd()
        globalTree.Write()
        globalTree = f_out.Get("global")

    for key in f_out.GetListOfKeys():
        if key.GetClassName() == "TTree" and key.GetName() != "global":
            globalTree.AddFriend(key.GetName())

    globalTree.AddFriend("corrected_amplitude")
    f_out.cd()
    globalTree.Write("", ROOT.TObject.kOverwrite)
    del globalTree

    f_out.Write("", ROOT.TObject.kOverwrite)
    f_out.Close()
    f_in.Close()
    print(f"  Done -> {out_path}")


# ===========================================================================
# DEBUG PLOT  (top pad: amplitude vs baseline coloured by segment;
#              middle pad: amplitude vs time  [rolling median + fit lines];
#              bottom pad: deviation vs time)
# ===========================================================================

def make_debug_plot(times, baselines, amplitude, rm, deviation,
                    all_segs, breakpoints, filepath):
    fname  = os.path.basename(filepath)
    outdir = DEBUG_DIR or os.path.dirname(filepath)
    os.makedirs(outdir, exist_ok=True)
    stem   = os.path.splitext(fname)[0]
    png    = os.path.join(outdir, f"{stem}_debug.png")

    thin  = PLOT_THIN
    x_min = float(times[0])
    x_max = float(times[-1])

    t_th  = np.ascontiguousarray(times[::thin],     dtype=np.float64)
    a_th  = np.ascontiguousarray(amplitude[::thin], dtype=np.float64)
    bl_th = np.ascontiguousarray(baselines[::thin], dtype=np.float64)
    rm_th = np.ascontiguousarray(rm[::thin],        dtype=np.float64)
    dv_th = np.ascontiguousarray(deviation[::thin], dtype=np.float64)

    a_min = float(np.min(amplitude)); a_max = float(np.max(amplitude))
    am    = (a_max - a_min) * 0.05
    a_min -= am; a_max += am
    d_max = float(np.max(deviation)) * 1.10

    c = ROOT.TCanvas(stem, stem, 1600, 1300)
    c.Divide(1, 3)
    store = []

    # ── pad 1: amplitude vs baseline, coloured by segment ───────────────────
    c.cd(1)
    ROOT.gPad.SetLeftMargin(0.07); ROOT.gPad.SetRightMargin(0.02)
    ROOT.gPad.SetBottomMargin(0.14)

    # build a per-event segment index array (thinned)
    seg_index = np.full(len(times), -1, dtype=int)
    for seg in all_segs:
        seg_index[seg["i_start"]:seg["i_end"]] = seg["seg_idx"]
    seg_index_th = seg_index[::thin]

    # draw one TGraph per segment so each gets its own colour
    first = True
    for seg in all_segs:
        mask = seg_index_th == seg["seg_idx"]
        if not np.any(mask):
            continue
        col = SEG_COLORS[(seg["seg_idx"] - 1) % len(SEG_COLORS)]
        g = tgraph(bl_th[mask], a_th[mask])
        g.SetMarkerStyle(7); g.SetMarkerColor(col)
        g.SetMarkerSize(0.3); g.SetLineWidth(0)
        if first:
            g.SetTitle(f"{fname}  baseline vs amplitude by segment"
                       f";heat_baseline;amplitude")
            g.GetYaxis().SetRangeUser(NOMINAL - 5.0 * RMS,
                                      NOMINAL + 5.0 * RMS)
            g.Draw("AP")
            first = False
        else:
            g.Draw("P SAME")
        store.append(g)

    # overlay the fit lines in baseline space
    for k, seg in enumerate(all_segs):
        if not seg.get("fit_ok"):
            continue
        col = SEG_COLORS[(seg["seg_idx"] - 1) % len(SEG_COLORS)]
        bl0   = seg["bl_lo"]
        bl1   = seg["bl_hi"]
        tc    = seg["bl_centre"]
        p0    = seg["intercept"]
        p1    = seg["slope"]
        y0    = p0 + p1 * (bl0 - tc)
        y1    = p0 + p1 * (bl1 - tc)
        fl    = tline(bl0, y0, bl1, y1, col, width=3, style=1)
        fl.Draw(); store.append(fl)

        mid_bl = 0.5 * (bl0 + bl1)
        mid_y  = 0.5 * (y0 + y1)
        lbl = ROOT.TLatex(
            mid_bl, mid_y + (a_max - a_min) * 0.03,
            f"S{seg['seg_idx']} s={p1:.3f}"
        )
        lbl.SetTextSize(0.042); lbl.SetTextColor(col)
        lbl.Draw(); store.append(lbl)

    # ── pad 2: amplitude vs time (rolling median + segment colouring) ────────
    c.cd(2)
    ROOT.gPad.SetLeftMargin(0.07); ROOT.gPad.SetRightMargin(0.02)
    ROOT.gPad.SetBottomMargin(0.12)

    g_raw = tgraph(t_th, a_th)
    g_raw.SetTitle(f"{fname}  amplitude vs time"
                   f";time_cumulative (s);amplitude")
    g_raw.SetMarkerStyle(7); g_raw.SetMarkerColor(ROOT.kAzure + 1)
    g_raw.SetMarkerSize(0.3); g_raw.SetLineWidth(0)
    g_raw.GetYaxis().SetRangeUser(NOMINAL - 5.0 * RMS, NOMINAL + 5.0 * RMS)
    g_raw.Draw("AP"); store.append(g_raw)

    g_rm = tgraph(t_th, rm_th)
    g_rm.SetMarkerStyle(7); g_rm.SetMarkerColor(ROOT.kBlack)
    g_rm.SetMarkerSize(0.5); g_rm.SetLineWidth(0)
    g_rm.Draw("P SAME"); store.append(g_rm)

    nl = tline(x_min, NOMINAL, x_max, NOMINAL,
               ROOT.kGray + 1, width=1, style=2)
    nl.Draw(); store.append(nl)

    for t_bp, label in breakpoints:
        vl = tline(t_bp, a_min, t_bp, a_max, ROOT.kRed, width=2, style=3)
        vl.Draw(); store.append(vl)
        lbl = ROOT.TLatex(
            t_bp + (x_max - x_min) * 0.004,
            a_max - (a_max - a_min) * 0.05,
            label,
        )
        lbl.SetTextColor(ROOT.kRed); lbl.SetTextSize(0.055)
        lbl.SetTextAngle(90); lbl.Draw(); store.append(lbl)

    # ── pad 3: deviation vs time ─────────────────────────────────────────────
    c.cd(3)
    ROOT.gPad.SetLeftMargin(0.07); ROOT.gPad.SetRightMargin(0.02)
    ROOT.gPad.SetBottomMargin(0.14)

    g_dv = tgraph(t_th, dv_th)
    g_dv.SetTitle(f"|rolling median - {NOMINAL:.0f}|"
                  f";time_cumulative (s);deviation")
    g_dv.SetMarkerStyle(7); g_dv.SetMarkerColor(ROOT.kOrange + 1)
    g_dv.SetMarkerSize(0.4); g_dv.SetLineWidth(0)
    g_dv.GetYaxis().SetRangeUser(0.0, d_max)
    g_dv.Draw("AP"); store.append(g_dv)

    thr = tline(x_min, DEVIATION_THRESHOLD, x_max, DEVIATION_THRESHOLD,
                ROOT.kRed, width=2, style=2)
    thr.Draw(); store.append(thr)

    thr_lbl = ROOT.TLatex(
        x_min + 0.01 * (x_max - x_min),
        DEVIATION_THRESHOLD * 1.04,
        f"threshold = {DEVIATION_THRESHOLD:.1f}",
    )
    thr_lbl.SetTextColor(ROOT.kRed); thr_lbl.SetTextSize(0.055)
    thr_lbl.Draw(); store.append(thr_lbl)

    for t_bp, label in breakpoints:
        vl = tline(t_bp, 0.0, t_bp, d_max, ROOT.kRed, width=2, style=3)
        vl.Draw(); store.append(vl)

    c.Update()
    c.SaveAs(png)
    c.Close()
    print(f"  Debug PNG : {png}")


# ===========================================================================
# PROCESS ONE FILE
# ===========================================================================

def process_file(filepath):
    fname = os.path.basename(filepath)
    print(f"\n{'─'*60}")
    print(f"  File      : {fname}")
    print(f"{'─'*60}")

    issignal_branch = detect_issignal_branch(filepath)
    print(f"  issignal branch : '{issignal_branch}'")

    times, baselines, amplitude = load_data(filepath)
    if times is None:
        return

    print(f"  Events    : {len(times)}")
    print(f"  Time      : [{times[0]:.1f}, {times[-1]:.1f}] s")
    print(f"  Baseline  : mean={np.mean(baselines):.2f}  "
          f"median={np.median(baselines):.2f}")
    print(f"  Amplitude : mean={np.mean(amplitude):.2f}  "
          f"median={np.median(amplitude):.2f}")
    print(f"  Nominal   : {NOMINAL:.1f}   threshold: +/-{DEVIATION_THRESHOLD:.1f}")
    print(f"  Finding segments ...", flush=True)

    all_segs, rm, deviation, breakpoints = find_segments(
        times, amplitude, fname
    )

    print(f"  Segments : {len(all_segs)}   breakpoints : {len(breakpoints)}")
    if breakpoints:
        for t_bp, lbl in breakpoints:
            print(f"    {lbl:6s}  t = {t_bp:.1f} s")

    print(f"  Robust-fitting {len(all_segs)} segment(s) "
          f"[ROB={ROB_FRACTION:.0%}] baseline vs amplitude ...", flush=True)
    fit_funcs = fit_segments(baselines, amplitude, all_segs)

    print(f"  Fit results  [amplitude = intercept + slope*(baseline - bl_centre)]  "
          f"(robust ROB={ROB_FRACTION:.0%}):")
    for seg in all_segs:
        if seg.get("intercept") is None:
            print(f"    Seg {seg['seg_idx']:2d}  [SKIPPED - no intercept]")
            continue
        if not seg.get("fit_ok"):
            print(f"    Seg {seg['seg_idx']:2d}  "
                  f"t=[{seg['t_start_s']:.0f}, {seg['t_end_s']:.0f}] s  "
                  f"[FLAT correction  intercept={seg['intercept']:.4f}]  "
                  f"reason: {seg.get('skip_reason','?')}")
            continue
        chi2_str = (f"chi2/ndf={seg['chi2']:.1f}/{seg['ndf']}"
                    if seg["ndf"] and seg["ndf"] > 0 else "ndf=0")
        print(f"    Seg {seg['seg_idx']:2d}  "
              f"t=[{seg['t_start_s']:.0f}, {seg['t_end_s']:.0f}] s  "
              f"intercept={seg['intercept']:.4f}  "
              f"slope={seg['slope']:+.6f}  "
              f"bl_centre={seg['bl_centre']:.2f}  "
              f"{chi2_str}")

    print(f"  Writing signal-only output file ...", flush=True)
    write_corrected_tree(filepath, all_segs, issignal_branch)

    if DEBUG_PLOT:
        make_debug_plot(times, baselines, amplitude, rm, deviation,
                        all_segs, breakpoints, filepath)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    directory = DIRECTORY
    if len(sys.argv) > 1:
        directory = sys.argv[1]

    root_files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".root") and not f.endswith("_corr.root")
    ])

    # --- Scarta i canali presenti in SKIP_CHANNELS --------------------------
    skip_set = {str(c) for c in SKIP_CHANNELS}
    if skip_set:
        kept = []
        for fp in root_files:
            m  = re.search(r"ch(\d+)", os.path.basename(fp))
            ch = m.group(1) if m else None
            if ch is not None and ch in skip_set:
                print(f"  [SKIP] {os.path.basename(fp)} (canale {ch} in SKIP_CHANNELS)")
                continue
            kept.append(fp)
        root_files = kept

        if not root_files:
            print(f"[ERR] Tutti i file sono stati saltati (SKIP_CHANNELS={sorted(skip_set, key=int)})")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  SEGMENT DETECTOR + ROBUST LINEAR FIT  v7 (baseline vs amp)")
    print(f"  Directory   : {directory}")
    print(f"  Files       : {len(root_files)}")
    print(f"  Window      : {WINDOW_EVENTS} events")
    print(f"  Nominal     : {NOMINAL}")
    print(f"  Threshold   : +/- {DEVIATION_THRESHOLD} (amplitude units)")
    print(f"  Min segment : {MIN_SEGMENT_EVENTS} events")
    print(f"  Robust frac : ROB={ROB_FRACTION:.0%} (inlier fraction)")
    print(f"  Fit thin    : every {FIT_THIN} point(s)")
    print(f"  Min fit pts : {MIN_FIT_EVENTS}  (else flat correction)")
    print(f"  Min BL range: {MIN_BL_RANGE}    (else flat correction)")
    print(f"  Max |slope| : {MAX_SLOPE_ABS}  (else flat correction)")
    print(f"  Debug plot  : {DEBUG_PLOT}")
    print(f"  Fit BL range: [{FIT_BL_MIN}, {FIT_BL_MAX}]")
    print(f"  Fit Amp range: [{FIT_AMP_MIN}, {FIT_AMP_MAX}]")
    print(f"  Fit axis    : baseline (x) vs amplitude (y)")
    print(f"  Output      : ALL trees filtered to issignal==1")
    print(f"                + corrected_amplitude TTree (signal only)")
    print(f"{'='*60}")

    t0 = time.time()
    for filepath in root_files:
        process_file(filepath)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  DONE   elapsed: {elapsed:.1f} s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()