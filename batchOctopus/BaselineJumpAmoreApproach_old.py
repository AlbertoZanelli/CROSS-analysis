"""
baseline_jump_detector.py  (v8 - blockwise pulser-mean normalization)
===========================================================================
Instead of segment detection + linear fit, this version:

  1. Loads heater/pulser events (time-ordered).
  2. Groups them into consecutive blocks of BLOCK_SIZE pulser events.
  3. For each block, computes the mean amplitude and a correction factor:
         corr_factor_block = NOMINAL / mean(amplitude_block)
     so that, after correction, the block's pulser mean becomes NOMINAL.
  4. For any event (signal or pulser) whose timestamp falls between the
     start time of block k and the start time of block k+1 (i.e. between
     two consecutive blocks' first pulser points), the correction factor
     of block k is applied:
         corrected_amp = amp * corr_factor_block_k

Events before the first block's start time use block 1's factor; events
after the last block's start time use the last block's factor.
"""

import sys, os, time, re
import numpy as np

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.Math.MinimizerOptions.SetDefaultMinimizer("Minuit2")

# ===========================================================================
# PARAMETERS  <- tune these
# ===========================================================================

DIRECTORY           = "/data/users/azanelli/octopus_work/CROSS/MergedRuns/"
DIRECTORYOUT        = "/data/users/azanelli/octopus_work/CROSS/MergedRuns/CorrectedAmp/"

NOMINAL             = 10000.0   # target mean amplitude for each block
BLOCK_SIZE          = 50        # number of pulser events per block

# Debug plot
DEBUG_PLOT          = True
PLOT_THIN           = 5
DEBUG_DIR           = "/data/users/azanelli/octopus_work/CROSS/MergedRuns/CorrectedAmp/PlotFit/"

# Channel filtering (il canale si legge dal nome file, es. ..._ch25_...)
CHANNELS_TO_PROCESS = []    # se NON vuota: processa SOLO questi canali (es. [25, 73])
SKIP_CHANNELS       = [25, 59] 

RMS = 0

# ===========================================================================
# HELPERS
# ===========================================================================

def tgraph(x_arr, y_arr):
    x = np.ascontiguousarray(x_arr, dtype=np.float64)
    y = np.ascontiguousarray(y_arr, dtype=np.float64)
    return ROOT.TGraph(len(x), x, y)


def tline(x1, y1, x2, y2, color, width=2, style=1):
    l = ROOT.TLine(float(x1), float(y1), float(x2), float(y2))
    l.SetLineColor(color); l.SetLineWidth(width); l.SetLineStyle(style)
    return l

def get_channel_from_filename(filepath):
    """Estrae il numero di canale dal nome file: '..._ch25_...' -> 25.
    Restituisce None se il pattern non viene trovato."""
    fname = os.path.basename(filepath)
    m = re.search(r"_ch(\d+)", fname)
    return int(m.group(1)) if m else None

SEG_COLORS = [
    ROOT.kRed + 1, ROOT.kBlue + 1, ROOT.kMagenta + 1,
    ROOT.kCyan + 2, ROOT.kOrange + 2, ROOT.kGreen + 2,
    ROOT.kViolet + 2, ROOT.kTeal + 2,
]

# ===========================================================================
# DATA LOADING  (pulser / heater events only, for block correction)
# ===========================================================================

def load_data(filepath):
    """Returns (times, baselines, amplitude) sorted by time, after quality cuts."""

    global RMS

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

    n_ok = len(amplitudes)
    if n_ok < BLOCK_SIZE:
        print(f"  [WARN] Only {n_ok} events after filtering -- skipping.")
        return None, None, None

    amplitudes = np.array(amplitudes, dtype=np.float64)
    RMS = np.sqrt(np.median((amplitudes - NOMINAL) ** 2))

    times = np.array(timestamps, dtype=np.float64)
    bls   = np.array(baselines,  dtype=np.float64)
    idx   = np.argsort(times)
    return times[idx], bls[idx], amplitudes[idx]


# ===========================================================================
# BLOCK CORRECTION  (group pulser events into blocks of BLOCK_SIZE,
#                    force each block's mean amplitude to NOMINAL)
# ===========================================================================
def build_blocks(times, amplitude):
    """
    Splits the time-ordered pulser/heater events into consecutive blocks of
    BLOCK_SIZE events. For each block computes:
        - t_start : timestamp of the block's first event
        - t_end   : timestamp of the block's last event
        - median_amp: median amplitude over the block
        - corr_factor = NOMINAL / median_amp

    Returns a list of block dicts, ordered by t_start.
    """
    n = len(times)
    blocks = []

    n_blocks = int(np.ceil(n / BLOCK_SIZE))
    for b in range(n_blocks):
        s = b * BLOCK_SIZE
        e = min(s + BLOCK_SIZE, n)
        if e - s < 1:
            continue

        a_blk = amplitude[s:e]
        t_blk = times[s:e]

        median_amp = float(np.median(a_blk))
        if abs(median_amp) < 1e-9:
            corr_factor = 1.0
            print(f"    [WARN] Block {b + 1:3d}: median amplitude ~0, "
                  f"using corr_factor=1.0")
        else:
            corr_factor = NOMINAL / median_amp

        blocks.append({
            "block_idx"   : b + 1,
            "i_start"     : s,
            "i_end"       : e,
            "n_pts"       : e - s,
            "t_start"     : float(t_blk[0]),
            "t_end"       : float(t_blk[-1]),
            "median_amp"  : median_amp,
            "corr_factor" : corr_factor,
        })

    return blocks


def build_correction_lookup(blocks):
    """
    Builds a sorted list of (t_start, corr_factor, block_idx) for use with
    get_correction_factor(). Blocks are already ordered by t_start since the
    input times are sorted.
    """
    return [(blk["t_start"], blk["corr_factor"], blk["block_idx"])
            for blk in blocks]


def get_correction_factor(ts, lookup):
    """
    For a given event timestamp `ts`, find the correction factor of the
    block whose t_start is the largest value <= ts (i.e. the block that
    "covers" the interval starting at its t_start up to the next block's
    t_start). Events before the first block's t_start use block 1's factor.

    `lookup` is a list of (t_start, corr_factor, block_idx) sorted by t_start.

    Returns (corr_factor, block_idx).
    """
    if not lookup:
        return None, -1

    if ts < lookup[0][0]:
        return lookup[0][1], lookup[0][2]

    # binary search for rightmost t_start <= ts
    lo, hi = 0, len(lookup) - 1
    ans = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if lookup[mid][0] <= ts:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return lookup[ans][1], lookup[ans][2]


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

def write_corrected_tree(filepath, lookup, issignal_branch):
    signal_cut = f"{issignal_branch} == 1"

    stem     = os.path.splitext(os.path.basename(filepath))[0]
    outdir   = DIRECTORYOUT or os.path.dirname(filepath)
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
    t_stab_out = f_out.Get("stabilization_all")
    t_ts_out   = f_out.Get("timestamp")
    t_bl_out   = f_out.Get("baseline")

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
    block_id_val = _array.array("i", [0])
    factor_val = _array.array("d", [0.0])

    corr_tree.Branch("heat_amplitude",  corr_val,     "heat_amplitude/D")
    corr_tree.Branch("time_cumulative", ts_val,       "time_cumulative/D")
    corr_tree.Branch("heat_baseline",   bl_val,       "heat_baseline/D")
    corr_tree.Branch("block_id",        block_id_val, "block_id/I")
    corr_tree.Branch("corr_factor",     factor_val,   "corr_factor/D")

    n_zero = 0
    for i in range(n_sig):
        t_stab_out.GetEntry(i)
        t_ts_out.GetEntry(i)
        t_bl_out.GetEntry(i)

        amp = float(t_stab_out.heat_amplitude)
        ts  = float(t_ts_out.time_cumulative)
        bl  = float(t_bl_out.heat_baseline)

        if not (np.isfinite(amp) and np.isfinite(ts) and np.isfinite(bl)):
            corr        = 0.0
            block_id    = -1
            factor      = 0.0
            n_zero += 1
        else:
            factor, block_id = get_correction_factor(ts, lookup)
            if factor is None:
                corr     = 0.0
                block_id = -1
                factor   = 0.0
                n_zero += 1
            else:
                corr = amp * factor

        corr_val[0]     = corr
        ts_val[0]       = ts
        bl_val[0]       = bl
        block_id_val[0] = block_id
        factor_val[0]   = factor
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
# DEBUG PLOT
#   pad 1: raw amplitude vs time (block boundaries marked)
#   pad 2: corrected amplitude vs time (using block factors)
#   pad 3: correction factor vs time (step function across blocks)
# ===========================================================================

def make_debug_plot(times, baselines, amplitude, blocks, lookup, filepath):
    fname  = os.path.basename(filepath)
    outdir = DEBUG_DIR or os.path.dirname(filepath)
    stem   = os.path.splitext(fname)[0]
    png    = os.path.join(outdir, f"{stem}_debug.png")

    thin  = PLOT_THIN
    x_min = float(times[0])
    x_max = float(times[-1])

    t_th  = np.ascontiguousarray(times[::thin],     dtype=np.float64)
    a_th  = np.ascontiguousarray(amplitude[::thin], dtype=np.float64)

    # corrected amplitude (thinned), using block lookup
    a_corr_th = np.empty_like(a_th)
    for i, ts in enumerate(t_th):
        factor, _ = get_correction_factor(ts, lookup)
        a_corr_th[i] = a_th[i] * (factor if factor is not None else 1.0)

    a_min = float(np.min(amplitude)); a_max = float(np.max(amplitude))
    am    = (a_max - a_min) * 0.05
    a_min -= am; a_max += am

    c = ROOT.TCanvas(stem, stem, 1600, 1300)
    c.Divide(1, 3)
    store = []

    # ── pad 1: raw amplitude vs time, block boundaries marked ──────────────
    c.cd(1)
    ROOT.gPad.SetLeftMargin(0.07); ROOT.gPad.SetRightMargin(0.02)
    ROOT.gPad.SetBottomMargin(0.12)

    g_raw = tgraph(t_th, a_th)
    g_raw.SetTitle(f"{fname}  raw amplitude vs time"
                   f";time_cumulative (s);amplitude")
    g_raw.SetMarkerStyle(7); g_raw.SetMarkerColor(ROOT.kAzure + 1)
    g_raw.SetMarkerSize(0.3); g_raw.SetLineWidth(0)
    g_raw.GetYaxis().SetRangeUser(NOMINAL - 5.0 * RMS, NOMINAL + 5.0 * RMS)
    g_raw.Draw("AP"); store.append(g_raw)

    nl = tline(x_min, NOMINAL, x_max, NOMINAL,
               ROOT.kGray + 1, width=1, style=2)
    nl.Draw(); store.append(nl)

    for blk in blocks:
        col = SEG_COLORS[(blk["block_idx"] - 1) % len(SEG_COLORS)]
        vl = tline(blk["t_start"], a_min, blk["t_start"], a_max, col, width=1, style=3)
        vl.Draw(); store.append(vl)

    # ── pad 2: corrected amplitude vs time ──────────────────────────────────
    c.cd(2)
    ROOT.gPad.SetLeftMargin(0.07); ROOT.gPad.SetRightMargin(0.02)
    ROOT.gPad.SetBottomMargin(0.12)

    g_corr = tgraph(t_th, a_corr_th)
    g_corr.SetTitle(f"{fname}  corrected amplitude vs time "
                   f"(block size={BLOCK_SIZE}, target={NOMINAL:.0f})"
                   f";time_cumulative (s);corrected amplitude")
    g_corr.SetMarkerStyle(7); g_corr.SetMarkerColor(ROOT.kGreen + 2)
    g_corr.SetMarkerSize(0.3); g_corr.SetLineWidth(0)
    g_corr.GetYaxis().SetRangeUser(NOMINAL - 5.0 * RMS, NOMINAL + 5.0 * RMS)
    g_corr.Draw("AP"); store.append(g_corr)

    nl2 = tline(x_min, NOMINAL, x_max, NOMINAL,
                ROOT.kGray + 1, width=1, style=2)
    nl2.Draw(); store.append(nl2)

    # ── pad 3: correction factor vs time (step function) ───────────────────
    c.cd(3)
    ROOT.gPad.SetLeftMargin(0.07); ROOT.gPad.SetRightMargin(0.02)
    ROOT.gPad.SetBottomMargin(0.14)

    f_x, f_y = [], []
    for k, (t_start, factor, _bidx) in enumerate(lookup):
        t_next = lookup[k + 1][0] if k + 1 < len(lookup) else x_max
        f_x.append(t_start); f_y.append(factor)
        f_x.append(t_next);  f_y.append(factor)

    f_x = np.array(f_x, dtype=np.float64)
    f_y = np.array(f_y, dtype=np.float64)

    g_fac = tgraph(f_x, f_y)
    g_fac.SetTitle("correction factor vs time;time_cumulative (s);corr_factor = NOMINAL / block mean")
    g_fac.SetLineColor(ROOT.kMagenta + 1); g_fac.SetLineWidth(2)
    g_fac.SetMarkerStyle(0)
    g_fac.Draw("AL"); store.append(g_fac)

    one_line = tline(x_min, 1.0, x_max, 1.0, ROOT.kGray + 1, width=1, style=2)
    one_line.Draw(); store.append(one_line)

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
    print(f"  Nominal   : {NOMINAL:.1f}")
    print(f"  Block size: {BLOCK_SIZE} pulser events/block")

    blocks = build_blocks(times, amplitude)
    n_blocks = len(blocks)
    print(f"  Blocks    : {n_blocks}")

    # for blk in blocks:
    #     print(f"    Block {blk['block_idx']:3d}  "
    #           f"n={blk['n_pts']:3d}  "
    #           f"t=[{blk['t_start']:.1f}, {blk['t_end']:.1f}] s  "
    #           f"mean_amp={blk['mean_amp']:.2f}  "
    #           f"corr_factor={blk['corr_factor']:.6f}")

    lookup = build_correction_lookup(blocks)

    print(f"  Writing signal-only output file ...", flush=True)
    write_corrected_tree(filepath, lookup, issignal_branch)

    if DEBUG_PLOT:
        make_debug_plot(times, baselines, amplitude, blocks, lookup, filepath)


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

    
    filtered = []
    for fp in root_files:
        ch = get_channel_from_filename(fp)
        if ch is None:
            print(f"  [WARN] Canale non riconosciuto in "
                  f"'{os.path.basename(fp)}' -- skip")
            continue
        if CHANNELS_TO_PROCESS and ch not in CHANNELS_TO_PROCESS:
            print(f"  [INFO] ch{ch:<3d}: non in CHANNELS_TO_PROCESS -- skip")
            continue
        if ch in SKIP_CHANNELS:
            print(f"  [INFO] ch{ch:<3d}: in SKIP_CHANNELS -- skip")
            continue
        filtered.append(fp)

    root_files = filtered
    if not root_files:
        print("[ERR] Nessun file rimasto dopo il filtro canali.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  BLOCKWISE PULSER-MEAN NORMALIZATION  v8")
    print(f"  Directory   : {directory}")
    print(f"  Files       : {len(root_files)}")
    print(f"  Nominal     : {NOMINAL}")
    print(f"  Block size  : {BLOCK_SIZE} pulser events/block")
    print(f"  Debug plot  : {DEBUG_PLOT}")
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