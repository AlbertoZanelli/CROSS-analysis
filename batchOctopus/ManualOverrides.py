#!/usr/bin/env python3
"""
===============================================================================
 ManualOverrides.py -- hand-tuned cuts for ThalliumStabilization.py
===============================================================================

 A small editor for the values the automatic analysis gets wrong on a few
 channels. It writes ONE CSV; ThalliumStabilization.py reads it at start-up and
 replaces the automatic value wherever the file has one. Nothing else changes:
 a channel with no row, or a field left empty, is analysed exactly as before.

 What can be overridden
 ----------------------
   correlation cut   the value the analysis puts at the CORR_CUT_PERCENTILE-th
                     percentile of the correlation distribution;
   light detector    which LD the LY cut is made on (the automatic choice is the
                     one with the higher discrimination factor);
   LY window         [min, max] of the light-yield acceptance;
   cleaning window   PER PARTITION, the [min, max] amplitude window of the
                     Gaussian pre-cleaning that decides which events enter the
                     stabilization fit (automatic: peak +/- HEAT_CLEAN_NSIGMA
                     sigma of the preliminary fit);
   partitions        extra baseline boundaries: when a channel has any, they
                     REPLACE the ones FindBaselinePartitions found, which only
                     separates clearly detached blocks;
   line from         PER PARTITION, use another partition's stabilization line
                     instead of the one fitted here;
   escludi           PER PARTITION, leave its events out of the combined
                     results (spectra, resolutions, both tables). The partition
                     is still analysed and drawn: what was thrown away stays
                     visible.

 The second tab edits the FIT of the thallium peak, one row per amplitude of the
 chain. Those settings live in chain_settings.csv, next to the programs, NOT in
 the dataset file: they describe how a channel's line is fitted, and the alpha
 program seeds its own from them. "Numero di bin" is a bin COUNT: leave it at 0
 to let the program derive it from the bin width, or type the count you want and
 it is used as it is.

 Which file it writes
 --------------------
 The cuts belong to the DATA, not to the program: a correlation cut tuned on the
 merged run means nothing on run 92. The file is therefore named after the
 dataset ThalliumStabilization.py is currently configured for
 (ANALYSIS_MODE / RUN_NUMBER, via its dataset_tag()):

     batchOctopus/manual_overrides_mergedrun.csv
     batchOctopus/manual_overrides_run92.csv

 Long format, one row per overridden value, so a channel with five partitions
 needs no extra columns and the file stays hand-editable:

     channel,partition,parameter,value
     57,,corr_cut,0.999814
     57,,chosen_ld,1
     57,0,heat_cut_min,8402
     57,0,heat_cut_max,8698

 An empty *partition* cell marks a channel-level value.

 Where the partitions come from
 ------------------------------
 The per-partition rows of the results table
 (<data>/ThalliumStabilizedAmp/thallium_partition_resolutions.csv), so the
 baseline range and the population of each partition are shown next to its
 fields. Run the analysis once, then tune. Without that table the editor starts
 with one partition and the count can be set by hand.

 Usage
 -----
     conda activate pyrootAlbi
     python ManualOverrides.py

 Then: pick a channel on the left, fill in only what has to change (empty =
 automatic), "Applica al canale", and "Salva su file" when done.

 "Salva e rilancia" saves and re-runs ThalliumStabilization.py on that channel
 alone, in a separate process (its output goes to the terminal this was started
 from). When it finishes, the partitions shown are re-read from the results
 table it has just rewritten -- so a change that splits the channel differently
 is visible right away.
===============================================================================
"""

import os
import sys
import csv
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

# The analysis program is the single source of the paths, the file naming and
# the CSV schema: importing it is what guarantees the editor and the analysis
# always agree on which file belongs to which dataset. (It pulls in ROOT, so the
# first start takes a couple of seconds.)
import ThalliumStabilization as TS


# ===========================================================================
# DATA
# ===========================================================================

def channels_available():
    """
    Channels of the dataset being analysed, from the .root file names in the
    folder ThalliumStabilization.py would scan. Returns a sorted list of channel
    strings, plus the folder (for the window title / error messages).
    """
    folder = TS.resolve_scan_dir()
    if not os.path.isdir(folder):
        return [], folder
    chs = set()
    for fn in os.listdir(folder):
        if not fn.endswith(".root") or "stabilized" in fn:
            continue
        ch = TS.parse_channel_id(fn)
        if ch is not None:
            chs.add(ch)
    return sorted(chs, key=lambda c: int(c) if c.isdigit() else 1 << 30), folder


def partitions_of(folder):
    """
    {channel: [(idx, baseline_lo, baseline_hi, n_events), ...]} from the
    per-partition results table written by the last analysis. Empty dict when
    the table is not there yet (the editor then starts with one partition).
    """
    path = os.path.normpath(os.path.join(folder, TS.RES_CSV_DIR_NAME,
                                         TS.PART_RES_CSV_NAME))
    out = {}
    if not os.path.exists(path):
        return out
    try:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                tag = str(r.get("partition", "")).strip()
                if not tag.startswith("P"):
                    continue                      # the "merged" rows
                ch = str(r.get("channel", "")).strip()
                try:
                    idx = int(tag[1:])
                except ValueError:
                    continue

                def num(k):
                    try:
                        return float(r.get(k, ""))
                    except (TypeError, ValueError):
                        return float("nan")

                seen = {p[0] for p in out.get(ch, [])}
                if idx not in seen:
                    out.setdefault(ch, []).append(
                        (idx, num("baseline_lo"), num("baseline_hi"), num("n_events")))
    except OSError as e:
        print(f"[!] Cannot read {path}: {e}", file=sys.stderr)
        return {}
    for ch in out:
        out[ch].sort()
    return out


def read_chain_settings():
    """{channel: {field: float}} of chain_settings.csv (only the fields we edit)."""
    out = {}
    if not os.path.exists(TS.CHAIN_CSV_PATH):
        return out
    try:
        with open(TS.CHAIN_CSV_PATH, newline="") as fh:
            for r in csv.DictReader(fh):
                ch = str(r.get("channel", "")).strip()
                if not ch:
                    continue
                vals = {}
                for f in TS.ChainSettings.FIELDS:
                    try:
                        vals[f] = float(r[f])
                    except (KeyError, TypeError, ValueError):
                        pass
                out[ch] = vals
    except OSError as e:
        print(f"[!] Cannot read {TS.CHAIN_CSV_PATH}: {e}", file=sys.stderr)
    return out


def write_chain_settings(data):
    """
    Update chain_settings.csv with the rows in *data* ({channel: {field: value}}),
    leaving every other channel exactly as it is. Missing cells are filled with
    the program defaults, the way ChainSettings reads them.
    """
    header = ["channel", *TS.ChainSettings.FIELDS]
    rows, seen = [], set()
    if os.path.exists(TS.CHAIN_CSV_PATH):
        with open(TS.CHAIN_CSV_PATH, newline="") as fh:
            for r in csv.DictReader(fh):
                ch = str(r.get("channel", "")).strip()
                if ch in data:
                    r = dict(r); r.update({k: v for k, v in data[ch].items()})
                    seen.add(ch)
                rows.append(TS.ChainSettings(r).as_row(ch))
    for ch, vals in data.items():
        if ch not in seen:
            rows.append(TS.ChainSettings(vals).as_row(ch))
    with open(TS.CHAIN_CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    return len(data)


def chain_bins_of(folder):
    """
    {channel: {variable: (n_bins, win_frac, res_exp)}} from the results table
    (energy row), for the "numero di bin" hint.

    *n_bins* is what the panel was really drawn with, and is None on the rows
    written before that column existed -- which is most of them until every
    channel is re-run. *win_frac* and *res_exp* have been written for far longer
    and are enough to recompute the automatic count (see auto_bins).
    """
    path = os.path.normpath(os.path.join(folder, TS.RES_CSV_DIR_NAME, TS.RES_CSV_NAME))
    out = {}
    if not os.path.exists(path):
        return out

    def num(r, k):
        try:
            return float(r.get(k, ""))
        except (TypeError, ValueError):
            return None

    try:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if str(r.get("row", "")) != "energy":
                    continue
                n = num(r, "n_bins")
                out.setdefault(str(r.get("channel", "")).strip(), {})[
                    str(r.get("variable", ""))] = (
                        int(n) if n and n == n else None,
                        num(r, "win_frac"), num(r, "res_exp"))
    except OSError:
        pass
    return out


def auto_bins(win_frac, res_exp, bin_div, sig_scale):
    """
    The bin count fit_peak_centred would compute, from the window and the
    expected width the last analysis recorded:

        nb = (hi - lo) / (sigma / bin_div),  hi - lo = 2*win_frac*mu,
                                             sigma  = res_exp*sig_scale*mu

    so the peak position cancels and only the two fractions are needed. Lets the
    hint work on the channels whose rows predate the n_bins column, and shows
    what the CURRENT bin_div would give. None when it cannot be computed.
    """
    if not (win_frac and res_exp and bin_div and sig_scale):
        return None
    n = round(2.0 * win_frac * bin_div / (res_exp * sig_scale))
    return int(min(max(n, TS.CHAIN_MIN_BINS), TS.CHAIN_MAX_BINS))


def write_overrides(path, data):
    """
    Write *data* ({channel: {param: value, "heat": {idx: {param: value}}}}) to
    the overrides CSV. Channels with nothing set are dropped, so removing every
    value of a channel is how it goes back to fully automatic.
    """
    rows = []
    for ch in sorted(data, key=lambda c: int(c) if str(c).isdigit() else 1 << 30):
        d = data[ch]
        for par in TS.MANUAL_CHANNEL_PARAMS:
            if d.get(par) is not None:
                rows.append({"channel": ch, "partition": "",
                             "parameter": par, "value": _fmt(d[par], par)})
        for k, edge in enumerate(sorted(d.get("edges", []))):
            rows.append({"channel": ch, "partition": k,
                         "parameter": TS.MANUAL_EDGE_PARAM, "value": _fmt(edge, "")})
        for idx in sorted(d.get("part", {})):
            for par in TS.MANUAL_PARTITION_PARAMS:
                v = d["part"][idx].get(par)
                if v is not None:
                    rows.append({"channel": ch, "partition": idx,
                                 "parameter": par, "value": _fmt(v, par)})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TS.MANUAL_OVERRIDES_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _fmt(v, par):
    """CSV text of a value: the LD is an index, everything else a number."""
    if par in ("chosen_ld", "line_from", "drop"):
        return str(int(v))
    # The correlation cut runs to the sixth decimal; amplitudes are ~10^4.
    return f"{float(v):.8g}"


# ===========================================================================
# EDITOR
# ===========================================================================

class OverridesEditor:
    """
    Channel list on the left, the overrides of the selected channel on the
    right. Values live in *self.data* until "Salva su file" is pressed; a field
    left empty means "keep the automatic value" and is not written at all.
    """

    # (key, label) of the channel-level fields, in the order they are drawn.
    CHANNEL_FIELDS = (
        ("corr_cut",   "Taglio in correlazione:"),
        ("ly_cut_min", "LY cut min:"),
        ("ly_cut_max", "LY cut max:"),
    )

    # (key, label) of the chain amplitudes, in the order of the canvas columns.
    CHAIN_ROWS = (("rough",      "Calibration rough"),
                  ("heater",     "Heater stabilized"),
                  ("corrected",  "Ampiezza corretta"),
                  ("stabilized", "Stabilizzata sul Tl"))
    # (key, label) of the channel-level fit settings.
    CHAIN_SCALARS = (("peak_nsigma", "Sigma del picco per la finestra:"),
                     ("sig_lo",      "Larghezza min (x attesa):"),
                     ("sig_hi",      "Larghezza max (x attesa):"))

    def __init__(self, channels, folder, parts, path):
        self.channels = channels
        self.folder   = folder
        self.parts    = parts
        self.path     = path
        self.data     = TS.load_manual_overrides(path)
        # What is ON FILE (read-only, to fill the fields) and what the user has
        # EDITED here. Only the latter is written back: chain_settings.csv is
        # hand-tuned, and rewriting rows nobody touched is how those edits get
        # quietly normalised away.
        self.chain_file = read_chain_settings()
        self.chain      = {}
        self.auto_bins = chain_bins_of(folder)
        self.current  = None
        self.part_entries = {}          # idx -> {"heat_cut_min": Entry, ...}
        self.part_vars    = {}          # idx -> {"line_from": Var, "drop": Var}

        self.root = tk.Tk()
        self.root.title(f"Modifiche manuali - {TS.dataset_tag()}")
        self.root.geometry("900x720")

        top = ttk.Label(self.root, wraplength=800, justify="left",
                        text=(f"Dati: {folder}\n"
                              f"File: {path}\n"
                              "Lascia vuoto un campo per tenere il valore automatico."))
        top.pack(fill="x", padx=10, pady=(10, 4))

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=10, pady=4)

        # --- left: channels --------------------------------------------------
        left = ttk.LabelFrame(body, text="Canale")
        left.pack(side="left", fill="y", padx=(0, 10))
        self.listbox = tk.Listbox(left, width=14, exportselection=False)
        self.listbox.pack(side="left", fill="y", padx=4, pady=4)
        sb = ttk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        sb.pack(side="left", fill="y", pady=4)
        self.listbox.config(yscrollcommand=sb.set)
        self.refresh_list()
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        # --- right: two tabs, the cuts and the peak fit ----------------------
        tabs = ttk.Notebook(body)
        tabs.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(tabs); tabs.add(right, text="Tagli")
        fitab = ttk.Frame(tabs); tabs.add(fitab, text="Fit del picco")

        ch_box = ttk.LabelFrame(right, text="Canale (tutte le partizioni)")
        ch_box.pack(fill="x", pady=(0, 8))
        self.entries = {}
        for key, label in self.CHANNEL_FIELDS:
            f = ttk.Frame(ch_box); f.pack(fill="x", padx=6, pady=3)
            ttk.Label(f, text=label, width=24).pack(side="left")
            e = ttk.Entry(f); e.pack(side="right", fill="x", expand=True)
            self.entries[key] = e

        f = ttk.Frame(ch_box); f.pack(fill="x", padx=6, pady=3)
        ttk.Label(f, text="LD per il taglio LY:", width=24).pack(side="left")
        self.ld_var = tk.StringVar(value="auto")
        ttk.Combobox(f, textvariable=self.ld_var, values=("auto", "1", "2"),
                     state="readonly", width=8).pack(side="left")

        # Baseline boundaries, as free text: their number changes with the
        # channel, and a list is quicker to read and to edit than N spin boxes.
        f = ttk.Frame(ch_box); f.pack(fill="x", padx=6, pady=3)
        ttk.Label(f, text="Confini di baseline:", width=24).pack(side="left")
        self.edges_entry = ttk.Entry(f)
        self.edges_entry.pack(side="right", fill="x", expand=True)
        self.edges_entry.bind("<FocusOut>", lambda _e: self.sync_npart())
        self.edges_entry.bind("<Return>",   lambda _e: self.sync_npart())
        ttk.Label(ch_box, foreground="#555555",
                  text="   separati da virgola; se ce n'e' almeno uno, "
                       "SOSTITUISCONO quelli automatici (N confini = N+1 partizioni)"
                  ).pack(anchor="w", padx=6)

        # --- partitions ------------------------------------------------------
        p_head = ttk.Frame(right); p_head.pack(fill="x")
        ttk.Label(p_head, text="Finestra di pulizia per partizione "
                               "(eventi che entrano nella stabilizzazione)"
                  ).pack(side="left")
        ttk.Label(p_head, text="n. partizioni:").pack(side="left", padx=(12, 4))
        self.npart_var = tk.IntVar(value=1)
        ttk.Spinbox(p_head, from_=1, to=12, width=4, textvariable=self.npart_var,
                    command=self.rebuild_partitions).pack(side="left")

        self.part_box = ttk.Frame(right)
        self.part_box.pack(fill="both", expand=True, pady=(4, 0))

        # --- fit tab: chain_settings.csv -------------------------------------
        ttk.Label(fitab, wraplength=560, justify="left", foreground="#555555",
                  text=(f"Fit del picco del tallio, un'ampiezza per riga. "
                        f"Vanno in {os.path.basename(TS.CHAIN_CSV_PATH)}, accanto "
                        f"ai programmi: NON dipendono dal dataset.\n"
                        f"Numero di bin: 0 = automatico (dalla larghezza del bin); "
                        f"un valore > 0 viene usato tale e quale.")
                  ).pack(fill="x", padx=6, pady=(8, 4))

        hdr = ttk.Frame(fitab); hdr.pack(fill="x", padx=6)
        for text, w in (("", 20), ("n. bin", 10), ("finestra (x)", 12),
                        ("largh. attesa (x)", 16)):
            ttk.Label(hdr, text=text, width=w).pack(side="left")

        self.chain_entries = {}
        for key, label in self.CHAIN_ROWS:
            row = ttk.Frame(fitab); row.pack(fill="x", padx=6, pady=2)
            ttk.Label(row, text=label, width=20).pack(side="left")
            self.chain_entries[key] = {}
            for field, w in ((f"bins_{key}", 10), (f"win_scale_{key}", 12),
                             (f"sig_scale_{key}", 16)):
                e = ttk.Entry(row, width=w - 2); e.pack(side="left", padx=(0, 8))
                self.chain_entries[key][field] = e
            self.chain_entries[key]["hint"] = ttk.Label(row, foreground="#777777",
                                                        width=18, text="")
            self.chain_entries[key]["hint"].pack(side="left")

        sc = ttk.LabelFrame(fitab, text="Tutto il canale")
        sc.pack(fill="x", padx=6, pady=(10, 0))
        self.chain_scalars = {}
        for field, label in self.CHAIN_SCALARS:
            f = ttk.Frame(sc); f.pack(fill="x", padx=6, pady=3)
            ttk.Label(f, text=label, width=30).pack(side="left")
            e = ttk.Entry(f, width=12); e.pack(side="left")
            self.chain_scalars[field] = e

        # --- buttons ---------------------------------------------------------
        btn = ttk.Frame(self.root); btn.pack(fill="x", padx=10, pady=10)
        self.b_apply = ttk.Button(btn, text="Applica al canale", command=self.apply_current)
        self.b_apply.pack(side="left")
        self.b_clear = ttk.Button(btn, text="Azzera canale", command=self.clear_current)
        self.b_clear.pack(side="left", padx=6)
        self.b_run = ttk.Button(btn, text="Salva e rilancia", command=self.rerun)
        self.b_run.pack(side="left")
        self.b_stop = ttk.Button(btn, text="Interrompi", command=self.stop_run,
                                 state="disabled")
        self.b_stop.pack(side="left", padx=6)
        self.b_save = ttk.Button(btn, text="Salva su file", command=self.save)
        self.b_save.pack(side="right")
        self.status = ttk.Label(btn, text="")
        self.status.pack(side="right", padx=10)
        self.proc = None

        if self.channels:
            self.listbox.selection_set(0)
            self.on_select()

    # -- channel list ---------------------------------------------------------
    def refresh_list(self):
        """Redraw the channel list, marking the channels that carry overrides."""
        sel = self.listbox.curselection()
        self.listbox.delete(0, "end")
        for ch in self.channels:
            mark = " *" if (self.data.get(ch) or self.chain.get(ch)
                            or self.chain_file.get(ch)) else ""
            self.listbox.insert("end", f"ch {ch}{mark}")
        if sel:
            self.listbox.selection_set(sel[0])

    def selected_channel(self):
        sel = self.listbox.curselection()
        return self.channels[sel[0]] if sel else None

    # -- partitions -----------------------------------------------------------
    def rebuild_partitions(self):
        """Draw one row of fields per partition, keeping what is already typed."""
        typed = {idx: {k: e.get() for k, e in d.items()}
                 for idx, d in self.part_entries.items()}
        typed_extra = {idx: {"line_from": v["line_from"].get(),
                             "drop": v["drop"].get()}
                       for idx, v in getattr(self, "part_vars", {}).items()}
        for w in self.part_box.winfo_children():
            w.destroy()
        self.part_entries = {}
        self.part_vars = {}

        info = {p[0]: p for p in self.parts.get(self.current, [])}
        for idx in range(int(self.npart_var.get())):
            p = info.get(idx)
            if p and p[1] == p[1]:            # not NaN
                head = (f"P{idx}   baseline [{p[1]:.4f}, {p[2]:.4f}]"
                        + (f"   {int(p[3])} eventi" if p[3] == p[3] else ""))
            else:
                head = f"P{idx}"
            box = ttk.LabelFrame(self.part_box, text=head)
            box.pack(fill="x", pady=2)
            row = ttk.Frame(box); row.pack(fill="x", padx=6, pady=3)
            self.part_entries[idx] = {}
            for key, label in (("heat_cut_min", "min:"), ("heat_cut_max", "max:")):
                ttk.Label(row, text=label, width=5).pack(side="left")
                e = ttk.Entry(row, width=12); e.pack(side="left", padx=(0, 12))
                e.insert(0, typed.get(idx, {}).get(key, ""))
                self.part_entries[idx][key] = e
            # Line of ANOTHER partition, and exclusion from the combined results.
            ttk.Label(row, text="retta da:").pack(side="left")
            others = ["-"] + [f"P{k}" for k in range(int(self.npart_var.get()))
                              if k != idx]
            v = tk.StringVar(value=typed_extra.get(idx, {}).get("line_from", "-"))
            ttk.Combobox(row, textvariable=v, values=others, state="readonly",
                         width=5).pack(side="left", padx=(2, 14))
            self.part_vars[idx] = {"line_from": v}
            d = tk.IntVar(value=typed_extra.get(idx, {}).get("drop", 0))
            ttk.Checkbutton(row, text="escludi dai risultati", variable=d
                            ).pack(side="left")
            self.part_vars[idx]["drop"] = d

    # -- edges ----------------------------------------------------------------
    def parse_edges(self):
        """(edges, bad_text) from the boundaries entry: numbers separated by
        commas or spaces. Unparsable items are returned, never guessed."""
        raw = self.edges_entry.get().replace(",", " ").split()
        edges, bad = [], []
        for t in raw:
            try:
                edges.append(float(t))
            except ValueError:
                bad.append(t)
        return sorted(edges), bad

    def sync_npart(self):
        """N boundaries make N+1 partitions: show that many rows of fields."""
        edges, _ = self.parse_edges()
        if edges and int(self.npart_var.get()) != len(edges) + 1:
            self.npart_var.set(len(edges) + 1)
            self.rebuild_partitions()

    # -- load / store ---------------------------------------------------------
    def on_select(self, _evt=None):
        """Commit the fields of the channel being left, then load the new one."""
        ch = self.selected_channel()
        if ch is None or ch == self.current:
            return
        if self.current is not None:
            self.store_current()
        self.current = ch
        d = self.data.get(ch, {})

        for key, _ in self.CHANNEL_FIELDS:
            self.entries[key].delete(0, "end")
            if d.get(key) is not None:
                self.entries[key].insert(0, _fmt(d[key], key))
        self.ld_var.set(str(int(d["chosen_ld"])) if d.get("chosen_ld") is not None
                        else "auto")
        self.edges_entry.delete(0, "end")
        if d.get("edges"):
            self.edges_entry.insert(0, ", ".join(_fmt(e, "") for e in d["edges"]))

        # Hand-set boundaries decide the count; otherwise what the last analysis
        # found, or what the file already carries.
        known = len(self.parts.get(ch, []))
        saved = max(d.get("part", {}), default=-1) + 1
        n_edge = len(d.get("edges", [])) + 1 if d.get("edges") else 0
        self.npart_var.set(max(n_edge, known, saved, 1))
        self.part_entries = {}
        self.part_vars = {}
        self.rebuild_partitions()
        for idx, vals in d.get("part", {}).items():
            if idx not in self.part_entries:
                continue
            for key, e in self.part_entries[idx].items():
                if vals.get(key) is not None:
                    e.delete(0, "end"); e.insert(0, _fmt(vals[key], key))
            if vals.get("line_from") is not None:
                self.part_vars[idx]["line_from"].set(f"P{int(vals['line_from'])}")
            if vals.get("drop"):
                self.part_vars[idx]["drop"].set(1)

        self.load_chain(ch)

    def load_chain(self, ch):
        """Fill the fit tab from chain_settings.csv, with the bin count the last
        analysis actually used shown next to each row."""
        cfg = TS.ChainSettings(self.chain.get(ch) or self.chain_file.get(ch))
        for key, _ in self.CHAIN_ROWS:
            for field, e in self.chain_entries[key].items():
                if field == "hint":
                    continue
                e.delete(0, "end")
                e.insert(0, _fmt(getattr(cfg, field), ""))
            n, win_frac, res_exp = self.auto_bins.get(ch, {}).get(key,
                                                                  (None, None, None))
            if n:
                txt = f"ultimo fit: {n} bin"
            else:
                n = auto_bins(win_frac, res_exp, cfg.bin_div(key), cfg.sig_scale(key))
                txt = f"automatico: {n} bin" if n else ""
            self.chain_entries[key]["hint"].config(text=txt)
        for field, e in self.chain_scalars.items():
            e.delete(0, "end")
            e.insert(0, _fmt(getattr(cfg, field), ""))

    def store_current(self):
        """
        Read the form into self.data for the current channel. Returns the list of
        fields that could not be parsed (left as they are, nothing is stored for
        them) so the caller can tell the user instead of dropping them quietly.
        """
        if self.current is None:
            return []
        bad, d = [], {}
        for key, label in self.CHANNEL_FIELDS:
            raw = self.entries[key].get().strip()
            if not raw:
                continue
            try:
                d[key] = float(raw)
            except ValueError:
                bad.append(label.rstrip(":"))
        if self.ld_var.get() != "auto":
            d["chosen_ld"] = int(self.ld_var.get())
        edges, bad_edges = self.parse_edges()
        bad += [f"confine '{t}'" for t in bad_edges]
        if edges:
            d["edges"] = edges

        part = {}
        for idx, fields in self.part_entries.items():
            vals = {}
            for key, e in fields.items():
                raw = e.get().strip()
                if not raw:
                    continue
                try:
                    vals[key] = float(raw)
                except ValueError:
                    bad.append(f"P{idx} {key}")
            src = self.part_vars[idx]["line_from"].get()
            if src.startswith("P"):
                vals["line_from"] = int(src[1:])
            if self.part_vars[idx]["drop"].get():
                vals["drop"] = 1
            if vals:
                part[idx] = vals
        if part:
            d["part"] = part

        # Fit settings: their own file, and always written in full (they are a
        # complete description of how the channel is fitted, not a sparse
        # override), so a row is only kept when it differs from the defaults.
        cfg_vals, defaults = {}, TS.ChainSettings()
        for key, _ in self.CHAIN_ROWS:
            for field, e in self.chain_entries[key].items():
                if field == "hint":
                    continue
                raw = e.get().strip()
                if not raw:
                    continue
                try:
                    cfg_vals[field] = float(raw)
                except ValueError:
                    bad.append(f"{key} {field.rsplit('_', 1)[0]}")
        for field, e in self.chain_scalars.items():
            raw = e.get().strip()
            if not raw:
                continue
            try:
                cfg_vals[field] = float(raw)
            except ValueError:
                bad.append(field)
        # Kept only when it says something the file does not already say.
        on_file = self.chain_file.get(self.current, {})
        if any(v != on_file.get(f, getattr(defaults, f))
               for f, v in cfg_vals.items()):
            self.chain[self.current] = cfg_vals
        else:
            self.chain.pop(self.current, None)

        if d:
            self.data[self.current] = d
        else:
            self.data.pop(self.current, None)
        return bad

    # -- buttons --------------------------------------------------------------
    def apply_current(self):
        bad = self.store_current()
        self.refresh_list()
        if bad:
            messagebox.showwarning("Valori non validi",
                                   "Non sono numeri e non sono stati salvati:\n  "
                                   + "\n  ".join(bad))
            return
        n = len(self.data.get(self.current, {})) if self.current else 0
        self.status.config(text=f"ch {self.current}: {n} voce/i in memoria")

    def clear_current(self):
        """Back to fully automatic for this channel."""
        if self.current is None:
            return
        self.data.pop(self.current, None)
        for e in self.entries.values():
            e.delete(0, "end")
        self.ld_var.set("auto")
        self.edges_entry.delete(0, "end")
        for fields in self.part_entries.values():
            for e in fields.values():
                e.delete(0, "end")
        for v in self.part_vars.values():
            v["line_from"].set("-"); v["drop"].set(0)
        # Back to the program defaults for the fit too, and WRITTEN as such on
        # save: showing defaults while the file keeps the old row would be a lie.
        self.chain_file.pop(self.current, None)
        self.chain[self.current] = {f: getattr(TS.ChainSettings(), f)
                                    for f in TS.ChainSettings.FIELDS}
        self.load_chain(self.current)
        self.refresh_list()
        self.status.config(text=f"ch {self.current}: automatico")

    def save(self):
        bad = self.store_current()
        if bad:
            messagebox.showwarning("Valori non validi",
                                   "Non sono numeri e non verranno scritti:\n  "
                                   + "\n  ".join(bad))
        try:
            n = write_overrides(self.path, self.data)
            n_fit = write_chain_settings(self.chain)
            self.chain_file.update(self.chain)
        except OSError as e:
            messagebox.showerror("Errore di scrittura", str(e))
            return
        self.refresh_list()
        self.status.config(text=f"salvate {n} riga/he + {n_fit} canale/i di fit")
        print(f">>> {n} row(s) written to {self.path}")
        print(f">>> {n_fit} channel(s) written to "
              f"{os.path.basename(TS.CHAIN_CSV_PATH)}")

    # -- re-run the analysis on this channel ----------------------------------
    def rerun(self):
        """
        Save, then re-run ThalliumStabilization.py on the selected channel in a
        SEPARATE process: exactly the command one would type, so the analysis
        cannot behave differently here, and ROOT never has to share a process
        with Tk. Its output goes to the terminal this editor was started from.
        """
        if self.proc is not None or self.current is None:
            return
        self.save()
        prog = os.path.abspath(TS.__file__)
        if prog.endswith(".pyc"):
            prog = prog[:-1]
        try:
            self.proc = subprocess.Popen([sys.executable, os.path.basename(prog),
                                          str(self.current)],
                                         cwd=os.path.dirname(prog))
        except OSError as e:
            messagebox.showerror("Impossibile lanciare l'analisi", str(e))
            self.proc = None
            return
        print(f">>> Re-running ch {self.current} "
              f"({os.path.basename(prog)}, pid {self.proc.pid})")
        self._set_busy(True, f"ch {self.current}: analisi in corso...")
        self.root.after(300, self.poll_run)

    def poll_run(self):
        """Wait for the analysis to end, then refresh the partitions it rewrote."""
        if self.proc is None:
            return
        rc = self.proc.poll()
        if rc is None:
            self.root.after(300, self.poll_run)
            return
        self.proc = None
        self._set_busy(False, "")
        self.parts = partitions_of(self.folder)
        self.auto_bins = chain_bins_of(self.folder)
        self.load_chain(self.current)
        n_before = int(self.npart_var.get())
        known = len(self.parts.get(self.current, []))
        if known:
            self.npart_var.set(known)
        self.rebuild_partitions()
        msg = (f"ch {self.current}: fatto" if rc == 0 else
               f"ch {self.current}: uscito con {rc}, vedi il terminale")
        if known and known != n_before:
            msg += f" ({known} partizioni)"
        self.status.config(text=msg)

    def stop_run(self):
        if self.proc is not None:
            self.proc.terminate()
            self.status.config(text="interrotto")

    def _set_busy(self, busy, msg):
        state = "disabled" if busy else "normal"
        widgets = [self.b_apply, self.b_clear, self.b_run, self.b_save,
                   self.listbox, self.edges_entry, *self.chain_scalars.values()]
        widgets += [e for d in self.chain_entries.values()
                    for f, e in d.items() if f != "hint"]
        for w in widgets:
            w.config(state=state)
        self.b_stop.config(state="normal" if busy else "disabled")
        self.status.config(text=msg)
        self.root.update_idletasks()

    def run(self):
        self.root.mainloop()


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    channels, folder = channels_available()
    path = TS.manual_overrides_path()
    print(f">>> Dataset: {TS.dataset_tag()}  (folder: {folder})")
    print(f">>> Overrides file: {path}")
    if not channels:
        print(f"[!] No .root file in {folder}: nothing to edit.\n"
              f"    Check ANALYSIS_MODE / BASE_DIR / RUN_NUMBER in "
              f"ThalliumStabilization.py.", file=sys.stderr)
        sys.exit(1)
    parts = partitions_of(folder)
    if not parts:
        print("    (no per-partition results table yet: the partitions of each "
              "channel are unknown, set their number by hand)")
    OverridesEditor(channels, folder, parts, path).run()


if __name__ == "__main__":
    main()
