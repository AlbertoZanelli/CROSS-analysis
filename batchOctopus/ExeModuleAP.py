# Created by: Roberto Serino
# Date: 2025-11-25
# Description: Automatized module for working point selection with fully nested tables

import argparse
import os
from tomlkit import document, table, array, dumps

def main():
    parser = argparse.ArgumentParser(description="Generate full TOML for reconstruction modules")

    # ---- Directories ----
    parser.add_argument("--rawdir", default="/Users/serino/Octopus/DATA/CROSS/")
    parser.add_argument("--triggerdir", default="/Users/serino/Octopus/Results/Triggered/CROSS/")
    parser.add_argument("--processeddir", default="/Users/serino/Octopus/Results/Processed/CROSS/")
    parser.add_argument("--BadIntervalPath", default="")

    # ---- RunConfig ----
    parser.add_argument("--prefix", nargs="+", default=["20251116T120036"])
    parser.add_argument("--run", nargs="+", default=["000048"])

    # ---- Settings ----
    parser.add_argument("--rawType", default="Cupid")
    parser.add_argument("--runType", default="reconstruct")
    parser.add_argument("--draw", action="store_true")
    parser.add_argument("--verbosity", type=int, default=5)
    parser.add_argument("--usemergedtriggers", action="store_true")
    parser.add_argument("--useflaggedtriggers", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    # ---- Channels ----
    parser.add_argument("--chan", nargs="+", type=int, default=[91])
    parser.add_argument("--waveformtype", default="continuous")
    parser.add_argument("--gain", type=int, default=2060, help="Amplifier gain")

    # ---- Module parameters ----
    parser.add_argument("--windowlength", nargs="+", type=float, default=[0.2])
    parser.add_argument("--pretrigger", nargs="+", type=float, default=[0.1])
    parser.add_argument("--risestart", nargs="+", type=float, default=[10.0])
    parser.add_argument("--risestop", nargs="+", type=float, default=[90.0])
    parser.add_argument("--decaystart", nargs="+", type=float, default=[90.0])
    parser.add_argument("--decaystop", nargs="+", type=float, default=[30.0])
    parser.add_argument("--namePulser", default="IsLED")

    # ---- Output handling ----
    parser.add_argument("--outdirCFG", default=".", help="Directory where the output TOML will be saved")
    parser.add_argument("-o", "--output", default="configModuleAP.toml", help="Base output filename")

    args = parser.parse_args()

    # ---- Ensure output directory exists ----
    os.makedirs(args.outdirCFG, exist_ok=True)

    # ---- Build final filename including channel(s) ----
    channel_str = "_".join(str(ch) for ch in args.chan)
    base_name, ext = os.path.splitext(args.output)
    final_output_path = os.path.join(args.outdirCFG, f"{base_name}_{channel_str}{ext}")

    # ---- TOML Document ----
    doc = document()

    # ===== Directories =====
    dirs = table()
    dirs.add("rawdir", args.rawdir)
    dirs.add("triggerdir", args.triggerdir)
    dirs.add("processeddir", args.processeddir)
    doc.add("directories", dirs)

    # ===== RunConfig =====
    rc = table()
    rc.add("filenamePrefix", args.prefix)
    rc.add("runNumber", args.run)
    doc.add("runConfig", rc)

    # ===== Settings =====
    settings = table()
    settings.add("rawType", args.rawType)
    settings.add("runType", args.runType)
    settings.add("draw", args.draw)
    settings.add("verbosity", args.verbosity)
    settings.add("overwrite", args.overwrite)
    settings.add("usemergedtriggers", args.usemergedtriggers)
    settings.add("useflaggedtriggers", args.useflaggedtriggers)
    doc.add("settings", settings)

    # ===== Channels =====
    channels = table()
    chan_table = table()
    chan_table.add("list", args.chan)
    chan_table.add("waveformtype", args.waveformtype)
    channels.add("light", chan_table)
    doc.add("channels", channels)

    # ===== Module =====
    module = table()

    # ---- module.module.light ----
    module_module = table()
    module_chan = table()
    module_chan.add("windowlength", args.windowlength)
    module_chan.add("pretrigger", args.pretrigger)
    module_chan.add("Draw", False)
    module_module.add("light", module_chan)
    module.add("module", module_module)

    # ---- module.timestamp ----
    ts = table()
    ts.add("Draw", False)
    module.add("timestamp", ts)

    # ---- module.badinterval ----
    BI = table()
    BI.add("TxtBadInterval", args.BadIntervalPath)
    BI.add("Draw", False)
    module.add("badinterval", BI)

    # ---- module.numberoftriggers.light ----
    notrig = table()
    notrig_chan = table()
    notrig_chan.add("Draw", False)
    notrig.add("light", notrig_chan)
    module.add("numberoftriggers", notrig)

    # ---- module.baseline.light ----
    baseline = table()
    baseline_chan = table()
    baseline_chan.add("Draw", False)
    baseline.add("light", baseline_chan)
    module.add("baseline", baseline)

    # ---- module.baselineslope.light ----
    bslope = table()
    bslope_chan = table()
    bslope_chan.add("Draw", False)
    bslope.add("light", bslope_chan)
    module.add("baselineslope", bslope)

    # ---- module.maxminusbaseline.light ----
    maxmb = table()
    maxmb_chan = table()
    maxmb_chan.add("Draw", False)
    maxmb.add("light", maxmb_chan)
    module.add("maxminusbaseline", maxmb)

    # ---- module.triggerdelay.light ----
    tdelay = table()
    tdelay_chan = table()
    tdelay_chan.add("Draw", False)
    tdelay.add("light", tdelay_chan)
    module.add("triggerdelay", tdelay)

    # ---- module.risetime.light ----
    risetime = table()
    risetime_chan = table()
    risetime_chan.add("StartPercentage", args.risestart)
    risetime_chan.add("StopPercentage", args.risestop)
    risetime_chan.add("Interpolation", True)
    risetime_chan.add("Draw", False)
    risetime.add("light", risetime_chan)
    module.add("risetime", risetime)

    # ---- module.decaytime.light ----
    decaytime = table()
    decaytime_chan = table()
    decaytime_chan.add("StartPercentage", args.decaystart)
    decaytime_chan.add("StopPercentage", args.decaystop)
    decaytime_chan.add("Interpolation", True)
    decaytime_chan.add("Draw", False)
    decaytime.add("light", decaytime_chan)
    module.add("decaytime", decaytime)

    # ---- Cuts helper ----
    def create_cut(select, cutvar=None, input_max=None, type_name=None,
               cutmode=None, select_above=None, select_inside=None):
        t = table()
        t.add("Select", select)
        if select_above:
            t.add("SelectAbove", select_above)
        if select_inside:
            t.add("SelectInside", select_inside)
        if cutvar:
            t.add("CutVariable", cutvar)
        if input_max:
            t.add("InputMaxminusbaselineModule", input_max)
        if type_name:
            t.add("Type", type_name)
        if cutmode:
            t.add("CutMode", cutmode)
        t.add("Draw", False)
        return t


    # ---- module.cuts_ap.light ----
    cuts_ap_table = table()
    cuts_ap_light = table()

    sel_signal = array()
    sel_signal.append(["module", "module", "issignal", True])
    sel_signal.append(["module", "numberoftriggers", "numberoftriggers", 1])
    sel_signal.append(["module", "badinterval", "badinterval", False])

    sel_inside_ap = array()
    sel_inside_ap.append(["module", "risetime", "risetime", [0.01], [0.02]])
    sel_inside_ap.append(["module", "maxminusbaseline", "amplitude", [0.01], [0.02]])

    cuts_ap_light.add(
        "signal",
        create_cut(
            sel_signal,
            select_inside=sel_inside_ap,
            input_max="maxminusbaseline",
            type_name="light",
            cutmode="SNratio"
        )
    )

    cuts_ap_table.add("light", cuts_ap_light)
    module.add("cuts_ap", cuts_ap_table)



    # ---- module.cuts_ap.light.slope ----
    cuts_table = table()
    cuts_light = table()

    sel_signal_slope = array()
    sel_signal_slope.append(["module","cuts_ap_signal","pass", True])
    cuts_light.add("signal_slope", create_cut(sel_signal_slope, cutvar=["baselineslope","slope"]))

    cuts_table.add("light", cuts_light)
    module.add("cuts", cuts_table)

    # ---- module.crosscorr.light ----
    crosscorr_table = table()
    crosscorr_light = table()
    sel_crosscorr = array()
    sel_crosscorr.append(["module","cuts_signal_slope","pass", True])
    crosscorr_light.add("Select", sel_crosscorr)
    crosscorr_light.add("CorrFraction", 0.35)
    # crosscorr_light.add("OnlineCutBottom", 0.85)
    crosscorr_light.add("CorrCut", 0.88)
    crosscorr_light.add("Draw", False)
    crosscorr_table.add("light", crosscorr_light)
    module.add("crosscorr", crosscorr_table)

   
    ################################
    # AP  — AVERAGE PULSE BLOCK
    ################################

    # ---- module.triggerdelaycorrection.light.ap ----
    tdc_ap_table = table()
    tdc_ap_light = table()
    tdc_ap = table()

    sel_ap = array()
    sel_ap.append(["module", "crosscorr", "pass", True])

    sel_ap_above = array()
    sel_ap_above.append(["module", "crosscorr", "avgcorr", [0.85]])

    tdc_ap.add("Select", sel_ap)
    tdc_ap.add("SelectAbove", sel_ap_above)
    tdc_ap.add("OutputWaveform", "triggerdelaycorrectedAP")
    tdc_ap.add("Draw", False)

    tdc_ap_light.add("ap", tdc_ap)
    tdc_ap_table.add("light", tdc_ap_light)
    module.add("triggerdelaycorrection", tdc_ap_table)

    # ---- module.baselinesubtraction.light.ap ----
    bs_ap_table = table()
    bs_ap_light = table()
    bs_ap = table()

    bs_ap.add("InputWaveform", "triggerdelaycorrectedAP")
    bs_ap.add("OutputWaveform", "baselinesubtracted")
    bs_ap.add("Select", sel_ap)
    bs_ap.add("RunOnlyOnGoodEvents", True)
    bs_ap.add("Draw", False)

    bs_ap_light.add("ap", bs_ap)
    bs_ap_table.add("light", bs_ap_light)
    module.add("baselinesubtraction", bs_ap_table)

    # ---- module.averagepulse.light.ap ----
    avgpulse_table = table()
    avgpulse_light = table()
    avgpulse = table()

    sel_ap_above_2 = array()
    sel_ap_above_2.append(["module", "crosscorr", "avgcorr", [0.85]])

    avgpulse.add("InputWaveform", "baselinesubtracted")
    avgpulse.add("SelectAbove", sel_ap_above_2)
    avgpulse.add("Normalize", True)
    avgpulse.add("NormalizeAverage", True)
    avgpulse.add("Select", sel_ap)
    avgpulse.add("PulseMin", [-0.005])
    avgpulse.add("PulseMax", [1.1])
    avgpulse.add("RunOnlyOnGoodEvents", True)
    avgpulse.add("Windowing", "FlatTop")
    avgpulse.add("Draw", False)

    avgpulse_light.add("ap", avgpulse)
    avgpulse_table.add("light", avgpulse_light)
    module.add("averagepulse", avgpulse_table)

    # ---- Add module to doc ----
    doc.add("module", module)

    # ---- Write final TOML ----
    with open(final_output_path, "w") as f:
        f.write(dumps(doc))

    print(f"TOML saved to: {final_output_path}")


if __name__ == "__main__":
    main()
