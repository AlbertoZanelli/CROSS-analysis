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
    parser.add_argument("-o", "--output", default="configModuleANPS.toml", help="Base output filename")

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

    variableComb = table()
    variableComb_chan = table()
    variableComb_chan.add("Operation", ["maxminusbaseline.windowamp - maxminusbaseline.amplitude "])
    variableComb_chan.add("NameVar", ["relAmp"])
    variableComb_chan.add("Draw", False)
    variableComb.add("light", variableComb_chan)
    module.add("variablecombination", variableComb)



    # ---- Cuts helper ----
    def create_cut(select, cutvar=None, threshold=None, input_max=None,
                type_name=None, cutmode=None, select_above=None):
        t = table()
        t.add("Select", select)

        if select_above:
            t.add("SelectAbove", select_above)
        if cutvar:
            t.add("CutVariable", cutvar)
        if threshold is not None:
            t.add("Threshold", threshold)
        if input_max:
            t.add("InputMaxminusbaselineModule", input_max)
        if type_name:
            t.add("Type", type_name)
        if cutmode:
            t.add("CutMode", cutmode)

        t.add("Draw", False)
        return t

    # ---- module.cuts.light ----
    cuts_table = table()
    cuts_light = table()


    sel_noise_relAmp = array()
    sel_noise_relAmp.append(["module","module","isnoise", True])
    sel_noise_relAmp.append(["module","numberoftriggers","numberoftriggers", 1])
    sel_noise_relAmp.append(["module","badinterval","badinterval", False])
    cuts_light.add("noise_relamp", create_cut(sel_noise_relAmp, cutvar=["variablecombination","relAmp"]))


    sel_noise_slope = array()
    sel_noise_slope.append(["module","cuts_noise_relamp","pass", True])
    cuts_light.add("noise_slope", create_cut(sel_noise_slope, cutvar=["baselineslope","slope"]))

    sel_noise_rms = array()
    sel_noise_rms.append(["module","cuts_noise_slope","pass", True])
    cuts_light.add(
        "noise_rms",
        create_cut(
            sel_noise_rms,
            cutvar=["baseline", "RMS"],
            threshold=2.2
        )
    )

    sel_noise_amp = array()
    sel_noise_amp.append(["module","cuts_noise_rms","pass", True])
    cuts_light.add(
        "noise_amplitude",
        create_cut(
            sel_noise_amp,
            cutvar=["maxminusbaseline", "amplitude"],
            threshold=1.5
        )
    )

    cuts_table.add("light", cuts_light)
    module.add("cuts", cuts_table)


    # ---- module.triggerdelaycorrection.light.noise ----
    tdc_table = table()
    tdc_light = table()
    sel_tdc = array()
    sel_tdc.append(["module","cuts_noise_amplitude","pass", True])
    tdc_light.add("OutputWaveform", "triggerdelaycorrected")
    tdc_light.add("Select", sel_tdc)
    tdc_table.add("noise", tdc_light)
    tdc_parent = table()
    tdc_parent.add("light", tdc_table)
    module.add("triggerdelaycorrection", tdc_parent)

    # ---- module.baselinesubtraction.light.noise ----
    bs_table = table()
    bs_light = table()
    bs_light_noise = table()
    sel_bs_noise = array()
    sel_bs_noise.append(["module","cuts_noise_amplitude","pass", True])
    bs_light_noise.add("InputWaveform", "triggerdelaycorrected")
    bs_light_noise.add("OutputWaveform", "baselinesubtracted")
    bs_light_noise.add("Select", sel_bs_noise)
    bs_light_noise.add("RunOnlyOnGoodEvents", True)
    bs_light_noise.add("Draw", False)
    bs_light.add("noise", bs_light_noise)
    bs_table.add("light", bs_light)
    module.add("baselinesubtraction", bs_table)

    # ---- module.fouriertransform.light.noise ----
    ft_table = table()
    ft_light = table()
    ft_light_noise = table()
    sel_ft_noise = array()
    sel_ft_noise.append(["module","cuts_noise_amplitude","pass", True])
    ft_light_noise.add("InputWaveform", "baselinesubtracted")
    ft_light_noise.add("OutputWaveform", "fft")
    ft_light_noise.add("Windowing", "FlatTop")
    ft_light_noise.add("Select", sel_ft_noise)
    ft_light_noise.add("RunOnlyOnGoodEvents", True)
    ft_light_noise.add("Draw", False)
    ft_light.add("noise", ft_light_noise)
    ft_table.add("light", ft_light)
    module.add("fouriertransform", ft_table)

        # ---- module.averagepowerspectrum.light.anps ----
    aps_table = table()
    aps_light = table()
    aps_anps = table()

    sel_aps = array()
    sel_aps.append(["module", "cuts_noise_amplitude", "pass", True])

    aps_anps.add("InputWaveform", "fft")
    aps_anps.add("PowerPlotMaxValue", [0.0001])
    aps_anps.add("Select", sel_aps)
    aps_anps.add("RunOnlyOnGoodEvents", True)
    aps_anps.add("Gain", args.gain)
    aps_anps.add("Normalize", False)
    aps_anps.add("Draw", False)

    aps_light.add("anps", aps_anps)
    aps_table.add("light", aps_light)
    module.add("averagepowerspectrum", aps_table)


    # ---- Add module to doc ----
    doc.add("module", module)

    # ---- Write final TOML ----
    with open(final_output_path, "w") as f:
        f.write(dumps(doc))

    print(f"TOML saved to: {final_output_path}")


if __name__ == "__main__":
    main()
