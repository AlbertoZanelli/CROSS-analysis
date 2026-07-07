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
    parser.add_argument("--gain", 2060)

    # ---- Module parameters ----
    parser.add_argument("--windowlength", nargs="+", type=float, default=[0.2])
    parser.add_argument("--pretrigger", nargs="+", type=float, default=[0.1])
    parser.add_argument("--risestart", nargs="+", type=float, default=[10.0])
    parser.add_argument("--risestop", nargs="+", type=float, default=[90.0])
    parser.add_argument("--decaystart", nargs="+", type=float, default=[90.0])
    parser.add_argument("--decaystop", nargs="+", type=float, default=[30.0])

    # ---- Output handling ----
    parser.add_argument("--outdirCFG", default=".", help="Directory where the output TOML will be saved")
    parser.add_argument("-o", "--output", default="configModule.toml", help="Base output filename")

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
               cutmode=None, select_above=None):
        t = table()
        t.add("Select", select)
        if select_above:
            t.add("SelectAbove", select_above)
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

    # ---- module.cuts.light ----
    cuts_table = table()
    cuts_light = table()

    sel_noise_slope = array()
    sel_noise_slope.append(["module","module","isnoise", True])
    sel_noise_slope.append(["module","numberoftriggers","numberoftriggers", 1])
    sel_noise_slope.append(["module","badinterval","badinterval", False])
    cuts_light.add("noise_slope", create_cut(sel_noise_slope, cutvar=["baselineslope","slope"]))

    sel_noise_rms = array()
    sel_noise_rms.append(["module","cuts_noise_slope","pass", True])
    cuts_light.add("noise_rms", create_cut(sel_noise_rms, cutvar=["baseline","RMS"]))

    sel_noise_amp = array()
    sel_noise_amp.append(["module","cuts_noise_rms","pass", True])
    cuts_light.add("noise_amplitude", create_cut(sel_noise_amp, cutvar=["maxminusbaseline","amplitude"]))

    cuts_table.add("light", cuts_light)
    module.add("cuts", cuts_table)

    # ---- module.cuts_ap.light ----
    cuts_ap_table = table()
    cuts_ap_light = table()

    sel_signal = array()
    sel_signal.append(["module", "module", "issignal", True])
    sel_signal.append(["module", "numberoftriggers", "numberoftriggers", 1])
    sel_signal.append(["module", "badinterval", "badinterval", False])

    sel_above_ap = array()
    sel_above_ap.append(["module", "decaytime", "decaytime", [0.008]])

    cuts_ap_light.add(
        "signal",
        create_cut(
            sel_signal,
            select_above=sel_above_ap,
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

    ############################################
    # Optimum Filter + Power + Gaussian Filter
    ############################################

    # ---- module.triggerdelaycorrection.light.all ----
    tdc_all_table = table()
    tdc_all_light = table()
    tdc_all = table()

    tdc_all.add("OutputWaveform", "triggerdelaycorrected")
    tdc_all.add("RunOnlyOnGoodEvents", True)
    tdc_all.add("Draw", False)

    tdc_all_light.add("all", tdc_all)
    tdc_all_table.add("light", tdc_all_light)
    module.add("triggerdelaycorrection", tdc_all_table)

    # ---- module.baselinesubtraction.light.all ----
    bs_all_table = table()
    bs_all_light = table()
    bs_all = table()

    bs_all.add("InputWaveform", "triggerdelaycorrected")
    bs_all.add("OutputWaveform", "baselinesubtracted")
    bs_all.add("RunOnlyOnGoodEvents", True)
    bs_all.add("Draw", False)

    bs_all_light.add("all", bs_all)
    bs_all_table.add("light", bs_all_light)
    module.add("baselinesubtraction", bs_all_table)

    # ---- module.fouriertransform.light.all ----
    ft_all_table = table()
    ft_all_light = table()
    ft_all = table()

    ft_all.add("InputWaveform", "baselinesubtracted")
    ft_all.add("OutputWaveform", "fft")
    ft_all.add("Windowing", "FlatTop")
    ft_all.add("CorrectDelay", False)
    ft_all.add("RunOnlyOnGoodEvents", True)
    ft_all.add("Draw", False)
    ft_all.add("DrawOption", "ReIm")

    ft_all_light.add("all", ft_all)
    ft_all_table.add("light", ft_all_light)
    module.add("fouriertransform", ft_all_table)

    # ---- module.optimumfilter.light.all ----
    of_all_table = table()
    of_all_light = table()
    of_all = table()

    of_all.add("InputWaveform", "fft")
    of_all.add("OutputWaveform", "OFFiltered")
    of_all.add("APModuleName", "averagepulse_ap")
    of_all.add("ANPSModuleName", "averagepowerspectrum_anps")
    of_all.add("AP", "averagepulse_ap_trimmedAP")
    of_all.add("ANPS", "averagepowerspectrum_anps_medianpower")
    of_all.add("Windowing", "FlatTop")
    of_all.add("RunOnlyOnGoodEvents", True)
    of_all.add("Draw", False)
    of_all.add("DrawOption", "Chi2")

    of_all_light.add("all", of_all)
    of_all_table.add("light", of_all_light)
    module.add("optimumfilter", of_all_table)

    ############################################
    # CORRELATION PARAMETER FOR PSD
    ############################################

    # ---- module.inversefouriertransform.light.corr ----
    ifft_corr_table = table()
    ifft_corr_light = table()
    ifft_corr = table()

    ifft_corr.add("InputWaveform", "OFFiltered")
    ifft_corr.add("OutputWaveform", "inversefft")
    ifft_corr.add("RunOnlyOnGoodEvents", True)
    ifft_corr.add("Draw", False)

    ifft_corr_light.add("corr", ifft_corr)
    ifft_corr_table.add("light", ifft_corr_light)
    module.add("inversefouriertransform", ifft_corr_table)

    # ---- module.correlation.light.corr ----
    corr_table = table()
    corr_light = table()
    corr = table()

    corr.add("InputWaveform", "inversefft")
    corr.add("AP", "FilteredAP")
    corr.add("APModuleName", "optimumfilter_all")
    corr.add("HalfWindow", 0.2)
    corr.add("RunOnlyOnGoodEvents", True)
    corr.add("Draw", False)

    corr_light.add("corr", corr)
    corr_table.add("light", corr_light)
    module.add("correlation", corr_table)

    # ---- module.calibration.light.all ----
    cal_all_table = table()
    cal_all_light = table()
    cal_all = table()

    cal_all.add("InputCalibrationModule", "optimumfilter_all")
    cal_all.add("Formula", ["[0]*x"])
    cal_all.add("RunOnlyOnGoodEvents", False)
    cal_all.add("Coefficients", [[1061.6341252728]])
    cal_all.add("Draw", False)

    cal_all_light.add("all", cal_all)
    cal_all_table.add("light", cal_all_light)
    module.add("calibration", cal_all_table)


    # ---- Add module to doc ----
    doc.add("module", module)

    # ---- Write final TOML ----
    with open(final_output_path, "w") as f:
        f.write(dumps(doc))

    print(f"TOML saved to: {final_output_path}")


if __name__ == "__main__":
    main()
