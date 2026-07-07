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
    parser.add_argument("--typeRun", default="C")

    parser.add_argument("--heaterless", nargs="+", type=int, required=False,
                    help="List of heaterless channels")

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
    parser.add_argument("--type", default="")


    # ---- Module parameters ----
    parser.add_argument("--windowlength", nargs="+", type=float, default=[0.2])
    parser.add_argument("--pretrigger", nargs="+", type=float, default=[0.1])
    parser.add_argument("--risestart", nargs="+", type=float, default=[10.0])
    parser.add_argument("--risestop", nargs="+", type=float, default=[90.0])
    parser.add_argument("--decaystart", nargs="+", type=float, default=[90.0])
    parser.add_argument("--decaystop", nargs="+", type=float, default=[30.0])
    parser.add_argument("--namePulser", default="IsLED")

    # ---- OF templates ----
    parser.add_argument("--templateAP", default="")
    parser.add_argument("--templateANPS", default="")
    parser.add_argument("--loadstab", default="")
    parser.add_argument("--sensitivityTXT", default="")

    



    # ---- Output handling ----
    parser.add_argument("--outdirCFG", default=".", help="Directory where the output TOML will be saved")
    parser.add_argument("-o", "--output", default="configModuleOF.toml", help="Base output filename")

    args = parser.parse_args()

    if args.templateAP:
        args.templateAP = f"{args.templateAP}{args.chan[0]}.root"

    if args.templateANPS:
        args.templateANPS = f"{args.templateANPS}{args.chan[0]}.root"

    if args.loadstab:
        args.loadstab = f"{args.loadstab}{args.chan[0]}.root"

    ## Set the path for the muon veto. The trigger has to be the same run 
    ## so we will use the triggerdir of the current run
    MuonPath = f"{args.triggerdir}/Trigger_{args.prefix[0]}_{args.run[0]}_65552.root"
    PulserPath = f"{args.triggerdir}/Trigger_{args.prefix[0]}_{args.run[0]}_1.root"
    pulser_lab = "heater"

    channels_heaterless = args.heaterless   # None or a list of ints


    ArrayEn = [2600.0, 2630.0]
    StringType = "ThManualPeak"
    StabValue = 2614.511

    if args.type=="light":
        ArrayEn = [0.05, 0.054]
        StringType = "MoManualPeak"
        PulserPath = f"{args.triggerdir}/Trigger_{args.prefix[0]}_{args.run[0]}_2.root"
        pulser_lab = "led"
        

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
    if args.type=="light":
        maxmb_chan.add("Window", [[0.005, 0.002]])
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


    if args.type=="light":
        # ---- The muon veto will use the light event time information (Fastest pulse)
        
        muon_table = table()
        muon_light = table()
        muon = table()

        muon.add("FlagPropagatorPath", MuonPath)
        muon.add("Type", "muon")
        muon.add("Draw", False)
        muon_light.add("muon", muon)
        muon_table.add("light", muon_light)
        module.add("flagpropagator", muon_table)


    pulser_table = table()
    pulser_light = table()
    pulser = table()

    pulser.add("FlagPropagatorPath", PulserPath)
    pulser.add("Draw", False)

    pulser_light.add(pulser_lab, pulser)
    pulser_table.add("light", pulser_light)
    module.add("flagpropagator", pulser_table)


    # ---- module.numberoftriggers.light ----
    notrig = table()
    notrig_chan = table()
    notrig_chan.add("Draw", False)
    notrig.add("light", notrig_chan)
    module.add("numberoftriggers", notrig)


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

    # ---- module.triggerdelaycorrection.light ----
    tdc_ap_table = table()
    tdc_ap_light = table()
    tdc_ap = table()

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
    bs_ap.add("Draw", False)

    bs_ap_light.add("ap", bs_ap)
    bs_ap_table.add("light", bs_ap_light)
    module.add("baselinesubtraction", bs_ap_table)

    # ---- module.fouriertransform.light.noise ----
    ft_table = table()
    ft_light = table()
    ft_light_noise = table()
    ft_light_noise.add("InputWaveform", "baselinesubtracted")
    ft_light_noise.add("OutputWaveform", "fft")
    ft_light_noise.add("Windowing", "FlatTop")
    ft_light_noise.add("Draw", False)

    ft_light.add("noise", ft_light_noise)
    ft_table.add("light", ft_light)
    module.add("fouriertransform", ft_table)

    # ---- module.optimumfilter.light.all ----
    of_all_table = table()
    of_all_light = table()
    of_all = table()

    of_all.add("InputWaveform", "fft")
    of_all.add("OutputWaveform", "OFFiltered")
    of_all.add("LoadAP", args.templateAP)
    of_all.add("LoadANPS", args.templateANPS)
    of_all.add("AP", "averagepulse_ap_medianAP")
    of_all.add("ANPS", "averagepowerspectrum_anps_medianpower")
     
    if args.type=="light":
        of_all.add("SkipMinimization", True)

    of_all.add("Windowing", "FlatTop")
    of_all.add("Draw", False)
    of_all.add("DrawOption", "Chi2")

    of_all_light.add("all", of_all)
    of_all_table.add("light", of_all_light)
    module.add("optimumfilter", of_all_table)

    # ---- module.inversefouriertransform.light.corr ----
    ifft_corr_table = table()
    ifft_corr_light = table()
    ifft_corr = table()

    ifft_corr.add("InputWaveform", "OFFiltered")
    ifft_corr.add("OutputWaveform", "inversefft")
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
    corr.add("Draw", False)

    corr_light.add("corr", corr)
    corr_table.add("light", corr_light)
    module.add("correlation", corr_table)

    # ---- module.chi2.light.corr ----
    chi2_table = table()
    chi2_light = table()
    chi2 = table()

    chi2.add("InputWaveform", "fft")
    chi2.add("AP", "AP_ReIm")
    chi2.add("APModuleName", "optimumfilter_all")
    chi2.add("InputAmplitudeModule", "optimumfilter_all")
    chi2.add("Domain", "frequency")
    chi2.add("Draw", False)

    chi2_light.add("chi2", chi2)
    chi2_table.add("light", chi2_light)
    module.add("chi2", chi2_table)





    if args.type!="light":


        cal_all_table = table()
        cal_all_light = table()
        cal_all = table()

        cal_all.add("LoadCalTxt", args.sensitivityTXT)
        cal_all.add("InputCalibrationModule", "optimumfilter_all")
        cal_all.add("Draw", False)


        cal_all_light.add("rough", cal_all)
        cal_all_table.add("light", cal_all_light)
        module.add("calibration", cal_all_table)



        

        # ---- module.stabilization.light.all ----
        stab_table = table()
        stab_light = table()
        stab = table()

        if args.chan[0] not in channels_heaterless:
            sel_stab = array()

            if args.typeRun=="B":
                sel_stab.append(["module","flagpropagator_heater","IsHeater", True])
                stab.add("Select", sel_stab)
                stab.add("InputAmplitude", "optimumfilter_all")
                stab.add("SegmentatedFit", True)
                stab.add("StabValue", 10000.0)
            else:
                stab.add("InputAmplitude", "calibration_rough")
                stab.add("StabValue", 2614.511)

                sel_stab.append(["module","calibration_rough","amplitude", [2590.0], [2630.0]])
                stab.add("SelectInside", sel_stab)

                sel_above_stab = array()
                sel_above_stab.append(["module","correlation_corr","correlation", [0.99]])
                stab.add("SelectAbove", sel_above_stab)

            
            stab.add("RemoveOutliers", 3.5)
            stab.add("Mode", "manual")
            stab.add("Draw", False)
            

        else:
            stab.add("InputAmplitude", "optimumfilter_all")
            stab.add("NameFunc", "stabilization_all_Stabilization")
            stab.add("LoadStab", args.loadstab)


        stab_light.add("all", stab)
        stab_table.add("light", stab_light)
        module.add("stabilization", stab_table)

    # ---- Add module to doc ----
    doc.add("module", module)

    # ---- Write final TOML ----
    with open(final_output_path, "w") as f:
        f.write(dumps(doc))

    print(f"TOML saved to: {final_output_path}")


if __name__ == "__main__":
    main()
