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
    parser.add_argument("--gain", type=int, default=2060, help="Amplifier gain")

    # ---- Channels ----
    parser.add_argument("--chan", nargs="+", type=int, default=[91])
    parser.add_argument("--waveformtype", default="continuous")
    parser.add_argument("--type", default="")

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

    # ---- Working Points info
    parser.add_argument("--workingpoints", nargs="+", type=float, required=False,
                    help="List of working points to process")
    
    parser.add_argument("--wp_duration", type=float, default=60.00,
                    help="Duration of each working point in seconds (default: 0.01s)")

    parser.add_argument("--offset", type=float, default=0.0,
                        help="Time offset to apply to working points in seconds (default: 0.0s)")


    args = parser.parse_args()

    # ---- Ensure output directory exists ----
    os.makedirs(args.outdirCFG, exist_ok=True)

    # ---- Build final filename including channel(s) ----
    channel_str = "_".join(str(ch) for ch in args.chan)
    base_name, ext = os.path.splitext(args.output)
    final_output_path = os.path.join(args.outdirCFG, f"{base_name}_{channel_str}{ext}")


    PulserPath = f"{args.triggerdir}/Trigger_{args.prefix[0]}_{args.run[0]}_1.root"
    pulser_lab = "heater"
    pulser_flag = "isHeater"

    if args.type=="light":
        PulserPath = f"{args.triggerdir}/Trigger_{args.prefix[0]}_{args.run[0]}_2.root"
        pulser_lab = "led"
        pulser_flag = "isLED"


    OFFSET = args.offset
    offset1 = (OFFSET*0.95)/2
    WP_VALUES = args.workingpoints
    num_wp = len(WP_VALUES)
    WP_DURATION = args.wp_duration

    startVec = [420.0, 1210.1, 2100.1, 2900.1, 3800.1, 4650.1, 5500.1, 6300.1, 7200.1, 8030.1, 8950.1, 9750.1, 10620.1, 11430.1, 12330.1, 13162.1, 14070.1, 14870.1, 15730.1, 16540.1, 17430.1, 18250.1, 19160.1, 19970.1, 20850.1, 21645.1, 22540.1, 23370.1, 24253.1, 25070.1, 25940.1]
    stopVec = [1150.1, 1970.1, 2840.1, 3650.1, 4550.1, 5370.1, 6250.1, 7080.1, 7960.1, 8780.1, 9650.1, 10500.1, 11390.1, 12200.1, 13110.1, 13920.1, 14780.1, 15610.1, 16490.1, 17300.1, 18150.1, 19000.1, 19850.1, 20700.1, 21580.1, 22400.1, 23310.1, 24120.1, 25000.1, 25830.1, 26700.1]
 
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


    pulser_table = table()
    pulser_light = table()
    pulser = table()

    # pulser.add("FlagPropagatorPath", PulserPath)
    # pulser.add("Draw", False)

    # pulser_light.add(pulser_lab, pulser)
    # pulser_table.add("light", pulser_light)
    # module.add("flagpropagator", pulser_table)


    # ---- Add module to doc ----
    doc.add("module", module)

    # Build the TOML content as a string
    toml_content = ""

    wps = args.workingpoints if args.workingpoints else [args.wp_duration]

    for suffix_num in range(0, len(wps) * 2):



        suffix = f"_wp{suffix_num}"


        # Working point center
        center = OFFSET + WP_DURATION/2 + suffix_num * (WP_DURATION+25)  ## usually it last a bit more

        # start = center - 0.5*WP_DURATION/2 
        # end   = center + 0.8*WP_DURATION/2
     
        # ======================
        start = startVec[suffix_num] 
        end   = stopVec[suffix_num] 

        # suffix_num = suffix_num+1

        toml_content += f"\n"
        toml_content += f"[module.cuts.light.noise_slope{suffix}]\n"
        toml_content += f"Select = [[\"module\", \"module\", \"isnoise\", true], [\"module\", \"numberoftriggers\", \"numberoftriggers\", 1],\n"
        toml_content += f"          [\"module\", \"module\", \"{pulser_flag}\", false]]\n"
        toml_content += f"SelectInside = [[\"module\", \"timestamp\", \"timefromstartrun\", [{start}], [{end}]]]\n"
        toml_content += "CutVariable = [\"baselineslope\", \"slope\"]\n"
        toml_content += "RunOnlyOnGoodEvents = true\n"
        toml_content += f"\n"
        
        toml_content += f"\n"
        toml_content += f"[module.cuts.light.noise_rms{suffix}]\n"
        toml_content += f"Select = [[\"module\", \"cuts_noise_slope{suffix}\", \"pass\", true]]\n"
        toml_content += f"SelectInside = [[\"module\", \"timestamp\", \"timefromstartrun\", [{start}], [{end}]]]\n"
        toml_content += f"Threshold = 2.2\n"
        toml_content += "CutVariable = [\"baseline\", \"RMS\"]\n"
        toml_content += "RunOnlyOnGoodEvents = true\n"
        toml_content += f"\n"
        
        toml_content += f"\n"
        toml_content += f"[module.cuts.light.noise_amplitude{suffix}]\n"
        toml_content += f"Select = [[\"module\", \"cuts_noise_rms{suffix}\", \"pass\", true]]\n"
        # toml_content += f"Select = [[\"module\", \"cuts_noise_slope{suffix}\", \"pass\", true]]\n"
        toml_content += f"SelectInside = [[\"module\", \"timestamp\", \"timefromstartrun\", [{start}], [{end}]]]\n"
        toml_content += f"Threshold = 2.5\n"
        toml_content += "CutVariable = [\"maxminusbaseline\", \"amplitude\"]\n"
        toml_content += "RunOnlyOnGoodEvents = true\n"
        toml_content += f"\n"
        
        # toml_content += f"\n"
        # toml_content += f"[module.cuts_ap.light.signal{suffix}]\n"
        # toml_content += f"Select = [[\"module\", \"module\", \"issignal\", true], [\"module\", \"flagpropagator_{pulser_lab}\", \"{pulser_flag}\", true]\n, [\"module\", \"numberoftriggers\", \"numberoftriggers\", 1]]\n"
        # toml_content += f"SelectInside = [[\"module\", \"timestamp\", \"timefromstartrun\", [{start}], [{end}]]]\n"
        # toml_content += "InputMaxminusbaselineModule = \"maxminusbaseline\"\n"
        # toml_content += "Type = \"light\"\n"
        # toml_content += "CutMode = \"SNratio\"\n"
        # toml_content += f"\n"


        toml_content += f"\n"
        toml_content += f"[module.cuts.light.signal_slope{suffix}]\n"
        toml_content += f"Select = [[\"module\", \"module\", \"{pulser_flag}\", true], [\"module\", \"numberoftriggers\", \"numberoftriggers\", 1]]\n"
        toml_content += f"SelectInside = [[\"module\", \"timestamp\", \"timefromstartrun\", [{start}], [{end}]]]\n"
        toml_content += "CutVariable = [\"baselineslope\", \"slope\"]\n"
        toml_content += "RunOnlyOnGoodEvents = true\n"
        toml_content += f"\n"

        # toml_content += f"\n"
        # toml_content += f"[module.cuts.light.manual{suffix}]\n"
        # toml_content += f"Select = [[\"module\", \"module\", \"isnoise\", true], [\"module\", \"numberoftriggers\", \"numberoftriggers\", 1]]\n"
        # # toml_content += f"Select = [[\"module\", \"module\", \"isnoise\", true]]\n"
        # toml_content += f"SelectInside = [[\"module\", \"timestamp\", \"timefromstartrun\", [{start}], [{end}]]]\n"
        # toml_content += "Mode = \"manual\"\n"
        # toml_content += "RunOnlyOnGoodEvents = true\n"
        # toml_content += f"\n"

        toml_content += f"\n"
        toml_content += f"[module.crosscorr.light.signal{suffix}]\n"
        toml_content += f"Select = [[\"module\", \"cuts_signal_slope{suffix}\", \"pass\", true]]\n"
        #toml_content += f"Select = [[\"module\", \"module\", \"{namePulser}\", true]]\n"
        toml_content += f"SelectInside = [[\"module\", \"timestamp\", \"timefromstartrun\", [{start}], [{end}]]]\n"
        toml_content += "CorrFraction = 0.3\n"
        toml_content += "CorrCut = 0.75\n"
        toml_content += "RunOnlyOnGoodEvents = true\n"
        toml_content += f"\n"

        ## ANPS

        toml_content += f"\n"
        toml_content += f"[module.triggerdelaycorrection.light.noise{suffix}]\n"
        toml_content += "OutputWaveform = \"triggerdelaycorrected\"\n"
        toml_content += f"Select = [[\"module\", \"cuts_noise_amplitude{suffix}\", \"pass\", true]]\n"
        toml_content += "RunOnlyOnGoodEvents = true\n"
        toml_content += f"\n"

        toml_content += f"\n"
        toml_content += f"[module.baselinesubtraction.light.noise{suffix}]\n"
        toml_content += "InputWaveform = \"triggerdelaycorrected\"\n"
        toml_content += "OutputWaveform = \"baselinesubtracted\"\n"
        toml_content += f"Select = [[\"module\", \"cuts_noise_amplitude{suffix}\", \"pass\", true]]\n"
        toml_content += "RunOnlyOnGoodEvents = true\n"
        toml_content += f"\n"

        toml_content += f"\n"
        toml_content += f"[module.fouriertransform.light.noise{suffix}]\n"
        toml_content += "InputWaveform = \"baselinesubtracted\"\n"
        toml_content += "OutputWaveform = \"fft\"\n"
        toml_content += "Windowing = \"FlatTop\"\n"
        toml_content += f"Select = [[\"module\", \"cuts_noise_amplitude{suffix}\", \"pass\", true]]\n"
        toml_content += "RunOnlyOnGoodEvents = true\n"
        toml_content += f"\n"

        toml_content += f"\n"
        toml_content += f"[module.averagepowerspectrum.light.noise{suffix}]\n"
        toml_content += "InputWaveform = \"fft\"\n"
        toml_content += f"Select = [[\"module\", \"cuts_noise_amplitude{suffix}\", \"pass\", true]]\n"
        toml_content += f"Gain = {args.gain}\n"
        toml_content += "RunOnlyOnGoodEvents = true\n"
        toml_content += f"\n"

        ## AP 

        toml_content += f"\n"
        toml_content += f"[module.triggerdelaycorrection.light.ap{suffix}]\n"
        toml_content += f"Select = [[\"module\", \"crosscorr_signal{suffix}\", \"pass\", true]]\n"
        # toml_content += f"Select = [[\"module\", \"cuts_signal_slope{suffix}\", \"pass\", true]]\n"
        # toml_content += f"Select = [[\"module\", \"module\", \"{namePulser}\", true]]\n"
        toml_content += "OutputWaveform = \"triggerdelaycorrectedAP\"\n"
        toml_content += "RunOnlyOnGoodEvents = true\n"
        toml_content += f"\n"

        toml_content += f"\n"
        toml_content += f"[module.baselinesubtraction.light.ap{suffix}]\n"
        toml_content += f"Select = [[\"module\", \"crosscorr_signal{suffix}\", \"pass\", true]]\n"
        # toml_content += f"Select = [[\"module\", \"cuts_signal_slope{suffix}\", \"pass\", true]]\n"
        # toml_content += f"Select = [[\"module\", \"module\", \"{namePulser}\", true]]\n"
        toml_content += "InputWaveform = \"triggerdelaycorrectedAP\"\n"
        toml_content += "OutputWaveform = \"baselinesubtracted\"\n"
        toml_content += "RunOnlyOnGoodEvents = true\n"
        toml_content += f"\n"

        toml_content += f"\n"
        toml_content += f"[module.averagepulse.light.ap{suffix}]\n"
        toml_content += "InputWaveform = \"baselinesubtracted\"\n"
        toml_content += f"Select = [[\"module\", \"crosscorr_signal{suffix}\", \"pass\", true]]\n"
        #toml_content += f"Select = [[\"module\", \"module\", \"{namePulser}\", true]]\n"
        # toml_content += f"Select = [[\"module\", \"cuts_signal_slope{suffix}\", \"pass\", true]]\n"
        toml_content += "Normalize = true\n"
        toml_content += "NormalizeAverage = true\n"
        toml_content += "Windowing = \"FlatTop\"\n"
        toml_content += "RunOnlyOnGoodEvents = true\n"
        toml_content += f"\n"

        ### Optimum Filter
        toml_content += f"\n"
        toml_content += f"[module.triggerdelaycorrection.light.{suffix}]\n"
        toml_content += "OutputWaveform = \"triggerdelaycorrected\"\n"
        toml_content += f"Select = [[\"module\", \"module\", \"{pulser_flag}\", true]]\n"
        toml_content += f"SelectInside = [[\"module\", \"timestamp\", \"timefromstartrun\", [{start}], [{end}]]]\n"
        toml_content += "RunOnlyOnGoodEvents = true\n"
        toml_content += f"\n"

        toml_content += f"\n"
        toml_content += f"[module.baselinesubtraction.light.{suffix}]\n"
        toml_content += "InputWaveform = \"triggerdelaycorrected\"\n"
        toml_content += "OutputWaveform = \"baselinesubtracted\"\n"
        toml_content += f"Select = [[\"module\", \"module\", \"{pulser_flag}\", true]]\n"
        toml_content += f"SelectInside = [[\"module\", \"timestamp\", \"timefromstartrun\", [{start}], [{end}]]]\n"
        toml_content += "RunOnlyOnGoodEvents = true\n"
        toml_content += f"\n"

        toml_content += f"\n"
        toml_content += f"[module.fouriertransform.light.{suffix}]\n"
        toml_content += "InputWaveform = \"baselinesubtracted\"\n"
        toml_content += "OutputWaveform = \"fft\"\n"
        toml_content += f"Select = [[\"module\", \"module\", \"{pulser_flag}\", true]]\n"
        toml_content += f"SelectInside = [[\"module\", \"timestamp\", \"timefromstartrun\", [{start}], [{end}]]]\n"
        toml_content += "Windowing = \"FlatTop\"\n"
        toml_content += "RunOnlyOnGoodEvents = true\n"
        toml_content += f"\n"


        toml_content += f"\n"
        toml_content += f"[module.optimumfilter.light.{suffix}]\n"
        toml_content += "InputWaveform = \"fft\"\n"
        toml_content += "OutputWaveform = \"OFFiltered\"\n"
        toml_content += f"Select = [[\"module\", \"module\", \"{pulser_flag}\", true]]\n"
        toml_content += f"SelectInside = [[\"module\", \"timestamp\", \"timefromstartrun\", [{start}], [{end}]]]\n"
        toml_content += f"APModuleName = \"averagepulse_ap{suffix}\"\n"
        toml_content += f"ANPSModuleName = \"averagepowerspectrum_noise{suffix}\"\n"
        toml_content += f"AP = \"averagepulse_ap{suffix}_trimmedAP\"\n"
        toml_content += f"ANPS = \"averagepowerspectrum_noise{suffix}_medianpower\"\n"
        toml_content += "Windowing = \"FlatTop\"\n"
        toml_content += "RunOnlyOnGoodEvents = true\n"
        toml_content += f"\n"



        
        # ... continue for all sections
    
     # # ---- Write final TOML ----
    with open(final_output_path, "w") as f:
        f.write(dumps(doc))
        f.write(toml_content)

#     # ---- module.averagepowerspectrum.light.anps ----
#     aps_table = table()
#     aps_light = table()
#     aps_anps = table()

#     sel_aps = array()
#     sel_aps.append(["module", "cuts_noise_amplitude", "pass", True])

#     aps_anps.add("InputWaveform", "fft")
#     aps_anps.add("PowerPlotMaxValue", [0.0001])
#     aps_anps.add("Select", sel_aps)
#     aps_anps.add("RunOnlyOnGoodEvents", True)
#     aps_anps.add("Gain", 910)
#     aps_anps.add("Normalize", False)
#     aps_anps.add("Draw", False)

#     aps_light.add("anps", aps_anps)
#     aps_table.add("light", aps_light)
#     module.add("averagepowerspectrum", aps_table)

#     ################################
#     # AP  — AVERAGE PULSE BLOCK
#     ################################

#     # ---- module.triggerdelaycorrection.light.ap ----
#     tdc_ap_table = table()
#     tdc_ap_light = table()
#     tdc_ap = table()

#     sel_ap = array()
#     sel_ap.append(["module", "crosscorr", "pass", True])

#     sel_ap_above = array()
#     sel_ap_above.append(["module", "crosscorr", "avgcorr", [0.85]])

#     tdc_ap.add("Select", sel_ap)
#     tdc_ap.add("SelectAbove", sel_ap_above)
#     tdc_ap.add("OutputWaveform", "triggerdelaycorrectedAP")
#     tdc_ap.add("Draw", False)

#     tdc_ap_light.add("ap", tdc_ap)
#     tdc_ap_table.add("light", tdc_ap_light)
#     module.add("triggerdelaycorrection", tdc_ap_table)

#     # ---- module.baselinesubtraction.light.ap ----
#     bs_ap_table = table()
#     bs_ap_light = table()
#     bs_ap = table()

#     bs_ap.add("InputWaveform", "triggerdelaycorrectedAP")
#     bs_ap.add("OutputWaveform", "baselinesubtracted")
#     bs_ap.add("Select", sel_ap)
#     bs_ap.add("RunOnlyOnGoodEvents", True)
#     bs_ap.add("Draw", False)

#     bs_ap_light.add("ap", bs_ap)
#     bs_ap_table.add("light", bs_ap_light)
#     module.add("baselinesubtraction", bs_ap_table)

#     # ---- module.averagepulse.light.ap ----
#     avgpulse_table = table()
#     avgpulse_light = table()
#     avgpulse = table()

#     sel_ap_above_2 = array()
#     sel_ap_above_2.append(["module", "crosscorr", "avgcorr", [0.85]])

#     avgpulse.add("InputWaveform", "baselinesubtracted")
#     avgpulse.add("SelectAbove", sel_ap_above_2)
#     avgpulse.add("Normalize", True)
#     avgpulse.add("NormalizeAverage", True)
#     avgpulse.add("Select", sel_ap)
#     avgpulse.add("PulseMin", [-0.005])
#     avgpulse.add("PulseMax", [1.1])
#     avgpulse.add("RunOnlyOnGoodEvents", True)
#     avgpulse.add("Windowing", "FlatTop")
#     avgpulse.add("Draw", False)

#     avgpulse_light.add("ap", avgpulse)
#     avgpulse_table.add("light", avgpulse_light)
#     module.add("averagepulse", avgpulse_table)

#     ############################################
#     # Optimum Filter + Power + Gaussian Filter
#     ############################################

#     # ---- module.triggerdelaycorrection.light.all ----
#     tdc_all_table = table()
#     tdc_all_light = table()
#     tdc_all = table()

#     tdc_all.add("OutputWaveform", "triggerdelaycorrected")
#     tdc_all.add("RunOnlyOnGoodEvents", True)
#     tdc_all.add("Draw", False)

#     tdc_all_light.add("all", tdc_all)
#     tdc_all_table.add("light", tdc_all_light)
#     module.add("triggerdelaycorrection", tdc_all_table)

#     # ---- module.baselinesubtraction.light.all ----
#     bs_all_table = table()
#     bs_all_light = table()
#     bs_all = table()

#     bs_all.add("InputWaveform", "triggerdelaycorrected")
#     bs_all.add("OutputWaveform", "baselinesubtracted")
#     bs_all.add("RunOnlyOnGoodEvents", True)
#     bs_all.add("Draw", False)

#     bs_all_light.add("all", bs_all)
#     bs_all_table.add("light", bs_all_light)
#     module.add("baselinesubtraction", bs_all_table)

#     # ---- module.fouriertransform.light.all ----
#     ft_all_table = table()
#     ft_all_light = table()
#     ft_all = table()

#     ft_all.add("InputWaveform", "baselinesubtracted")
#     ft_all.add("OutputWaveform", "fft")
#     ft_all.add("Windowing", "FlatTop")
#     ft_all.add("CorrectDelay", False)
#     ft_all.add("RunOnlyOnGoodEvents", True)
#     ft_all.add("Draw", False)
#     ft_all.add("DrawOption", "ReIm")

#     ft_all_light.add("all", ft_all)
#     ft_all_table.add("light", ft_all_light)
#     module.add("fouriertransform", ft_all_table)

#     # ---- module.optimumfilter.light.all ----
#     of_all_table = table()
#     of_all_light = table()
#     of_all = table()

#     of_all.add("InputWaveform", "fft")
#     of_all.add("OutputWaveform", "OFFiltered")
#     of_all.add("APModuleName", "averagepulse_ap")
#     of_all.add("ANPSModuleName", "averagepowerspectrum_anps")
#     of_all.add("AP", "averagepulse_ap_trimmedAP")
#     of_all.add("ANPS", "averagepowerspectrum_anps_medianpower")
#     of_all.add("Windowing", "FlatTop")
#     of_all.add("RunOnlyOnGoodEvents", True)
#     of_all.add("Draw", False)
#     of_all.add("DrawOption", "Chi2")

#     of_all_light.add("all", of_all)
#     of_all_table.add("light", of_all_light)
#     module.add("optimumfilter", of_all_table)

#     ############################################
#     # CORRELATION PARAMETER FOR PSD
#     ############################################



    # # ---- Add module to doc ----
    # doc.add("module", module)

    # # # ---- Write final TOML ----
    # # with open(final_output_path, "w") as f:
    # #     f.write(dumps(doc))

    print(f"TOML saved to: {final_output_path}")


if __name__ == "__main__":
    main()
