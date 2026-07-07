# Created by: Roberto Serino
# Date: 2025-11-19
# Description: Automatized trigger for working point selection

import argparse
import os
from tomlkit import document, dumps, table


def main():

    parser = argparse.ArgumentParser(description="Generate TOML configuration.")

    # ---- Directories ----
    parser.add_argument("--rawdir", default="/Users/serino/Octopus/DATA/CROSS/")
    parser.add_argument("--triggerdir", default="/Users/serino/Octopus/Results/Triggered/CROSS/")

    # ---- Run Config ----
    parser.add_argument("--prefix", nargs="+", default=["20251114T093015"])
    parser.add_argument("--run", nargs="+", default=["000045"])

    # ---- Channels ----
    parser.add_argument("--channel", nargs="+", type=int, default=[32])
    parser.add_argument("--waveformtype", default="continuous")
    parser.add_argument("--isPulser", action="store_true")
    parser.add_argument("--isVeto", action="store_true")
    parser.add_argument("--pulsertype", default="IsLED")

  # ---- Settings ----
    parser.add_argument("--rawType", default="Cupid")
    parser.add_argument("--runType", default="trigger")
    parser.add_argument("--draw", action="store_true")           
    parser.add_argument("--applyFilter", action="store_true")
    parser.add_argument("--noiseTriggerPeriod", type=float, default=5)
    parser.add_argument("--verbosity", type=int, default=5)

    # ---- Trigger (linear interpolation) ----
    parser.add_argument("--NRMSList", nargs="+", type=float, default=[2.8])
    parser.add_argument("--NAboveList", nargs="+", type=float, default=[5])
    parser.add_argument("--DeadTimeList", nargs="+", type=float, default=[0.02])
    parser.add_argument("--BufferLengthList", nargs="+", type=float, default=[0.03])
    parser.add_argument("--NoisePeriodList", nargs="+", type=float, default=[10.123456])
    parser.add_argument("--ReadFromTxt" , default="")
    parser.add_argument("--triggerDraw", default=False)
    parser.add_argument("--UseFixedThreshold", action="store_true", help="Use fixed threshold instead of dynamic one")
    parser.add_argument("--FixedThreshold", nargs="+", type=float, default=[0.0])
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--triggertype", default="linearinterpolation",
                    help="Type of trigger (e.g., linearinterpolation, derivative)")


    # ---- Output handling ----
    parser.add_argument("--outdirCFG", default=".",
                        help="Directory where the output TOML will be saved")
    parser.add_argument("-o", "--output", default="configTrigg.toml",
                        help="Base output filename")

    args = parser.parse_args()

    # ---- Ensure output directory exists ----
    os.makedirs(args.outdirCFG, exist_ok=True)


    # ---- Build final filename including channel(s) ----
    channel_str = "_".join(str(ch) for ch in args.channel)
    base_name, ext = os.path.splitext(args.output)

    final_output_name = f"{base_name}_{channel_str}{ext}"
    final_output_path = os.path.join(args.outdirCFG, final_output_name)

    # ---- Build TOML using tomlkit ----
    doc = document()

    # directories
    directories = table()
    directories.add("rawdir", args.rawdir)
    directories.add("triggerdir", args.triggerdir)
    doc.add("directories", directories)

    # runConfig
    runConfig = table()
    runConfig.add("filenamePrefix", args.prefix)
    runConfig.add("runNumber", args.run)
    doc.add("runConfig", runConfig)

    # settings
    settings = table()
    settings.add("rawType", args.rawType)
    settings.add("runType", args.runType)
    settings.add("draw", args.draw)
    settings.add("noiseTriggerPeriod", args.noiseTriggerPeriod)
    settings.add("overwrite", True)
    settings.add("verbosity", args.verbosity)
    settings.add("applyFilter", args.applyFilter)
    doc.add("settings", settings)

    # channels.heat
    channels = table()
    heat = table()
    heat.add("list", args.channel)
    heat.add("waveformtype", args.waveformtype)
    if args.isPulser:
        heat.add("PulserLabel", args.pulsertype)
        heat.add("IsPulser", args.isPulser)
    channels.add("heat", heat)
    doc.add("channels", channels)

    # trigger.linearinterpolation.heat
    trigger_type = args.triggertype 
    trig = table()
    lin = table()
    trig_heat = table()

    if args.isVeto:
        trig_heat.add("AuxChannel", 65680)
        trig_heat.add("Mode", "sector")
    else:
        if not args.isPulser:
            trig_heat.add("NRMSList", args.NRMSList)
            trig_heat.add("NAboveList", args.NAboveList)
            trig_heat.add("DeadTimeList", args.DeadTimeList)
            trig_heat.add("ReadFromTxt", args.ReadFromTxt)     
        else:
            trig_heat.add("UseFixedThreshold", [args.UseFixedThreshold])
            trig_heat.add("FixedThrList", args.FixedThreshold)

        trig_heat.add("BufferLengthList", args.BufferLengthList)
        trig_heat.add("NoisePeriodList", args.NoisePeriodList)
        trig_heat.add("Draw", args.triggerDraw)


    lin.add("heat", trig_heat)
    trig.add(trigger_type, lin)
    doc.add("trigger", trig)

    # ---- Write final TOML ----
    with open(final_output_path, "w") as f:
        f.write(dumps(doc))

    print(f"TOML saved to: {final_output_path}")


if __name__ == "__main__":
    main()
