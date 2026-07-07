# Created by: Roberto Serino
# Date: 2025-11-19
# Description: Automatized trigger merger for working point selection (updated structure)

import argparse
import os
from tomlkit import document, dumps, table, comment

def main():
    parser = argparse.ArgumentParser(description="Generate TOML configuration.")

    # ---- Directories ----
    parser.add_argument("--rawdir", default="/home/gbenato/Cross/data/raw/")
    parser.add_argument("--triggerdir", default="/home/gbenato/Cross/data/triggered/")

    # ---- Run Config ----
    parser.add_argument("--prefix", nargs="+", default=["20251116T120036"])
    parser.add_argument("--run", nargs="+", default=["000048"])

    # ---- Channels ----
    parser.add_argument("--ch", nargs="+", type=int, default=[49])
    parser.add_argument("--lmo", nargs="+", type=int, default=[])
    parser.add_argument("--ld", nargs="+", type=int, default=[])
    parser.add_argument("--channelHeater", nargs="+", type=int)
    parser.add_argument("--channelLED", nargs="+", type=int)

    parser.add_argument("--waveformtype", default="continuous")

    # ---- Optional Pulser Labels ----
    parser.add_argument("--PulserLabelHeater", default=None,
                        help="Optional label for heater pulser (default: Heater)")
    parser.add_argument("--PulserLabelLED", default=None,
                        help="Optional label for LED pulser (default: LED)")

    # ---- Settings ----
    parser.add_argument("--rawType", default="Cupid")
    parser.add_argument("--draw", action="store_true")
    parser.add_argument("--propagateflag", action="store_true")
    parser.add_argument("--verbosity", type=int, default=5)
    parser.add_argument("--timewindow", type=float, default=0.0015)

    # ---- Output ----
    parser.add_argument("--outdirCFG", default=".")
    parser.add_argument("-o", "--output", default="configMerger.toml")

    parser.add_argument("--detectormapPath", default="/home/gbenato/Cross/Octopus/data/detectormap/DetectorMap_Cupid_LMO_LD.toml")


    args = parser.parse_args()
    os.makedirs(args.outdirCFG, exist_ok=True)


    channel_str = "_".join(str(ch) for ch in args.ch)
    base_name, ext = os.path.splitext(args.output)

    final_output_name = f"{base_name}_{channel_str}{ext}"
    final_output_path = os.path.join(args.outdirCFG, final_output_name)

    list_lmo = args.lmo if hasattr(args, 'lmo') else []
    list_ld = args.ld if hasattr(args, 'ld') else []

    # ---- Create TOML ----
    doc = document()

    # === [directories] ===
    directories = table()
    directories.add("rawdir", args.rawdir)
    directories.add("triggerdir", args.triggerdir)
    doc.add("directories", directories)

    # === [runConfig] ===
    runConfig = table()
    runConfig.add("filenamePrefix", args.prefix)
    runConfig.add("runNumber", args.run)
    doc.add("runConfig", runConfig)

    # === [settings] ===
    settings = table()
    settings.add("rawType", args.rawType)
    settings.add("verbosity", args.verbosity)
    settings.add("draw", args.draw)
    settings.add("overwrite", True)
    settings.add("timewindow", args.timewindow)
    doc.add("settings", settings)

    propagate_heater = False
    propagate_led = False

    # === [channels] ===
    channels = table()
    if args.propagateflag:
        # ---- Heater ----
        if args.channelHeater is not None:
            if len(args.channelHeater) == 1 and args.channelHeater[0] < 0:
                # Disable heater
                doc.add(comment("[channels.heater] list=[]  # disabled"))
                propagate_heater = False
            else:
                # Enabled heater
                heater_label = args.PulserLabelHeater if args.PulserLabelHeater else "Heater"

                heater = table()
                heater.add("list", args.channelHeater)
                heater.add("waveformtype", args.waveformtype)
                heater.add("IsPulser", True)
                heater.add("PulserLabel", heater_label)
                channels.add("heater", heater)

                propagate_heater = True
        else:
            # No heater specified at all → no section, no propagate
            propagate_heater = False


        # ---- LED ----
        if args.channelLED is not None:
            if len(args.channelLED) == 1 and args.channelLED[0] < 0:
                doc.add(comment("[channels.led] list=[]  # disabled"))
                propagate_led = False
            else:
                led_label = args.PulserLabelLED if args.PulserLabelLED else "LED"

                led = table()
                led.add("list", args.channelLED)
                led.add("waveformtype", args.waveformtype)
                led.add("IsPulser", True)
                led.add("PulserLabel", led_label)
                channels.add("led", led)

                propagate_led = True
        else:
            propagate_led = False

    # ---- Lmo channels ----
    lmo = table()
    lmo.add("list", list_lmo)
    lmo.add("waveformtype", args.waveformtype)
    lmo.add("PropagateHeater", propagate_heater)
    lmo.add("PropagateLED", propagate_led)
    channels.add("lmo", lmo)

    # doc.add("channels", channels)

    # ---- LD channels ----
    ld = table()
    ld.add("list", list_ld)
    ld.add("waveformtype", args.waveformtype)
    ld.add("Skip", True)
    channels.add("ld", ld)

    doc.add("channels", channels)


    # === [detectormap] ===
    detectormap = table()
    detectormap_vector = [["lmo", "ld"]]
    detectormap.add("PropagationScheme", detectormap_vector)
    detectormap.add("DetectorMap", args.detectormapPath)
    detectormap.add("NumberOfFloors", 7)
    doc.add("detectormap", detectormap)

    # ---- Write file ----
    with open(final_output_path, "w") as f:
        f.write(dumps(doc))

    print(f"TOML saved to: {final_output_path}")


if __name__ == "__main__":
    main()
