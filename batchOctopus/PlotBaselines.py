import ROOT
import plotly.graph_objects as go
import os
import argparse
from tomlkit import document, table, array, dumps

def main():
    parser = argparse.ArgumentParser(description="Generate full TOML for reconstruction modules")

    # ---- Directories ----
    parser.add_argument("--rawdir", default="/Users/serino/Octopus/DATA/CROSS/")
    parser.add_argument("--outputdir", default="/Users/serino/Octopus/DATA/CROSS/")

    # ---- RunConfig ----
    parser.add_argument("--prefix", nargs="+", default=["20251116T120036"])
    parser.add_argument("--run", nargs="+", default=["000048"])


    # ---- Settings ----
    parser.add_argument("--rawType", default="Cupid")
    parser.add_argument("--draw", action="store_false")
    parser.add_argument("--verbosity", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--saveWaveforms", action="store_true")
    parser.add_argument("--stride", type=int, default=10)

    parser.add_argument("--startTime", type=float, default=0.0)
    parser.add_argument("--stopTime", type=float, default=-1.0)

    # ---- Channels ----
    parser.add_argument("--chan", nargs="+", type=int, default=[91])
    parser.add_argument("--waveformtype", default="continuous")



    # ---- Output handling ----
    parser.add_argument("--outdirCFG", default=".", help="Directory where the output TOML will be saved")
    parser.add_argument("-o", "--output", default="configPlotData.toml", help="Base output filename")

    args = parser.parse_args()

        

    # ---- Ensure output directory exists ----
    os.makedirs(args.outdirCFG, exist_ok=True)

    # ---- Build final filename including channel(s) ----
    channel_str = "_".join(str(ch) for ch in args.chan)
    base_name, ext = os.path.splitext(args.output)
    final_output_path = os.path.join(args.outdirCFG, f"{base_name}_{0}{ext}")

    # ---- TOML Document ----
    doc = document()

    # ===== Directories =====
    dirs = table()
    dirs.add("rawdir", args.rawdir)
    doc.add("directories", dirs)

    # ===== RunConfig =====
    rc = table()
    rc.add("filenamePrefix", args.prefix)
    rc.add("runNumber", args.run)
    doc.add("runConfig", rc)

    # ===== Channels =====
    channels = table()
    chan_table = table()
    chan_table.add("list", args.chan)
    chan_table.add("waveformtype", args.waveformtype)
    channels.add("light", chan_table)
    doc.add("channels", channels)

    # ===== Settings =====
    settings = table()
    settings.add("rawType", args.rawType)
    settings.add("verbosity", args.verbosity)
    settings.add("startTime", args.startTime)
    settings.add("stopTime", args.stopTime)
    settings.add("stride", args.stride)
    settings.add("plotTrigger", False)

    if args.saveWaveforms:
        settings.add("SaveDir", args.outputdir)
    
    doc.add("settings", settings)


    # ---- Write final TOML ----
    with open(final_output_path, "w") as f:
        f.write(dumps(doc))

    print(f"TOML saved to: {final_output_path}")


if __name__ == "__main__":
    main()
