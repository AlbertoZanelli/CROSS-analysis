import ROOT
import os
import argparse
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument("--cross", required=True)
parser.add_argument("--runs", nargs="+", required=True)
parser.add_argument("--outdir", default=".")
parser.add_argument("--channel", type=int, default=None)
parser.add_argument("--channelLD1", type=int, default=None)
parser.add_argument("--channelLD2", type=int, default=None)


args = parser.parse_args()


def format_run(run):
    run = str(run)

    if run.startswith("RUN"):
        return run

    # convert 102 -> RUN000102
    return f"RUN{int(run):06d}"


def merge_runs(cross_dir, runs, channel=None, channelLD1=None, channelLD2=None, output_dir="."):
    import ROOT
    from collections import defaultdict
    from array import array
    import os

    if len(runs) < 2:
        raise ValueError("Need at least 2 runs")

    print("Merging runs:", runs)
    print("Channel:", channel if channel is not None else "ALL")

    files_by_channel = defaultdict(list)

    # --------------------------------------------
    # Collect files
    # --------------------------------------------
    for run in runs:
        run_str = str(run)
        if run_str.startswith("RUN"):
            run_number = int(run_str.replace("RUN", ""))
        else:
            run_number = int(run_str)

        run_fmt = f"RUN{run_number:06d}"
        ofdata_path = os.path.join(cross_dir, run_fmt, "Coincidence")

        if not os.path.isdir(ofdata_path):
            print(f"Warning: missing {ofdata_path}")
            continue

        for fname in os.listdir(ofdata_path):
            if not fname.endswith(".root"):
                continue

            if channel is None or channelLD1 is None or channelLD2 is None:
                continue

            parts = fname.replace(".root", "").split("_")

            ch_name = ""
            run_in_file = ""

            requested_ch = ""

            if len(parts) == 4:
                ch_name = f"{parts[0]}_{parts[1]}_{parts[2]}"
                run_in_file = int(parts[3])
                requested_ch = f"{channel}_{channelLD1}_{channelLD2}"
            elif len(parts) == 3:
                ch_name = f"{parts[0]}_{parts[1]}"
                run_in_file = int(parts[2])
                requested_ch = f"{channel}_{channelLD1}"
            else:
                continue

            if run_in_file != run_number:
                continue

            if ch_name != requested_ch:
                continue

            fullpath = os.path.join(ofdata_path, fname)
            print(fullpath)
            files_by_channel[ch_name].append(fullpath)

    if not files_by_channel:
        print("No matching files found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    out_base = "_".join(runs)

    # --------------------------------------------
    # Merge with cumulative time
    # --------------------------------------------
    for ch, flist in sorted(files_by_channel.items()):
        if len(flist) != len(runs):
            print(f"Skipping channel {ch} (missing in some runs)")
            continue

        print(f"\nMerging channel {ch}")

        f0 = ROOT.TFile.Open(flist[0])
        tree_names = [k.GetName() for k in f0.GetListOfKeys() if k.GetClassName() == "TTree"]
        f0.Close()

        if not tree_names:
            print("No TTrees found")
            continue

        outname = os.path.join(output_dir, f"{out_base}_ch{ch}.root")
        outfile = ROOT.TFile(outname, "RECREATE")

        for treename in tree_names:
            chain = ROOT.TChain(treename)
            for f in flist:
                chain.Add(f)

            # Only for timestamp tree, add cumulative branch
            if treename == "timestamp":
                # Clone structure but empty
                newtree = chain.CloneTree(0)

                # New branch
                time_cum = array('d', [0.])
                newbranch = newtree.Branch("time_cumulative", time_cum, "time_cumulative/D")

                # Precompute last time of each file and entries
                last_times = []
                entries_per_file = []
                for f in flist:
                    tf = ROOT.TFile.Open(f)
                    t = tf.Get("timestamp")
                    entries_per_file.append(t.GetEntries())
                    if t.GetEntries() > 0:
                        t.GetEntry(t.GetEntries() - 1)
                        last_times.append(t.heat_timefromstartrun)
                    else:
                        last_times.append(0.0)
                    tf.Close()

                cumulative_offset = 0
                file_index = 0
                entry_start = 0
                entry_end = entries_per_file[file_index]

                # Loop over all entries
                for i, event in enumerate(chain):
                    if i >= entry_end and file_index < len(flist) - 1:
                        cumulative_offset += last_times[file_index]
                        file_index += 1
                        entry_start = entry_end
                        entry_end += entries_per_file[file_index]

                    # Fill cumulative branch
                    time_cum[0] = event.heat_timefromstartrun + cumulative_offset + 10000
                    newtree.Fill()

                outfile.cd()
                newtree.Write()

            else:
                # Merge all other trees normally
                chain.Merge(outfile, 0, "keep")

        outfile.Close()
        print("  ->", outname)

    print("\nDone.")

merge_runs(
    cross_dir=args.cross,
    runs=args.runs,
    channel=args.channel,
    channelLD1=args.channelLD1,
    channelLD2=args.channelLD2,
    output_dir=args.outdir
)
