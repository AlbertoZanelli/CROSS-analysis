import re
import tempfile
import subprocess
import os
import time
import threading
import sys
import random
import ROOT
import plotly.graph_objects as go
import json




RunTerminal = False


ramUsage = 4       # GB per job, adjust as needed

max_parallel_jobs = 100
sleep_interval = 20
KILL_TIMEOUT = 100000

_shared_config = None

channels_LD = None
channels_LMO = None

def SetChannels(vectorLMO, vectorLD):
    global channels_LD, channels_LMO

    channels_LD = vectorLD
    channels_LMO = vectorLMO

    return
    

def set_shared_config(cfg):
    global _shared_config
    _shared_config = cfg
    return



def GetTypeChannel(channel):

    type_crystal = ""

    if channel in channels_LD:
        type_crystal = "light"
        _shared_config["typeCrystal"] = "light"
    elif channel in channels_LMO:
        type_crystal = "heat"
        _shared_config["typeCrystal"] = "heat"
    else:
        type_crystal = "undefined"
        _shared_config["typeCrystal"] = "heat"
        type_crystal = "heat"

        print("Type of crystal not found, default will be heat")

    return


def SetRunOnTerminal(ExecuteOnTerminal):
    global RunTerminal
    RunTerminal = ExecuteOnTerminal
    return

def get_channel_vector(file_path):
    """Return vector where index=vertical_position, value=channel."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find all [channel, position] pairs, skipping comment lines
    matches = []
    for line in lines:
        # Skip comment lines
        if line.strip().startswith('#'):
            continue
        
        # Find [channel, position] pattern
        match = re.search(r'\[(\d+),\s*(\d+)\]', line)
        if match:
            matches.append((int(match.group(1)), int(match.group(2))))
    
    if not matches:
        return []
    
    # Find max position for vector size
    max_pos = max(pos for _, pos in matches)
    
    # Create and fill vector
    channel_vector = [None] * (max_pos + 1)
    for channel, pos in matches:
        channel_vector[pos] = channel
    
    return channel_vector

def build_neighbor_map(channel_vector, nFloors=14):
    neighbor_map = {}
    """
    Build a neighbor map based on channel positions in a tower.
    
    Args:
        channel_vector: List where index is vertical position, value is channel (None for empty positions)
        nFloors: Number of floors per tower (default 14)
    
    Returns:
        Dictionary mapping channel -> list of neighbor channels
    """
    
    for i in range(len(channel_vector)):
        # Skip empty positions
        if channel_vector[i] is None:
            continue
            
        neighbor = []
        tower_size = nFloors*2
        # Check if position is at top or bottom of a tower
        # Bottom of tower: position % tower_size == 0
        # Top of tower: position % tower_size == (tower_size - 1) = 13
        position_in_tower = i % tower_size
        
        is_bottom = (position_in_tower == 0)  # First floor
        is_top = (position_in_tower == tower_size - 1)  # Top floor (13)
        
        # For bottom positions: only neighbor above
        if is_bottom:
            if i < len(channel_vector) - 1 and channel_vector[i+1] is not None:
                neighbor.append(channel_vector[i+1])
        
        # For top positions: only neighbor below
        elif is_top:
            if i > 0 and channel_vector[i-1] is not None:
                neighbor.append(channel_vector[i-1])
        
        # For middle positions: neighbors above and below
        else:
            if i > 0 and channel_vector[i-1] is not None:
                neighbor.append(channel_vector[i-1])
            if i < len(channel_vector) - 1 and channel_vector[i+1] is not None:
                neighbor.append(channel_vector[i+1])
        
        neighbor_map[channel_vector[i]] = neighbor
    
    return neighbor_map


# ---------- Helper functions --------------------------------

# --------------------------
# Create temporary shell script
def create_sh(lines):
    tmp = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.sh')
    tmp.write("#!/bin/bash\n")
    tmp.write("\n".join(lines))
    tmp.close()
    os.chmod(tmp.name, 0o755)
    return tmp.name


def extract_running_job_ids(qstat_output):
    """Return a list of numeric job ids currently running (as strings)."""
    running = []
    for line in qstat_output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 0:
            continue
        token = parts[0].split(".")[0]
        if token.isdigit():
            running.append(token)
    return running


def get_current_running_ids():
    """Run qstat and return running numeric job ids."""
    try:
        result = subprocess.run("qstat -u $USER", shell=True,
                                capture_output=True, text=True, check=False)
        return extract_running_job_ids(result.stdout)
    except Exception as e:
        # If qstat fails for some reason, return empty list (safe fallback)
        print("[WARN] qstat failed:", e)
        return []

# Wait until running jobs < max_parallel_jobs
def wait_for_slot():
    user = os.environ['USER']  # ensures we get the correct username
    while True:
        cmd = f"qstat -u {user} | grep 'xmaster' | wc -l"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        running = int(result.stdout.strip())
        if running < max_parallel_jobs:
            break
        print(f"Max parallel jobs reached ({running}). Waiting {sleep_interval}s...")
        time.sleep(sleep_interval)

def submit_job(channel, sh_file, job_map):
    """
    Submit the sh_file for channel if no job for the channel is still running.
    job_map is modified in-place.

    Returns:
        True  -> new job submitted (and registered in job_map)
        False -> existing job still running (no new submission)
        "error" -> submission failed (qsub error)
    """

    ## To test single channel process
    if RunTerminal is True:
        os.chmod(sh_file, 0o755)
        subprocess.run([sh_file])
        return True
        # sys.exit()  # exits the script here

    running_ids = get_current_running_ids()

    # If a job was already submitted for this channel, check it
    if channel in job_map:
        existing_id = job_map[channel]
        if existing_id in running_ids:
            # still running -> don't submit another
            return False
        else:
            # previous job finished -> clear from map and allow re-submission
            del job_map[channel]

    # Submit new job
    try:
        cmd = f"qsub -q cupid -o localhost:/mnt/disk1/data/users/azanelli/octopus_work/CROSS/output/ -e localhost:/mnt/disk1/data/users/azanelli/octopus_work/CROSS/error/ -l walltime=24:00:00 -l mem={ramUsage}G {sh_file}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    except Exception as e:
        print(f"[ERROR] Exception when running qsub for channel {channel}: {e}")
        return "error"

    # -m abe -M roberto.serino@ijclab.in2p3.fr

    if result.returncode != 0:
        print(f"[ERROR] qsub failed for channel {channel}. stderr:\n{result.stderr}")
        return "error"

    full_jobid = result.stdout.strip()
    jobid = full_jobid.split(".")[0] if full_jobid else ""

    if not jobid.isdigit():
        print(f"[ERROR] qsub returned invalid job id for channel {channel}: '{full_jobid}'")
        return "error"

    job_map[channel] = jobid
    print(f"[OK] Submitted channel {channel} with job ID {jobid}")
    return True

# At this point, all channels have been submitted to the queue.
# If you want to wait for all tracked jobs to finish before exiting, you can poll qstat:
def wait_for_all_jobs(job_map, poll_interval=10):
    """Block until all job ids in job_map are no longer running."""
    while job_map:
        running_ids = get_current_running_ids()
        # remove finished entries
        finished = [ch for ch, jid in job_map.items() if jid not in running_ids]
        for ch in finished:
            print(f"[INFO] job for channel {ch} (id {job_map[ch]}) finished.")
            del job_map[ch]
        if job_map:
            time.sleep(poll_interval)

def try_submit_pass(pending_list, make_lines_fn, job_map, append_jobids_list=None, sleep_on_slot=False, check = True):


    ## To test single channel process
    if RunTerminal is True:
        for channel in pending_list[:]:
            GetTypeChannel(channel)
            sh_file = create_sh(make_lines_fn(channel))
            os.chmod(sh_file, 0o755)
            subprocess.run([sh_file])

            sys.exit()  # exits the script here, only one channel on terminal

            

        

    """
    One pass trying to submit each channel in pending_list (non-destructive).
    - make_lines_fn(channel) -> list_of_lines for create_sh
    - pending_list will be mutated: channels that were successfully submitted are removed
    - append_jobids_list (optional) will be appended with job ids for successful submissions
    Returns number of submissions performed in this pass.
    """
    submitted_count = 0
    # iterate over a shallow copy to allow removing from pending_list
    for channel in pending_list[:]:
        if sleep_on_slot:
            wait_for_slot()

        if check:
            GetTypeChannel(channel)

        lines = make_lines_fn(channel)
        sh_file = create_sh(lines)
        os.chmod(sh_file, 0o755)

        if 0 in job_map:
            time.sleep(sleep_interval)
            continue   # skip submission for now

        status = submit_job(channel, sh_file, job_map)

        if status is True:
            # record submitted job id if requested
            if append_jobids_list is not None:
                append_jobids_list.append(job_map[channel])
            pending_list.remove(channel)
            submitted_count += 1
        elif status == "error":
            # don't remove the channel; print a warning and continue
            print(f"[WARN] submission error for channel {channel}; will retry later.")
        else:
            # status is False -> job already running for this channel; leave it in pending
            # no print here to avoid spam, but could log if desired
            pass

    return submitted_count
    #return False





def get_random_timestamp(base_path, run_number, min_size_mb=100):
    """
    base_path: e.g. "/data2/LSC/DATA/RUN14"
    run_number: e.g. "000102" or 102
    min_size_mb: minimum file size in MB (default 100)
    """

    # Ensure run number is zero-padded to 6 digits
    run_str = f"{int(run_number):06d}"

    if not os.path.isdir(base_path):
        print(f"Run directory not found: {base_path}")
        return None

    min_size_bytes = min_size_mb * 1024 * 1024

    # Collect valid files
    valid_files = []
    for fname in os.listdir(base_path):
        if not fname.endswith(".bin"):
            continue

        full_path = os.path.join(base_path, fname)

        if os.path.getsize(full_path) >= min_size_bytes:
            valid_files.append(fname)

    if not valid_files:
        print(f"No .bin files larger than {min_size_mb} MB found")
        return None

    # Pick one randomly
    chosen_file = random.choice(valid_files)

    # Extract timestamp
    # Expected pattern: 000102_20251231T132124_65680_002.bin
    match = re.search(r'_(\d{8}T\d{6})_', chosen_file)

    if not match:
        print(f"Could not extract timestamp from {chosen_file}")
        return None

    return match.group(1)



def generate_html(input_dir, output_file, run0=""):
    # Input file
    input_file = f"{input_dir}/PlotDataStream_Run{run0}.root"

    # Output
    outfile = output_file

    # Open ROOT file
    f = ROOT.TFile.Open(input_file)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open file {input_file}")

    # --------------------------------------------------
    # Extract all canvases from the file
    # --------------------------------------------------
    figures_html = []
    
    # Get all keys in the file
    keys = f.GetListOfKeys()
    
    for key in keys:
        obj_name = key.GetName()
        obj = key.ReadObj()
        
        # Check if it's a canvas
        if not obj.InheritsFrom("TCanvas"):
            continue
        
        canvas = obj
        canvas_title = canvas.GetTitle()
        
        print(f"Processing canvas: {obj_name} (Title: {canvas_title})")
        
        # Create figure for this canvas
        fig = go.Figure()
        
        # Get all primitives in the canvas
        primitives = canvas.GetListOfPrimitives()
        
        for primitive in primitives:
            # Handle TGraph objects
            if primitive.InheritsFrom("TGraph"):
                g = primitive
                x = [g.GetX()[i] for i in range(g.GetN())]
                y = [g.GetY()[i] for i in range(g.GetN())]
                
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=g.GetName() or f"Graph_{len(fig.data)}"
                ))
            
            # Handle TMultiGraph objects
            elif primitive.InheritsFrom("TMultiGraph"):
                multigraph = primitive
                graphs = multigraph.GetListOfGraphs()
                
                for g in graphs:
                    x = [g.GetX()[i] for i in range(g.GetN())]
                    y = [g.GetY()[i] for i in range(g.GetN())]
                    
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        mode="lines+markers",
                        name=g.GetName()
                    ))
        
        # Skip empty figures
        if len(fig.data) == 0:
            print(f"  Warning: No data found in canvas {obj_name}")
            continue
        
        # Layout per canvas
        fig.update_layout(
            title=f"<b>{canvas_title or obj_name}</b>",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white",
            height=400,
            margin=dict(t=60, b=40)
        )
        
        # Convert figure to HTML snippet (no full page)
        figures_html.append(fig.to_html(full_html=False, include_plotlyjs=False))
    
    f.Close()
    
    if not figures_html:
        raise RuntimeError(f"No valid canvases found in {input_file}")
    
    # --------------------------------------------------
    # Write ONE HTML with all figures
    # --------------------------------------------------
    with open(outfile, "w") as fhtml:
        fhtml.write("""
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .figure-container { margin-bottom: 50px; border-bottom: 2px solid #ccc; padding-bottom: 20px; }
        </style>
    </head>
    <body>
    """)
        
        for i, fig_html in enumerate(figures_html, 1):
            fhtml.write(f"<div class='figure-container' id='canvas_{i}'>")
            fhtml.write(fig_html)
            fhtml.write("</div>")

        fhtml.write("""
    </body>
    </html>
    """)

    print(f"\nSaved HTML with {len(figures_html)} canvases:\n  {outfile}")