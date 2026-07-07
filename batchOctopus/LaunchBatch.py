#!/usr/bin/env python3
# Author: Roberto Serino
# Date: 2025-11-20
# Description: Automated Working Point selection with PBS job submission,
#              dependency management, parallel job limiting, and watchdog killer.


import subprocess
import os
import time
import threading
import sys
import helpers


# Environment varibles
pythonVenv = "/home/zanelli/env/"  # path where the python venv is located ( with all the needed libraries installed )
sourceOcto = "/home/zanelli/LoadOctopus.sh"  # script to load octopus environment

# --------------------------
# # Channel Configuration

# ch TeO2 13, 14, 15, 16, 17, 18,
channels_heaterless = [25, 53, 59, 66]

channels_LD = [43, 31, 32, 33, 34, 35, 36, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 37, 38, 39, 40, 41, 42, 79, 80, 81, 82, 83, 84, 91, 92, 93, 94, 95, 96]  # List of the Light Detector CROSSS
channels_LMO = [19, 20, 21, 22, 23, 24, 49, 50, 51, 52, 53, 54, 25, 26, 27, 28, 29, 30, 55, 56, 57, 58, 59, 60, 85, 86, 87, 88, 89, 90, 61, 62, 63, 64, 65, 66] # List of the Heat Detector CROSSS  
channels_no_ntl = [32, 43, 69, 70, 72, 76, 84, 93]


channelMuon = 65552
channelHeater = 1
channelLED = 2
 
# channels_LMO = [89, 90, 61, 62, 63, 64, 65, 66] 

#channels_LMO = [19, 20, 21, 22, 23, 24, 49, 50, 51, 52, 53, 54] # first tower 
#channels_LD = [31, 32, 33, 34, 35, 36, 67, 68, 69, 70, 71, 72] # first tower

#channels_LMO = [20, 21, 22, 23, 24] # first tower 


channels_LMO = [25 , 26, 27, 28, 29, 30, 55, 56, 57, 58, 59, 60] #  tower 2
channels_LD = [73, 74, 75, 76, 77, 78, 37, 38, 39, 40, 41, 42] # first tower 2

# channels_LMO = [85, 86, 87, 88, 89, 90, 61, 62, 63, 64, 65, 66] #  tower 3
# channels_LD = [79, 80, 81, 82, 83, 84, 91, 92, 93, 94, 95, 96] #  tower 3



# channels_WP = [31, 34, 37, 40 , 41, 71, 83, 91, 94]

channels_WP = channels_no_ntl

channels_WP = [25] #channels_LMO

# Run 


# Put run0 = runfin if you want to run only one run, 
# otherwise it will run from run0 to runfin (included)
run0 = "000100"
runFin = "000101"       # problem with the noise from 140 on
ComputeAll = False
# If false, it computes only the first run (run0), 
# otherwise it computes all the runs from run0 to runFin (included)

runList = ["000097", "000099", "000100", "000101", "000102", "000103", "000104", "000105", "000106", "000120", "000123", "000124", "000125", "000126", "000128"]    # if it's not empty, it will use these runs
#runList = ["000099", "000100", "000101", "000106"]

skipRuns = ["000098", "000107", "000108", "000109", "000110","000111", "000112",
             "000113", "000114","000115","000116", "000117", "000118","000119", "000121","000127",
             "000129", "000130", "000131","000132","000133","000134","000135","000136", "000137",
             "000138","000139", "000140", "000158", "000201", "000203"]  # list of runs to skip, if ComputeAll is True

for i in range(149, 158):
    skipRuns.append(f"{i:06d}")

for i in range(160, 170):
    skipRuns.append(f"{i:06d}")

for i in range(172, 198):
    skipRuns.append(f"{i:06d}")



NTL = True

PlotBaselines = False

ComputeTriggerPulsers = False
ComputeMuonVeto = False

ComputeTriggerScan = False
ComputeTrigger = False

ComputeMerger = False
PropagateFlag = False

ComputeAnalysis = False
ComputeAnalysisScript = False

ComputeAP = False
ComputeANPS = False
ComputeOF = False       # And calibration

MergeProduction = False

UseMergedTriggers = False
UseFlaggedTriggers = False
Overwrite = True

ComputeMergeRuns = True
OutputDir = "/mnt/disk1/data/users/azanelli/octopus_work/CROSS/MergedRuns"
path_cross = "/mnt/disk1/data/users/azanelli/octopus_work/CROSS/"
plotLY = "/data/users/azanelli/octopus_work/CROSS/MergedRuns/PlotLY/"

GenerateCFG = True 
ExeOctopus = True 
ExecuteOnTerminal = True


typeCrystal = "heat"
typeRun = "B"
overwrite = ""
usemergedtriggers = ""   
useflaggedtriggers = ""
saveWaveforms = ""


# 1: Heater, 2: LED   

# Just initialising values
templateAP = ""
templateANPS = ""
loadStab = ""
window = 1.0  # in seconds
pretrigger = 0.5  # in seconds
deadTime = 0.12
BufferLength = 0.1
pulserModule = ""


## Trigger Variables
triggerTyp = "linearinterpolation"
triggerTypePuls = "derivative"
triggerTypeMuon = "MuonCROSS"
noisePeriod = 10.0123456
NRMSList = 3.2
FixedThreshold = 3.5 # in Volts

Gain = 910

## Trigger Scan Variables
startThresh = 2
endThresh = 8.0
Step = 0.05
TimeScan = 500.0
NAboveInit = 4
NAboveFin = 8

# Plot Baselines Variables
stride = 10000
startTime = 0.0
stopTime = -1 # if it's negative, it will take the whole run
SavePlot = True
channels_plot = [49, 54, 25, 60, 66, 61]
channels_plot = [31, 34, 37, 40 , 41, 71, 83, 91, 94]
OutputPlot_dir = "/data/users/rserino//CROSS/PlotDataStream/"

# channels_plot = [31]


#### Working Point Info
workingPointList = [0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40, 50] 

# # workingPointList = [0.4, 0.6]
wp_duration = 800
offset = 300



if UseFlaggedTriggers:
    useflaggedtriggers = "--useflaggedtriggers"

if UseMergedTriggers:
    usemergedtriggers = "--usemergedtriggers"

if Overwrite:
    overwrite = "--overwrite"

if SavePlot:
    saveWaveforms = "--saveWaveforms"


helpers.SetChannels(channels_LMO, channels_LD)

detectormapPath = "/data/users/azanelli/Octopus/detectormaps/Cross.toml"
nFloors = 7

def SetVarTypeCrystal(typeC):

    global templateAP, templateANPS, window, pretrigger, deadTime, BufferLength, pulserModule, loadStab

    if typeC == "heat":
        templateAP = "/data/users/ploaiza/Octopus/Octopus_work/RUN000092/AP/Processed_20251211T191513_000092_"
        templateANPS = "/data/users/ploaiza/Octopus/Octopus_work/RUN000089/ANPS/Processed_20251210T182259_000089_"
        loadStab = "/data/users/ploaiza/Octopus/Octopus_work/RUN000092/OFData_0/Processed_20251211T191513_000092_"
        window = 1.0  # in seconds
        pretrigger = 0.5  # in seconds
        deadTime = 0.12
        BufferLength = 0.1
        pulserModule = "Heater"


    if typeC == "light":
        window = 1.0  # in seconds
        pretrigger = 0.5  # in seconds
        #window = 0.4  # in seconds
        #pretrigger = 0.2  # in seconds
        loadStab = "/data/users/rserino/CROSS/RUN000096/OFData/Processed_20251215T103142_000096_"
        deadTime = 0.06
        BufferLength = 0.05
        pulserModule = "LED"

        if NTL is True:
            templateAP = "/data/users/ploaiza/Octopus/Octopus_work/RUN000096/AP/Processed_20251215T103142_000096_"
            templateANPS = "/data/users/ploaiza/Octopus/Octopus_work/RUN000096/ANPS/Processed_20251215T103142_000096_"
        else:
            templateAP = "/data/users/rserino/CROSS/RUN000090/AP/Processed_20251210T221410_000090_"
            templateANPS = "/data/users/rserino/CROSS/RUN000089/ANPS/Processed_20251210T182259_000089_"

    #templateAP = "/data/users/rserino/CROSS/RUN000204/AP/Processed_20260405T115934_000204_"
    #templateANPS = "/data/users/rserino/CROSS/RUN000206/ANPS/Processed_20260407T011059_000206_"


        


shared_config = {
    "typeCrystal": None
}

helpers.set_shared_config(shared_config)

if runList == []:
    print(f"Running runs from {run0} to {runFin}, excluding runs in skipRuns list.")
    runList = [f"{i:06d}" for i in range(int(run0), int(runFin)+1)]

    if run0 == runFin:
        print(f"Running only run {run0}")
        runList = [f"{run0}"]

    if ComputeAll is False:
        runList = [f"{run0}"]

print("Initial run list (before skipping): ", runList)


ListRuns = [item for item in runList if item not in skipRuns]


channel_run_map = {}

for run in runList:

    
  

    helpers.wait_for_slot()


    runNumber = run
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    if not PlotBaselines and not ComputeTriggerPulsers and not ComputeMuonVeto and not ComputeTriggerScan and not ComputeTrigger and not ComputeMerger and not ComputeAnalysis and not ComputeAnalysisScript and not ComputeAP and not ComputeANPS and not ComputeOF and not MergeProduction:
        continue


    ### Global information and path, IF THE DIR IT'S NOT THERE, IT WILL BE CREATED
    OctoBuild_dir = "/mnt/disk1/data/users/azanelli/Octopus_build/"
    RawData_dir = f"/data2/LSC/DATA/RUN14/{runNumber}/"
    prefixRun = helpers.get_random_timestamp(RawData_dir, runNumber, 10)  # just to check that the run exists and has data, and to get a timestamp for the log file name if needed
    
    if prefixRun is None:
        print(f"Error: No data found for run {runNumber} in {RawData_dir}. Skipping this run.")
        continue
    OutputData_dir = f"/mnt/disk1/data/users/azanelli/octopus_work/CROSS/RUN{runNumber}/Processed/"
    OutputOF_dir = f"/mnt/disk1/data/users/azanelli/octopus_work/CROSS/RUN{runNumber}/OFData/"
    OutputAP_dir =  f"/mnt/disk1/data/users/azanelli/octopus_work/CROSS/RUN{runNumber}/AP/"
    OutputANPS_dir =  f"/mnt/disk1/data/users/azanelli/octopus_work/CROSS/RUN{runNumber}/ANPS/"
    CFG_dir = f"/mnt/disk1/data/users/azanelli/octopus_work/CROSS/RUN{runNumber}/cfg/"    ##### Thresholds written by the triggerscan
    triggerScandir = f"/mnt/disk1/data/users/azanelli/octopus_work/CROSS/RUN{runNumber}/TriggerScan/"
    ThresholdTXT = "/data/users/azanelli/Octopus/detectormaps/ThresholdsCROSS.txt"
    SensitivitiesTXT = "/mnt/disk1/data/users/azanelli/octopus_work/CROSS/SensitivitiesCROSS.txt"
    Trigger_dir = f"/data/users/azanelli/octopus_work/CROSS/RUN{runNumber}/Trigger/"
    
    BadInterval_dir = "/data/users/rserino/CROSS/BadInterval.txt"
    coincidence_dir = f"/data/users/azanelli/octopus_work/CROSS/RUN{runNumber}/Coincidence/"
    verbosity = 5

    if run in skipRuns:
        print(f"Skipping run {run} as it's in the skipRuns list.")
        continue

    if run == "000136":
        print("Skipping run 136, it's a load curve run")
        continue

    print(f"Processing run: {runNumber}")
    print(f"Using prefix: {prefixRun}")

    ## Create Directoris if they are not there
    dirs = [OutputData_dir, CFG_dir, triggerScandir, Trigger_dir, OutputAP_dir, OutputOF_dir, coincidence_dir, OutputANPS_dir, OutputPlot_dir]

    for d in dirs:
        if not os.path.exists(d):
            print(f"Creating: {d}")
            os.makedirs(d, exist_ok=True)



    helpers.SetRunOnTerminal(ExecuteOnTerminal)


    # ---------- Begin corrected workflow --------------------------------

    channel_jobs_map = {}


    # --------------------------
    # Step 0 — Pulser trigger
    if PlotBaselines is True:

        print(f"Generating TOML for PlotDataStream for run {runNumber}...")

        pending_merge = [0] # only one iteration for merger

        def make_plot_lines(channel):
            if GenerateCFG is True:

                lines = [
                            f"source {pythonVenv}/bin/activate",
                            f"source {sourceOcto}",
                            f"argsAnal='--rawdir {RawData_dir} --outdirCFG {CFG_dir} --stride {stride} "
                            f"--chan {' '.join(map(str, channels_plot))}  --prefix {prefixRun} --run {runNumber} "
                            f"--startTime {startTime}  --stopTime {stopTime} {saveWaveforms} ",
                            f"--verbosity {verbosity} {overwrite} --outputdir {OutputPlot_dir}'",
                            f"{pythonVenv}/bin/python {SCRIPT_DIR}/PlotBaselines.py $argsAnal"
                        ]
            else:
                lines = [""]

            if ExeOctopus is True:
                lines.append(f"{OctoBuild_dir}/bin/PlotDataStream {CFG_dir}/configPlotData_0.toml")


            lines.append(
                f"""{pythonVenv}/bin/python -c "import sys; sys.path.append('{SCRIPT_DIR}'); import helpers; print('PlotDataStream completed. Generating HTML files in dir {OutputPlot_dir}'); helpers.generate_html('{OutputPlot_dir}', '{OutputPlot_dir}/PlotDataStream_{runNumber}.html', '{runNumber}'); print('Generating HTML complete. Files in dir {OutputPlot_dir}')"
            """
            )

            lines.append(f"rm -f {OutputPlot_dir}/PlotDataStream_Run{runNumber}.root")

            return lines

        submitted = helpers.try_submit_pass(pending_merge, make_plot_lines, channel_jobs_map, None, False, False)
        


    # --------------------------
    # Step 0 — Pulser trigger
    if ComputeTriggerPulsers is True:
        # Trigger Heater
        channelSelected = 1  # LED
        pulserModule = "Heater"

        if GenerateCFG is True:
            pulser_lines = [
                f"source {pythonVenv}/bin/activate",
                f"source {sourceOcto}",
                f"argsTrigPuls='--rawdir {RawData_dir} --triggerdir {Trigger_dir} "
                f"--DeadTimeList {deadTime} --BufferLengthList {BufferLength} "
                f"--channel {channelSelected} --prefix {prefixRun} --run {runNumber} --NRMSList 8.0 "
                f"--outdirCFG {CFG_dir} --NoisePeriodList 10000000 --verbosity {verbosity} "
                f"--UseFixedThreshold --FixedThreshold {FixedThreshold} "
                f"--triggertype {triggerTypePuls} --pulsertype {pulserModule} --isPulser'",
                f"{pythonVenv}/bin/python {SCRIPT_DIR}/ExeTrigger.py $argsTrigPuls"]
        else:
            pulser_lines = [""]

        if ExeOctopus is True:
            pulser_lines.append(f"{OctoBuild_dir}/bin/Trigger {CFG_dir}/configTrigg_{channelSelected}.toml")

        pulser_script = helpers.create_sh(pulser_lines)
        os.chmod(pulser_script, 0o755)

        # Option 1: run locally and then submit (I preserved your subprocess.run if you want immediate local run)
        # If you want only to submit to qsub, comment out the next line.
        # subprocess.run([pulser_script])

        # Submit pulser job to the queue and track it
        pulser_status = helpers.submit_job(channelSelected, pulser_script, channel_jobs_map)
        if pulser_status is True:
            print(f"Heater trigger script submitted for channel {channelSelected} (job {channel_jobs_map[channelSelected]})")
        elif pulser_status == "error":
            print("Heater trigger submission failed; check qsub output.")
        else:
            print("Heater trigger job already running; not submitting another.")

        
        # Trigger LED

        pulserModule = "LED"
        channelSelected = 2  # LED
        FixedThreshold = 1.2

        if GenerateCFG is True:
            pulser_lines = [
                f"source {pythonVenv}/bin/activate",
                f"source {sourceOcto}",
                f"argsTrigPuls='--rawdir {RawData_dir} --triggerdir {Trigger_dir} "
                f"--DeadTimeList {deadTime} --BufferLengthList {BufferLength} "
                f"--channel {channelSelected} --prefix {prefixRun} --run {runNumber} --NRMSList 8.0 "
                f"--outdirCFG {CFG_dir} --NoisePeriodList 10000000 --verbosity {verbosity} "
                f"--UseFixedThreshold --FixedThreshold {FixedThreshold} "
                f"--triggertype {triggerTypePuls} --pulsertype {pulserModule} --isPulser'",
                f"{pythonVenv}/bin/python {SCRIPT_DIR}/ExeTrigger.py $argsTrigPuls"]
        else:
            pulser_lines = [""]

        if ExeOctopus is True:
            pulser_lines.append(f"{OctoBuild_dir}/bin/Trigger {CFG_dir}/configTrigg_{channelSelected}.toml")

        pulser_script = helpers.create_sh(pulser_lines)
        os.chmod(pulser_script, 0o755)

        # Option 1: run locally and then submit (I preserved your subprocess.run if you want immediate local run)
        # If you want only to submit to qsub, comment out the next line.
        # subprocess.run([pulser_script])

        # Submit pulser job to the queue and track it
        pulser_status = helpers.submit_job(channelSelected, pulser_script, channel_jobs_map)
        if pulser_status is True:
            print(f"LED trigger script submitted for channel {channelSelected} (job {channel_jobs_map[channelSelected]})")
        elif pulser_status == "error":
            print("LED trigger submission failed; check qsub output.")
        else:
            print("LED trigger job already running; not submitting another.")


    if ComputeMuonVeto is True:

        if GenerateCFG is True:
            pulser_lines = [
                f"source {pythonVenv}/bin/activate",
                f"source {sourceOcto}",
                f"argsTrigPuls='--rawdir {RawData_dir} --triggerdir {Trigger_dir} "
                f"--channel {channelMuon} --prefix {prefixRun} --run {runNumber} "
                f"--outdirCFG {CFG_dir} --verbosity {verbosity} "
                f"--triggertype {triggerTypeMuon} --isVeto '",
                f"{pythonVenv}/bin/python {SCRIPT_DIR}/ExeTrigger.py $argsTrigPuls"]
        else:
            pulser_lines = [""]

        if ExeOctopus is True:
            pulser_lines.append(f"{OctoBuild_dir}/bin/Trigger {CFG_dir}/configTrigg_{channelMuon}.toml")

        pulser_script = helpers.create_sh(pulser_lines)
        os.chmod(pulser_script, 0o755)

        # Option 1: run locally and then submit (I preserved your subprocess.run if you want immediate local run)
        # If you want only to submit to qsub, comment out the next line.
        # subprocess.run([pulser_script])

        # Submit pulser job to the queue and track it
        pulser_status = helpers.submit_job(channelMuon, pulser_script, channel_jobs_map)
        if pulser_status is True:
            print(f"Muon veto script submitted for channel {channelMuon} (job {channel_jobs_map[channelMuon]})")
        elif pulser_status == "error":
            print("Muon veto submission failed; check qsub output.")
        else:
            print("Muon veto job already running; not submitting another.")

    # --------------------------
    # # Step 1 — Trigger Scan (channels_LD)
    if ComputeTriggerScan is True:
        pending_trigger_scan = channels_WP.copy()

        print("Submitting trigger scan jobs for channels: ", pending_trigger_scan)

        def make_triggerscan_lines(channel):

            if GenerateCFG is True:

                typeCrystal = shared_config["typeCrystal"]
                SetVarTypeCrystal(typeCrystal)
                
                lines = [
                    f"source {pythonVenv}/bin/activate",
                    f"source {sourceOcto}",
                    f"argsTrig='--rawdir {RawData_dir} --NAboveFin {NAboveFin}  "
                    f"--DeadTimeList {deadTime} --BufferLengthList {BufferLength} "
                    f"--channel {channel} --prefix {prefixRun} --run {runNumber} "
                    f"--outdirCFG {CFG_dir}  --verbosity {verbosity} --triggerdir {Trigger_dir} "
                    f"--startThresh {startThresh} --endThresh {endThresh} "
                    f"--Step {Step} --TimeScan {TimeScan} --NAboveInit {NAboveInit} "
                    f"--triggertype {triggerTyp} --triggerScandir {triggerScandir}'",
                    f"{pythonVenv}/bin/python {SCRIPT_DIR}/ExeTriggerScan.py $argsTrig"
                    ]
            else:
                lines = [""]

            if ExeOctopus is True:
                lines.append(f"{OctoBuild_dir}/bin/TriggerScan {CFG_dir}/configTriggScan_{channel}.toml")

            return lines

        # Try submitting trigger-scan jobs in passes until all submitted
        while pending_trigger_scan:
            submitted = helpers.try_submit_pass(pending_trigger_scan, make_triggerscan_lines, channel_jobs_map)
            if submitted == 0:
                # nothing new submitted this pass: either all are already running or qsub errored
                # sleep a bit and retry, so slow channels won't block others
                time.sleep(5)

    # --------------------------
    # #Step 1b — Trigger channels (from results of scan)
    if ComputeTrigger is True:
        pending_trigger_channels = channels_WP.copy()

        print("Submitting trigger jobs for channels: ", pending_trigger_channels)

        def make_trigger_lines(channel):

            if GenerateCFG is True:

                typeCrystal = shared_config["typeCrystal"]
                SetVarTypeCrystal(typeCrystal)

                lines = [
                    f"source {pythonVenv}/bin/activate",
                    f"source {sourceOcto}",
                    f"argsTrig='--rawdir {RawData_dir} --triggerdir {Trigger_dir} "
                    f"--DeadTimeList {deadTime} --BufferLengthList {BufferLength} "
                    f"--channel {channel} --prefix {prefixRun} --run {runNumber} {overwrite} "
                    f"--outdirCFG {CFG_dir} --NoisePeriodList {noisePeriod} --verbosity {verbosity} "
                    f"--triggertype {triggerTyp} --NRMSList {NRMSList} --ReadFromTxt {ThresholdTXT}'",   #
                    f"{pythonVenv}/bin/python {SCRIPT_DIR}/ExeTrigger.py $argsTrig"
                    ]
            else:
                lines = [""]

            if ExeOctopus is True:
                lines.append(f"{OctoBuild_dir}/bin/Trigger {CFG_dir}/configTrigg_{channel}.toml")

            return lines

        while pending_trigger_channels:
            submitted = helpers.try_submit_pass(pending_trigger_channels, make_trigger_lines, channel_jobs_map)
            if submitted == 0:
                time.sleep(5)

    # # --------------------------
    # # Step 2 — Merge channels (channels_WP)
    if ComputeMerger is True:
        helpers.wait_for_all_jobs(channel_jobs_map)
        pending_merge = [0] # only one iteration for merger

        print("Submitting trigger merger. ")

        # print(" Ch heater: ",channelHeater)
        # print(" Ch LED: ",channelLED)

        lmo_str = " ".join(map(str, channels_LMO))
        ld_str  = " ".join(map(str, channels_LD))


        if PropagateFlag is True:
            propagate = "--propagateflag"
        else:
            propagate = ""

        # propagate = "--propagateflag"

        def make_merge_lines(channel):
            if GenerateCFG is True:

                typeCrystal = shared_config["typeCrystal"]
                SetVarTypeCrystal(typeCrystal)

                lines = [
                    f"source {pythonVenv}/bin/activate",
                    f"source {sourceOcto}",
                    f"argsMerge='--rawdir {RawData_dir} --triggerdir {Trigger_dir} "
                    f"--ch {channel} --channelHeater {channelHeater} --channelLED {channelLED} "
                    f"--run {runNumber} --outdirCFG {CFG_dir} --lmo {lmo_str} --ld {ld_str} "
                    f"--verbosity {verbosity} --prefix {prefixRun} {propagate} --detectormapPath {detectormapPath} '",
                    f"{pythonVenv}/bin/python {SCRIPT_DIR}/ExeMerger.py $argsMerge"
                    ]
            else:
                lines = [""]

            if ExeOctopus is True:
                lines.append(f"{OctoBuild_dir}/bin/TriggerMerger {CFG_dir}/configMerger_{channel}.toml")

            return lines

        while pending_merge:
            submitted = helpers.try_submit_pass(pending_merge, make_merge_lines, channel_jobs_map)
            if submitted == 0:
                time.sleep(5)

    # --------------------------
    # Step 3 — Analyse Workin Points (channels_WP)
    if ComputeAnalysis is True:
        pending_analysis = channels_WP.copy()
        analyse_channel_jobs = []  # collect job ids if you want

        print("Submitting analysis jobs for channels: ", pending_analysis)


        def make_analysis_lines(channel):
            if GenerateCFG is True:

                typeCrystal = shared_config["typeCrystal"]
                SetVarTypeCrystal(typeCrystal)

                lines = [
                    f"source {pythonVenv}/bin/activate",
                    f"source {sourceOcto}",
                    f"argsAnal='--rawdir {RawData_dir} --triggerdir {Trigger_dir} "
                    f"--chan {channel} --prefix {prefixRun} --run {runNumber} "
                    f"--outdirCFG {CFG_dir} --processeddir {OutputData_dir} --type {typeCrystal} "
                    f"--windowlength {window} --pretrigger {pretrigger} --gain {Gain} "
                    f"--workingpoints {' '.join(map(str, workingPointList))}  "
                    f"--wp_duration {wp_duration} --offset {offset} "
                    f"--verbosity {verbosity} {usemergedtriggers} {useflaggedtriggers} {overwrite}'",
                    f"{pythonVenv}/bin/python {SCRIPT_DIR}/ExeModuleWP.py $argsAnal"
                    ]
            else:
                lines = [""]

            if ExeOctopus is True:
                lines.append(f"{OctoBuild_dir}/bin/Octopus {CFG_dir}/configModule_{channel}.toml")

            return lines

            

        while pending_analysis:
            # for analysis jobs I'd suggest respecting slot availability before creating & submitting
            submitted = helpers.try_submit_pass(pending_analysis, make_analysis_lines, channel_jobs_map, append_jobids_list=analyse_channel_jobs, sleep_on_slot=True)
            if submitted == 0:
                time.sleep(5)

    if ComputeAP is True:
        pending_analysis = channels_WP.copy()
        analyse_channel_jobs = []  # collect job ids if you want

        print("Submitting AP jobs for channels: ", pending_analysis)

        def make_analysis_lines(channel):
            if GenerateCFG is True:

                typeCrystal = shared_config["typeCrystal"]
                SetVarTypeCrystal(typeCrystal)

                lines = [
                    f"source {pythonVenv}/bin/activate",
                    f"source {sourceOcto}",
                    f"argsAnal='--rawdir {RawData_dir} --triggerdir {Trigger_dir} "
                    f"--chan {channel} --prefix {prefixRun} --run {runNumber} "
                    f"--outdirCFG {CFG_dir} --processeddir {OutputAP_dir} "
                    f"--windowlength {window} --pretrigger {pretrigger} --gain {Gain} "
                    f"--namePulser {pulserModule}  --BadIntervalPath {BadInterval_dir}  "
                    f"--verbosity {verbosity} {usemergedtriggers} {useflaggedtriggers} {overwrite}'",
                    f"{pythonVenv}/bin/python {SCRIPT_DIR}/ExeModuleAP.py $argsAnal"
                    ]
            else:
                lines = [""]

            if ExeOctopus is True:
                lines.append(f"{OctoBuild_dir}/bin/Octopus {CFG_dir}/configModuleAP_{channel}.toml")

            return lines

        while pending_analysis:
            # for analysis jobs I'd suggest respecting slot availability before creating & submitting
            submitted = helpers.try_submit_pass(pending_analysis, make_analysis_lines, channel_jobs_map, append_jobids_list=analyse_channel_jobs, sleep_on_slot=True)
            if submitted == 0:
                time.sleep(5)

    if ComputeANPS is True:
        pending_analysis = channels_WP.copy()
        analyse_channel_jobs = []  # collect job ids if you want

        print("Submitting ANPS jobs for channels: ", pending_analysis)

        def make_analysis_lines(channel):
            if GenerateCFG is True:

                typeCrystal = shared_config["typeCrystal"]
                SetVarTypeCrystal(typeCrystal)

                lines = [
                    f"source {pythonVenv}/bin/activate",
                    f"source {sourceOcto}",
                    f"argsAnal='--rawdir {RawData_dir} --triggerdir {Trigger_dir} "
                    f"--chan {channel} --prefix {prefixRun} --run {runNumber} "
                    f"--outdirCFG {CFG_dir} --processeddir {OutputANPS_dir} "
                    f"--windowlength {window} --pretrigger {pretrigger} --gain {Gain} "
                    f" --BadIntervalPath {BadInterval_dir} "
                    f"--verbosity {verbosity} {usemergedtriggers} {useflaggedtriggers} {overwrite}'",
                    f"{pythonVenv}/bin/python {SCRIPT_DIR}/ExeModuleANPS.py $argsAnal"
                    ]
            else:
                lines = [""]

            if ExeOctopus is True:
                lines.append(f"{OctoBuild_dir}/bin/Octopus {CFG_dir}/configModuleANPS_{channel}.toml")

            return lines

        while pending_analysis:
            # for analysis jobs I'd suggest respecting slot availability before creating & submitting
            submitted = helpers.try_submit_pass(pending_analysis, make_analysis_lines, channel_jobs_map, append_jobids_list=analyse_channel_jobs, sleep_on_slot=True)
            if submitted == 0:
                time.sleep(5)



    if ComputeOF is True:
        pending_analysis = channels_WP.copy()
        analyse_channel_jobs = []  # collect job ids if you want

        print("Submitting OF jobs for channels: ", pending_analysis)

        def make_analysis_lines(channel):
            if GenerateCFG is True:

                typeCrystal = shared_config["typeCrystal"]
                SetVarTypeCrystal(typeCrystal)

                lines = [
                    f"source {pythonVenv}/bin/activate",
                    f"source {sourceOcto}",
                    f"argsAnal='--rawdir {RawData_dir} --triggerdir {Trigger_dir} "
                    f"--chan {channel} --prefix {prefixRun} --run {runNumber} "
                    f"--outdirCFG {CFG_dir} --processeddir {OutputOF_dir} --type {typeCrystal} "
                    f"--windowlength {window} --pretrigger {pretrigger} --gain {Gain} --sensitivityTXT {SensitivitiesTXT} "
                    f" --BadIntervalPath {BadInterval_dir} --typeRun {typeRun} --loadstab {loadStab} "
                    f"--templateAP {templateAP} --templateANPS {templateANPS} --heaterless {' '.join(map(str, channels_heaterless))} "
                    f"--verbosity {verbosity} {usemergedtriggers} {useflaggedtriggers} {overwrite}'",
                    f"{pythonVenv}/bin/python {SCRIPT_DIR}/ExeModuleOF.py $argsAnal"
                    ]
            else:
                lines = [""]

            if ExeOctopus is True:
                lines.append(f"{OctoBuild_dir}/bin/Octopus {CFG_dir}/configModuleOF_{channel}.toml")

            return lines

        while pending_analysis:
            # for analysis jobs I'd suggest respecting slot availability before creating & submitting
            submitted = helpers.try_submit_pass(pending_analysis, make_analysis_lines, channel_jobs_map, append_jobids_list=analyse_channel_jobs, sleep_on_slot=True)
            if submitted == 0:
                time.sleep(5)


    if MergeProduction is True:
        helpers.wait_for_all_jobs(channel_jobs_map)
        pending_analysis = channels_WP.copy()
        analyse_channel_jobs = []  # collect job ids if you want

        print("Submitting Merging jobs for channels: ", pending_analysis)
        
        neighbor_map = {} 
        # Main execution
        channel_vector = helpers.get_channel_vector(detectormapPath)
        neighbor_map = helpers.build_neighbor_map(channel_vector, nFloors)
        neighborCh = []

        # print(neighbor_map)

        for ch in pending_analysis:
            neighborCh = neighbor_map[ch]

            if len(neighborCh) == 1:
                neighborCh.append(-1)         # Non existing channel, the macro will handle it
        

            lines = [f"source {pythonVenv}/bin/activate", f"source {sourceOcto}"]
            heat_ch = f"{OutputOF_dir}/Processed_{prefixRun}_{runNumber}_{ch}.root"
            LD1_ch = f"{OutputOF_dir}/Processed_{prefixRun}_{runNumber}_{neighborCh[0]}.root"
            LD2_ch = f"{OutputOF_dir}/Processed_{prefixRun}_{runNumber}_{neighborCh[1]}.root"
            output = f"{coincidence_dir}/{ch}_{neighborCh[0]}_{neighborCh[1]}_{runNumber}.root"

            if neighborCh[1] == -1:
                output = f"{coincidence_dir}/{ch}_{neighborCh[0]}_{runNumber}.root"


            if ExeOctopus is True:
                lines.append(f"{pythonVenv}/bin/python {SCRIPT_DIR}/MergeProduction.py {heat_ch} {LD1_ch} {LD2_ch} {output}")

            merge_script = helpers.create_sh(lines)
            os.chmod(merge_script, 0o755)

            string_ch = ""

            if neighborCh[1] != -1:
                string_ch = f"{ch}_{neighborCh[0]}_{neighborCh[1]}"
            else:
                string_ch = f"{ch}_{neighborCh[0]}"


            if ExecuteOnTerminal is True:
                if neighborCh[1] != -1:
                    print(f"Computing channels {string_ch}")
                else:
                    print(f"Computing channels {string_ch}")
                subprocess.run([merge_script])
                continue
                
            
            # Submit new job
            try:
                cmd = f"qsub -q cupid -l mem=2G {merge_script}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            except Exception as e:
                print(f"[ERROR] Exception when running qsub for channels {string_ch}: {e}")

            if result.returncode != 0:
                print(f"[ERROR] qsub failed for channels {string_ch}. stderr:\n{result.stderr}")

            full_jobid = result.stdout.strip()
            jobid = full_jobid.split(".")[0] if full_jobid else ""

            if not jobid.isdigit():
                print(f"[ERROR] qsub returned invalid job id for channels {string_ch}: '{full_jobid}'")

            print(f"[OK] Submitted channels {string_ch} with job ID {jobid}")
            



    # # # # # # --------------------------
    # # Step 4 — Final Result Analysis
    # # Optionally wait for all jobs we submitted:
    if ComputeAnalysisScript is True:
        # helpers.wait_for_all_jobs(channel_jobs_map)

        typeCrystal = shared_config["typeCrystal"]
        SetVarTypeCrystal(typeCrystal)


        lines = [
            f"source {pythonVenv}/bin/activate",
            f"{pythonVenv}/bin/python AnalyseResult.py --inputdir {OutputData_dir} "
            f"--outputdir {OutputData_dir} --run {runNumber} "
            f"--channels {' '.join(map(str, channels_WP))} "
            f"--workingpoints {' '.join(map(str, workingPointList))} "
            f"--wp_duration {wp_duration} --offset {offset} --outputdiranps {OutputANPS_dir} "
            f"--outputdirap {OutputAP_dir}"
        ]
        sh_file = helpers.create_sh(lines)
        os.chmod(sh_file, 0o755)
        subprocess.run([sh_file])

    

    # # # --------------------------
    # # # Start watchdog
    # # threading.Thread(target=watchdog,
    # #                  args=(analyse_channel_jobs,
    # #                        KILL_TIMEOUT),
    # #                  daemon=True).start()

    # # print("\nAll jobs submitted successfully!")
    # # print("Watchdog started in background.\n")




if ComputeMergeRuns is True:

    helpers.wait_for_all_jobs(channel_run_map)

    run_str = " ".join(ListRuns)

    neighbor_map = {} 
    # Main execution
    channel_vector = helpers.get_channel_vector(detectormapPath)
    neighbor_map = helpers.build_neighbor_map(channel_vector, nFloors)

    print("Submitting merge runs jobs for channels: ", channels_WP)

    for ch in channels_WP:
        neighborCh = neighbor_map[ch]

        lines = [
            f"source {pythonVenv}/bin/activate",
            f"source {sourceOcto}"
        ]


        if len(neighborCh) == 1:
            neighborCh.append(-1)         # Non existing channel, the macro will handle it

        if neighborCh[1] != -1:
            string_ch = f"{ch}_{neighborCh[0]}_{neighborCh[1]}"
        else:
            string_ch = f"{ch}_{neighborCh[0]}"
    
        lines = [f"source {pythonVenv}/bin/activate", f"source {sourceOcto}"]
        LD1_ch = neighborCh[0]
        LD2_ch = neighborCh[1]

        lines.append(
            f"{pythonVenv}/bin/python {SCRIPT_DIR}/MergeRunsUnbin.py "
            f"--cross {path_cross} "
            f"--runs {run_str} "
            f"--channel {ch} "
            f"--channelLD1 {LD1_ch} "
            f"--channelLD2 {LD2_ch} "
            f"--outdir {OutputDir} "
            f"--outPlot {plotLY}"
        )

        merge_script = helpers.create_sh(lines)
        os.chmod(merge_script, 0o755)

        # ----------------------------------
        # Run locally in terminal
        # ----------------------------------
        if ExecuteOnTerminal is True:
            print(f"Merging runs for channels {string_ch}")
            subprocess.run([merge_script])
            continue

        # ----------------------------------
        # Submit to farm
        # ----------------------------------
        try:
            cmd = f"qsub -q cupid -l mem=3G {merge_script}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        except Exception as e:
            print(f"[ERROR] Exception submitting channel {string_ch}: {e}")
            continue

        if result.returncode != 0:
            print(f"[ERROR] qsub failed for channels {string_ch}\n{result.stderr}")
            continue

        full_jobid = result.stdout.strip()
        jobid = full_jobid.split(".")[0] if full_jobid else ""

        if not jobid.isdigit():
            print(f"[ERROR] invalid job id for channel {string_ch}: '{full_jobid}'")
            continue

        print(f"[OK] Submitted merge for channel {string_ch} with job ID {jobid}")
