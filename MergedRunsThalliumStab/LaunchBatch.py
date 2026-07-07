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
print(f"DEBUG: Sto usando il file helpers situato in: {helpers.__file__}")

# Environment varibles
pythonVenv = "/mnt/disk1/home/zanelli/env/"  # path where the python venv is located ( with all the needed libraries installed )
sourceOcto = "/mnt/disk1/home/zanelli/LoadOctopus.sh"  # script to load octopus environment

# --------------------------
# # Channel Configuration
channels_LD = [43, 44, 45, 46, 47, 48, 67, 68, 69, 70, 71, 72, 31, 32, 33, 34, 35, 36, 73, 74, 75, 76, 77, 78, 37, 38, 39, 40, 41, 42, 79, 80, 81, 82, 83, 84, 91, 92, 93, 94, 95, 96]  # List of the Light Detector CROSSS
channels_LMO = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 49, 50, 51, 52, 53, 54, 25, 26, 27, 28, 29, 30, 55, 56, 57, 58, 59, 60, 85, 86, 87, 88, 89, 90, 61, 62, 63, 64, 65, 66] # List of the Heat Detector CROSSS  
channels_LMO = [16, 17, 18, 19, 20, 21, 22, 23, 24, 49, 50, 51, 52, 53, 54, 25, 26, 27, 28, 29, 30, 55, 56, 57, 58, 59, 60, 85, 86, 87, 88, 89, 90, 61, 62, 63, 64, 65, 66] # List of the Heat Detector CROSSS  


channelMuon = 65552
channelHeater = 1
channelLED = 2

# channels_LMO = [19, 20, 21, 22, 23, 24, 49, 50, 51, 52, 53, 54] #first tower 1
# channels_LD = [31, 32, 33, 34, 35, 36, 67, 68, 69, 70, 71, 72] #first tower


channels_WP =  channels_LMO + channels_LD
channels_WP = [20] 

## Run 
# runNumber = "000058"

# runNumber = "000089"

# runNumber = "000090"

# runNumber = "000092"

# runNumber = "000096"

#runNumber = "000097"

runNumber = "000102"

#runNumber = "000103"

#runNumber = "000104"

#runNumber = "000105"


#runNumber = "000107"

# runNumber = "000116"


#runNumber = "000136"



NTL = False


ComputeTriggerPulsers = False
ComputeMuonVeto = False

ComputeTriggerScan = False
ComputeTrigger = False

ComputeMerger = False

ComputeAnalysis = False
ComputeAnalysisScript = False

ComputeAP = False
ComputeANPS = False
ComputeOF = False      # And calibration

MergeProduction = False

UseMergedTriggers = False
UseFlaggedTriggers = False
Overwrite = False

ComputeMergeRuns = True
ListRuns = ["102", "103", "104", "105"]
OutputDir = "/data/users/azanelli/octopus_work/MergedRunsThalliumStab"
path_cross = "/data/users/rserino/CROSS"


GenerateCFG = False 
ExecuteOnTerminal = True 


typeCrystal = "heat"
typeRun = "B"
overwrite = ""
usemergedtriggers = ""   
useflaggedtriggers = ""

if UseFlaggedTriggers:
    useflaggedtriggers = "--useflaggedtriggers"

if UseMergedTriggers:
    usemergedtriggers = "--usemergedtriggers"

if Overwrite:
    overwrite = "--overwrite"


# 1: Heater, 2: LED   

# Just initialising values
templateAP = ""
templateANPS = ""
window = 0.0  # in seconds
pretrigger = 0.0  # in seconds
deadTime = 0.0
BufferLength = 0.0


helpers.SetChannels(channels_LMO, channels_LD)

def SetVarTypeCrystal(typeC):

    global templateAP, templateANPS, window, pretrigger, deadTime, BufferLength

    if typeC == "heat":
        templateAP = "/data/users/rserino/CROSS/RUN000092/AP/Processed_20251211T191513_000092_"
        templateANPS = "/data/users/rserino/CROSS/RUN000089/ANPS/Processed_20251210T182259_000089_"
        window = 1.0  # in seconds
        pretrigger = 0.5  # in seconds
        deadTime = 0.02
        BufferLength = 0.1
        


    if typeC == "light":
        window = 1.0  # in seconds
        pretrigger = 0.5  # in seconds
        deadTime = 0.01
        BufferLength = 0.05
        

        if NTL is True:
            templateAP = "/data/users/rserino/CROSS/RUN000096/AP/Processed_20251215T103142_000096_"
            templateANPS = "/data/users/rserino/CROSS/RUN000096/ANPS/Processed_20251215T103142_000096_"
        else:
            templateAP = "/data/users/rserino/CROSS/RUN000090/AP/Processed_20251210T221410_000090_"
            templateANPS = "/data/users/rserino/CROSS/RUN000089/ANPS/Processed_20251210T182259_000089_"




shared_config = {
    "typeCrystal": None
}

helpers.set_shared_config(shared_config)
    



### Global information and path, IF THE DIR IT'S NOT THERE, IT WILL BE CREATED
OctoBuild_dir = "/mnt/disk1/data/users/rserino/Octopus-build/"
RawData_dir = f"/data2/LSC/DATA/RUN14/{runNumber}/"
prefixRun = helpers.get_random_timestamp(RawData_dir, runNumber)  # just to check that the run exists and has data, and to get a timestamp for the log file name if needed
print(f"Using prefix: {prefixRun}")
OutputData_dir = f"/mnt/disk1/data/users/rserino//CROSS/RUN{runNumber}/Processed/"
OutputOF_dir = f"/mnt/disk1/data/users/rserino//CROSS/RUN{runNumber}/OFData/"
OutputAP_dir =  f"/mnt/disk1/data/users/rserino/CROSS/RUN{runNumber}/AP/"
OutputANPS_dir =  f"/mnt/disk1/data/users/rserino/CROSS/RUN{runNumber}/ANPS/"
CFG_dir = f"/mnt/disk1/data/users/rserino/CROSS/RUN{runNumber}/cfg/"    ##### Thresholds written by the triggerscan
triggerScandir = f"/mnt/disk1/data/users/rserino/CROSS/RUN{runNumber}/TriggerScan/"
ThresholdTXT = "/data/users/rserino/Octopus/detectormaps/ThresholdsCROSS.txt"
Trigger_dir = f"/data/users/rserino/CROSS/RUN{runNumber}/Triggered/"
BadInterval_dir = "/data/users/rserino/CROSS/BadInterval.txt"
detectormapPath = "/data/users/rserino/Octopus/detectormaps/Cross.toml"
coincidence_dir = f"/data/users/rserino//CROSS/RUN{runNumber}/Coincidence/"
verbosity = 5
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
nFloors = 7


## Create Directoris if they are not there
dirs = [OutputData_dir, CFG_dir, triggerScandir, Trigger_dir, OutputAP_dir, OutputOF_dir, coincidence_dir]

for d in dirs:
    if not os.path.exists(d):
        print(f"Creating: {d}")
        os.makedirs(d, exist_ok=True)
    else:
        print(f"Exists:   {d}")


## Trigger Variables
triggerTyp = "linearinterpolation"
triggerTypePuls = "derivative"
triggerTypeMuon = "MuonCROSS"
noisePeriod = 2.0123456
NRMSList = 1.6
FixedThreshold = 3.5 # in Volts


Gain = 910



## Trigger Scan Variables
startThresh = 1.5
endThresh = 8.0
Step = 0.05
TimeScan = 500.0
NAboveInit = 4
NAboveFin = 9


# ### Working Point Info
workingPointList = [0.3, 0.6, 1.0, 1.4, 1.8, 2, 3, 4, 5, 6, 8, 10, 20, 26, 30, 40, 50]
# # workingPointList = [0.4, 0.6]
wp_duration = 900
offset = 1250


helpers.SetRunOnTerminal(ExecuteOnTerminal)


# ---------- Begin corrected workflow --------------------------------

channel_jobs_map = {}

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

    def make_trigger_lines(channel):

        if GenerateCFG is True:

            typeCrystal = shared_config["typeCrystal"]
            SetVarTypeCrystal(typeCrystal)

            lines = [
                f"source {pythonVenv}/bin/activate",
                f"source {sourceOcto}",
                f"argsTrig='--rawdir {RawData_dir} --triggerdir {Trigger_dir} "
                f"--DeadTimeList {deadTime} --BufferLengthList {BufferLength} "
                f"--channel {channel} --prefix {prefixRun} --run {runNumber} "
                f"--outdirCFG {CFG_dir} --NoisePeriodList {noisePeriod} --verbosity {verbosity} "
                f"--triggertype {triggerTyp} --NRMSList {NRMSList} --ReadFromTxt {ThresholdTXT}'",   #
                f"{pythonVenv}/bin/python {SCRIPT_DIR}/ExeTrigger.py $argsTrig"
                ]
        else:
            lines = [""]

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

    # print(" Ch heater: ",channelHeater)
    # print(" Ch LED: ",channelLED)

    lmo_str = " ".join(map(str, channels_LMO))
    ld_str  = " ".join(map(str, channels_LD))

    # if PropagateFlag is True:
    #     propagate = "--propagateflag"
    # else:
    #     propagate = ""

    propagate = "--propagateflag"

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
                f"--windowlength {window} --pretrigger {pretrigger} --gain {Gain} "
                f" --BadIntervalPath {BadInterval_dir} --typeRun {typeRun} "
                f"--templateAP {templateAP} --templateANPS {templateANPS} "
                f"--verbosity {verbosity} {usemergedtriggers} {useflaggedtriggers} {overwrite}'",
                f"{pythonVenv}/bin/python {SCRIPT_DIR}/ExeModuleOF.py $argsAnal"
                ]
        else:
            lines = [""]

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
    
    neighbor_map = {} 
    # Main execution
    channel_vector = helpers.get_channel_vector(detectormapPath)
    neighbor_map = helpers.build_neighbor_map(channel_vector, nFloors)
    neighborCh = []

    # print(neighbor_map)

    for ch in channels_WP:
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
            sys.exit()  # exits the script here
        
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
        


if ComputeMergeRuns is True:

    helpers.wait_for_all_jobs(channel_jobs_map)

    run_str = " ".join(ListRuns)

    neighbor_map = {} 
    # Main execution
    channel_vector = helpers.get_channel_vector(detectormapPath)
    neighbor_map = helpers.build_neighbor_map(channel_vector, nFloors)

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
        output = f"{coincidence_dir}/{string_ch}_{run_str}.root"

        lines.append(
            f"{pythonVenv}/bin/python {SCRIPT_DIR}/MergeRuns.py "
            f"--cross {path_cross} "
            f"--runs {run_str} "
            f"--channel {ch} "
            f"--channelLD1 {LD1_ch} "
            f"--channelLD2 {LD2_ch} "
            f"--outdir {OutputDir}"
        )

        merge_script = helpers.create_sh(lines)
        os.chmod(merge_script, 0o755)

        # ----------------------------------
        # Run locally in terminal
        # ----------------------------------
        if ExecuteOnTerminal is True:
            print(f"Merging runs for channels {string_ch}")
            subprocess.run([merge_script])
            sys.exit()

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


# # # # # # --------------------------
# # Step 4 — Final Result Analysis
# # Optionally wait for all jobs we submitted:
# if ComputeAnalysisScript is True:
#     helpers.wait_for_all_jobs(channel_jobs_map)

#     lines = [
#         f"source {pythonVenv}/bin/activate",
#         f"{pythonVenv}/bin/python AnalyseResult.py --inputdir {OutputData_dir} "
#         f"--outputdir {OutputData_dir} --run {runNumber} "
#         f"--channels {' '.join(map(str, channels_WP))} "
#         f"--workingpoints {' '.join(map(str, workingPointList))} "
#         f"--wp_duration {wp_duration} --offset {offset} --outputdiranps {OutputANPS_dir}"
#         f"--pulsertype {pulserModule} --outputdirap {OutputAP_dir}"
#     ]
#     sh_file = helpers.create_sh(lines)
#     os.chmod(sh_file, 0o755)
#     subprocess.run([sh_file])

# # # --------------------------
# # # Start watchdog
# # threading.Thread(target=watchdog,
# #                  args=(analyse_channel_jobs,
# #                        KILL_TIMEOUT),
# #                  daemon=True).start()

# # print("\nAll jobs submitted successfully!")
# # print("Watchdog started in background.\n")
