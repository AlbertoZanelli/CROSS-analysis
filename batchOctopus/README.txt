# Author: Roberto Serino
# Date: 2025-11-20

Do not modify the file in the Octopus directory, make a copy in your build.

The scripts are created for CROSS batch analysis, you can customise them if you want to use them
for other detectors.

If the dir are not there they will be created

You can modify the trigger/merger/module/templates option adding parameter when you call the 
trigger/merger/module (look in the ExeModuleOF/ExeModuleAP/ExeModuleANPS/ExeModule/ExeMerger/ExeTrigger) script how to pass the variables.

To launch all the chain you only need to set some environment variables, put true
to the things you want to compute and then LaunchWP.py and then execute:

    python3 LaunchWP.py 
