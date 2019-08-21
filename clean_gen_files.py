import sys
import os
# Deletes all files generated from previous executions of our algorithms
# If called with -a, also deletes all generated plots and .csv files

if len(sys.argv) >= 2:
    if(sys.argv[1] == "-a"):
        if [f for f in os.listdir("plots") if not f.startswith('.')] == []:
            print("No plots or .csv files to be deleted")
        else :
            if os.system('rm plots/*') == 0:
                print("Deleted all plots and .csv files")
            else:
                print("Error while deleting plot files")

if [f for f in os.listdir("_tmp") if not f.startswith('.')] == []:
    print("No temporary dataset files to be deleted")
else :
    if os.system('rm _tmp/*') == 0:
        print("Deleted all temporary dataset files")
    else:
        print("Error while deleting temporary dataset files")

if [f for f in os.listdir("gen_files/compas") if not f.startswith('.')] == [] and [f for f in os.listdir("gen_files/adult") if not f.startswith('.')] == [] and [f for f in os.listdir("gen_files/german_credit") if not f.startswith('.')] == [] and [f for f in os.listdir("gen_files/default_credit") if not f.startswith('.')] == []:
    print("No CORELS logs to be deleted")
else :
    if os.system('rm -r gen_files/*/*') == 0:
        print("Deleted all generated CORELS logs")
    else:
        print("Error while deleting CORELS log files")
