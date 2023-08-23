import os, sys, subprocess

def jobmaker(jobname, options):
    f = open("UQPFIN-run.slurm", "r")
    scriptname = "UQPFIN-run-{}.slurm".format(jobname) 
    f2w = open(scriptname, "w")
    for line in f:
        line = line.replace("<JOBNAME>", jobname).replace("<OPTIONS>", options).strip() + "\n"
        f2w.write(line)
    f.close()
    f2w.close()
    return scriptname



# topdata_nominal_ptmask575

options = "--epochs 2 --label topdata_nominal_ptmask575 --data-type topdata --KLcoef nominal --pt-range OR:575,625 --batch-mode"
jobname = "topdata_nominal_ptmask575"
scriptname = jobmaker(jobname, options)
subprocess.call("sbatch " + scriptname, shell=True)


# jetnet_0.1_skipwz

options = "--epochs 2 --label jetnet_0.1_skipwz --data-type jetnet --KLcoef 0.1 --skip-labels 3,4 --batch-mode"
jobname = "jetnet_0.1_skipwz"
scriptname = jobmaker(jobname, options)
subprocess.call("sbatch " + scriptname, shell=True)

