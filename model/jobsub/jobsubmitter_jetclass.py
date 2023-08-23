import os, sys, subprocess

def jobmaker(jobname, options):
    f = open("UQPFIN-jetclass-run.slurm", "r")
    scriptname = "UQPFIN-jetclass-run-{}.slurm".format(jobname) 
    f2w = open(scriptname, "w")
    for line in f:
        line = line.replace("<JOBNAME>", jobname).replace("<OPTIONS>", options).strip() + "\n"
        f2w.write(line)
    f.close()
    f2w.close()
    return scriptname


jobname = "jetclass20M_0_baseline"
options = " --label {} --data-type jetclass --KLcoef 0 --batch-size 2048 --epochs 30 --Phi-nodes 100,100,100,128 --F-nodes 128,100,100,100 --NPhiI 256 --nodes 1 --gpus 4 --ndata-per-gpu 5  --batch-mode".format(jobname)
scriptname = jobmaker(jobname, options)
subprocess.call("sbatch " + scriptname, shell=True)

