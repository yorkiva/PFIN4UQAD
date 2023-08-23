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




##############################################################

## All jobs for jetclass

## Ensemble: baseline
for ii in range(10):
    jobname = "jetclass_softmax_baseline_{}".format(ii)
    options = "--epochs 30 --label {} --data-type jetclass --KLcoef 0 --batch-size 2500 --use-softmax --batch-mode".format(jobname)
    scriptname = jobmaker(jobname, options)
    subprocess.call("sbatch " + scriptname, shell=True)

## MCDO: baseline
jobname = "jetclass_softmax_dropout_baseline"
options = "--epochs 30 --label {} --data-type jetclass --KLcoef 0 --batch-size 2500 --use-softmax --use-dropout --batch-mode".format(jobname)
scriptname = jobmaker(jobname, options)
subprocess.call("sbatch " + scriptname, shell=True)


## Ensemble: skiptop
for ii in range(10):
    jobname = "jetclass_softmax_skiptop_{}".format(ii)
    options = "--epochs 30 --label {} --data-type jetclass --KLcoef 0 --batch-size 2500 --skip-labels 8,9 --use-softmax --batch-mode".format(jobname)
    scriptname = jobmaker(jobname, options)
    subprocess.call("sbatch " + scriptname, shell=True)

## MCDO: skiptop
jobname = "jetclass_softmax_dropout_skiptop"
options = "--epochs 30 --label {} --data-type jetclass --KLcoef 0 --batch-size 2500 --skip-labels 8,9 --use-softmax --use-dropout --batch-mode".format(jobname)
scriptname = jobmaker(jobname, options)
subprocess.call("sbatch " + scriptname, shell=True)

## Ensemble: skipwz
for ii in range(10):
    jobname = "jetclass_softmax_skipwz_{}".format(ii)
    options = "--epochs 30 --label {} --data-type jetclass --KLcoef 0 --batch-size 2500 --skip-labels 6,7 --use-softmax --batch-mode".format(jobname)
    scriptname = jobmaker(jobname, options)
    subprocess.call("sbatch " + scriptname, shell=True)

## MCDO: skipwz
jobname = "jetclass_softmax_dropout_skipwz"
options = "--epochs 30 --label {} --data-type jetclass --KLcoef 0 --batch-size 2500 --skip-labels 6,7 --use-softmax --use-dropout --batch-mode".format(jobname)
scriptname = jobmaker(jobname, options)
subprocess.call("sbatch " + scriptname, shell=True)

## Ensemble: skiptwz
for ii in range(10):
    jobname = "jetclass_softmax_skiptwz_{}".format(ii)
    options = "--epochs 30 --label {} --data-type jetclass --KLcoef 0 --batch-size 2500 --skip-labels 6,7,8,9 --use-softmax --batch-mode".format(jobname)
    scriptname = jobmaker(jobname, options)
    subprocess.call("sbatch " + scriptname, shell=True)

## MCDO: skiptwz
jobname = "jetclass_softmax_dropout_skiptwz"
options = "--epochs 30 --label {} --data-type jetclass --KLcoef 0 --batch-size 2500 --skip-labels 6,7,8,9 --use-softmax --use-dropout --batch-mode".format(jobname)
scriptname = jobmaker(jobname, options)
subprocess.call("sbatch " + scriptname, shell=True)

## Ensemble: skiph
for ii in range(10):
    jobname = "jetclass_softmax_skiph_{}".format(ii)
    options = "--epochs 30 --label {} --data-type jetclass --KLcoef 0 --batch-size 2500 --skip-labels 1,2,3,4,5 --use-softmax --batch-mode".format(jobname)
    scriptname = jobmaker(jobname, options)
    subprocess.call("sbatch " + scriptname, shell=True)

## MCDO: skiph
jobname = "jetclass_softmax_dropout_skiph"
options = "--epochs 30 --label {} --data-type jetclass --KLcoef 0 --batch-size 2500 --skip-labels 1,2,3,4,5 --use-softmax --use-dropout --batch-mode".format(jobname)
scriptname = jobmaker(jobname, options)
subprocess.call("sbatch " + scriptname, shell=True)


for kl in ['0', '0.1', 'nominal']:
    # baseline
    jobname = "jetclass_{}_baseline".format(kl)
    options = "--epochs 30 --label {} --data-type jetclass --batch-size 2500 --KLcoef {} --batch-mode".format(jobname, kl)
    scriptname = jobmaker(jobname, options)
    subprocess.call("sbatch " + scriptname, shell=True)
    
    # skiptop
    jobname = "jetclass_{}_skiptop".format(kl)
    options = "--epochs 30 --label {} --data-type jetclass --batch-size 2500 --KLcoef {} --skip-labels 8,9 --batch-mode".format(jobname, kl)
    scriptname = jobmaker(jobname, options)
    subprocess.call("sbatch " + scriptname, shell=True)


    # skipwz
    jobname = "jetclass_{}_skipwz".format(kl)
    options = "--epochs 30 --label {} --data-type jetclass --batch-size 2500 --KLcoef {} --skip-labels 6,7 --batch-mode".format(jobname, kl)
    scriptname = jobmaker(jobname, options)
    subprocess.call("sbatch " + scriptname, shell=True)

    # skiptwz
    jobname = "jetclass_{}_skiptwz".format(kl)
    options = "--epochs 30 --label {} --data-type jetclass --batch-size 2500 --KLcoef {} --skip-labels 6,7,8,9 --batch-mode".format(jobname, kl)
    scriptname = jobmaker(jobname, options)
    subprocess.call("sbatch " + scriptname, shell=True)

    # skiph
    jobname = "jetclass_{}_skiph".format(kl)
    options = "--epochs 30 --label {} --data-type jetclass --batch-size 2500 --KLcoef {} --skip-labels 1,2,3,4,5 --batch-mode".format(jobname, kl)
    scriptname = jobmaker(jobname, options)
    subprocess.call("sbatch " + scriptname, shell=True)
