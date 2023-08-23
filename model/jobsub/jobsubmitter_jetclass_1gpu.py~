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


# # topdata_nominal_baseline

# options = "--epochs 50 --label topdata_nominal_baseline --data-type topdata --KLcoef nominal --batch-mode"
# jobname = "topdata_nominal_baseline"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# # topdata_nominal_0

# options = "--epochs 50 --label topdata_nominal_0 --data-type topdata --KLcoef 0 --batch-mode"
# jobname = "topdata_nominal_0"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)


# # topdata_nominal_massmask200

# options = "--epochs 50 --label topdata_nominal_massmask200 --data-type topdata --KLcoef nominal --mass-range AND:0,200 --batch-mode"
# jobname = "topdata_nominal_massmask200"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# # topdata_nominal_ptmask575

# options = "--epochs 50 --label topdata_nominal_ptmask575 --data-type topdata --KLcoef nominal --pt-range OR:575,625 --batch-mode"
# jobname = "topdata_nominal_ptmask575"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)


# # topdata_0.1_baseline

# options = "--epochs 50 --label topdata_0.1_baseline --data-type topdata --KLcoef 0.1 --batch-mode"
# jobname = "topdata_0.1_baseline"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# # topdata_0.1_massmask200

# options = "--epochs 50 --label topdata_0.1_massmask200 --data-type topdata --KLcoef 0.1 --mass-range AND:0,200 --batch-mode"
# jobname = "topdata_0.1_massmask200"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# # topdata_0.1_ptmask575

# options = "--epochs 50 --label topdata_0.1_ptmask575 --data-type topdata --KLcoef 0.1 --pt-range OR:575,625 --batch-mode"
# jobname = "topdata_0.1_ptmask575"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# # jetnet_nominal_baseline

# options = "--epochs 300 --label jetnet_nominal_baseline --data-type jetnet --KLcoef nominal --batch-mode"
# jobname = "jetnet_nominal_baseline"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# # jetnet_nominal_0

# options = "--epochs 300 --label jetnet_nominal_0 --data-type jetnet --KLcoef 0 --batch-mode"
# jobname = "jetnet_nominal_0"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# # jetnet_nominal_skiptop

# options = "--epochs 300 --label jetnet_nominal_skiptop --data-type jetnet --KLcoef nominal --skip-labels 2 --batch-mode"
# jobname = "jetnet_nominal_skiptop"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# # jetnet_nominal_skipwz

# options = "--epochs 300 --label jetnet_nominal_skipwz --data-type jetnet --KLcoef nominal --skip-labels 3,4 --batch-mode"
# jobname = "jetnet_nominal_skipwz"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# # jetnet_0.1_baseline

# options = "--epochs 300 --label jetnet_0.1_baseline --data-type jetnet --KLcoef 0.1 --batch-mode"
# jobname = "jetnet_0.1_baseline"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# # jetnet_0.1_skiptop

# options = "--epochs 300 --label jetnet_0.1_skiptop --data-type jetnet --KLcoef 0.1 --skip-labels 2 --batch-mode"
# jobname = "jetnet_0.1_skiptop"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# # jetnet_0.1_skipwz

# options = "--epochs 300 --label jetnet_0.1_skipwz --data-type jetnet --KLcoef 0.1 --skip-labels 3,4 --batch-mode"
# jobname = "jetnet_0.1_skipwz"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)


# # jetnet_0.1_skiptwz

# options = "--epochs 300 --label jetnet_0.1_skiptwz --data-type jetnet --KLcoef 0.1 --skip-labels 2,3,4 --batch-mode"
# jobname = "jetnet_0.1_skiptwz"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)


# # jetnet_nominal_skiptwz

# options = "--epochs 300 --label jetnet_nominal_skiptwz --data-type jetnet --KLcoef nominal --skip-labels 2,3,4 --batch-mode"
# jobname = "jetnet_nominal_skiptwz"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)


# # jetnet_0_skiptop

# options = "--epochs 300 --label jetnet_0_skiptop --data-type jetnet --KLcoef 0 --skip-labels 2 --batch-mode"
# jobname = "jetnet_0_skiptop"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# # jetnet_0_skipwz

# options = "--epochs 300 --label jetnet_0_skipwz --data-type jetnet --KLcoef 0 --skip-labels 3,4 --batch-mode"
# jobname = "jetnet_0_skipwz"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)


# # jetnet_0_skiptwz

# options = "--epochs 300 --label jetnet_0_skiptwz --data-type jetnet --KLcoef 0 --skip-labels 2,3,4 --batch-mode"
# jobname = "jetnet_0_skiptwz"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# for ii in range(10):
#     jobname = "jetnet_softmax_baseline_{}".format(ii)
#     options = "--epochs 100 --label {} --data-type jetnet --KLcoef 0 --use-softmax --batch-mode".format(jobname)
#     scriptname = jobmaker(jobname, options)
#     subprocess.call("sbatch " + scriptname, shell=True)


# for ii in range(10):
#     jobname = "jetnet_softmax_skiptop_{}".format(ii)
#     options = "--epochs 100 --label {} --data-type jetnet --KLcoef 0 --skip-labels 2 --use-softmax --batch-mode".format(jobname)
#     scriptname = jobmaker(jobname, options)
#     subprocess.call("sbatch " + scriptname, shell=True)

# for ii in range(10):
#     jobname = "jetnet_softmax_skiptwz_{}".format(ii)
#     options = "--epochs 100 --label {} --data-type jetnet --KLcoef 0 --skip-labels 2,3,4 --use-softmax --batch-mode".format(jobname)
#     scriptname = jobmaker(jobname, options)
#     subprocess.call("sbatch " + scriptname, shell=True)

# for ii in range(10):
#     jobname = "jetnet_softmax_skipwz_{}".format(ii)
#     options = "--epochs 100 --label {} --data-type jetnet --KLcoef 0 --skip-labels 3,4 --use-softmax --batch-mode".format(jobname)
#     scriptname = jobmaker(jobname, options)
#     subprocess.call("sbatch " + scriptname, shell=True)

# jobname = "jetnet_softmax_dropout_baseline"
# options = "--epochs 100 --label {} --data-type jetnet --KLcoef 0 --use-softmax --use-dropout --batch-mode".format(jobname)
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)


# jobname = "jetnet_softmax_dropout_skiptop"
# options = "--epochs 100 --label {} --data-type jetnet --KLcoef 0 --skip-labels 2 --use-softmax --use-dropout --batch-mode".format(jobname)
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# jobname = "jetnet_softmax_dropout_skiptwz"
# options = "--epochs 100 --label {} --data-type jetnet --KLcoef 0 --skip-labels 2,3,4 --use-softmax --use-dropout --batch-mode".format(jobname)
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# jobname = "jetnet_softmax_dropout_skipwz"
# options = "--epochs 100 --label {} --data-type jetnet --KLcoef 0 --skip-labels 3,4 --use-softmax --use-dropout --batch-mode".format(jobname)
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# jobname = "jetnet_0.1_baseline_ocmodel"
# options = "--epochs 30 --label {} --data-type jetnet --KLcoef 0.1  --preload --preload-file trained_models/UQPFIN_best_jetnet_0_baseline --batch-mode".format(jobname)
# scriptname = jobmaker(jobname, options)                                                                                                                             
# subprocess.call("sbatch " + scriptname, shell=True)


# jobname = "jetnet_0.5_baseline_ocmodel"
# options = "--epochs 30 --label {} --data-type jetnet --KLcoef 0.5  --preload --preload-file trained_models/UQPFIN_best_jetnet_0_baseline --batch-mode".format(jobname)
# scriptname = jobmaker(jobname, options)                                                                                                                             
# subprocess.call("sbatch " + scriptname, shell=True)


# jobname = "jetnet_0.1_skiptop_ocmodel"
# options = "--epochs 30 --label {} --data-type jetnet --KLcoef 0.1 --skip-labels 2 --preload --preload-file trained_models/UQPFIN_best_jetnet_0_skiptop --batch-mode".format(jobname)
# scriptname = jobmaker(jobname, options)                                                                                                                             
# subprocess.call("sbatch " + scriptname, shell=True)

# jobname = "jetnet_0.1_skiptwz_ocmodel"
# options = "--epochs 30 --label {} --data-type jetnet --KLcoef 0.1 --skip-labels 2,3,4 --preload --preload-file trained_models/UQPFIN_best_jetnet_0_skiptwz --batch-mode".format(jobname)
# scriptname = jobmaker(jobname, options)                                                                                                                             
# subprocess.call("sbatch " + scriptname, shell=True)

# jobname = "jetnet_0.1_skipwz_ocmodel"
# options = "--epochs 30 --label {} --data-type jetnet --KLcoef 0.1 --skip-labels 3,4 --preload --preload-file trained_models/UQPFIN_best_jetnet_0_skipwz --batch-mode".format(jobname)
# scriptname = jobmaker(jobname, options)                                                                                                                             
# subprocess.call("sbatch " + scriptname, shell=True)


# jobname = "jetclass_trial0"
# options = " --label jetclass_trial0 --data-type jetclass --KLcoef 0 --use-softmax --batch-size 2500 --epochs 10 --Phi-nodes 100,100,100,128 --F-nodes 128,100,100,100 --NPhiI 256"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# jobname = "jetclass_trial0_partdata"
# options = " --label jetclass_trial0_partdata --data-type jetclass --KLcoef 0 --use-softmax --batch-size 2500 --epochs 50 --Phi-nodes 100,100,100,128 --F-nodes 128,100,100,100 --NPhiI 256 --batch-mode"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# jobname = "jetclass_trial0_20Mdata_2"
# options = " --label jetclass_trial0_20Mdata_2 --data-type jetclass --KLcoef 0 --use-softmax --batch-size 2048 --epochs 30 --Phi-nodes 100,100,100,128 --F-nodes 128,100,100,100 --NPhiI 256 --preload --preload-file trained_models/UQPFIN_best_jetclass_trial0_20Mdata --batch-mode"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)

# jobname = "jetclass_trial1_20Mdata"
# options = " --label jetclass_trial1_20Mdata --data-type jetclass --KLcoef 0 --use-softmax --batch-size 2048 --epochs 50 --Phi-nodes 100,100,100,256 --F-nodes 256,100,100,100 --NPhiI 256  --batch-mode"
# scriptname = jobmaker(jobname, options)
# subprocess.call("sbatch " + scriptname, shell=True)



##############################################################

## All jobs for JNqgmerged

## Ensemble: baseline
for ii in range(10):
    jobname = "JNqgmerged_softmax_baseline_{}".format(ii)
    options = "--epochs 100 --label {} --data-type JNqgmerged --KLcoef 0 --use-softmax --batch-mode".format(jobname)
    scriptname = jobmaker(jobname, options)
    subprocess.call("sbatch " + scriptname, shell=True)

## MCDO: baseline
jobname = "JNqgmerged_softmax_dropout_baseline"
options = "--epochs 100 --label {} --data-type JNqgmerged --KLcoef 0 --use-softmax --use-dropout --batch-mode".format(jobname)
scriptname = jobmaker(jobname, options)
subprocess.call("sbatch " + scriptname, shell=True)


## Ensemble: skiptop
for ii in range(10):
    jobname = "JNqgmerged_softmax_skiptop_{}".format(ii)
    options = "--epochs 100 --label {} --data-type JNqgmerged --KLcoef 0 --skip-labels 1 --use-softmax --batch-mode".format(jobname)
    scriptname = jobmaker(jobname, options)
    subprocess.call("sbatch " + scriptname, shell=True)

## MCDO: skiptop
jobname = "JNqgmerged_softmax_dropout_skiptop"
options = "--epochs 100 --label {} --data-type JNqgmerged --KLcoef 0 --skip-labels 1 --use-softmax --use-dropout --batch-mode".format(jobname)
scriptname = jobmaker(jobname, options)
subprocess.call("sbatch " + scriptname, shell=True)

## Ensemble: skipwz
for ii in range(10):
    jobname = "JNqgmerged_softmax_skipwz_{}".format(ii)
    options = "--epochs 100 --label {} --data-type JNqgmerged --KLcoef 0 --skip-labels 2,3 --use-softmax --batch-mode".format(jobname)
    scriptname = jobmaker(jobname, options)
    subprocess.call("sbatch " + scriptname, shell=True)

## MCDO: skipwz
jobname = "JNqgmerged_softmax_dropout_skipwz"
options = "--epochs 100 --label {} --data-type JNqgmerged --KLcoef 0 --skip-labels 2,3 --use-softmax --use-dropout --batch-mode".format(jobname)
scriptname = jobmaker(jobname, options)
subprocess.call("sbatch " + scriptname, shell=True)


for kl in ['0', '0.1', 'nominal']:
    # baseline
    jobname = "JNqgmerged_{}_baseline".format(kl)
    options = "--epochs 100 --label {} --data-type JNqgmerged --KLcoef {} --batch-mode".format(jobname, kl)
    scriptname = jobmaker(jobname, options)
    subprocess.call("sbatch " + scriptname, shell=True)
    
    # skiptop
    jobname = "JNqgmerged_{}_skiptop".format(kl)
    options = "--epochs 100 --label {} --data-type JNqgmerged --KLcoef {} --skip-labels 1 --batch-mode".format(jobname, kl)
    scriptname = jobmaker(jobname, options)
    subprocess.call("sbatch " + scriptname, shell=True)


    # skipwz
    jobname = "JNqgmerged_{}_skipwz".format(kl)
    options = "--epochs 100 --label {} --data-type JNqgmerged --KLcoef {} --skip-labels 2,3 --batch-mode".format(jobname, kl)
    scriptname = jobmaker(jobname, options)
    subprocess.call("sbatch " + scriptname, shell=True)
