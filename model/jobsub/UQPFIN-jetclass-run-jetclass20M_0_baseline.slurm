#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output="/u/avroy/UQ_XAI_Studies/models/PFIN_UQ/model/jobouts/PFIN.jetclass20M_0_baseline.%j.%N.out"
#SBATCH --cpus-per-task=4    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbhj-delta-gpu
#SBATCH --job-name=jetclass20M_0_baseline
#SBATCH --time=48:00:00      # hh:mm:ss for the job
#SBATCH --constraint="projects"
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest     # <- or closest
#SBATCH --mail-user=avroy@illinois.edu
#SBATCH --mail-type="FAIL"

module reset
module load modtree/gpu gcc anaconda3_gpu
source activate toptagger_env
module list  # job documentation and metadata
pwd
ls -lh
cd /projects/bbhj/avroy/UQ_XAI_Studies/models/PFIN_UQ/model
pwd
ls -lh
echo "job is starting on `hostname`"
conda env list
export HDF5_USE_FILE_LOCKING=FALSE
which python
python train_mod.py  --label jetclass20M_0_baseline --data-type jetclass --KLcoef 0 --batch-size 2048 --epochs 30 --Phi-nodes 100,100,100,128 --F-nodes 128,100,100,100 --NPhiI 256 --nodes 1 --gpus 4 --ndata-per-gpu 5  --batch-mode
