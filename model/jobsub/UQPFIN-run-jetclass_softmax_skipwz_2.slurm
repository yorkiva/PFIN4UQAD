#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --output="/u/avroy/UQ_XAI_Studies/models/PFIN_UQ/model/jobouts/PFIN.jetclass_softmax_skipwz_2.%j.%N.out"
#SBATCH --cpus-per-task=4    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbhj-delta-gpu
#SBATCH --job-name=jetclass_softmax_skipwz_2
#SBATCH --time=48:00:00      # hh:mm:ss for the job
#SBATCH --constraint="projects"
### GPU options ###
#SBATCH --gpus-per-node=1
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
python train.py --epochs 30 --label jetclass_softmax_skipwz_2 --data-type jetclass --KLcoef 0 --batch-size 2500 --skip-labels 6,7 --use-softmax --batch-mode
