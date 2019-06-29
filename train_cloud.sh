#!/usr/bin/bash
#SBATCH -N 1
#SBATCH -p skx-normal
#SBATCH -J train
#SBATCH -e slurm-%j.err
#SBATCH -t 8:0:0

module load intel/17.0.4
module load python3/
pip3 install --user tensorflow==2.0.0-beta1
pip3 install --user keras h5py==2.8.0
module load phdf5

export HDF5_USE_FILE_LOCKING=FALSE
export MKL_NUM_THREADS=96
export GOTO_NUM_THREADS=96
export OMP_NUM_THREADS=96
# export openmp=0

set +e
python3 scripts/train2.py --num_consider 100 --batch_size 16 --num_data 160 --ckpt_every 25 --num_epochs 1
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 25 --batch_size 16 --num_data 160 --ckpt_every 25 --num_epochs 1
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 25 --batch_size 16 --num_data 160 --ckpt_every 25 --num_epochs 1
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 25 --batch_size 16 --num_data 160 --ckpt_every 25 --num_epochs 1
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 25 --batch_size 16 --num_data 160 --ckpt_every 25 --num_epochs 1
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 25 --batch_size 16 --num_data 160 --ckpt_every 25 --num_epochs 1
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 25 --batch_size 16 --num_data 160 --ckpt_every 25 --num_epochs 1
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 25 --batch_size 16 --num_data 160 --ckpt_every 25 --num_epochs 1
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 25 --batch_size 16 --num_data 160 --ckpt_every 25 --num_epochs 1
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 25 --batch_size 16 --num_data 160 --ckpt_every 25 --num_epochs 1
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 25 --batch_size 16 --num_data 160 --ckpt_every 25 --num_epochs 1
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 25 --batch_size 16 --num_data 160 --ckpt_every 25 --num_epochs 1
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 25 --batch_size 16 --num_data 160 --ckpt_every 25 --num_epochs 1
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 25 --batch_size 16 --num_data 160 --ckpt_every 25 --num_epochs 1
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 25 --batch_size 16 --num_data 160 --ckpt_every 25 --num_epochs 1
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 25 --batch_size 16 --num_data 160 --ckpt_every 25 --num_epochs 1
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 25 --batch_size 16 --num_data 160 --ckpt_every 25 --num_epochs 1
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 25 --batch_size 16 --num_data 160 --ckpt_every 25 --num_epochs 1
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
python3 scripts/train2.py --num_consider 1000 --batch_size 16 --num_data 640 --ckpt_every 25 --num_epochs 2
set -e