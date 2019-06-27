#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -p development
#SBATCH -J gnormal
#SBATCH -e slurm-%j.err
#SBATCH -t 0:120:0


module load intel/17.0.4
module load python3/3.6.3
pip3 install --user keras h5py==2.8.0
module load hdf5/1.8.16

export HDF5_USE_FILE_LOCKING=FALSE
export MKL_NUM_THREADS=2
export GOTO_NUM_THREADS=2
export OMP_NUM_THREADS=2
export openmp=1

cd /work/05863/mgarciat/stampede2/AlphaTextWorld/
srun -N 1 -n 1 python3 play_remote.py --temperature 0.5 --subtrees 100 --subtree_depth 5
wait
