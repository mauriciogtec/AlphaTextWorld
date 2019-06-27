#!/bin/bash

#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 1
#SBATCH -p normal
#SBATCH -J gnormal
#SBATCH -e slurm-%j.err
#SBATCH -t 4:0:0


module load intel/17.0.4
module load python3/3.6.3
#pip3 install --user tensorflow==2.0.0-beta1
pip3 install --user keras h5py==2.8.0
module load phdf5

export HDF5_USE_FILE_LOCKING=FALSE
# export MKL_NUM_THREADS=2
# export GOTO_NUM_THREADS=2
# export OMP_NUM_THREADS=2
# export openmp=1

cd /work/05863/mgarciat/stampede2/AlphaTextWorld/
srun -N 1 -n 1 python3 play_remote.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --min_time 15 &
srun -N 1 -n 1 python3 play_remote.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --min_time 15
wait
