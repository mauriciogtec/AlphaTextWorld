#!/bin/bash

#SBATCH -N 16
#SBATCH -n 32
#SBATCH -c 1
#SBATCH -p normal
#SBATCH -J gfull
#SBATCH -e slurm-%j.err
#SBATCH -t 4:0:0


module load intel/17.0.4
module load python3/3.6.3
#pip3 install --user tensorflow==2.0.0-beta1
pip3 install --user keras h5py==2.8.0
module load phdf5

export HDF5_USE_FILE_LOCKING=FALSE


cd /work/05863/mgarciat/stampede2/AlphaTextWorld/
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 & 
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 --min_time 15 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0
--min_time 15 wait

