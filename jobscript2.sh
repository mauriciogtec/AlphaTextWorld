#!/bin/bash

#SBATCH -N 16
#SBATCH -n 32
#SBATCH -c 1
#SBATCH -p normal
#SBATCH -J gfull
#SBATCH -e slurm-%j.err
#SBATCH -t 2:0:0


module load intel/17.0.4
module load python3/3.6.3
module load hdf5/1.8.16

cd /work/05863/mgarciat/stampede2/AlphaTextWorld/
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0 &
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 0
wait

