#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --tasks-per-node 1
#SBATCH -p development
#SBATCH -J gameplay2
#SBATCH -e slurm-%j.err
#SBATCH -t 1:0:0


module load intel/17.0.4
module load python3/3.6.3


cd /work/05863/mgarciat/stampede2/AlphaTextWorld/
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 1
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 1
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 1
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 1
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 1
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 1
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 1
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 1
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 1
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 1
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 1
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 1
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 1
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 1
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 1
python3 play_remote2.py --temperature 0.5 --subtrees 100 --subtree_depth 5 --verbose 1

