#!/bin/sh

#SBATCH --job-name=DQN_LSTM_Mario

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

#SBATCH --mail-type=END
#SBATCH --mail-user=mathieu.pont@irit.fr

#srun tf-py3 ./myTestE.sh
srun tf-py3 python myTest.py