#!/bin/sh

#SBATCH	--job-name=DQN_LSTM_tr8
#SBATCH --ntasks=1
#SBATCH	--cpus-per-task=4
#SBATCH	--partition=GPUNodes
#SBATCH	--gres=gpu:1
#SBATCH	--gres-flags=enforce-binding

#SBATCH --mail-type=END
#SBATCH --mail-user=mathieu.pont@irit.fr

cd ./src/main/java/amico/python/JavaPy
srun tf-py3 ./run.sh -o "-ag MarioDQNAgent2.MarioDQNAgent.py" -nc


