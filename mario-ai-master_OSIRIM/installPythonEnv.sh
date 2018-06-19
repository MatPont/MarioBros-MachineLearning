#!/bin/sh
#SBATCH --job-name=install_virtual_env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

srun tf-py3 virtualenv --system-site-packages /users/smac/mpont/venv
