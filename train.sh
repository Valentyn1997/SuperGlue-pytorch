#!/bin/bash
#SBATCH --output=./train.log  # send stdout to outfile
#SBATCH -p debug
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH -w worker-2

source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:./
ssh -N -f -L localhost:5100:localhost:5100 ubuntu@10.195.1.136
PYTHONPATH=. python3 train.py data.batch_size.train=32