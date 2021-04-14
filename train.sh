#!/bin/bash
#SBATCH --output=./train2.log  # send stdout to outfile
#SBATCH -p debug
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH -w worker-2

source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:./
ssh -N -f -L localhost:5100:localhost:5100 ubuntu@10.195.1.136
PYTHONPATH=. python3 train.py data.batch_size.train=16 model.superglue.min_keypoints=256 model.superglue.trainable_bin_score=False