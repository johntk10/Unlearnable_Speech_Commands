#!/bin/bash
#SBATCH --job-name=speech_classifier
#SBATCH --mem=16G ## memory that you need for your code
#SBATCH --gres=gpu:1 ## change this according to your need. It can go up to 4 GPUs.
#SBATCH --output=train_model_out%j.txt
#SBATCH --error=train_model_err%j.txt
source .env/bin/activate ## creating a virtual environment is a must
python3 -u poisoned_main.py
