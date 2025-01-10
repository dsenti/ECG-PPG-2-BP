#!/bin/bash
#SBATCH --job-name=M_l_ft_wh_1kep    # Create a short name for your job
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=72            # CPU cores per task
#SBATCH --time=24:00:00               # Total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:4                  # Number of GPUs per node
#SBATCH --account=lp12

# Change to the home directory
cd $HOME

# Activate the virtual environment
source /users/<username>/TimeFMvenv/bin/activate

# double check
which python
python --version
which pip

# Change to the TimeFM directory
cd $HOME/code/TimeFM

# Run the finetuning script
python -u train.py --wandb_name 'M_l_ft_wh_1000ep' --dataset 'MIMIC' --model_size 'large' --num_epochs 1000 --pretrained True --freeze_backbone False --patience 50 --learning_rate 1e-4 --batch_size 1024
