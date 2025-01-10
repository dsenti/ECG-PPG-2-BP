#!/bin/bash
#SBATCH --job-name=testing    # Create a short name for your job
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=72            # CPU cores per task
#SBATCH --time=1:00:00               # Total run time limit (HH:MM:SS)
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

# python split_data_correlated_MIMIC.py
# Run the finetuning script
python -u test.py --model_checkpoint 'checkpoint_M_l_ft_wh_1000ep.pt' --dataset 'MIMIC' --model_size 'large' --batch_size 1024