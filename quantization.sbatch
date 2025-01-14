#!/bin/bash
#SBATCH --job-name=dynamic_quant       # Create a short name for your job
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=72            # CPU cores per task
#SBATCH --time=05:00:00               # Total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:0                  # Number of GPUs per node
#SBATCH --account=lp12

# Change to the home directory
cd $HOME

# Activate the virtual environment
source /users/<username>/TimeFMvenv/bin/activate

# double check
which python
python --version
which pip

# Change to the quantization directory
cd $HOME/quantization/

# Run the finetuning script
python -u ptq_dynamic.py --model_size 'large' --dataset 'MIMIC' --model_checkpoint 'final_model_M_l_ft_wh_100ep.ckpt' --batch_size 64
