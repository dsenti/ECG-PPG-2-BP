# Blood Pressure Estimation from ECG and PPG Signals

This project focuses on blood pressure estimation using ECG and PPG signals. The model is a fine-tuned transformer, originally pretrained on EEG data (the TimeFM project, hence the TimeFM references throughout this repository). For more information, please refer to our report.

**Disclaimer:** To ensure transparency, all components that are not our own work will be explicitly labeled as being sourced from the TimeFM repository by the README.md in their folder or parent folder.

## Project Structure

The repository is organized as follows:

- **`training/`**
  - Contains scripts for training and testing the model.
  - Includes functionality for data preprocessing, model fine-tuning, and evaluation on validation/test datasets.
  
- **`quantization/`**
  - Provides tools for model quantization to reduce its size and improve efficiency.
  - Useful as a first step towards deploying the model on resource-constrained devices.


## Usage

This code is designed to run on the CSCS cluster, Piz Daint. Refer to the section on how to set up an environment below.

1. **Training:**
   - To train the model, submit the `training` job script using the `sbatch` command.
     ```bash
     sbatch train.sbatch
     ```
   - The `train.sbatch` script is configured to:
     - Use 1 node with 4 GPUs.
     - Train the model for up to 24h with parameters specified in the script.
   - Example SLURM settings in `train.sbatch`:
     ```bash
     #SBATCH --job-name=finetuning
     #SBATCH --nodes=1
     #SBATCH --ntasks-per-node=1
     #SBATCH --cpus-per-task=72
     #SBATCH --time=24:00:00
     #SBATCH --gres=gpu:4
     #SBATCH --account=lp12
     ```

   - `train.sbatch` calls `train.py`. Here is an example command:
     ```bash
     python -u train.py --wandb_name 'M_l_ft_wh_1000ep' --dataset 'MIMIC' --model_size 'large' --num_epochs 1000 --pretrained True --freeze_backbone False --patience 50 --learning_rate 1e-4 --batch_size 1024
     ```

2. **Testing:**
   - Testing is included in the training process, but if the job is interrupted, you can test a specific checkpoint using the `test` job script.
     ```bash
     sbatch test.sbatch
     ```
   - Example SLURM settings are available in the Training section for reference.

   -  An example of how `test.py` can be called is given in `test.sbatch`:
     ```bash
     python -u test.py --model_checkpoint 'checkpoint_M_l_ft_wh_1000ep.pt' --dataset 'MIMIC' --model_size 'large' --batch_size 1024
     ```

3. **Quantization:**
   - To quantize the model for deployment, submit the `quantization` job script.
     ```bash
     sbatch quantization.sbatch
     ```
   - Example SLURM settings are available in the Training section for reference.

   - The script performs quantization by calling `ptq_dynamic.py` or `ptq_static.py`. Here is an example usage:
     ```bash
     python -u ptq_dynamic.py --model_size 'large' --dataset 'MIMIC' --model_checkpoint 'final_model_M_l_ft_wh_100ep.ckpt' --batch_size 64
     ```

---

## Dataset

The dataset used in this project was sourced from **PulseDB**, a publicly available repository including ECG and PPG data. We specifically utilized data from the two subsets: **MIMIC** and **VitalDB**.

To prepare the dataset for model training and evaluation, verify that both subsets contain all the required columns accessed by the `ECGPPGDataset` class. These columns included:

1. **ECG Features:** Columns named `ECG_F_1` to `ECG_F_1250`, representing ECG signals sampled over a specified time window.
2. **PPG Features:** Columns named `PPG_F_1` to `PPG_F_1250`, representing PPG signals aligned with the ECG signals.
3. **Target Values:** Columns `SegDBP_AVG` and `SegSBP_AVG`, representing the average diastolic and systolic blood pressure values for each sample.

To facilitate loading by `ecg_ppg_dataset.py` the data should be organized as follows:
- All data files must be in `.csv` format.
- Each file should begin with either `Vital` or `MIMIC` in its name, corresponding to the respective dataset subset.
- All `.csv` files should be stored together in a **single folder**.


# Setup

Follow the steps below to set up your environment and prepare for running the training, testing, and quantization scripts on the CSCS cluster. This code is for a Windows PC so you might have to adjust certain commands.

## Steps

1. **Set up the CSCS Environment:**
   - Activate the CSCS-specific virtual environment. This environment and the necessary Python scripts (e.g., `cscs-keygen.py`) are provided by CSCS staff after account creation.
     ```bash
     conda activate cscs
     ```

2. **Start the SSH Agent:**
   - Ensure the SSH agent is running to manage your SSH keys.
     ```bash
     Start-Service ssh-agent
     ```

3. **Generate SSH Keys:**
   - Run the provided script to generate SSH keys for accessing the CSCS cluster.
     ```bash
     python cscs-keygen.py
     ```

4. **Add the Generated Key to the SSH Agent:**
   - Add the SSH key to the agent.
     ```bash
     ssh-add <path_to_your_generated_key>
     ```

5. **Connect to the CSCS Gateway:**
   - Use SSH to connect to the CSCS gateway.
     ```bash
     ssh -A <your_username>@ela.cscs.ch
     ```

6. **Connect to the Computing Node:**
   - From the CSCS gateway, connect to the desired node (e.g., Daint).
     ```bash
     ssh daint.alps.cscs.ch
     ```

7. **Create a Virtual Environment:**
   - Once on the cluster, create a virtual environment and install the required dependencies (`requirements.txt` is available in this folder):
     ```bash
     python -m venv <your_venv_name>
     source <your_venv_name>/bin/activate
     pip install -r requirements.txt
     ```

8. **Adjust Paths in Scripts:**
   - Before running any scripts, update all file paths (e.g., for datasets or model checkpoints) to match your specific directory structure.

By completing these steps, you will have an environment ready to run this project on the CSCS cluster.
