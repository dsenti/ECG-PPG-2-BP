import pandas as pd
import glob
import os

input_folder_path = "/capstor/scratch/cscs/<username>/dataset/MIMIC_all" 
output_folder_path = "/capstor/scratch/cscs/<username>/dataset/MIMIC_all_no_bmi"

os.makedirs(output_folder_path, exist_ok=True)

csv_files = glob.glob(os.path.join(input_folder_path, "*.csv"))
total_files = len(csv_files) 

for idx, file in enumerate(csv_files, start=1):  # Use enumerate for a counter
    df = pd.read_csv(file) 
    if 'BMI' in df.columns:
        df.drop(columns=['BMI'], inplace=True)
    output_file = os.path.join(output_folder_path, os.path.basename(file))

    df.to_csv(output_file, index=False)
    print(f"Processed {idx}/{total_files} files: {file}")