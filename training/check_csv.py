import os
import pandas as pd

def analyze_first_csv(directory):
    for file in os.listdir(directory):
        if file.endswith('p002772.csv'):
            csv_path = os.path.join(directory, file)
            print(f"Analyzing file: {csv_path}")

            try:
                data = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Error reading file: {e}")
                return
            
            print("\n--- File Info ---")
            print(data.info())
            print("\n--- Summary Statistics ---")
            print(data.describe())
            print("\n--- First Few Rows ---")
            print(data.head())
            
            return

    print("No CSV files in the dir")

directory_path = '/capstor/scratch/cscs/<username>/dataset/Vital_all'
analyze_first_csv(directory_path)
