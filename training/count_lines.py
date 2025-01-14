import os
import subprocess

folder_path = "/capstor/scratch/cscs/<username>/dataset/all_files"

def count_lines_in_folder(folder_path):
    print(f"Counting lines in files in folder: {folder_path}")
    total_lines = 0

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path):
            # Call wc -l for file
            result = subprocess.run(['wc', '-l', file_path], capture_output=True, text=True)

            try:
                line_count = int(result.stdout.split()[0])
                total_lines += line_count
                print(f"{file_name}: {line_count} lines")
            except (IndexError, ValueError):
                print(f"Error processing {file_name}")

    print(f"Total lines in all files: {total_lines}")

count_lines_in_folder(folder_path)
