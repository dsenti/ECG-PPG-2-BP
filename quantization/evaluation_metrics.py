import torch
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import time

def calibrate(model, data_loader):
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inputs = batch['input']
            model(inputs)  # Forward pass to collect statistics
            # if i >= 10:  # Use only 10 batches for calibration to save time
            #     break

def calculate_model_size(model, filename):
    filename = filename if filename.endswith(".pt") else f"{filename}.pt"  # Ensure the .pt extension
    torch.save(model.state_dict(), filename)  # Save the model in .pt format
    size = os.path.getsize(filename) / (1024 ** 2)  # Convert size from bytes to MB
    os.remove(filename)  # Clean up the saved file
    return size

def calculate_and_print_metrics(true_values, predictions):
    true_dbp, true_sbp = true_values[:, 0], true_values[:, 1]
    pred_dbp, pred_sbp = predictions[:, 0], predictions[:, 1]

    mse_dbp = mean_squared_error(true_dbp, pred_dbp)
    mae_dbp = mean_absolute_error(true_dbp, pred_dbp)
    r2_dbp = r2_score(true_dbp, pred_dbp)
    var_dbp = np.var(true_dbp - pred_dbp)
    std_dbp = np.std(true_dbp - pred_dbp)

    mse_sbp = mean_squared_error(true_sbp, pred_sbp)
    mae_sbp = mean_absolute_error(true_sbp, pred_sbp)
    r2_sbp = r2_score(true_sbp, pred_sbp)
    var_sbp = np.var(true_sbp - pred_sbp)
    std_sbp = np.std(true_sbp - pred_sbp)

    print("Metrics for DBP (Diastolic Blood Pressure):")
    print(f"  Test MSE: {mse_dbp:.4f}")
    print(f"  Test MAE: {mae_dbp:.4f}")
    print(f"  Test R^2: {r2_dbp:.4f}")
    print(f"  Error Variance: {var_dbp:.4f}")
    print(f"  Error Standard Deviation: {std_dbp:.4f}")

    print("Metrics for SBP (Systolic Blood Pressure):")
    print(f"  Test MSE: {mse_sbp:.4f}")
    print(f"  Test MAE: {mae_sbp:.4f}")
    print(f"  Test R^2: {r2_sbp:.4f}")
    print(f"  Error Variance: {var_sbp:.4f}")
    print(f"  Error Standard Deviation: {std_sbp:.4f}")

def calculate_and_print_bhs_grades(true_values, predictions):
    true_dbp, true_sbp = true_values[:, 0], true_values[:, 1]
    pred_dbp, pred_sbp = predictions[:, 0], predictions[:, 1]

    error_dbp = np.abs(true_dbp - pred_dbp)
    error_sbp = np.abs(true_sbp - pred_sbp)

    within_5_dbp = np.mean(error_dbp <= 5) * 100
    within_10_dbp = np.mean(error_dbp <= 10) * 100
    within_15_dbp = np.mean(error_dbp <= 15) * 100

    bhs_grade_dbp = 'D'
    if within_5_dbp >= 60 and within_10_dbp >= 85 and within_15_dbp >= 95:
        bhs_grade_dbp = 'A'
    elif within_5_dbp >= 50 and within_10_dbp >= 75 and within_15_dbp >= 90:
        bhs_grade_dbp = 'B'
    elif within_5_dbp >= 40 and within_10_dbp >= 65 and within_15_dbp >= 85:
        bhs_grade_dbp = 'C'

    within_5_sbp = np.mean(error_sbp <= 5) * 100
    within_10_sbp = np.mean(error_sbp <= 10) * 100
    within_15_sbp = np.mean(error_sbp <= 15) * 100

    bhs_grade_sbp = 'D'
    if within_5_sbp >= 60 and within_10_sbp >= 85 and within_15_sbp >= 95:
        bhs_grade_sbp = 'A'
    elif within_5_sbp >= 50 and within_10_sbp >= 75 and within_15_sbp >= 90:
        bhs_grade_sbp = 'B'
    elif within_5_sbp >= 40 and within_10_sbp >= 65 and within_15_sbp >= 85:
        bhs_grade_sbp = 'C'

    print("BHS Grades for DBP (Diastolic Blood Pressure):")
    print(f"  <=5mmHg: {within_5_dbp:.2f}%, <=10mmHg: {within_10_dbp:.2f}%, <=15mmHg: {within_15_dbp:.2f}% -> Grade {bhs_grade_dbp}")

    print("BHS Grades for SBP (Systolic Blood Pressure):")
    print(f"  <=5mmHg: {within_5_sbp:.2f}%, <=10mmHg: {within_10_sbp:.2f}%, <=15mmHg: {within_15_sbp:.2f}% -> Grade {bhs_grade_sbp}")

def calculate_and_print_aami_validation(true_values, predictions):
    true_dbp, true_sbp = true_values[:, 0], true_values[:, 1]
    pred_dbp, pred_sbp = predictions[:, 0], predictions[:, 1]

    error_dbp = true_dbp - pred_dbp
    error_sbp = true_sbp - pred_sbp

    mean_error_dbp = np.abs(np.mean(error_dbp))
    std_error_dbp = np.std(error_dbp)
    aami_valid_dbp = mean_error_dbp < 5 and std_error_dbp < 8

    mean_error_sbp = np.abs(np.mean(error_sbp))
    std_error_sbp = np.std(error_sbp)
    aami_valid_sbp = mean_error_sbp < 5 and std_error_sbp < 8

    print("AAMI Validation for DBP (Diastolic Blood Pressure):")
    print(f"  Mean Error = {mean_error_dbp:.2f}, STD Error = {std_error_dbp:.2f}, Valid = {aami_valid_dbp}")

    print("AAMI Validation for SBP (Systolic Blood Pressure):")
    print(f"  Mean Error = {mean_error_sbp:.2f}, STD Error = {std_error_sbp:.2f}, Valid = {aami_valid_sbp}")


def calculate_and_print_mse(true_values, predictions):
    mse = mean_squared_error(true_values, predictions)
    print(f"MSE: {mse:.4f}")
    return mse

def collect_predictions_and_ground_truths_GPU(model, dataloader, device="cuda"):
    predictions, ground_truths = [], []
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)
            model_outputs = model(inputs)

            predictions.append(model_outputs.cpu().numpy())
            ground_truths.append(labels.cpu().numpy())

            # count += 1
            # if count >= 4:
            #     break

    predictions = np.vstack(predictions)
    ground_truths = np.vstack(ground_truths)
    return predictions, ground_truths

def collect_predictions_and_ground_truths(model, dataloader):
    predictions, ground_truths = [], []
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input']
            labels = batch['label']
            model_outputs = model(inputs)

            predictions.append(model_outputs.cpu().numpy())
            ground_truths.append(labels.cpu().numpy())

            # count += 1
            # if count >= max_batches:
            #     break

    predictions = np.vstack(predictions)
    ground_truths = np.vstack(ground_truths)
    return predictions, ground_truths

def calculate_and_print_aami_validation(true_values, predictions):
    true_dbp, true_sbp = true_values[:, 0], true_values[:, 1]
    pred_dbp, pred_sbp = predictions[:, 0], predictions[:, 1]

    error_dbp = true_dbp - pred_dbp
    error_sbp = true_sbp - pred_sbp

    mean_error_dbp = np.abs(np.mean(error_dbp))
    std_error_dbp = np.std(error_dbp)
    aami_valid_dbp = mean_error_dbp < 5 and std_error_dbp < 8

    mean_error_sbp = np.abs(np.mean(error_sbp))
    std_error_sbp = np.std(error_sbp)
    aami_valid_sbp = mean_error_sbp < 5 and std_error_sbp < 8

    print("AAMI Validation for DBP (Diastolic Blood Pressure):")
    print(f"  Mean Error = {mean_error_dbp:.2f}, STD Error = {std_error_dbp:.2f}, Valid = {aami_valid_dbp}")

    print("AAMI Validation for SBP (Systolic Blood Pressure):")
    print(f"  Mean Error = {mean_error_sbp:.2f}, STD Error = {std_error_sbp:.2f}, Valid = {aami_valid_sbp}")


def calculate_inference_time(model, dataloader, device='cpu', num_batches=64):
    model.to(device)
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = batch['input'].to(device)
            model(inputs)
            if i + 1 == num_batches:  # Limit to num_batches for timing
                break
    end_time = time.time()
    return (end_time - start_time) / num_batches

import torch
import time

def calculate_inference_time_GPU(model, dataloader, device='cuda', num_batches=64):
    model.to(device)
    model.eval()
    torch.cuda.empty_cache()  # Clear GPU cache for accurate timing
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_time.record()  # Start timing
        for i, batch in enumerate(dataloader):
            inputs = batch['input'].to(device)
            model(inputs)
            if i + 1 == num_batches:  # Limit to num_batches for timing
                break
        end_time.record()  # End timing
    
    torch.cuda.synchronize()
    return start_time.elapsed_time(end_time) / num_batches / 1000  # Convert to seconds
