import torch
import argparse
import our_model as our_model
import ecg_ppg_dataset as dataset
from torch.utils.data import random_split, DataLoader
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfig, QConfigMapping, MinMaxObserver, PerChannelMinMaxObserver, HistogramObserver, MovingAveragePerChannelMinMaxObserver, MovingAverageMinMaxObserver, PlaceholderObserver
from evaluation_metrics import *
from torch.utils.data import Subset
import random

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Quantization.")

    parser.add_argument(
        "--dataset",
        type=str,
        default="Vital",
        help="Dataset to use Vital or MIMIC(default: Vital)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)"
    )

    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        help="Model size (options: [small, base, large] default: small)"
    )

    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="final_model_M_l_ft_wh_100ep.ckpt",
        help="Model checkpoint, default=final_model_M_l_ft_wh_100ep.ckpt"
    )

    return parser.parse_args()


def evaluate_static_quantized_model(model, quantization_config, test_loader, calib_loader):
    
    config_description = repr(quantization_config.global_qconfig)
    print(f"Testing Configuration: {config_description}")
    
    
    example_inputs = next(iter(test_loader))['input']
    model.eval()
    prepared_model = prepare_fx(model, quantization_config, example_inputs)
    
    print("Calibrating the model for static quantization")
    calibrate(prepared_model, calib_loader)
   
    # Convert the model to a quantized version
    quantized_model = convert_fx(prepared_model)

    # Collect predictions and ground truths
    pred_q, true_q = collect_predictions_and_ground_truths(quantized_model, test_loader)

    print("\nMSE:")
    # print("---Original---")
    # calculate_and_print_mse(true_o, pred_o)
    print("---Quantized---")
    calculate_and_print_mse(true_q, pred_q)
    print("----------------------------------\n")

    # Statistical Metrics
    print("Statistical Metrics:")
    #print("---Original---")
    #calculate_and_print_metrics(true_o, pred_o)
    print("---Quantized---")
    calculate_and_print_metrics(true_q, pred_q)
    print("----------------------------------\n")

    # BHS Grading
    print("BHS Grading:")
    #print("---Original---")
    #calculate_and_print_bhs_grades(true_o, pred_o)
    print("---Quantized---")
    calculate_and_print_bhs_grades(true_q, pred_q)
    print("----------------------------------\n")

    # AAMI Validation
    print("AAMI Validation:")
    #print("---Original---")
    #calculate_and_print_aami_validation(true_o, pred_o)
    print("---Quantized---")
    calculate_and_print_aami_validation(true_q, pred_q)
    print("----------------------------------\n")

    # Model Size
    quantized_model_size = calculate_model_size(quantized_model, "static_quantized_model.pt")
    print(f"Static Quantized Model Size: {quantized_model_size:.2f} MB")
    print("------------------------------------------------------------------------------------------------------\n")

    # Inference Times
    #print("Inference Times:")
    #device = "cpu"
    #original_inference_time = calculate_inference_time(model, test_loader, device)
    #quantized_inference_time = calculate_inference_time(quantized_model, test_loader, device)
    #print(f"Original Model Inference Time (per batch): {original_inference_time:.4f} seconds")
    #print(f"Quantized Model Inference Time (per batch): {quantized_inference_time:.4f} seconds")
    #speedup_ratio = original_inference_time / quantized_inference_time
    #print(f"Speedup Ratio: {speedup_ratio:.2f}x")
    #print("----------------------------------")
    
args = parse_args()

torch.backends.quantized.engine = 'qnnpack'

device = "cpu"

seed = 42
torch.manual_seed(seed)


# Create DataLoaders for each split
batch_size = args.batch_size  # Use the same batch size as the original DataLoader
print("Loading dataset")
our_dataset = dataset.ECGPPGDataset(csv_folder=f"/capstor/scratch/cscs/<username>/dataset/{args.dataset}_all", finetune=True, minmax=True, cache_size=1000, dataset=args.dataset)

# Define the split sizes
total_size = len(our_dataset)
train_size = int(0.8 * total_size)  # 80% for training
val_size = int(0.1 * total_size)    # 10% for validation
test_size = total_size - train_size - val_size  # Remaining for testing

# Perform the initial split
train_dataset, val_dataset, test_dataset = random_split(our_dataset, [train_size, val_size, test_size])

# Combine train and val datasets for calibration sampling
combined_indices = train_dataset.indices + val_dataset.indices

# Determine the size of the calib_dataset (3% of the combined set)
calib_size = int(0.03 * len(combined_indices))

# Randomly sample indices for the calib_dataset
calib_indices = random.sample(combined_indices, calib_size)

# Create the calib_dataset
calib_dataset = Subset(our_dataset, calib_indices)

# Create DataLoaders
calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Total dataset size: {total_size}")
print(f"Calibration dataset size: {len(calib_dataset)}")
print(f"Test dataset size: {len(test_dataset)} \n")

print("Loading model and weights")

# Check the model size argument and load the appropriate model
if args.model_size == "small":
    model = our_model.FinetunePretrainedModelSmall(quantize=False)
elif args.model_size == "base":
    model = our_model.FinetunePretrainedModelBase(quantize=False)
elif args.model_size == "large":
    model = our_model.FinetunePretrainedModelLarge(quantize=False)
else:
    raise ValueError(f"Unknown model size: {args.model_size}")

print(model)
model = model.to(torch.float32)
model.eval()

model.to(device)
print(args.model_checkpoint)
print("========================")
checkpoint = torch.load(f'/users/<username>/code/TimeFM/{args.model_checkpoint}')
print(checkpoint.keys())
state_dict = {key.replace("module.", ""): value for key, value in checkpoint['model_state_dict'].items()}
missing_keys, unexpected_keys = model.load_state_dict(state_dict)

# Print warnings for missing keys (these will retain their initialized values)
if missing_keys:
    print(f"Missing keys (kept in their initialized state): {missing_keys}")
if unexpected_keys:
    print(f"Unexpected keys (ignored): {unexpected_keys}")

print("Weights loaded successfully!")

original_model_size = calculate_model_size(model, "original_model.pt")
print(f"Original Model Size: {original_model_size:.2f} MB \n")

pred_o, true_o = collect_predictions_and_ground_truths(model, test_loader)
print("Original MSE:")
calculate_and_print_mse(true_o, pred_o)

print("------------------------------------------------------------------------------------------------------\n")
print("MinMaxObserver & MinMaxObserver \n")

activation_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

print("-------------------------------------- \n")
print("MinMaxObserver & MovingAverageMinMaxObserver \n")

activation_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

print("-------------------------------------- \n")
print("MinMaxObserver & PerChannelMinMaxObserver \n")

activation_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

print("-------------------------------------- \n")
print("MinMaxObserver & MovingAveragePerChannelMinMaxObserver \n")

activation_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

print("-------------------------------------- \n")
print("MovingAverageMinMaxObserver & MinMaxObserver \n")

activation_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

print("-------------------------------------- \n")
print("MovingAverageMinMaxObserver & MovingAverageMinMaxObserver \n")

activation_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
config_dict = {"": qconfig}
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

print("-------------------------------------- \n")
print("MovingAverageMinMaxObserver & PerChannelMinMaxObserver \n")

activation_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

print("-------------------------------------- \n")
print("MovingAverageMinMaxObserver & MovingAveragePerChannelMinMaxObserver \n")

activation_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

print("-------------------------------------- \n")
print("HistogramObserver & MinMaxObserver \n")

activation_observer = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

print("-------------------------------------- \n")
print("HistogramObserver & MovingAverageMinMaxObserver \n")

activation_observer = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

print("-------------------------------------- \n")
print("HistogramObserver & PerChannelMinMaxObserver \n")

activation_observer = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

print("-------------------------------------- \n")
print("HistogramObserver & MovingAveragePerChannelMinMaxObserver \n")

activation_observer = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_symmetric)
weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)

activation_observer = HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader, calib_loader)


