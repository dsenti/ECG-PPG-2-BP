import torch
import argparse
import our_model as our_model
import ecg_ppg_dataset as dataset
from torch.utils.data import random_split, DataLoader
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfig, QConfigMapping, MinMaxObserver, PerChannelMinMaxObserver, MovingAveragePerChannelMinMaxObserver, MovingAverageMinMaxObserver, PlaceholderObserver
from evaluation_metrics import *


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


def evaluate_static_quantized_model(model, quantization_config, test_loader):
    
    config_description = repr(quantization_config.global_qconfig)
    print(f"Testing Configuration: {config_description}")
    
    
    example_inputs = next(iter(test_loader))['input']
    model.eval()
    prepared_model = prepare_fx(model, quantization_config, example_inputs)
   
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

# # Perform the split
train_dataset, val_dataset, test_dataset = random_split(our_dataset, [train_size, val_size, test_size])

# Create DataLoaders for each split
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

activation_observer = PlaceholderObserver.with_args(
    dtype=torch.quint8,
    quant_min=0,
    quant_max=255,
    is_dynamic=True,
)

weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader)

weight_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader)

weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader)

weight_observer = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader)

weight_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader)

weight_observer = MovingAverageMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader)

weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader)

weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine)
qconfig = QConfig(activation=activation_observer, weight=weight_observer)
qconfig_mapping = QConfigMapping().set_global(qconfig)
evaluate_static_quantized_model(model, qconfig_mapping, test_loader)