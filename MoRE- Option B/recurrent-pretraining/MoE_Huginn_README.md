# MoE-Huginn Training Guide

## Overview

This document provides a comprehensive guide to setting up, training, and evaluating the MoE-Huginn model.

This model integrates a Mixture-of-Experts (MoE) architecture with a shared expert into the recurrent-depth framework of the Huginn model. The implementation is designed to be compatible with both NVIDIA and AMD hardware.

## 1. Setup and Dependencies

### 1.1. Clone the Repository
First, clone the project repository to your local machine:
```bash
git clone <your-repository-url>
cd recurrent-pretraining
```

### 1.2. Install Dependencies
It is recommended to use a Python virtual environment. This project requires Python 3.11 or newer.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install core dependencies
pip install numpy einops lightning jsonargparse requests tensorboard torchmetrics lm-eval wandb sentencepiece tokenizers safetensors datasets transformers pandas plotly ninja torchdata

# Install PyTorch (select the command for your hardware)
# For NVIDIA GPUs (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For AMD GPUs (ROCm)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# Install hardware-specific acceleration libraries
# For NVIDIA GPUs, install FlashAttention for optimal performance
pip install flash-attn --no-build-isolation
```

## 2. Dataset

The model is configured to train on the `tomg-group-umd/huginn-dataset`, which was created by the authors of the original Huginn paper.

The training script will automatically handle downloading this dataset from the Hugging Face Hub when you start a training run. No manual download is required. The configuration is set to train on **800 billion tokens** from this dataset, which is more than sufficient for the ~3.8B parameter model.

## 3. Training

### 3.1. Weights & Biases Login
This project uses Weights & Biases (W&B) for logging metrics. Before starting, log in to your W&B account:
```bash
wandb login
```
You will be prompted for your API key. Training metrics, including the MoE and recurrence-specific metrics, will be viewable at your configured `wandb_project`, which is set to "moe-huginn" in the config file.

### 3.2. Launching Training
The recommended way to launch multi-GPU training is using `torchrun`. The training process is controlled by the `config/moe_huginn_config.yaml` file.

To start training on a machine with 8 GPUs, use the following command:
```bash
torchrun --nproc_per_node=8 train.py --config_file config/moe_huginn_config.yaml
```
The script will automatically configure the distributed training environment, and each GPU will work on a piece of the batch.

## 4. Evaluation and Testing

### 4.1. Evaluating with a Specific Recurrence Depth (r=8)

You do **not** need to modify training to test at a specific recurrence depth. The model is trained on a randomly sampled number of recurrences for each batch, which allows it to generalize well to any recurrence count at test time.

To evaluate a trained checkpoint with a specific recurrence depth, such as `r=8`, you can use the `evaluate_raven/quick_checkpoint_eval.py` script.

Example command:
```bash
python evaluate_raven/quick_checkpoint_eval.py \
    --checkpoint_name "/path/to/your/checkpoint.pth" \
    --tokenizer_path "/path/to/your/tokenizer" \
    --tasks "arc_easy,hellaswag,tinyMMLU" \
    --recurrence 8 \
    --batch_size 16
```
- `--checkpoint_name`: Path to the saved model checkpoint file.
- `--tokenizer_path`: Path to the tokenizer used during training.
- `--tasks`: A comma-separated list of evaluation tasks.
- `--recurrence`: The number of recurrent steps to use for the evaluation. Set this to `8` for your test case.

### 5. Hardware Compatibility (NVIDIA & AMD)

This implementation is designed to be hardware-agnostic and works on both NVIDIA and AMD GPUs.

- **NVIDIA:** For the best performance on NVIDIA hardware, make sure you install the `flash-attn` library as specified in the dependency section. PyTorch 2.0+ will automatically use it as the backend for scaled dot-product attention, resulting in significant speedups.
- **AMD:** The code is also compatible with AMD's ROCm stack. The core logic uses standard PyTorch functions that are portable across hardware.

### 6. `MoEHuginnConfig700M` Configuration

A new configuration is available for a 700M parameter version of the MoE-Huginn model. You can use it as follows:

```python
from recpre.config_dynamic import MoEHuginnConfig700M

# Instantiate the configuration
config = MoEHuginnConfig700M()

# You can now use this config object to initialize the model.
# For example, if you were creating the model directly:
# from recpre.moe_model_dynamic import MoEHuginnModel
# model = MoEHuginnModel(config)
```
This model is designed to provide a smaller, more manageable version of the MoE-Huginn architecture for faster experimentation and deployment on smaller hardware.