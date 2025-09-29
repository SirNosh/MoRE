# MoRE: Mixture of Recurrent Experts

This project provides an implementation of a standard Mixture of Experts (MoE) model and four variants of a "Mixture of Recurrent Experts" (MoRE) architecture, based on the research plans provided. Each model is designed to be trained and evaluated on the WikiText-103 dataset.

The goal is to explore how replacing or augmenting standard feed-forward experts with recurrent blocks affects model performance and training dynamics.

## Implemented Architectures

1.  **Standard MoE (`base`)**: The baseline model with standard feed-forward network (FFN) experts.
2.  **MoRE Option A (`more_a`)**: Each FFN expert is completely replaced by a Huginn-style recurrent block.
3.  **MoRE Option B (`more_b`)**: A "sandwich" model where a GRU recurrently processes the high-dimensional state between two linear layers.
4.  **MoRE Option C (`more_c`)**: Each expert has two parallel branches (FFN and recurrent) whose outputs are combined by a learned gate.
5.  **MoRE Option D (`more_d`)**: A sequential model where the output of an FFN is iteratively refined by a recurrent block.

## Setup

1.  Navigate to the `MoRE` project directory.
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Training

The main `train.py` script is used to train any of the implemented models. You can select the model architecture and the number of recurrent steps for the MoRE variants.

### Basic Training Commands

*   **Train the baseline MoE model**:
    ```bash
    python train.py --model_type base
    ```

*   **Train MoRE Option A with 3 recurrent steps**:
    ```bash
    python train.py --model_type more_a --num_recurrences 3
    ```

*   **Train MoRE Option B with 2 recurrent steps**:
    ```bash
    python train.py --model_type more_b --num_recurrences 2
    ```

*   **Train MoRE Option C with 2 recurrent steps**:
    ```bash
    python train.py --model_type more_c --num_recurrences 2
    ```

*   **Train MoRE Option D with a single refinement step**:
    ```bash
    python train.py --model_type more_d --num_recurrences 1
    ```

### Training Arguments

You can customize the training run with several command-line arguments:

*   `--model_type`: (Required) Choose from `base`, `more_a`, `more_b`, `more_c`, `more_d`.
*   `--num_recurrences`: The number of steps for the recurrent experts (default: 1).
*   `--epochs`: Number of training epochs (default: 1).
*   `--batch_size`: Training batch size (default: 4).
*   `--seq_length`: Model sequence length (default: 256).
*   `--lr`: Learning rate (default: 1e-4).
*   `--seed`: Random seed for reproducibility (default: 42).
*   `--log_interval`: How often to log to TensorBoard (default: 50 steps).

**Example of a custom run:**
```bash
python train.py --model_type more_a --num_recurrences 4 --epochs 3 --batch_size 8 --lr 5e-5
```

## Visualization

This project uses TensorBoard to visualize training metrics and analyze model behavior, which is crucial for understanding MoE architectures.

1.  **Launch TensorBoard**: While training is running (or after it finishes), open a new terminal in the `MoRE` directory and run:
    ```bash
    tensorboard --logdir=runs
    ```
2.  **View in Browser**: Open the URL provided by TensorBoard (usually `http://localhost:6006/`) in your web browser.

In TensorBoard, you can track:
*   Loss, perplexity, and learning rate.
*   **Expert Load Distribution**: Histograms showing which experts are being used in each layer. This is critical for identifying "expert collapse," where some experts are over- or under-utilized.
*   **Router Entropy**: A measure of the router's confidence.
*   **Load Balance CV**: A quantitative measure of how balanced the token distribution is across experts. 