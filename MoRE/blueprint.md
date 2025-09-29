# MoRE Project: End-to-End Research Blueprint

This document provides a comprehensive step-by-step guide to setting up, training, and evaluating the models in the Mixture of Recurrent Experts (MoRE) project.

## 1. Project Setup

First, ensure you have a suitable Python environment. Then, install all the necessary dependencies from the root of the `MoRE` directory.

```bash
# Navigate to the project's root directory
cd MoRE

# Install all required packages
pip install -r requirements.txt
```

## 2. Phase 1: Training the Models

The core of the training process is the `train.py` script. It is designed to train any of the five architectures. For our baseline experiments, we will train all recurrent models (`more_a` through `more_d`) with **4 recurrent steps**.

The script will automatically save TensorBoard logs to the `runs/` directory and model checkpoints to the `checkpoints/` directory, organized by model type and number of recurrences.

### Standard Training Commands

Execute these commands from the `MoRE` directory to train each model.

*   **1. Train the Baseline Standard MoE Model (`base`)**:
    *(Note: `num_recurrences` has no effect here but is included for consistency in run names).*
    ```bash
    python train.py --model_type base --num_recurrences 1
    ```

*   **2. Train MoRE Option A (Fully Recurrent) with 4 Recurrences**:
    ```bash
    python train.py --model_type more_a --num_recurrences 4
    ```

*   **3. Train MoRE Option B (Sandwiched Recurrent) with 4 Recurrences**:
    ```bash
    python train.py --model_type more_b --num_recurrences 4
    ```

*   **4. Train MoRE Option C (Parallel Branches) with 4 Recurrences**:
    ```bash
    python train.py --model_type more_c --num_recurrences 4
    ```

*   **5. Train MoRE Option D (Sequential Refinement) with 4 Recurrences**:
    ```bash
    python train.py --model_type more_d --num_recurrences 4
    ```

## 3. Phase 2: Monitoring Training
 
It is crucial to monitor the training process, especially for MoE models, to ensure they are learning effectively and that experts are being utilized correctly.

1.  **Launch TensorBoard**: While a model is training (or after it has finished), open a **new terminal** in the `MoRE` directory and run:
    ```bash
    tensorboard --logdir=runs
    ```

2.  **Analyze in Browser**: Open the URL provided by TensorBoard (e.g., `http://localhost:6006/`) to view the training dashboards.

    **Key Metrics to Watch:**
    *   `Loss/train` and `Perplexity`: Should decrease over time.
    *   `Layer_*/Expert_Load_Distribution`: Check if all experts are receiving tokens. Bars of similar height are ideal. A "collapsed" expert will have a bar at or near zero.
    *   `Layer_*/Router_Entropy`: A high, flat entropy may indicate the router is not learning to specialize. A decreasing entropy is often a sign of healthy specialization.
    *   `Layer_*/Load_Balance_CV`: A quantitative measure of load balance. Lower is generally better, but not at the cost of high router entropy.

## 4. Phase 3: Evaluating Trained Models

After a model has been trained and a checkpoint is saved, use the `evaluate.py` script to assess its performance on a range of tasks. This script is designed to test both general language modeling ability and the specific benefits of recurrence.

### Standard Evaluation Command

To run the evaluation, you need to provide the model type and the path to the saved checkpoint file.

1.  **Find the Checkpoint Path**: After training, the script will print the path to the saved model. It will look something like this:
    `checkpoints/more_a_rec4/final_model.pth`

2.  **Run Evaluation**: Use the path from the previous step in the command below.

    ```bash
    # Replace the model_type and checkpoint_path with your specific model's details
    python evaluate.py --model_type [model_type] --num_recurrences [num_recurrences] --checkpoint_path [path/to/your/checkpoint.pth]
    ```

### Full Workflow Example (for `more_a`)

Here is a complete, end-to-end example for training and then evaluating the `more_a` model.

*   **Step 1: Train the model.**
    ```bash
    python train.py --model_type more_a --num_recurrences 4
    ```
    *Let this run to completion. At the end, it will print the path to the saved checkpoint, for example: `checkpoints/more_a_rec4/final_model.pth`.*

*   **Step 2: Evaluate the trained model.**
    *(Use the actual path from the output of the previous command).*
    ```bash
    python evaluate.py --model_type more_a --num_recurrences 4 --checkpoint_path checkpoints/more_a_rec4/final_model.pth
    ```

The evaluation script will run through all the tests and print the results for:
*   **Perplexity** on the WikiText test set.
*   **Accuracy on bAbI tasks 2 & 3** to test multi-hop reasoning.
*   **Accuracy on an arithmetic task** to test algorithmic reasoning.
*   A **qualitative text generation sample** to check for long-context coherence.

Repeat this train-and-evaluate cycle for all model variants to gather the data needed for a full comparison.

## 5. Phase 4: Advanced Evaluation and Visualization

For a deeper, more qualitative analysis, the `evaluate_for_llm.py` script provides two key features: generating a CSV of model predictions for analysis by another LLM and visualizing the expert routing for a given input.

### Generating Evaluation Data for LLM Analysis

This script runs a bAbI evaluation and saves the context, question, true answer, and predicted answer to a CSV file. This format is ideal for feeding into a more powerful LLM (like GPT-4) to get a qualitative assessment of the model's reasoning abilities.

1.  **Run the Advanced Evaluation Script**:
    ```bash
    # Example for more_b model
    python MoRE/evaluate_for_llm.py \
      --model_type more_b \
      --checkpoint_path MoRE/checkpoints/more_b_rec4/final_model.pth \
      --num_recurrences 4 \
      --babi_task_id 2 \
      --output_csv evaluation_results.csv
    ```

2.  **Analyze the Output**:
    The script will generate `evaluation_results.csv`, which you can then inspect or use in another analysis pipeline.

### Visualizing Expert Token Routing

To better understand how the model is utilizing its experts, the same script can generate a **Sankey diagram** that shows the path of each token through the MoE layers.

*   When you run the `evaluate_for_llm.py` script, it will automatically generate and display a Sankey diagram for the **first sample** in the evaluation set.
*   This visualization helps you intuitively grasp which experts are being activated for different parts of the input text, providing a much clearer picture of expert specialization and load balancing than raw metrics alone. 