import argparse
import torch
import random
import re
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
from transformers import AutoTokenizer

from train import get_model, VOCAB_SIZE
from data.babi import get_babi_task, BabiTask
from datasets import load_dataset


def load_model_for_eval(model_type, checkpoint_path, num_recurrences, aux_loss_weight):
    """Loads a model and its checkpoint for evaluation."""
    print(f"Loading model '{model_type}' from checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate model architecture
    model = get_model(model_type, num_recurrences, aux_loss_weight)
    
    # Load the saved weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def generate_sankey_diagram(routing_indices, tokenizer, input_text):
    """Generates a Sankey diagram to visualize token routing."""
    
    # routing_indices is a list of tensors, one for each layer
    # Each tensor shape: [1, seq_len, 1]
    
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))
    num_layers = len(routing_indices)
    seq_len = len(tokens)

    labels = []
    # Input tokens
    labels.extend([f"L0_T{i}: {tokens[i]}" for i in range(seq_len)])
    # Experts in each layer
    for layer_idx in range(num_layers):
        num_experts = routing_indices[layer_idx].max().item() + 1
        labels.extend([f"L{layer_idx+1}_E{expert_idx}" for expert_idx in range(num_experts)])

    source = []
    target = []
    value = []
    
    node_map = {label: i for i, label in enumerate(labels)}

    # From input tokens to first layer of experts
    for token_idx in range(seq_len):
        expert_idx = routing_indices[0][0, token_idx, 0].item()
        source.append(node_map[f"L0_T{token_idx}: {tokens[token_idx]}"])
        target.append(node_map[f"L1_E{expert_idx}"])
        value.append(1)

    # Between expert layers
    for layer_idx in range(num_layers - 1):
        num_experts_prev_layer = routing_indices[layer_idx].max().item() + 1
        
        # Calculate the starting index of the current layer's nodes
        prev_layers_expert_count = sum(routing_indices[l].max().item() + 1 for l in range(layer_idx))
        source_layer_start_idx = seq_len + prev_layers_expert_count

        # Calculate the starting index of the next layer's nodes
        target_layer_start_idx = source_layer_start_idx + num_experts_prev_layer

        for token_idx in range(seq_len):
            source_expert = routing_indices[layer_idx][0, token_idx, 0].item()
            target_expert = routing_indices[layer_idx+1][0, token_idx, 0].item()
            
            source.append(source_layer_start_idx + source_expert)
            target.append(target_layer_start_idx + target_expert)
            value.append(1)


    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = labels,
          color = "blue"
        ),
        link = dict(
          source = source,
          target = target,
          value = value
      ))])

    fig.update_layout(title_text="Token Routing through MoE Layers", font_size=10)
    fig.show()


def evaluate_and_store_babi(model, device, tokenizer, task_id, output_csv_path):
    """Evaluates the model on a bAbI task and stores the results in a CSV."""
    print(f"\n--- Evaluating on bAbI Task {task_id} and Storing Results ---")
    babi_task = BabiTask(task_id)
    task_data = babi_task.get_split('test')

    results = []
    correct = 0
    total = len(task_data)

    with torch.no_grad():
        for item in tqdm(task_data, desc=f"bAbI Task {task_id}"):
            prompt = f"Context:\n{item['context']}\n\nQuestion: {item['question']}\n\nAnswer:"
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

            # Check if the model has the 'generate_with_routing' method
            if hasattr(model, 'generate_with_routing'):
                 outputs = model.generate_with_routing(
                    input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                 output_ids = outputs["sequences"]
                 routing_indices = outputs["routing_indices"]
            else:
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                routing_indices = None

            generated_text = tokenizer.decode(output_ids[0][input_ids.size(1):], skip_special_tokens=True)
            predicted_answer = generated_text.split('\n')[0].strip().lower()
            
            is_correct = (predicted_answer == item['answer'].lower())
            if is_correct:
                correct += 1

            results.append({
                "context": item['context'],
                "question": item['question'],
                "true_answer": item['answer'],
                "predicted_answer": predicted_answer,
                "is_correct": is_correct
            })

            # Generate and show Sankey diagram for the first item
            if routing_indices and len(results) == 1:
                # We need to get the routing for the input prompt, not the generated part
                with torch.no_grad():
                    prompt_output = model(input_ids, output_routing_indices=True)
                prompt_routing = prompt_output["routing_indices"]
                generate_sankey_diagram(prompt_routing, tokenizer, prompt)

    accuracy = (correct / total) * 100
    print(f"--> bAbI Task {task_id} Accuracy: {accuracy:.2f}%")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MoRE model and store results.")
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['base', 'synthesis_i_option_a', 'synthesis_i_option_b', 'synthesis_ii'],
                        help="The type of model to evaluate.")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the saved model checkpoint (.pth file).")
    parser.add_argument('--num_recurrences', type=int, default=4,
                        help="Global number of recurrences for recurrent modules.")
    parser.add_argument('--aux_loss_weight', type=float, default=0.01,
                        help="Weight for the auxiliary load balancing loss.")
    parser.add_argument('--babi_task_id', type=int, default=2,
                        help="The bAbI task ID to evaluate.")
    parser.add_argument('--output_csv', type=str, default='evaluation_results.csv',
                        help="Path to save the evaluation results CSV file.")
    
    args = parser.parse_args()

    # --- Load Model and Tokenizer ---
    model, device = load_model_for_eval(args.model_type, args.checkpoint_path, args.num_recurrences, args.aux_loss_weight)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Run Evaluation and Store Results ---
    evaluate_and_store_babi(model, device, tokenizer, args.babi_task_id, args.output_csv)

if __name__ == '__main__':
    main() 