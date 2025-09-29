import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Project-specific imports
from data.dataset import create_wikitext_dataloader
from models.moe import BaseMoEModel
from models.more_synthesis_i_option_a import MoREModelSynthesisIOptionA
from models.more_synthesis_i_option_b import MoREModelSynthesisIOptionB
from models.more_synthesis_ii import MoREModelSynthesisII

# --- Model Configuration ---
VOCAB_SIZE = 50257
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 6
NUM_EXPERTS = 8
TOP_K = 1
DROPOUT = 0.1

def set_seed(seed):
    """Set a random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(model_type, num_recurrences, aux_loss_weight=0.01):
    """Factory function to get the correct model."""
    common_args = {
        "vocab_size": VOCAB_SIZE,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "dropout": DROPOUT,
    }
    
    if model_type == 'base':
        print("Initializing Base MoE Model...")
        return BaseMoEModel(**common_args, aux_loss_weight=aux_loss_weight)
    elif model_type == 'synthesis_i_option_a':
        print(f"Initializing MoRE Synthesis I Option A (Independent Recurrent Experts) with {num_recurrences} recurrences...")
        return MoREModelSynthesisIOptionA(**common_args, num_recurrences=num_recurrences, aux_loss_weight=aux_loss_weight)
    elif model_type == 'synthesis_i_option_b':
        print(f"Initializing MoRE Synthesis I Option B (Shared Recurrent Block with Projections) with {num_recurrences} recurrences...")
        return MoREModelSynthesisIOptionB(**common_args, num_recurrences=num_recurrences, aux_loss_weight=aux_loss_weight)
    elif model_type == 'synthesis_ii':
        print(f"Initializing MoRE Synthesis II (MoE Layer as Recurrent Unit) with {num_recurrences} iterations...")
        return MoREModelSynthesisII(**common_args, num_iters=num_recurrences, aux_loss_weight=aux_loss_weight)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train(args):
    """Main training loop with visualization."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- TensorBoard Setup ---
    run_name = f"{args.model_type}_rec{args.num_recurrences}"
    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # --- Checkpoint Setup ---
    checkpoint_dir = os.path.join("checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    # --- Data ---
    print("Setting up dataloader...")
    dataloader = create_wikitext_dataloader(split='train', seq_length=args.seq_length, batch_size=args.batch_size)
    
    # --- Model ---
    model = get_model(args.model_type, args.num_recurrences, args.aux_loss_weight)
    model.to(device)
    
    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader))

    # --- Training Loop ---
    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']
            
            if loss is None:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # --- Logging and Visualization ---
            if global_step % args.log_interval == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Loss/auxiliary', outputs['aux_loss'].item(), global_step)
                writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)
                writer.add_scalar('Perplexity', torch.exp(loss).item(), global_step)

                # Log expert utilization metrics
                for i, layer_logits in enumerate(outputs['router_logits']):
                    # Expert Load: Which expert gets how many tokens?
                    expert_choices = torch.argmax(layer_logits, dim=-1)
                    writer.add_histogram(f'Layer_{i}/Expert_Load_Distribution', expert_choices, global_step, max_bins=NUM_EXPERTS)
                    
                    # Router Entropy: How confident is the router?
                    probs = torch.softmax(layer_logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
                    writer.add_scalar(f'Layer_{i}/Router_Entropy', entropy.item(), global_step)

                    # Coefficient of Variation (CV) for load balancing
                    expert_counts = torch.bincount(expert_choices, minlength=NUM_EXPERTS).float()
                    if expert_counts.mean() > 0:
                        cv = expert_counts.std() / expert_counts.mean()
                        writer.add_scalar(f'Layer_{i}/Load_Balance_CV', cv.item(), global_step)

                # Log model-specific metrics
                if args.model_type == 'synthesis_ii':
                    # For Synthesis II, log iteration-specific metrics
                    if 'expert_usage' in outputs and outputs['expert_usage']:
                        for layer_idx, layer_usage in enumerate(outputs['expert_usage']):
                            if isinstance(layer_usage, list) and len(layer_usage) > 0:
                                # Log usage across iterations
                                for iter_idx, usage in enumerate(layer_usage):
                                    writer.add_scalar(f'Layer_{layer_idx}/Iteration_{iter_idx}/Expert_Usage', usage.mean().item(), global_step)

            global_step += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    # --- Save Final Model ---
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    
    writer.close()
    print("\nTraining finished.")
    print(f"Final model saved to {final_model_path}")
    print(f"To view logs, run: tensorboard --logdir={os.path.join(os.getcwd(), 'runs')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Mixture of Recurrent Experts (MoRE) model.")
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=['base', 'synthesis_i_option_a', 'synthesis_i_option_b', 'synthesis_ii'],
                        help="The type of model to train.")
    parser.add_argument('--num_recurrences', type=int, default=1,
                        help="Global number of recurrences for recurrent modules (or iterations for Synthesis II).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training.")
    parser.add_argument('--seq_length', type=int, default=256, help="Sequence length for training.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--log_interval', type=int, default=50, help="Log progress to TensorBoard every N steps.")
    parser.add_argument('--aux_loss_weight', type=float, default=0.01, help="Weight for the auxiliary load balancing loss.")

    args = parser.parse_args()
    # Correct the directory for the tensorboard command
    # We need to cd into the MoRE directory first.
    # The command should be run from the MoRE directory.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train(args) 