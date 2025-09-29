import argparse
import torch
import random
import re
from tqdm import tqdm
from transformers import AutoTokenizer

from train import get_model, VOCAB_SIZE
from data.babi import get_babi_task
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

def evaluate_perplexity(model, device):
    """Evaluates the model's perplexity on the WikiText test set."""
    print("\n--- Evaluating Perplexity on WikiText Test Set ---")
    test_data = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    encodings = tokenizer("\n\n".join(test_data["text"]), return_tensors="pt")
    
    max_length = model.positional_embedding.num_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    with torch.no_grad():
        for i in tqdm(range(0, seq_len, stride), desc="Perplexity"):
            begin_loc = max(0, i + stride - max_length)
            end_loc = min(i + stride, seq_len)
            trg_len = end_loc - i
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs['loss']
            nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"--> Test Perplexity: {ppl.item():.4f}")
    return ppl.item()

def evaluate_babi(model, device, tokenizer, task_id):
    """Evaluates the model on a specific bAbI task, testing multi-hop reasoning."""
    print(f"\n--- Evaluating on bAbI Task {task_id} (Recurrence Check) ---")
    task_data = get_babi_task(task_id, split='test')
    
    correct = 0
    total = len(task_data)
    
    with torch.no_grad():
        for item in tqdm(task_data, desc=f"bAbI Task {task_id}"):
            prompt = f"Context:\n{item['context']}\n\nQuestion: {item['question']}\n\nAnswer:"
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            output_ids = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(output_ids[0][input_ids.size(1):], skip_special_tokens=True)
            answer = generated_text.split('\n')[0].strip().lower()
            
            if answer == item['answer'].lower():
                correct += 1
                
    accuracy = (correct / total) * 100
    print(f"--> bAbI Task {task_id} Accuracy: {accuracy:.2f}%")
    return accuracy

def evaluate_arithmetic(model, device, tokenizer, num_problems=100):
    """Evaluates the model's ability to perform simple arithmetic."""
    print(f"\n--- Evaluating on Algorithmic Arithmetic (Recurrence Check) ---")
    
    # Seed the random number generator to ensure the same questions are generated each time
    random.seed(42) # A fixed seed for reproducibility

    correct = 0
    for _ in tqdm(range(num_problems), desc="Arithmetic"):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        expected_answer = a + b
        
        prompt = f"Question: What is {a} + {b}?\nAnswer:"
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            
        generated_text = tokenizer.decode(output_ids[0][input_ids.size(1):], skip_special_tokens=True)
        
        try:
            # Extract the first number from the generated text
            answer = int(re.findall(r'\d+', generated_text)[0])
            if answer == expected_answer:
                correct += 1
        except (ValueError, IndexError):
            continue
            
    accuracy = (correct / num_problems) * 100
    print(f"--> Arithmetic Accuracy: {accuracy:.2f}%")
    return accuracy

def generate_long_context_text(model, device, tokenizer):
    """Generates text to qualitatively check for long-context coherence."""
    print("\n--- Generating Long-Context Text (Coherence Check) ---")
    prompt = (
        "The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. "
        "It was formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud. "
        "The vast majority of the system's mass is in the Sun, with most of the remaining mass contained in the planet Jupiter. "
        "The four inner terrestrial planets—Mercury, Venus, Earth and Mars—are composed primarily of rock and metal. "
        "The four outer giant planets are substantially more massive than the terrestrials. "
        "The two largest, Jupiter and Saturn, are gas giants, being composed mainly of hydrogen and helium; "
        "the two outermost planets, Uranus and Neptune, are ice giants, being composed mostly of substances with relatively high melting points "
        "compared with hydrogen and helium, called volatiles, such as water, ammonia and methane. All eight planets have nearly circular orbits "
        "that lie within a nearly flat disc called the ecliptic. One of the most interesting aspects of the outer solar system is"
    )
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=300,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("--> Generated Text:")
    print(generated_text)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MoRE model.")
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=['base', 'synthesis_i_option_a', 'synthesis_i_option_b', 'synthesis_ii'],
                        help="The type of model to evaluate.")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to the saved model checkpoint (.pth file).")
    parser.add_argument('--num_recurrences', type=int, default=1,
                        help="Global number of recurrences for recurrent modules.")
    parser.add_argument('--aux_loss_weight', type=float, default=0.01,
                        help="Weight for the auxiliary load balancing loss (must match training).")

    args = parser.parse_args()

    # --- Load Model and Tokenizer ---
    model, device = load_model_for_eval(args.model_type, args.checkpoint_path, args.num_recurrences, args.aux_loss_weight)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Run Evaluations ---
    evaluate_perplexity(model, device)
    
    # Recurrence-specific evaluations
    evaluate_babi(model, device, tokenizer, task_id=2) # Two-supporting facts
    evaluate_babi(model, device, tokenizer, task_id=3) # Three-supporting facts
    evaluate_arithmetic(model, device, tokenizer)

    # Qualitative check
    generate_long_context_text(model, device, tokenizer)

if __name__ == '__main__':
    main() 