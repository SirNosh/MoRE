import os
from datasets import load_dataset, Dataset

# --- Configuration ---
SEED = 42
NUM_TOKENS_TARGET = 160_000_000_000
# The Huginn model uses a block size of 4096, so we assume each example is roughly this many tokens.
TOKENS_PER_EXAMPLE = 4096
# IMPORTANT: Use a path in a shared, high-performance filesystem.
# /data/temp-scratch is a good choice. Replace 'dvyas4' with your username.
SAVE_PATH = "/data/temp-scratch/dvyas4/huginn_160B_seeded"

# --- Calculation ---
num_samples_to_select = int(NUM_TOKENS_TARGET / TOKENS_PER_EXAMPLE)

print(f"Loading original dataset 'tomg-group-umd/huginn-dataset'...")
# Load the dataset in streaming mode to avoid downloading the entire thing at once
original_dataset = load_dataset("tomg-group-umd/huginn-dataset", split="train", streaming=True)

print(f"Shuffling with seed {SEED} and selecting {num_samples_to_select:,} samples for ~160B tokens...")
# This step iterates through the dataset, shuffles the order, and takes the required number of samples.
subset_iterable = original_dataset.shuffle(seed=SEED).take(num_samples_to_select)

# Convert the iterable into a standard list. This will trigger the download and can take time.
print("Downloading and materializing the subset...")
subset_list = list(subset_iterable)
subset_dataset = Dataset.from_list(subset_list)
print("Download complete.")

print(f"Saving the final 160B token dataset to: {SAVE_PATH}")
# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
subset_dataset.save_to_disk(SAVE_PATH)

print("\nPreprocessing complete!")
print(f"Your reproducible dataset is now saved and ready for all training runs at: {SAVE_PATH}")


