import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

class WikiTextDataset(Dataset):
    def __init__(self, split, seq_length=256):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        
        text = "\n\n".join(filter(None, dataset['text']))
        self.encodings = self.tokenizer(text, return_tensors='pt', truncation=False).input_ids[0]
        
        self.seq_length = seq_length
        # Calculate the number of full sequences we can create
        self.num_sequences = (len(self.encodings) - 1) // self.seq_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        
        input_ids = self.encodings[start_idx:end_idx].clone()
        # The labels are the next token, so we shift the input
        labels = self.encodings[start_idx+1:end_idx+1].clone()
        
        return {"input_ids": input_ids, "labels": labels}

def create_wikitext_dataloader(split, seq_length=256, batch_size=4):
    """Creates a DataLoader for the WikiText dataset."""
    dataset = WikiTextDataset(split=split, seq_length=seq_length)
    # Shuffle only for the training set
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), drop_last=True)
    return dataloader

if __name__ == '__main__':
    # This part is for testing the dataset script independently
    print("Testing WikiText dataloader creation...")
    train_loader = create_wikitext_dataloader(split='train', seq_length=256, batch_size=2)
    
    for batch in train_loader:
        print("Batch shapes:")
        print("Input IDs:", batch['input_ids'].shape)
        print("Labels:", batch['labels'].shape)
        assert torch.equal(batch['input_ids'][0, 1:], batch['labels'][0, :-1])
        print("Test passed: Labels are correctly shifted.")
        break

    print("\nTesting bAbI data loading...")
    from data.babi import get_babi_task
    babi_task_2 = get_babi_task(task_id=2, split='test')
    print(f"Loaded {len(babi_task_2)} QA pairs from bAbI task 2.")
    print("Sample:")
    print(babi_task_2[0])