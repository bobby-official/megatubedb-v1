import csv
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset

class SQLDataset(Dataset):
    """Custom Dataset class for tokenized natural language to SQL data."""
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

def load_data(file_path):
    """Loads data from a CSV file."""
    dataset = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Ensure required columns exist
            if "nl_query" in row and "sql_query" in row:
                dataset.append(row)
            else:
                print(f"Skipping invalid row: {row}")
    return dataset

def tokenize_data(dataset, tokenizer):
    """Tokenizes the dataset using the specified tokenizer."""
    tokenized_data = []
    for example in dataset:
        input_text = example["nl_query"]  # Natural language query
        target_text = example["sql_query"]  # Corresponding SQL query

        # Tokenize input and target
        inputs = tokenizer(input_text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        targets = tokenizer(target_text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")

        tokenized_data.append({
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": targets["input_ids"].squeeze(0),
        })
    return tokenized_data

def fine_tune_model(tokenized_data):
    """Fine-tunes the T5 model on the tokenized dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    train_loader = DataLoader(SQLDataset(tokenized_data), batch_size=8, shuffle=True)
    
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass and calculate loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} loss: {total_loss / len(train_loader)}")

    print("Fine-tuning completed!")

if __name__ == "__main__":
    # Load your dataset
    dataset_file = "nl_to_sql_log_dataset.csv"
    dataset = load_data(dataset_file)
    
    if not dataset:
        print(f"No valid data found in {dataset_file}. Exiting...")
    else:
        print(f"Loaded {len(dataset)} rows from {dataset_file}.")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        tokenized_dataset = tokenize_data(dataset, tokenizer)
        print(f"Tokenized dataset with {len(tokenized_dataset)} entries.")
        fine_tune_model(tokenized_dataset)
