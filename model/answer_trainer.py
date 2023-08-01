import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import json

class AnswerDataset(Dataset):
    def __init__(self, tokenizer, texts, summaries, max_len=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.summaries = summaries
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = str(self.texts[index])
        summary = str(self.summaries[index])
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        
        labels = self.tokenizer.encode_plus(
            summary,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": labels["input_ids"].flatten(),
        }

with open('model/summarized_data.json', 'r') as rawData:
    summarized_data = json.load(rawData)
with open('model/preprocessed_data.json', 'r') as rawData:
    preprocessed_data = json.load(rawData)

train_size = 100
train_texts = [preprocessed_data[f"pair{i+1}"][f"answer{i+1}"] for i in range(0, train_size)]
train_summaries = [summarized_data[f"pair{i+1}"][f"answer{i+1}"] for i in range(0, train_size)]

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

max_seq_length = 512

train_dataset = AnswerDataset(tokenizer, train_texts, train_summaries, max_len=max_seq_length)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(3):
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.save_pretrained("fine_tuned_answer_model/")
tokenizer.save_pretrained("fine_tuned_answer_model/")
