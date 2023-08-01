import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from rouge import Rouge
import json

tokenizer = BartTokenizer.from_pretrained("fine_tuned_question_model/")
model = BartForConditionalGeneration.from_pretrained("fine_tuned_question_model/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with open('model/summarized_data.json', 'r') as rawData:
    summarized_data = json.load(rawData)
with open('model/reprocessed_data.json', 'r') as rawData:
    preprocessed_data = json.load(rawData)

questions = [preprocessed_data[f"pair{i+1}"][f"question{i+1}"] for i in range(100, 130)]
ground_truth_summaries = [summarized_data[f"pair{i+1}"][f"question{i+1}"] for i in range(100, 130)]

rouge_scores = []
generated_summaries = []

for i in range(0, len(questions)):
    inputs = tokenizer(questions[i], return_tensors="pt", max_length=1024, padding='max_length', truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    summary_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    generated_summaries.append(summary)

    # Validation
    rouge = Rouge()
    rouge_score = rouge.get_scores(summary, ground_truth_summaries[i])[0]["rouge-l"]["f"]
    rouge_scores.append(rouge_score)

print(f"Average Rouge Score: {sum(rouge_scores)/len(rouge_scores):.4f}")

