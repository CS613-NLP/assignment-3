import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
device

from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("Nokzendi/sst2_finetuned_model")
model = BertForSequenceClassification.from_pretrained(
    "Nokzendi/sst2_finetuned_model"
).to(device)

from datasets import load_dataset, load_metric

train_dataset = load_dataset("glue", "sst2", split="train")

dataset = train_dataset.train_test_split(
    test_size=0.2, stratify_by_column="label", seed=1
)

train_data = dataset["train"]
test_data = dataset["test"]

with torch.no_grad():
    logit_out = model(
        **tokenizer(test_data["sentence"], padding=True, return_tensors="pt").to(device)
    ).logits

preds = logit_out.argmax(dim=1)

import evaluate

acc = evaluate.load("accuracy")
recall = evaluate.load("recall")
prec = evaluate.load("precision")
f1 = evaluate.load("f1")

res = (
    acc.compute(predictions=preds.tolist(), references=test_data["label"]),
    recall.compute(predictions=preds.tolist(), references=test_data["label"]),
    prec.compute(predictions=preds.tolist(), references=test_data["label"]),
    f1.compute(predictions=preds.tolist(), references=test_data["label"]),
)

print(res)
