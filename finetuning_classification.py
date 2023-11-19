import os
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, BertTokenizer, BertConfig, TFBertModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from datasets import load_dataset,load_metric


####################################################
# MAKE DIRECTORY

## Load the dataset (CAN CHANGE TO GET THE ENTIRE DATASET BY CONCATENATING DIFFERENT SPLITS)
train_dataset = load_dataset('glue', 'sst2', split='train')

dataset = train_dataset.train_test_split(test_size=0.2, stratify_by_column="label" , seed=1)

train_data = dataset['train']
test_data = dataset['test']


train_dataset = train_data.map(lambda examples: {'labels': examples['label']}, batched=True)
test_dataset = test_data.map(lambda examples: {'labels': examples['label']}, batched=True)

test_dataset = test_dataset.remove_columns(['label'])
train_dataset = train_dataset.remove_columns(['label'])

# tokenizer = BertTokenizer.from_pretrained('/content/drive/MyDrive/ES613_Assignment_3')
tokenizer = AutoTokenizer.from_pretrained("Skratch99/bert-pretrained")

import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print(device)

model = BertForSequenceClassification.from_pretrained("Skratch99/bert-pretrained").to(device)

# Change the max_length according to your wish
MAX_LENGTH = 256
train_dataset = train_dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
test_dataset = test_dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

# train_dataset = train_dataset.to(device)
# test_dataset = test_dataset.to(device)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


#start training
training_args = TrainingArguments(
    output_dir='sst2_results',          #output directory
    learning_rate=1e-4,
    num_train_epochs=10,
    per_device_train_batch_size=32,                #batch size per device during training
    per_device_eval_batch_size=32,                #batch size for evaluation
    logging_dir='sst2_logs',
    logging_steps=100,
    do_train=True,
    do_eval=False,
    no_cuda=False,
    load_best_model_at_end=False,
    save_strategy = "epoch",
    evaluation_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    compute_metrics=compute_metrics
)

train_out = trainer.train()

PATH = "bert_sst2_finetuned"
# PATH = None # Set the path to the huggingface here to save the model

model.push_to_hub(PATH)

results = trainer.evaluate(eval_dataset = test_dataset)
print("Test results on test_dataset:", results)


train_results = trainer.evaluate(eval_dataset = train_dataset)
print("Test results on train_dataset:", train_results)