import warnings

warnings.filterwarnings("ignore")

from transformers import AutoTokenizer
from transformers import LineByLineTextDataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
)
from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback, TrainerState, TrainerControl
import os
import math
import torch

our_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
our_tokenizer.train_new_from_iterator("wikitext_2_raw_v1.txt", 30522)

dataset = LineByLineTextDataset(
    tokenizer=our_tokenizer, file_path="wikitext_2_raw_v1.txt", block_size=128
)

validation_dataset = LineByLineTextDataset(
    tokenizer=our_tokenizer, file_path="wikitext_2_raw_v1_test.txt", block_size=128
)

print("No. of lines: ", len(dataset))


config = BertConfig()

model = BertForMaskedLM(config)
print("No of parameters: ", model.num_parameters())

data_collator = DataCollatorForLanguageModeling(
    tokenizer=our_tokenizer, mlm=True, mlm_probability=0.15
)

perplexities = []

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


class PerplexityCallback(TrainerCallback):
    def __init__(self, model):
        self.epoch = 0
        self.model = model

    def on_epoch_end(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        self.epoch += 1
        print(f"Epoch {self.epoch}")
        print(f"state.log_history: {state.log_history}")


perplexity_callback = PerplexityCallback(model)


training_args = TrainingArguments(
    output_dir="checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    report_to="none",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    logging_first_step=True,
    logging_dir="./some_logs",
    eval_accumulation_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=validation_dataset,
    callbacks=[PerplexityCallback(model)],
)

print("Training started ......")

trainer.train()
#### Please add your Hugging Face repo here:
PATH = "Skratch99/bert-pretrained"
our_tokenizer.push_to_hub(PATH)
model.push_to_hub(PATH)

print("Training completed ......")

# Saving the perplexity values to a text file

with open("perplexities.txt", "w") as f:
    for perplexity in perplexities:
        f.write(str(perplexity) + "\n")

isExist = os.path.exists("pre_trained_model")
if not isExist:
    os.makedirs("pre_trained_model")

trainer.save_model("pre_trained_model")
