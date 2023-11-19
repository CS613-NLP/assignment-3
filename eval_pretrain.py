from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
)
from transformers import BertForMaskedLM

our_tokenizer = AutoTokenizer.from_pretrained("Skratch99/bert-pretrained")

validation_dataset = LineByLineTextDataset(
    tokenizer=our_tokenizer, file_path="wikitext_2_raw_v1_test.txt", block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=our_tokenizer, mlm=True, mlm_probability=0.15
)


model_id = "Skratch99/bert-pretrained"
model = BertForMaskedLM.from_pretrained(model_id)
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=16,
    ),
    data_collator=data_collator,
)

results = trainer.evaluate(eval_dataset=validation_dataset)
print(results)
