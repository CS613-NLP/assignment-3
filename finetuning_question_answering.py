from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from transformers import TFAutoModelForQuestionAnswering
from transformers import DefaultDataCollator
from transformers import create_optimizer
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf

raw_datasets = load_dataset("squad_v2")

raw_datasets["train"].filter(lambda x: len(x["answers"]["text"]) != 1)


def remove_no_answers(row):
    if len(row["answers"]["text"]) > 0:
        return True
    else:
        return False


def remove_no_start(row):
    if len(row["answers"]["answer_start"]) > 0:
        return True
    else:
        return False


raw_datasets = raw_datasets.filter(remove_no_answers)
raw_datasets = raw_datasets.filter(remove_no_start)

t1 = raw_datasets["train"]
t2 = raw_datasets["validation"]

raw_datasets = concatenate_datasets([t1, t2])

split_dataset = raw_datasets.class_encode_column("title").train_test_split(
    test_size=0.2, stratify_by_column="title", seed=1
)


tokenizer = AutoTokenizer.from_pretrained("Skratch99/bert-pretrained")

max_length = 384
stride = 128


def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


train_dataset = split_dataset["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=split_dataset["train"].column_names,
)


def preprocess_test_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


test_dataset = split_dataset["test"].map(
    preprocess_test_examples,
    batched=True,
    remove_columns=split_dataset["test"].column_names,
)


model = TFAutoModelForQuestionAnswering.from_pretrained("Skratch99/bert-pretrained")

data_collator = DefaultDataCollator(return_tensors="tf")

tf_train_dataset = model.prepare_tf_dataset(
    train_dataset,
    collate_fn=data_collator,
    shuffle=True,
    batch_size=32,
)
tf_eval_dataset = model.prepare_tf_dataset(
    test_dataset,
    collate_fn=data_collator,
    shuffle=False,
    batch_size=32,
)


# The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
# by the total number of epochs. Note that the tf_train_dataset here is a batched tf.data.Dataset,
# not the original Hugging Face Dataset, so its len() is already num_samples // batch_size.
num_train_epochs = 5
num_train_steps = len(tf_train_dataset) * num_train_epochs
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
)
model.compile(optimizer=optimizer)

# Train in mixed-precision float16
tf.keras.mixed_precision.set_global_policy("mixed_float16")

callback = PushToHubCallback(output_dir="bert-finetuned-squadv2", tokenizer=tokenizer)

# We're going to do validation afterwards, so no validation mid-training

print("Training started")

model.fit(tf_train_dataset, callbacks=[callback], epochs=num_train_epochs, verbose=1)

print("Training ended")
