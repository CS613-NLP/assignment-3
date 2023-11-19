from transformers import (
    BertForQuestionAnswering,
    BertForPreTraining,
    BertForSequenceClassification,
)

model_pretrained = BertForPreTraining.from_pretrained("Skratch99/bert-pretrained")
model_classify = BertForSequenceClassification.from_pretrained(
    "Nokzendi/bert_sst2_finetuned"
)
model_qna = BertForQuestionAnswering.from_pretrained(
    "Skratch99/finetuned-bert-squadv2", from_tf=True
)

print("Parameters of pre trained model:", model_pretrained.num_parameters())
print(
    "Parameters of fine tuned on classification model", model_classify.num_parameters()
)
print("Parameters of fine tuned on QnA model:", model_qna.num_parameters())
