# Assignment-3
This repository contains the python scripts and notebooks for Assignment-3 NLP.

The assignment is divided into 3 main sections:
- **Pretraining** the model(bert-base-uncased) on given dataset (wikitext_raw_2_v1)
- **Finetuning** the pretrained model for specific tasks (Classification and Question & Answering)
- **Evaluation** of finetuned models using specified metrics (Classification: Accuracy, Precision, Recall, F1 ; Question-Answering: squad_v2, F1, METEOR, BLEU, ROUGE, exact-match)

To find all the respective code files, refer to the below file structure:

### Pre-training
- Pre-training Dataset: [wikitext_raw_2_v1](wikitext_raw_2_v1.txt)
- Pre-training Dataset Eval: [wikitext_raw_2_v1_test](wikitext_raw_2_v1_test.txt)
- Pre-training Code File: [pretraining](pretraining.py)
- Pre-training Perplexities on Test Dataset: [pretrained_perplexities](pretrained_perplexities.txt)

### Fine-tuning
- Classification Task Finetuning: [finetuning_classification](finetuning_classification.py)
- Question & Answering Task Finetuning: [finetuning_question_answering](finetuning_question_answering.py)

### Evaluation
- Evaluation Pre-trained model: [eval_pretrain](eval_pretrain.py)
- Evaluation Classification Finetuned model: [eval_classification](eval_classification.py)
- Evaluation Question & Answering Finetuned model: [eval_question_answering](eval_question_answering.py)

### Miscellaneous
- Calculating Parameters for models: [parameter_calculation](parameter_calculation.py)


Our pre-trained model is pushed on Hugging Face as: [Skratch99/bert-pretrained](https://huggingface.co/Skratch99/bert-pretrained)

Our fine-tuned model for classification is pushed as: [Nokzendi/bert_sst2_finetuned](https://huggingface.co/Nokzendi/bert_sst2_finetuned)

Our fine-tuned model for question & answering is pushed as: [Skratch99/finetuned-bert-squadv2](https://huggingface.co/Skratch99/finetuned-bert-squadv2)

The assignment [report](report.pdf) can be found here.
