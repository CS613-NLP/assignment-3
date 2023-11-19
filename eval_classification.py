import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset, load_metric
import evaluate
from tqdm import tqdm
import concurrent.futures


device='cuda' if torch.cuda.is_available() else 'cpu'


tokenizer = BertTokenizer.from_pretrained('Skratch99/bert-pretrained')
model = BertForSequenceClassification.from_pretrained("Nokzendi/bert_sst2_finetuned").to(device)


train_dataset = load_dataset('glue', 'sst2', split='train')


dataset = train_dataset.train_test_split(test_size=0.2, stratify_by_column="label" , seed=1)


train_data = dataset['train']
test_data = dataset['test']


acc = evaluate.load('accuracy')
recall = evaluate.load('recall')
prec = evaluate.load('precision')
f1 = evaluate.load('f1')


predictions = []


def process_example(example):

    with torch.no_grad():
      logit_out = model(**tokenizer(example['sentence'], padding=True, return_tensors='pt').to(device)).logits

    preds = logit_out.argmax(dim=1).detach().tolist()
    predictions.append({'prediction': preds[0], 'answer': example['label']})


num_processes = 4

with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
    list(tqdm(executor.map(process_example, test_data), total=len(test_data['sentence'])))


preds = []
theo = []
for i in predictions:
  preds.append(i['prediction'])
  theo.append(i['answer'])


res = (acc.compute(predictions=preds, references=theo),
recall.compute(predictions=preds, references=theo),
prec.compute(predictions=preds, references=theo),
f1.compute(predictions=preds, references=theo))


print(res)

