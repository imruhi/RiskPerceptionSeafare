from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import json
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification,Trainer, TrainingArguments, AutoConfig
import torch
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("params.json", 'r') as f:
    PARAMS = json.load(f)

def evaluate_model(label2id, id2label, val_data):
    model_id = PARAMS["classi_finetune_model"]
    model_id_finetuned = f'{PARAMS["save_model"]}{model_id.split("/")[-1]}_finetuned'
    tokenizer = AutoTokenizer.from_pretrained(model_id, max_length = 512, padding=True, Truncation=True)
    config = AutoConfig.from_pretrained(model_id, num_labels=len(label2id))
    config.label2id = label2id
    config.id2label = id2label

    model_baseline = AutoModelForSequenceClassification.from_pretrained(model_id, config=config)
    pipeline_baseline = pipeline(task="text-classification", model=model_baseline, device=device, tokenizer=tokenizer, max_length=512, truncation=True)

    model_finetuned = AutoModelForSequenceClassification.from_pretrained(model_id_finetuned, config=config)
    pipeline_finetuned = pipeline(task="text-classification", model=model_finetuned, device=device, tokenizer=tokenizer, max_length=512, truncation=True)

    baseline_answers = []
    for out in pipeline_baseline(KeyDataset(val_data, "text"), batch_size=8):
        baseline_answers.append(label2id[out["label"]])    

    finetuned_answers = []
    for out in pipeline_finetuned(KeyDataset(val_data, "text"), batch_size=8):
        finetuned_answers.append(label2id[out["label"]])    

    baseline_cr = classification_report(val_data["label"], baseline_answers)
    f = open(model_id_finetuned+'/baseline_cr.txt', 'w')
    f.write('{}\n\nClassification Report\n\n{}'.format(label2id, baseline_cr))
    f.close()

    finetuned_cr = classification_report(val_data["label"], finetuned_answers)
    f = open(model_id_finetuned+'/finetuned_cr.txt', 'w')
    f.write('{}\n\nClassification Report\n\n{}'.format(label2id, finetuned_cr))
    f.close()


    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=val_data["label"],
        y_pred=finetuned_answers,
        normalize='true'
    )
    disp.plot()
    plt.title("Finetuned")
    plt.savefig(model_id_finetuned+"/finetuned.png")

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=val_data["label"],
        y_pred=baseline_answers,
        normalize="true"
    )
    disp.plot()
    plt.title("Baseline")
    plt.savefig(model_id_finetuned+"/baseline.png")