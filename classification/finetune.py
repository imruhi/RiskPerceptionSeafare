import pandas as pd
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification,Trainer, TrainingArguments, AutoConfig
import torch
from collections import Counter
import evaluate 
import json
from .utils_finetune import load_dataset, plot_tokens, split_dataset
from .evaluate_model import evaluate_model
from sklearn.metrics import accuracy_score as accuracy_score_sklearn, precision_recall_fscore_support
import numpy as np
import warnings
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"Emptying torch cache")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score_sklearn(labels, preds)
    classes = np.unique(labels)
    class_accs = {}
    cm = confusion_matrix(labels, preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    class_acc = cm.diagonal()
    for i, c in enumerate(class_acc):
        class_accs.update({f"acc_{i}": c})

    mets = {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
    mets.update(class_accs)
    return mets

with open("params.json", 'r') as f:
    PARAMS = json.load(f)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
    model_id = PARAMS["classi_finetune_model"]
    dataset = load_dataset()
    labels = list(dataset["label"].unique())
    save_path = f'{PARAMS["save_model"]}{model_id.split("/")[-1]}_finetuned'

    tokenizer_len = AutoTokenizer.from_pretrained(model_id, max_length = 512)
    tokenizer = AutoTokenizer.from_pretrained(model_id, max_length = 512, padding=True, Truncation=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def tokenization(batched_text):
        return tokenizer(batched_text['text'], padding = False, truncation=True,)


    plot_tokens(tokenizer_len, dataset)
    label2id, id2label, train_data, test_data, val_data = split_dataset(dataset, labels, train_size=PARAMS["train_split"], val_size=PARAMS["val_split"])


    config = AutoConfig.from_pretrained(model_id, num_labels=len(labels))
    config.label2id = label2id
    config.id2label = id2label
    model = AutoModelForSequenceClassification.from_pretrained(model_id, config=config).to(device)
    
    train_data = train_data.map(tokenization, batched = True, num_proc=4)
    test_data = test_data.map(tokenization, batched = True, num_proc=4)

    print(f'Train: {Counter(train_data["label"])}')
    print(f'Test: {Counter(test_data["label"])}')
    print(f'Val: {Counter(val_data["label"])}')

    train_model = True if PARAMS["btrain_model"].lower() in ["true", "t"] else False

    if train_model:
      
        training_args = TrainingArguments(
                                                output_dir=save_path+PARAMS["output_dir"],
                                                num_train_epochs=PARAMS["train_epochs"],
                                                per_device_train_batch_size=PARAMS["batch_size"],
                                                per_device_eval_batch_size=PARAMS["batch_size"],
                                                eval_strategy="steps",
                                                eval_steps=50,
                                                save_strategy="steps",
                                                save_steps=200,
                                                logging_strategy="steps",
                                                logging_steps=50,
                                                disable_tqdm=False, 
                                                load_best_model_at_end=True,
                                                warmup_steps=PARAMS["warmup_steps"],
                                                weight_decay=PARAMS["weight_decay"],
                                                learning_rate=PARAMS["lr"],
                                                fp16=True,
                                                logging_dir=save_path+PARAMS["logging_dir"],
                                                dataloader_num_workers=8,
                                                metric_for_best_model=PARAMS["metric"],
                                                greater_is_better=True,
                                                run_name = 'classification',
                                                report_to=["tensorboard"]
                                        )
                                        
        trainer = Trainer(
                            model=model,
                            args=training_args,
                            compute_metrics=compute_metrics,
                            train_dataset=train_data,
                            eval_dataset=test_data,
                            data_collator=data_collator, 
                        )
        print("Training")

        trainer.train()
        trainer.save_model(save_path) 
        print(f'Model saved at {save_path}')
    print("Evaluating")
    evaluate_model(label2id, id2label, val_data)


if __name__ == '__main__':

    train()