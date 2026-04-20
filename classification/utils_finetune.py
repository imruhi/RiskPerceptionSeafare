from evaluate import evaluator
from evaluate import Metric
import datasets
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd 
from datasets import Dataset
from sklearn.model_selection import train_test_split
import re
from data_gathering.utils.clean_text import clean_text

with open("params.json", 'r') as f:
    PARAMS = json.load(f)


def load_dataset():
    dataset_path = f'{PARAMS["roberta_data_path"]}_{PARAMS["word_window"]}_filtered'
    all_excerpts = Dataset.load_from_disk(dataset_path).to_pandas()
    # after topic_modeling choose the topics wanted 
    topics = PARAMS["interested_topics"]
    all_excerpts = all_excerpts.query("topic in @topics")
    dataset = pd.DataFrame({"text":all_excerpts["text"], "label":all_excerpts["label"]}).dropna().drop_duplicates()
    print("Cleaning text")
    dataset["text"] = [clean_text(x) for x in tqdm(dataset["text"])]

    return dataset


def split_dataset(dataset, labels, train_size, val_size):
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    print(label2id)  
    print(id2label)
    train_data, test_data = train_test_split(dataset, train_size=train_size, stratify=dataset["label"], random_state=42)
    test_data, val_data = train_test_split(test_data, train_size=val_size, stratify=test_data["label"], random_state=42)
    train_data["label"] = [label2id[x] for x in train_data["label"]]
    test_data["label"] = [label2id[x] for x in test_data["label"]]
    val_data["label"] = [label2id[x] for x in val_data["label"]]
    train_data = Dataset.from_pandas(train_data).remove_columns(["__index_level_0__"])
    test_data = Dataset.from_pandas(test_data).remove_columns(["__index_level_0__"])
    val_data = Dataset.from_pandas(val_data).remove_columns(["__index_level_0__"])
    
    return label2id, id2label, train_data, test_data, val_data


def plot_tokens(tokenizer, dataset):
    token_lens = []
    for t in dataset["text"]:
        tokens = tokenizer(t, truncation=False, padding=False)
        token_lens.append(len(tokens["input_ids"]))

    dataset["token_len"] = token_lens
    dataset = dataset[dataset["token_len"] < 512]
    dataset = dataset.drop(columns="token_len")
    plt.hist(token_lens, bins=100)
    plt.axvline(512, color="r")
    plt.savefig(f'{PARAMS["images"]}finetune_token_distr.png')
    plt.clf()
