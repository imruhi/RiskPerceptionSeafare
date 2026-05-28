import shap
import torch
import numpy as np
import sys
import os
import json
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
from datasets import Dataset
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import string
from collections import defaultdict
import pickle

project_root = os.path.abspath("..")
sys.path.insert(0, project_root)

with open(project_root + "/params.json", 'r') as f:
    PARAMS = json.load(f)

model_id = PARAMS["classi_finetune_model"]
model_path = f'{PARAMS["save_model"]}{model_id.split("/")[-1]}_finetuned_32_8' # best model
model_base = PARAMS["classi_finetune_model"]
tokenizer = AutoTokenizer.from_pretrained(model_base)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model = model.to("cuda")
model.eval()

device = "cuda"

def get_tokenizer():
    return tokenizer

def get_model():
    return model

def predict_proba(texts):
    all_probs = []
    batch_size = 32  # tune based on your GPU memory

    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        # Move all inputs to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        all_probs.append(probs.cpu().numpy())  # back to CPU for SHAP

    return np.vstack(all_probs)

def merge_tokens(tokens, values):
    merged_tokens = []
    merged_values = []

    current_token = ""
    current_value = None

    for t, v in zip(tokens, values):

        # RoBERTa word start token
        if t.startswith("▁") and t not in string.punctuation:

            # flush previous token
            if current_token != "":
                merged_tokens.append(current_token.lower())
                merged_values.append(current_value)

            current_token = t[1:]  # remove _
            current_value = v.copy()

        else:
            if t not in string.punctuation:
                # continuation of same word
                current_token += t
                current_value += v

    # flush last token
    if current_token:
        merged_tokens.append(current_token.lower())
        merged_values.append(current_value)

    return merged_tokens, np.array(merged_values)

def group_values(tokens, values, group=2):
    paired_tokens = []
    paired_values = []

    for i in range(0, len(tokens), group):
        pair_tokens = tokens[i:i+group]
        pair_values = values[i:i+group]

        # join tokens with space
        paired_tokens.append(" ".join(pair_tokens))

        paired_values.append(np.sum(pair_values, axis=0))

    return paired_tokens, np.array(paired_values)

def get_class_contribs(all_data, all_class_shap, group=2):
    context_size = 6
    class_contribs = [defaultdict(list) for _ in range(3)]
    class_contexts = [defaultdict(list) for _ in range(3)]
    for d, sv in zip(all_data, all_class_shap):
        merged_tokens, merged_values = merge_tokens(d, sv)
        merged_tokens, merged_values = group_values(merged_tokens, merged_values, group)
        for i, token in enumerate(merged_tokens):
            for c in range(3):
                # get context around max value words 
                if i == 0:
                    context = " ".join([f"[[{merged_tokens[i]}]]"]
                                + merged_tokens[i + 1: i + context_size])
                else:
                    context = " ".join(
                                    merged_tokens[i - context_size: i - 1]
                                    + [f"[[{merged_tokens[i]}]]"]
                                    + merged_tokens[i + 1: i + context_size]
                            )
                
                class_contribs[c][token].append(merged_values[i, c])
                class_contexts[c][token].append(context)      
    return class_contribs, class_contexts        
            

def print_attr(class_contribs, class_contexts, id2label, top_k=10, attr="positive"):

    agg = []
    output = f""
    for c in range(3):
        token_scores = {
            token: np.mean(vals)
            for token, vals in class_contribs[c].items()
        }
        agg.append(token_scores)
        

    for c in range(3):
        output += "\n"+"-"*40
        output = output + f"\n\nTop words for class {id2label[c]}:"
        if attr == "positive":
            sorted_tokens = sorted(
                [(token, score) for token, score in agg[c].items() if score > 0],
                key=lambda x: x[1],
                reverse=True
            )
        else:
            sorted_tokens = sorted(
                [(token, score) for token, score in agg[c].items() if score < 0],
                key=lambda x: x[1],
                reverse=False
            )

        for token, score in sorted_tokens[:top_k]:

            # choose one example context
            example_context = class_contexts[c][token][:3]

            output = output + f"\n\n{token}: {score:.4f}\n"
            examples = "\n\n    ".join(example_context)
            output = output + f"    {examples}"
    
    return output