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
import matplotlib.pyplot as plt

project_root = os.path.abspath("..")
sys.path.insert(0, project_root)

with open(project_root + "/params.json", 'r') as f:
    PARAMS = json.load(f)

# model_id = PARAMS["classi_finetune_model"]
# model_path = f'{PARAMS["save_model"]}{model_id.split("/")[-1]}_finetuned_32_8' # best model
model_base = PARAMS["classi_finetune_model"]
tokenizer = AutoTokenizer.from_pretrained(model_base)
# model = AutoModelForSequenceClassification.from_pretrained(model_path)
# model = model.to("cuda")
# model.eval()

device = "cuda"

def get_tokenizer():
    return tokenizer

# def get_model():
#     return model

# def predict_proba(texts):
#     all_probs = []
#     batch_size = 32  # tune based on your GPU memory
#
#     for i in range(0, len(texts), batch_size):
#         batch = list(texts[i:i + batch_size])
#         inputs = tokenizer(
#             batch,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=512
#         )
#         # Move all inputs to GPU
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#
#         with torch.no_grad():
#             outputs = model(**inputs)
#
#         probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#         all_probs.append(probs.cpu().numpy())  # back to CPU for SHAP
#
#     return np.vstack(all_probs)

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
    class_idxs = [defaultdict(list) for _ in range(3)]
    for idx, (d, sv) in enumerate(zip(all_data, all_class_shap)):
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
                class_idxs[c][token].append(idx)
    return class_contribs, class_contexts, class_idxs
            

def print_attr(class_contribs, class_contexts, class_idxs, id2label, top_k=10, attr="positive"):
    agg = []
    output = f""
    for c in range(3):
        token_scores = {
            token: {
                "score": np.mean(vals),
                "idx": idx
            }
            for (token, vals), (token_, idx) in zip(class_contribs[c].items(), class_idxs[c].items())
        }
        agg.append(token_scores)


    for c in range(3):
        output += "\n"+"-"*40
        output = output + f"\n\nTop words for class {id2label[c]}:"
        if attr == "positive":
            sorted_tokens = sorted(
                [
                    (token, data["score"], data["idx"])
                    for token, data in agg[c].items()
                    if data["score"] > 0
                ],
                key=lambda x: x[1],
                reverse=True
            )
        else:
            sorted_tokens = sorted(
                [
                    (token, data["score"], data["idx"])
                    for token, data in agg[c].items()
                    if data["score"] < 0
                ],
                key=lambda x: x[1],
                reverse=False
            )

        for token, score, idx in sorted_tokens[:top_k]:

            # choose one example context
            example_context = class_contexts[c][token][:3]

            output = output + f"\n\n{token} (idx={idx}): {score:.4f}\n"
            examples = "\n\n    ".join(example_context)
            output = output + f"    {examples}"

    return output

def plot_token_attributions(
        title,
        tokens,
        scores,
        figsize=(14, 2),
        positive_color="#008000",  # green
        negative_color="#ff4d4d",  # red
        neutral_color="#f2f2f2",  # beige
        normalize=True,
        spacing=0.03,
        fontsize=11
):
    scores = np.array(scores)
    if normalize:
        max_abs = np.max(np.abs(scores))
        if max_abs > 0:
            scores = scores / max_abs

    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((1, 0))
    px_per_data = p1[0] - p0[0]

    x = 0
    y = 0.4

    for token, score in zip(tokens, scores):
        if  score > 0:
            alpha = min(abs(score), 1.0)
            color = positive_color
        elif score < 0.06:
            alpha = min(abs(score), 1.0)
            color = negative_color
        else:
            alpha = 0.15
            color = neutral_color

        txt = ax.text(
            x, y,
            token,
            fontsize=fontsize,
            ha="left",
            va="center",
            transform=ax.transData,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=color,
                edgecolor="none",
                alpha=alpha
            ),
            color="black"
        )

        fig.canvas.draw()
        bbox = txt.get_window_extent(renderer=renderer)
        width_axes = bbox.width / px_per_data

        # move x based on actual token width
        x += width_axes + spacing

    ax.text(
        0.2, 0.7, title,
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=fontsize,
        # fontweight="bold"
    )
    plt.show()

def merge_roberta_tokens(tokens, scores):

    merged_tokens = []
    merged_scores = []

    current_token = ""
    current_scores = []

    for token, score in zip(tokens, scores):

        # new word starts
        if token.startswith("▁"):

            # save previous word
            if current_token:
                merged_tokens.append(current_token)
                merged_scores.append(np.mean(current_scores))

            # start new word
            current_token = token.replace("▁", "")
            current_scores = [score]

        else:
            # continue subword
            current_token += token
            current_scores.append(score)

    # add final token
    if current_token:
        merged_tokens.append(current_token)
        merged_scores.append(np.mean(current_scores))

    return merged_tokens, np.array(merged_scores)