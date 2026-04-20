import pickle
import pandas as pd
from tqdm import tqdm
import re
import unicodedata
from datasets import Dataset
from transformers import RobertaTokenizerFast
import matplotlib.pyplot as plt
import re
from pathlib import Path
import json
from collections import Counter
from clean_text import clean_text
with open("params.json", 'r') as f:
    PARAMS = json.load(f)

def get_context(text, target, n):
    words = text.split()
    
    for i, word in enumerate(words):
        if re.sub(r'[^a-zA-Z]', '', word).lower() == re.sub(r'[^a-zA-Z]', '', target).lower():
            start = max(i - n, 0)
            end = min(i + n + 1, len(words))
            return " ".join(words[start:end]).lstrip().rstrip()
    
    return None


def get_concordance(dataset_path, save_path, place_filter=True, word_window=100):
    
    all_dfs_with_context = Dataset.load_from_disk(dataset_path).to_pandas()
    places = all_dfs_with_context["newOBJECTID"].unique()
    print(f"# of unique places: {len(places)}")
    print(f"len of data: {len(all_dfs_with_context)}")
    if place_filter:
        all_dfs_with_context = all_dfs_with_context[all_dfs_with_context["type"]=="place"]  
        # all_dfs_with_context = all_dfs_with_context[all_dfs_with_context["category"]=="geography"]
        print(f"    len of data (place filter): {len(all_dfs_with_context)}")
        print(f"    categories: {Counter(all_dfs_with_context['category'])}")

    count = 1 

    new_rows = []
    for place in tqdm(places):
        df = all_dfs_with_context[all_dfs_with_context["newOBJECTID"]==place]
        place_names = [re.findall(r"(<b>.*<\/b>)", y)[0] for y in df["no_context_text"]]
        place_names = [re.sub(r"<\/?b>", '', y) for y in place_names]
        place_names = list(set(place_names))
        # print(place_names)
        # print("-"*50)
        for idx, row in df.iterrows():
            for place in place_names:
                windowed_text = get_context(row["text"], place, word_window)
                if windowed_text:
                    # print(row)
                    new_row = list(row[["text_id", "workID", "auth_title_display", "type", "category", "original_lang", "edate", "placeID", "newOBJECTID", "AoE", "level_shipwreck", "num_shipwrecks"]])
                    new_row.extend([place, windowed_text])
                    new_rows.append(new_row)
                    # print(f"{place}: {windowed_text}")
        # print("-"*50)

    context_df = pd.DataFrame(new_rows, columns=["text_id", "workID", "auth_title_display", "type", "category", "original_lang", "edate", "placeID", "newOBJECTID", "AoE", "level_shipwreck", "num_shipwrecks", "place_name", f"text_{word_window}"])
    context_df = context_df.dropna().drop_duplicates().reset_index(drop=True)
    # how the data will be used
    context_df_idx = pd.DataFrame({f"text_{word_window}":context_df[f"text_{word_window}"], "label":context_df["level_shipwreck"]}).dropna().drop_duplicates().index
    context_df = context_df.loc[context_df_idx]
    Dataset.from_pandas(context_df).save_to_disk(save_path)
    plot_lens(context_df, word_window)
    print(f"len of contextual data: {len(context_df)}")

def plot_lens(context_df, word_window):
    
    model_id = "bowphs/PhilBerta"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_id, max_length = 512)

    texts = [clean_text(x) for x in context_df[f"text_{word_window}"]]
    token_lens = []
    for t in texts:
        tokens = tokenizer(t, truncation=False, padding=False)
        token_lens.append(len(tokens["input_ids"]))

    context_df["token_len"] = token_lens
    plt.hist(context_df["token_len"], bins=100)
    plt.axvline(512, color="r")
    plt.title(f"Token length distr ({model_id}, window={word_window})")
    plt.savefig(f"{PARAMS['images']}roberta_data_{word_window}.png")


if __name__ == '__main__':


    # make dataset
    get_concordance(dataset_path=PARAMS["roberta_data_path"], 
                    save_path=f'{PARAMS["roberta_data_path"]}_{PARAMS["word_window"]}',
                    place_filter=True if PARAMS["bplace_filter"].lower() in ["true", "t"] else False, 
                    word_window=PARAMS["word_window"])