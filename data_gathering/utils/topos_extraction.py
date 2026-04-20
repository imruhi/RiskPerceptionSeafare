import pandas as pd
import pickle 
import urllib.request
import json 
import lxml.html.clean
import re
import requests
from bs4 import BeautifulSoup
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import unicodedata
from datasets import Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

with open("params.json", 'r') as f:
    PARAMS = json.load(f)

def load_graauw_data(paths, aoes):
    dfs = []
    for path, aoe in zip(paths, aoes):
        df = pd.read_excel(path)
        df["AoE"] = aoe
        dfs.append(df)
    
    dfs = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
    dfs = dfs[dfs["TOPOSText"].notna()].reset_index(drop=True)
    dfs["newOBJECTID"] = dfs.index
    print(f"# of ports: {len(dfs['newOBJECTID'])}")
    dfs.reset_index(drop=True).to_csv(PARAMS["all_ports_aoe_topos"])
    return dfs


def get_full_text(pid):
    url = f"https://topostext.org/api/paragraph/readone.php?paraID={pid}"
    r = requests.get(url)
    return r.json()


def get_text(text_):
    # valid to do, only one text field and only one p field 
    return unicodedata.normalize('NFKC', BeautifulSoup(text_).find_all('p')[0].text)


def get_topos_texts(all_ports_aoe_topos):
    all_dfs = []

    # extract table data (full paragraphs)
    for index, row in tqdm(all_ports_aoe_topos.iterrows()):
        html = row["TOPOSText"]
        id_ = row["newOBJECTID"]

        req = urllib.request.Request(url=html)
        f = urllib.request.urlopen(req)
        xhtml = f.read().decode("utf-8")
        table_url = re.findall("'https:\/\/topostext\.org\/api\/.*'", xhtml)[0].replace("'","")
        data = requests.get(table_url, headers=PARAMS["headers"]).json()
        records = data["records"]
        
        paragraph_ids = [r["paragraph_id"] for r in records]
        df = []
        for r in records:
            context = get_full_text(r["paragraph_id"])
            context.update({"no_context_text": r["text"]})
            df.append(context)
        df = pd.DataFrame(df)
        response = requests.request('GET', table_url, headers=PARAMS["headers"])
        data = response.json()
        texts_df = pd.DataFrame.from_dict(data['records'])
        texts_df["newOBJECTID"] = id_
        try:
            # sometimes a place has no associated text to it!
            texts_df = texts_df[["type", "category", "original_lang", "newOBJECTID", "edate", "paragraph_id", "placeID", "index_id"]]
            texts_df = pd.concat([df, texts_df], axis=1).reset_index(drop=True)
            all_dfs.append(texts_df)
        except:
            print(f"No text found for: {html}")
            continue

    with open(PARAMS["dataset_raw"], 'wb') as f:  
        pickle.dump(all_dfs, f) # serialize the list

def extract_topos_text(all_ports_aoe_topos):
    with open(PARAMS["dataset_raw"], 'rb') as f:  
        all_dfs = pickle.load(f) 
    all_excerpts = pd.concat(all_dfs).reset_index(drop=True)

    all_excerpts["text"] = [get_text(x).replace('&sect;','§').replace('&nbsp;', ' ') for x in all_excerpts["text"]]
    all_excerpts["no_context_text"] = [x.replace('&sect;','§').replace('&nbsp;', ' ') for x in all_excerpts["no_context_text"]]
    
    id_aoe_dict = dict(zip(all_ports_aoe_topos['newOBJECTID'], all_ports_aoe_topos['AoE']))
    all_excerpts["AoE"] = [id_aoe_dict[x] for x in all_excerpts["newOBJECTID"]]
    Dataset.from_pandas(all_excerpts).save_to_disk(PARAMS["dataset_extracted"])
    print(f"ToposText dataset length: {len(all_excerpts)}")


# TODO: add part to remove duplicates? there were 17 potential ones


def get_info(all_ports_aoe_topos, ids):
    return all_ports_aoe_topos[all_ports_aoe_topos["newOBJECTID"].isin(ids)][["TOPOSText", "NAME", "LATITUDE", "LONGITUDE"]]


def plot_graphs(all_ports_aoe_topos):
    dataset = Dataset.load_from_disk(PARAMS["dataset_extracted"]).to_pandas()

    # plot aoe counts
    ax = sns.countplot(x='AoE', data=dataset)
    ax.bar_label(ax.containers[0])
    plt.title("AoE counts in dataset")
    plt.savefig("figs/dataset_aoe_counts.png")
    ax = sns.countplot(x='AoE', data=all_ports_aoe_topos)
    ax.bar_label(ax.containers[0])
    plt.title("AoE counts based on port/equivalent")
    plt.savefig("figs/port_aoe_counts.png")

    # plot texts per port + most popular ones
    plt.figure(figsize=(30,8))
    sns.countplot(dataset, x="newOBJECTID")
    plt.title("# texts per port")
    plt.xticks(rotation=90)
    plt.savefig("figs/texts_per_port.png")
    common = Counter(dataset["newOBJECTID"]).most_common(5)
    print(f"Most common ports: {common}")
    print(get_info(all_ports_aoe_topos, [int(x[0]) for x in Counter(dataset["newOBJECTID"]).most_common(5)]))
    print()

def extract(paths, aoes):
    all_ports_aoe_topos = load_graauw_data(paths=paths, aoes=aoes) 
    get_topos_texts(all_ports_aoe_topos) # get texts from topos text api + save
    extract_topos_text(all_ports_aoe_topos) # extract them into a df + save
    plot_graphs(all_ports_aoe_topos) # for visualization


if __name__ == '__main__':


    paths = PARAMS["degraauw_data_paths"].values()
    aoes = PARAMS["degraauw_data_paths"].keys()

    extract(paths, aoes)
