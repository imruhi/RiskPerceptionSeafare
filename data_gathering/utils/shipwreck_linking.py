import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np
from datasets import Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import json
import math
with open("params.json", 'r') as f:
    PARAMS = json.load(f)

def preprocess_coord(coord):
    coord = str(coord).split('.')
    coord = '.'.join(coord[:-1])
    return float(coord)

def level_shipwreck(count, high, low):
    if 0 < count <= low:
        return "LOW"
    elif low < count <= high:
        return "MEDIUM"
    elif high < count:
        return "HIGH"
    else:
        return "COUNT_MISSING"


def get_df():
    geodatabase_shipwrecks = pd.read_excel(PARAMS["shipwreck_path"], sheet_name='GeoDatabase', header=1).drop(columns='Unnamed: 0')
    len_original = len(geodatabase_shipwrecks)
    geodatabase_shipwrecks = geodatabase_shipwrecks[geodatabase_shipwrecks['Latitude'].notna() & geodatabase_shipwrecks['Longitude'].notna()]
    remove = []
    missing = 1
    for idx, row in geodatabase_shipwrecks.iterrows():
        try:
            lat, long = float('{:,.1f}'.format(row['Latitude'])), float('{:,.1f}'.format(row['Longitude']))            
        except:
            missing+=1
            remove.append(idx)
    geodatabase_shipwrecks = geodatabase_shipwrecks.drop(remove)
    geodatabase_shipwrecks["Latitude"] = [float(x) for x in geodatabase_shipwrecks["Latitude"]]
    geodatabase_shipwrecks["Longitude"] = [float(x) for x in geodatabase_shipwrecks["Longitude"]]

    print(f"{len_original-len(geodatabase_shipwrecks)+missing} shipwrecks (of {len_original}) do not have lat and long")
    
    return geodatabase_shipwrecks
    

def link(low, high):
    geodatabase_shipwrecks = get_df()
    all_excerpts = Dataset.load_from_disk(PARAMS["dataset_extracted"]).to_pandas()
    print(all_excerpts.columns)
    all_ports_aoe_topos = pd.read_csv(PARAMS["all_ports_aoe_topos"])
    len_original = len(all_ports_aoe_topos)

    coord_port = all_ports_aoe_topos[["LATITUDE", "LONGITUDE", "newOBJECTID"]].dropna().reset_index(drop=True)
    print(f"{len_original-len(coord_port)} ports do not have lat and long")

    port_aoe = all_ports_aoe_topos[["newOBJECTID", "AoE"]]
    port_aoe = port_aoe.set_index('newOBJECTID')["AoE"].to_dict()

    # Convert to radians (required for haversine distance)
    df1_rad = np.radians(coord_port[['LATITUDE', 'LONGITUDE']])
    df2_rad = np.radians(geodatabase_shipwrecks[['Latitude', 'Longitude']])

    # Build BallTree
    tree = BallTree(df1_rad, metric='haversine')

    # Query nearest neighbor
    dist, ind = tree.query(df2_rad, k=1)

    # connect shipwrecks to ports (using lat long)
    geodatabase_shipwrecks['newOBJECTID'] = coord_port.iloc[ind.flatten()]['newOBJECTID'].values
    geodatabase_shipwrecks["AoE"] = [port_aoe[x] for x in geodatabase_shipwrecks["newOBJECTID"]]
    object_id_port = all_ports_aoe_topos.set_index('newOBJECTID')["NAME"].to_dict()
    # for display purposes
    geodatabase_shipwrecks["port_name"] = [object_id_port[x].split(', ')[0] for x in geodatabase_shipwrecks["newOBJECTID"]]

    # descretisize
    port_shipwrecks_ = geodatabase_shipwrecks.groupby("newOBJECTID").size().reset_index(name="num_shipwrecks")
    port_shipwrecks_["level_shipwreck"] = [level_shipwreck(int(x), low=low, high=high) for x in port_shipwrecks_["num_shipwrecks"]]

    # keep texts for which we know level of shipwreck
    port_shipwrecks = port_shipwrecks_[["newOBJECTID", "level_shipwreck"]]
    port_shipwrecks = port_shipwrecks.set_index('newOBJECTID')["level_shipwreck"].to_dict()
    possible_ports = list(port_shipwrecks.keys())
    all_excerpts_ = all_excerpts.query("newOBJECTID in @possible_ports")
    all_excerpts_["level_shipwreck"] = [port_shipwrecks[x] for x in all_excerpts_["newOBJECTID"]]
    port_shipwrecks = port_shipwrecks_[["newOBJECTID", "num_shipwrecks"]]
    port_shipwrecks_num = port_shipwrecks.set_index('newOBJECTID')["num_shipwrecks"].to_dict()
    all_excerpts_["num_shipwrecks"] = [port_shipwrecks_num[x] for x in all_excerpts_["newOBJECTID"]]
    all_excerpts_["text_id"] = [f"id{i+1}" for i in range(len(all_excerpts_))]
    # save collated data for finetuning model
    Dataset.from_pandas(all_excerpts_).save_to_disk(PARAMS["roberta_data_path"])

    print(f'# excerpts found: {len(all_excerpts_)} (for {len(all_excerpts_["newOBJECTID"].unique())} ports/equivalent)')
    print(f'    shipwreck percentiles 95/75 {np.percentile(all_excerpts_["num_shipwrecks"],q=95)}, {np.percentile(all_excerpts_["num_shipwrecks"],q=75)}')
    print(f"    counts: {Counter(all_excerpts_['level_shipwreck'])}")
    return geodatabase_shipwrecks

def plots(geodatabase_shipwrecks, low, high):
    all_ports_aoe_topos = pd.read_csv(PARAMS["all_ports_aoe_topos"])
    all_excerpts_ = Dataset.load_from_disk(PARAMS["roberta_data_path"]).to_pandas()
    all_excerpts_ = all_excerpts_[["text", "num_shipwrecks", "level_shipwreck", "newOBJECTID", "AoE", "edate"]].dropna().drop_duplicates()
    all_ports = all_ports_aoe_topos[all_ports_aoe_topos["newOBJECTID"].isin(all_excerpts_["newOBJECTID"].unique())]
    all_ports["NAME"] = [x.split(', ')[0] for x in all_ports["NAME"]]
    name_aoe = all_ports.set_index('NAME')["AoE"].to_dict()
    ax = sns.countplot(all_excerpts_, x="level_shipwreck")
    ax.bar_label(ax.containers[0])
    plt.xticks(rotation=45)
    plt.title("Finetuning data")
    plt.savefig(PARAMS["images"]+"textcounts_levelshipwreck_finetuning.png")
    plt.clf()

    plt.hist(all_excerpts_["num_shipwrecks"], log=False, bins=50)
    plt.axvline(low, color="r")
    plt.axvline(high, color="r")
    plt.title("Distribution # texts associated with of # of shipwrecks")
    plt.savefig(PARAMS["images"]+"numtexts_numshipwrecks.png")
    plt.clf()

    ax=sns.countplot(all_excerpts_, x="AoE")
    ax.bar_label(ax.containers[0])
    plt.xticks(rotation=45)
    plt.title("Finetuning data")
    plt.savefig(PARAMS["images"]+"textcounts_aoe_finetuning.png")
    plt.clf()

    # shipwrecks in x
    sns.scatterplot(data=geodatabase_shipwrecks, x=np.radians(geodatabase_shipwrecks["Latitude"]), y=np.radians(geodatabase_shipwrecks["Longitude"]), 
                    marker='x', color="black", alpha=0.1, legend=False, hue=geodatabase_shipwrecks["AoE"], palette=PARAMS["colours"])
    sns.scatterplot(data=all_ports, x=np.radians(all_ports["LATITUDE"]), y=np.radians(all_ports["LONGITUDE"]), hue=all_ports["AoE"], s=50, palette=PARAMS["colours"])
    plt.savefig(PARAMS["images"]+"viz_shipwrecks_ports.png")
    plt.clf()

  
    counts = geodatabase_shipwrecks['port_name'].value_counts().reset_index()
    counts.columns = ['port_name', 'count']
    counts['log_count'] = np.log1p(counts['count'])  # log(1 + x) to avoid log(0)
    
    counts['AoE'] = [name_aoe[x] for x in counts["port_name"]]
    plt.figure(figsize=(20,12))
    sns.barplot(data=counts, x='port_name', y='log_count', hue="AoE")
    plt.axhline(math.log(low), color='r')
    plt.axhline(math.log(high), color='r')
    plt.xlabel("Port/equiv")
    plt.ylabel("log counts # of wrecks")
    plt.title("Shipwrecks data")
    plt.xticks(rotation=90)
    plt.savefig(PARAMS["images"]+"shipwrecks_per_port_log.png")
    plt.clf()
    
    sns.countplot(geodatabase_shipwrecks, x="port_name", hue="AoE", order = geodatabase_shipwrecks['port_name'].value_counts().index, palette=PARAMS["colours"])
    plt.axhline(low, color='r')
    plt.axhline(high, color='r')
    plt.xlabel("Port/equiv")
    plt.ylabel("# of wrecks")
    plt.title("Shipwrecks data")
    plt.xticks(rotation=90)
    plt.savefig(PARAMS["images"]+"shipwrecks_per_port.png")
    plt.clf()



if __name__ == '__main__':
    print(f'LOW: {int(PARAMS["low"])}, HIGH: {int(PARAMS["high"])}')
    geodatabase_shipwrecks = link(low=int(PARAMS["low"]), high=int(PARAMS["high"]))
    plots(geodatabase_shipwrecks, low=int(PARAMS["low"]), high=int(PARAMS["high"]))