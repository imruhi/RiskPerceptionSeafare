from glob import glob
from datasets import Dataset
from tqdm import tqdm
import pickle
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from nltk.corpus import stopwords
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired, PartOfSpeech
from bertopic.vectorizers import  ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import re
import json
import torch
from sentence_transformers import SentenceTransformer
from clean_text import clean_text

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")


with open("params.json", 'r') as f:
    PARAMS = json.load(f)

def get_params():
    params = {  
                # TFIDF
                "reduce_frequent_words": True, "bm25_weighting": True,   
                "seed_words": PARAMS["seed_words"],
                "seed_multiplier": 6,
                # UMAP
                "n_neighbors": 10, "n_components": 5, "min_dist": 0.0, "metric_umap": "cosine", "random_state": 42,
                # HDBSCAN (change min_cluster_size for more/less topics?, default is 10, recommended to only increase above 10)
                "min_cluster_size": 100, "metric_hbd": "euclidean", "cluster_selection_method": "eom", "prediction_data": True,
                # Vectorizer model
                "min_df": 2, "ngram_range": (1,2),
                # Representation models
                "diversity": 0.7
            }
    return params

def train_model():
    save_path =  PARAMS["topic_model_save"]
    data_path = f'{PARAMS["roberta_data_path"]}_{PARAMS["word_window"]}'
    embedding_model = SentenceTransformer(PARAMS["sentence_model"], model_kwargs={"torch_dtype": "float16"}, device="cuda")
    
    all_excerpts = Dataset.load_from_disk(data_path).to_pandas().reset_index(drop=True)
    print(f"Original size: {len(all_excerpts)}")
    dataset = pd.DataFrame({"text_id": all_excerpts["text_id"],"text":all_excerpts[f'text_{PARAMS["word_window"]}'], "label":all_excerpts["level_shipwreck"]})
    dataset["text_cleaned"] = [clean_text(x) for x in tqdm(dataset["text"])]
    texts = list(dataset["text_cleaned"])
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    params = get_params()

    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=params["reduce_frequent_words"], bm25_weighting=params["bm25_weighting"], 
                                seed_words=params['seed_words'], seed_multiplier=params["seed_multiplier"])

    umap_model = UMAP(n_neighbors=params["n_neighbors"], 
                    n_components=params["n_components"], 
                    min_dist=params["min_dist"], 
                    metric=params["metric_umap"], 
                    random_state=params["random_state"])

    hdbscan_model = HDBSCAN(min_cluster_size=params["min_cluster_size"],
                            metric=params["metric_hbd"], 
                            cluster_selection_method=params["cluster_selection_method"], 
                            prediction_data=params["prediction_data"])

    vectorizer_model = CountVectorizer(stop_words="english", min_df=params["min_df"], 
                                    ngram_range=params["ngram_range"])

    representation_models = {
                                "KeyBERT": KeyBERTInspired(), 
                                "MMR": MaximalMarginalRelevance(diversity=params["diversity"]),
                                "POSSpacy": PartOfSpeech("en_core_web_sm"),
                            }


    topic_model = BERTopic(

        # Pipeline models
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_models,
        top_n_words=10,
        verbose=True,
        ctfidf_model=ctfidf_model,
        calculate_probabilities=True,
    )

    # Train model
    topics, probs = topic_model.fit_transform(texts, embeddings)
    topic_model.save(save_path, serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

    dataset["topic"] = topics
    topic_interested = []
    for idx, row in topic_model.get_topic_info().iterrows():
        if any(i in row["Name"] for i in params['seed_words']):
            topic_interested.append(int(row["Topic"]))
    dataset_roberta = dataset[dataset["topic"].isin(topic_interested)]
    print(f"Filtered size: {len(dataset_roberta)}")

    Dataset.from_pandas(dataset_roberta.drop(columns="text_cleaned")).save_to_disk(data_path+"_filtered")

if __name__ == '__main__':

    train_model()