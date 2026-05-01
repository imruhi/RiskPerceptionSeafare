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
from wordcloud import WordCloud
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")


with open("params.json", 'r') as f:
    PARAMS = json.load(f)

def get_params():
    params = {  
                # TFIDF
                "reduce_frequent_words": True, "bm25_weighting": True,   
                "seed_words": PARAMS["seed_words"],
                "seed_multiplier": 10,
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

    Dataset.from_pandas(dataset.drop(columns="text_cleaned")).save_to_disk(data_path+"_filtered")

def knn_clustering(dataset):
    data_path = f'{PARAMS["roberta_data_path"]}_{PARAMS["word_window"]}_filtered'

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def clean_text_(text):
        text = clean_text(text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
        text = text.lower()  # Convert to lowercase
        words = text.split()  # Split into words
        words = [word for word in words if word not in stop_words]  # Remove stop words
        return ' '.join(words)

    def normalize_text(text, method='lemmatization'):
        words = text.split()
        if method == 'stemming':
            words = [stemmer.stem(word) for word in words]
        elif method == 'lemmatization':
            words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    def preprocess_text(text, method='lemmatization'):
        text = clean_text_(text)
        text = normalize_text(text, method)
        return text
    
    embedding_model = SentenceTransformer(PARAMS["sentence_model"], model_kwargs={"torch_dtype": "float16"}, device="cuda")
    dataset["text_cleaned"] = [preprocess_text(x) for x in dataset["text"]]
    embeddings = embedding_model.encode(list(dataset["text_cleaned"]) , show_progress_bar=True)

    def perform_kmeans_clustering(embeddings, n_clusters=5):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        print(f"K-Means clustering with {n_clusters} clusters: Silhouette Score = {silhouette_avg}")

        return cluster_labels, silhouette_avg, kmeans
    kmeans_labels, kmeans_silhouette, kmeans_model = perform_kmeans_clustering(embeddings, n_clusters=4)

    dataset["label_knn"] = kmeans_labels
    label_to_keep = []

    for x in dataset["label_knn"].unique():
        df = dataset[dataset["label_knn"]==x]
        print(f"CLuster size: ", len(df))
        t = list(df.head()["text"])
        all_texts = " ".join(list(df["text_cleaned"]))
        text = ' '.join(word for word in all_texts.split() if word not in stop_words)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        top_10 = [v for v in wordcloud.words_.keys()][:10]
        print(", ".join(top_10))
        if "sea" in top_10:
            label_to_keep.append(x)

    return dataset, label_to_keep
    
def filter():
    save_path =  PARAMS["topic_model_save"]
    data_path = f'{PARAMS["roberta_data_path"]}_{PARAMS["word_window"]}_filtered'
    embedding_model = SentenceTransformer(PARAMS["sentence_model"], model_kwargs={"torch_dtype": "float16"}, device="cuda")
    
    dataset = Dataset.load_from_disk(data_path).to_pandas().reset_index(drop=True)

    # most common topic from bertopic should be relating to sea
    dataset = dataset[dataset["topic"].isin([0])]

    # knn filtering
    if PARAMS["bKnn"].lower() == "true":
        dataset, label_to_keep = knn_clustering(dataset=dataset)
        dataset = dataset[dataset["label_knn"].isin(label_to_keep)].drop(columns=["label_knn", "text_cleaned"])
        
    print(f"Filtered size: {len(dataset)}")
    Dataset.from_pandas(dataset).save_to_disk(data_path)
    print(f"saved at {data_path}")

if __name__ == '__main__':

    train_model()
    filter()