from sentence_transformers import SentenceTransformer
from .utils_finetune import load_dataset, split_dataset
from .evaluate_model import evaluate_model
import json 
import torch
from collections import Counter
import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer

with open("params.json", 'r') as f:
    PARAMS = json.load(f)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", 
                            min_df=5,       
                            max_df=0.5,     
                            stop_words="english")
dataset = load_dataset()
print(f"Size of dataset: {len(dataset)}")
labels = list(dataset["label"].unique())
save_path = f'classification/regression_model/'


label2id, id2label, train_data, test_data, val_data = split_dataset(dataset, labels, train_size=PARAMS["train_split"], val_size=PARAMS["val_split"])

weights = compute_class_weight('balanced', classes=np.array([0,1,2]), y=train_data["label"])
class_weights = torch.tensor(weights, dtype=torch.float).to(device)
class_weights = {i: w.item() for i, w in enumerate(class_weights)}


train_data = train_data.to_pandas()
val_data = val_data.to_pandas()

X_train = vectorizer.fit_transform(list(train_data["text"])).toarray()
y_train = train_data["label"]

X_test = vectorizer.transform(list(val_data["text"])).toarray()
y_test = val_data["label"]

with open(save_path + 'vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print(f'Train: {Counter(train_data["label"])}')
print(f'Test: {Counter(test_data["label"])}')
print(f'Val: {Counter(val_data["label"])}')

train_model = True if PARAMS["btrain_model"].lower() in ["true", "t"] else False

if train_model:
    model = LogisticRegression(solver='lbfgs', l1_ratio=0, 
                               C=0.8, class_weight=class_weights, 
                               max_iter=100).fit(X_train, y_train)

    # define the model evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(model, X_test, y_test, scoring='f1_weighted', cv=cv, n_jobs=-1)
    # report the model performance
    print('Mean f1: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    filename = save_path + 'regression_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    
    answers = model.predict(X_test)

    finetuned_cr = classification_report(y_test, answers)
    f = open(save_path + 'finetuned_cr.txt', 'w')
    f.write('{}\n\nClassification Report\n\n{}'.format(label2id, finetuned_cr))
    f.close()

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y_test,
        y_pred=answers,
        normalize='true'
    )
    disp.plot()
    plt.title("Finetuned")
    plt.savefig(save_path + "finetuned.png")