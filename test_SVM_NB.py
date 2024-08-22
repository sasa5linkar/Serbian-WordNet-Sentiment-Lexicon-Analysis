# Modified script with diagnostic code integrated

"""
Created on Wed Feb 15 18:58:02 2023

This script applies pre-trained sentiment analysis models, including SVM (Support Vector Machines) and NB (Naive Bayes), to estimate sentiment polarity in Serbian WordNet synsets.

Author: Sasa Petalinkar

"""

from srpskiwordnet import SrbWordNetReader
from joblib import load
import pandas as pd


def estimate_polarity(text, est_POS, est_NEG):
    a = est_POS.predict(text)
    b = est_NEG.predict(text)
    return (a * (1 - b), b * (1 - a))  


def estimate_polarity_table(text, df): 
    sentiment = list()
    for t in TYPE:
        filter2 = df["Tip"] == t
        for i in ITERATION:
            filter1 = df["Iteracija"] == i
            filter3 = df["Polaritet"] == "NEG"
            est_NEG = estimators.where(filter1 & filter2 & filter3).dropna().reset_index()["Model"][0]
            filter3 = df["Polaritet"] == "POS"
            est_POS = estimators.where(filter1 & filter2 & filter3).dropna().reset_index()["Model"][0]
            sentiment.append(estimate_polarity(text, est_POS, est_NEG))

    res = tuple([sum(ele) / len(sentiment) for ele in zip(*sentiment)])
    return res


def check_model_predictions(definicije, estimators):
    results = []
    definicije_sample = definicije.sample(min(100, len(definicije)))

    for _, row in estimators.iterrows():
        estimator = row["Model"]
        polarity = row["Polaritet"]
        model_type = row["Tip"]
        iteration = row["Iteracija"]
        
        predictions = estimator.predict(definicije_sample)
        unique_predictions = set(predictions)
        
        for pred in unique_predictions:
            if pred not in [0, 1]:
                results.append({
                    "Model Type": model_type,
                    "Iteration": iteration,
                    "Polarity": polarity,
                    "Unexpected Prediction": pred
                })
                
    return pd.DataFrame(results) if results else "All predictions are binary."


# Your existing code
RES_DIR = ".\\resources\\"
MOD_DIR = ".\\ml_models\\"

swordnet = SrbWordNetReader(RES_DIR, "wnsrp30.xml") 

POLARITY = ["POS", "NEG"]
TYPE = ["SVM", "Bern"]
ITERATION = ["0", "2", "4", "6"]  
pom = list()

for p in POLARITY:
    for t in TYPE:
        for i in ITERATION:
            pom2 = dict()
            pom2["Iteracija"] = i
            pom2["Tip"] = t
            pom2["Polaritet"] = p
            pom2["Model"] = load(MOD_DIR + t + "-klas-" + p + "-" + i + ".joblit") 
            pom.append(pom2)
estimators = pd.DataFrame(pom)

sword = pd.read_csv(RES_DIR + "definicije_lematizone.csv", index_col=0)
definicije = sword["Definicija"].fillna(' ')
a = estimate_polarity_table(definicije, estimators)  
sword["POS"], sword["NEG"] = a

# Diagnostic step
unexpected_predictions = check_model_predictions(definicije, estimators)
print(unexpected_predictions)



