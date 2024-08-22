# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:47:59 2022.

@author: "Petalinkar Saša"

script for for genartion test and train sets for traning machine learning models for sentiment anasys of Serbain Wordnet. 
"""
# Importing Required Libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from SerbainTagger import SrbTreeTagger
from srpskiwordnet import SrbWordNetReader
from srppolsets import PolaritySets, syn2gloss

# Constants
ROOT_DIR = ""
RES_DIR = os.path.join(ROOT_DIR, "resources")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_sets")

# Function Definitions
def load_file_into_list(filename):
    """
    Load the contents of a file into a list, stripping leading and trailing whitespaces.
    :param filename: Path of the file
    :return: List containing lines from the file
    """
    with open(filename, mode="r", encoding="utf-16") as file:
        return [line.strip() for line in file]

from typing import List, Set

def get_synset_ids_from_csv(path: str, srb_wordnet_reader) -> Set[str]:
    """
    Reads a CSV file containing words and finds their corresponding synset IDs using a given SrbWordNetReader instance.
    
    Parameters:
    - path (str): The path to the CSV file containing the words.
    - srb_wordnet_reader (SrbWordNetReader): An instance of the SrbWordNetReader class for Serbian WordNet.
    
    Returns:
    - Set[str]: A set of unique synset IDs corresponding to the words in the CSV file.
    """
    # Read the CSV into a Pandas DataFrame
    df = pd.read_csv(path, header=None, sep =";")

    
    # Initialize an empty set to store unique synset IDs
    synset_ids = set()
    
    # Loop through the words in the first column of the DataFrame
    for word in df.iloc[:, 0]:
        # Find synsets corresponding to the word
        synsets = srb_wordnet_reader.synsets(word)
        
        # Add the IDs of the found synsets to the set
        for synset in synsets:
            synset_ids.add(synset.ID())
            
    return synset_ids

# Example usage (uncomment the following lines to run the example)
# path_to_csv = "path/to/your/csv/file.csv"
# srb_wordnet = SrbWordNetReader("path/to/your/wordnet", "fileids")
# synset_ids = get_synset_ids_from_csv(path_to_csv, srb_wordnet)


def preprocess_and_split_data(pol_sets, polarity, preprocess_fns, random_state=13):
    """
    Preprocesses the data and splits it into training and testing sets.
    :param pol_sets: PolaritySets object containing sentiment information
    :param polarity: Sentiment polarity ("POS" or "NEG")
    :param preprocess_fns: List of preprocessing functions
    :param random_state: Random state for reproducibility
    :return: X_train, X_test, y_train, y_test DataFrames
    """
    X, y = pol_sets.getXY(polarity, preprocess=preprocess_fns)
    return train_test_split(X, y, stratify=y, random_state=random_state)

def save_datasets(X_train, X_test, y_train, y_test, prefix):
    """
    Save training and testing datasets to CSV files.
    :param X_train, X_test, y_train, y_test: DataFrames containing the data
    :param prefix: Prefix for filenames
    """
    X_train.to_csv(os.path.join(TRAIN_DIR, f"X_train_{prefix}.csv"))
    y_train.to_csv(os.path.join(TRAIN_DIR, f"y_train_{prefix}.csv"))
    X_test.to_csv(os.path.join(TRAIN_DIR, f"X_test_{prefix}.csv"))
    y_test.to_csv(os.path.join(TRAIN_DIR, f"y_test_{prefix}.csv"))

# ... (previous imports and function definitions remain the same)

def generate_and_save_datasets(all_pol_sets, preprocess_fns, prefix):
    """
    Generate and save the training and testing datasets based on polarity sets and preprocessing functions.
    :param all_pol_sets: List of all polarity sets
    :param preprocess_fns: List of preprocessing functions, None for unprocessed
    """
    for i, pol_set in enumerate(all_pol_sets):
        for polarity in ["POS", "NEG"]:
            X, y = pol_set.getXY(polarity, preprocess=preprocess_fns)
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=13)
            save_datasets(X_train, X_test, y_train, y_test, f"{prefix}{polarity}{i}")

# Main Code
if __name__ == "__main__":
    # Initialize Serbian WordNet and TreeTagger
    swordnet = SrbWordNetReader(RES_DIR, "wnsrp30.xml")
    tree_tagger = SrbTreeTagger()

    # Load Positive and Negative Seed Words and IDs
    pos_df = pd.read_csv(os.path.join(RES_DIR, "pos.csv"))
    neg_df = pd.read_csv(os.path.join(RES_DIR, "neg.csv"))


    print("OBJ from SWN")


    positive_seed_words = ["dobar",
                        "dobrota",
                        "lep",
                        "čudesno",
                        "dragocen",
                        "anđeoski",
                        "izobilje",
                        "izbavljenje",
                        "tešiti",
                        "ispravnost",
                        "oduševiti se",
                        "slast",
                        "uveseljavajući",
                        "napredovati",
                        "proslavljen",
                        "usrećiti",
                        "uspešnost"]

    negative_seed_words = ["zao",
                        "antipatija",
                        "beda",
                        "bedan",
                        "bol",
                        "laž",
                        "lažno",
                        "korupcija",
                        "krivica",
                        "prezreti",
                        "tuga",
                        "nauditi",
                        "sebičnost",
                        "paćeništvo",
                        "ukloniti s ovog sveta",
                        "masakr",
                        "ratovanje"]
    positive_seed_IDS = ["ENG30-01828736-v",
                        "ENG30-13987905-n",
                        "ENG30-01777210-v",
                        "ENG30-13987423-n",
                        "ENG30-01128193-v",
                        "ENG30-02355596-v",
                        "ENG30-00271946-v",
                        ]

    negative_seed_IDS = ["ENG30-01510444-a",
                        "ENG30-01327301-v",
                        "ENG30-00735936-n",
                        "ENG30-00220956-a",
                        "ENG30-02463582-a",
                        "ENG30-01230387-a",
                        "ENG30-00193480-a",
                        "ENG30-00364881-a",
                        "ENG30-14213199-n",
                        "ENG30-01792567-v",
                        "ENG30-07427060-n",
                        "ENG30-14408086-n",
                        "ENG30-14365741-n",
                        "ENG30-02466111-a",
                        "ENG30-14204950-n",
                        "ENG30-10609960-n",
                        "ENG30-02319050-v",
                        "ENG30-02495038-v",
                        "ENG30-01153548-n",
                        "ENG30-00751944-n",
                        ]


    print("POS and NEG manuely chosen")
    rem_obj = ["ENG30-05528604-n",
    "ENG30-00749767-n",
    "ENG30-09593651-n",
    "ENG30-13250542-n",
    "ENG30-13132338-n",
    "ENG30-05943066-n",
    "ENG30-03123143-a",
    "ENG30-10104209-n",
    "ENG30-12586298-n",
    "ENG30-01971094-n",
    "ENG30-00759269-v",
    "ENG30-00948206-n",
    "ENG30-01039307-n",
    "ENG30-02041877-v",
    "ENG30-00023271-n",
    "ENG30-13509196-n",
    "ENG30-09450866-n",
    "ENG30-03947798-n",
    "ENG30-08589140-n",
    "ENG30-09569709-n",
    "ENG30-00223268-n",
    "ENG30-00220409-n",
    "ENG30-00224936-n",
    "ENG30-00222248-n",
    "ENG30-00221981-n",
    "ENG30-00223362-n",
    "ENG30-00222485-n",
    "ENG30-00223720-n",
    "ENG30-00225593-n",
    "ENG30-00221596-n",
    "ENG30-00223268-n",
    "ENG30-02485451-v",
    "ENG30-02574205-v"

            ]
    
    pol_set = PolaritySets(swordnet, 0)

    #add all whihc have sentonint 0,0 in english, by direct mapping
    pol_set.addWSWN()
    # rmeove all whihc from polarity lexikon Senti-Pol-sr
    path_to_csv = os.path.join(RES_DIR, "recnikPolariteta.csv")
    not_objective_synset_ids = get_synset_ids_from_csv(path_to_csv,swordnet )
    print (len(not_objective_synset_ids))
    pol_set.removeSynIDs(not_objective_synset_ids)

    pol_set.addPOSall(positive_seed_words)
    pol_set.addNEGall(negative_seed_words)
    pol_set.addPOSIDall(positive_seed_IDS)
    pol_set.addNEGIDall(negative_seed_IDS)
    pol_set.addPOSIDall(pos_df["ID"])
    pol_set.addNEGIDall(neg_df["ID"])

    pol_set.removeSynIDs(rem_obj)

    # pol_set.addWNopAll()
    # print("POS, NEG and OBJ from WN-OP")
    pol_set.updateDataFrame()

    all_pol_sets = [pol_set]
    for _ in range(6):
        all_pol_sets.append(all_pol_sets[-1].next_iteration())

    # Generate and Save Lemmatized Datasets
    preprocess_fns = [syn2gloss, tree_tagger.lemmarizer]
    generate_and_save_datasets(all_pol_sets, preprocess_fns, "LM")

    # Generate and Save Unprocessed Datasets
    preprocess_fns = [syn2gloss]
    generate_and_save_datasets(all_pol_sets,preprocess_fns, "UP" )
        


