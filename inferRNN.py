#inferRNN

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:00:00 2020
Last modified on Mon Jun  1 12:00:00 2020
Author: Petalinkar Saša
Description: Script for sentiment analysis of all synsets in SrbWordNet
unsing RNN model and saving the result in a csv file
""" 

from srpskiwordnet import SrbWordNetReader
import pandas as pd
import os
import tensorflow as tf



def sentiment_analyze_df(swn):
    """
    Function that returns a dataframe with sentiment analysis of all synsets in SrbWordNet
    :param swn: SrbWordNetReader object 
    :return: dataframe with sentiment analysis of all synsets in SrbWordNet
    """  
    syns_list = list()
    for sifra in swn._synset_dict:
        syn = swn._synset_dict[sifra]
        el = dict()
        el["ID"] = sifra
        el["Lemme"] = ",".join(syn._lemma_names)
        el["Vrsta"] = syn.POS()
        syns_list.append(el)
    return pd.DataFrame(syns_list)


# Define constants
ROOT_DIR = ""
RES_DIR = os.path.join(ROOT_DIR, "resources")
MOD_DIR = os.path.join(ROOT_DIR, "ml_models")
POLARITY = ["POS", "NEG"]
DATASET_ITERATIONS = [0, 2, 4, 6]  # Dataset iterations to process

def main():
    """
    Function that performs sentiment analysis of all synsets in SrbWordNet and saves the result in a csv file
    :return: None
    """
    # load the file wnsrp30.xml and create a SrbWordNetReader object from RES_DIR
    swordnet = SrbWordNetReader(RES_DIR, "wnsrp30.xml") 
    # load the file definicije_lematizone.csv and create a dataframe from RES_DIR
    definitions_leammatized = pd.read_csv(os.path.join(RES_DIR, "definicije_lematizone.csv"), index_col=0)
    """
    Sample of that file 
    ,ID,Definicija
    0,ENG30-03574555-n,zgrada u koji se nalaziti organizacioni jedinica neki grana javan poslovanje 
    1,ENG30-07810907-n,pripremljen dodatak jela za poboljšanje ukus 
    2,ENG30-00721431-n,"u nečiji prilika , mogućnost "
    3,ENG30-00473799-v,ostati jesam još samo da se doterati neki finesa u igra plaviti . 
    """
    # replace all na in column Definicija with empty string
    definitions_leammatized["Definicija"].fillna("", inplace=True)

    # Repeat the process for each dataset iteration and polarity
    for i in DATASET_ITERATIONS:
        for polarity in POLARITY:
            model_folder_name = f"RNN_model_{polarity}_{i}.tf"
            model_path = os.path.join(MOD_DIR, model_folder_name)
            model =tf.keras.models.load_model(model_path)
            # infer from columan Definicija and save result in column polarity_i in dataframe definitions_leammatized
            definitions_leammatized[f"{polarity}_{i}"] = model.predict(definitions_leammatized["Definicija"])
        # apply corection to polarity columns in dataframe definitions_leammatized pos_i = pos_i * (1 - neg_i) and neg_i = neg_i * (1 - pos_i)
        # and save result in column polarity_i in dataframe definitions_leammatized
        # Create temporary columns to store the original values
        definitions_leammatized[f"TEMP_POS_{i}"] = definitions_leammatized[f"POS_{i}"]
        definitions_leammatized[f"TEMP_NEG_{i}"] = definitions_leammatized[f"NEG_{i}"]

        # Perform the adjustments using the original (temporary) values
        definitions_leammatized[f"POS_{i}"] = definitions_leammatized[f"TEMP_POS_{i}"] * (1 - definitions_leammatized[f"TEMP_NEG_{i}"])
        definitions_leammatized[f"NEG_{i}"] = definitions_leammatized[f"TEMP_NEG_{i}"] * (1 - definitions_leammatized[f"TEMP_POS_{i}"])

        # Remove the temporary columns
        definitions_leammatized.drop([f"TEMP_POS_{i}", f"TEMP_NEG_{i}"], axis=1, inplace=True)

    # calculate mean of POS_i and NEG_i and save result in column POS and NEG in dataframe definitions_leammatized
    # delte columns POS_i and NEG_i from dataframe definitions_leammatized
    definitions_leammatized["POS"] = definitions_leammatized[[f"POS_{i}" for i in DATASET_ITERATIONS]].mean(axis=1)
    definitions_leammatized["NEG"] = definitions_leammatized[[f"NEG_{i}" for i in DATASET_ITERATIONS]].mean(axis=1)
    definitions_leammatized.drop(columns=[f"POS_{i}" for i in DATASET_ITERATIONS], inplace=True)
    definitions_leammatized.drop(columns=[f"NEG_{i}" for i in DATASET_ITERATIONS], inplace=True)
    # like in function sentiment_analyze_df fill dataframe definitions_leammatized missing collumns with data from swordnet
    # and save result in dataframe definitions_leammatized
    sword = sentiment_analyze_df(swordnet)
    definitions_leammatized = pd.merge(sword, definitions_leammatized, on="ID", how="left")
    # save dataframe definitions_leammatized in file srbsentiwordnet_a3.csv in RES_DIR
    definitions_leammatized.to_csv(os.path.join(RES_DIR, "srbsentiwordnet_a3.csv"))
    #drop column Vrsta from dataframe definitions_leammatized and save result in dataframe definitions_leammatized
    definitions_leammatized.drop(columns=["Vrsta"], inplace=True)
    # save dataframe definitions_leammatized in file srbsentiwordnet3.csv in RES_DIR
    definitions_leammatized.to_csv(os.path.join(RES_DIR, "srbsentiwordnet3.csv"))
if __name__ == "__main__":
    main()