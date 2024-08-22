# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 10:17:42 2023

@author: "Petalinkar Saša"

-------------------------------------------------------
Serbian PolaritySet Loader Python Module
-------------------------------------------------------

This Python module contains a single function to load a preset Serbian polarity set. This set is a manually 
curated collection of synsets derived from the Serbian WordNet, specifically designed to aid in the training of 
sentiment analysis models for Serbian synsets.

Key function: 

1. load_preset_Serbian_pol_set: This function is responsible for loading the curated Serbian polarity set. 
   This collection includes positive, negative, and objective synsets that form a foundation for 
   sentiment analysis tasks.

This module acts as an interface to the polarity set, providing an efficient method to access the data without 
interacting directly with the data files. 

Please note that understanding this module assumes familiarity with the concepts of Natural Language Processing, 
particularly the use of WordNet synsets.

For any additional queries or suggestions, please contact sasa5linkar@gmail.com.
"""

from srppolsets import PolaritySets
import pandas as pd

#swordnet = SrbWordNetReader("F:\Serbian Corpora\wnsrp30","wnsrp30.xml")
def load_preset_Serbian_pol_set(wordnet):
    """
    Function: load_preset_Serbian_pol_set
    
    This function loads a preset Serbian polarity set and modifies it according to predefined rules. 
    It specifically targets sentiment analysis models for Serbian synsets.
    
    Parameters: 
    wordnet (obj): The wordnet object to be used for creating the sentiment set.
    I should be instance of Serbian Wordnet 
    
    Returns: 
    Synset_Sentiment_set (obj): An instance of the PolaritySets class with an updated dataframe.
    
    Workflow:
    1. It reads positive and negative sentiment data from CSV files in a specified resources directory.
    2. Initializes an instance of the PolaritySets class, and adds Word Sense Disambiguation Network (WSWN) synsets to it.
    3. Adds manually defined seed words and WordNet IDs for both positive and negative sentiments to the sentiment set.
    4. Reads additional positive and negative synset IDs from the CSV files and adds them to the sentiment set.
    5. Removes specific objective synset IDs from the sentiment set.
    6. Updates the dataframe in the PolaritySets instance to reflect the changes made.
    """
    RES_DIR = ".\\resources\\"
    
    pos_df = pd.read_csv(RES_DIR + "pos.csv" )
    neg_df = pd.read_csv(RES_DIR + "neg.csv" )
    
    
    Synset_Sentiment_set = PolaritySets(wordnet, 0)
    Synset_Sentiment_set.addWSWN()
    Synset_Sentiment_set.addWSWN()
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
    
    Synset_Sentiment_set.addPOSall(positive_seed_words)
    Synset_Sentiment_set.addNEGall(negative_seed_words)
    Synset_Sentiment_set.addPOSIDall(positive_seed_IDS)
    Synset_Sentiment_set.addNEGIDall(negative_seed_IDS)
    Synset_Sentiment_set.addPOSIDall(pos_df["ID"])
    Synset_Sentiment_set.addNEGIDall(neg_df["ID"])
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
    "ENG30-08589140-n"
              ]
    Synset_Sentiment_set.removeSynIDs(rem_obj)
    
    # Synset_Sentiment_set.addWNopAll()
    # print("POS, NEG and OBJ from WN-OP")
    Synset_Sentiment_set.updateDataFrame()
    return Synset_Sentiment_set
    