# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 12:14:54 2022

@author: "Petalinkar Saša"
"""


import copy
from pprint import pprint
from time import time
import logging
import numpy as np
import treetaggerwrapper as ttpw


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

from nltk import FreqDist
from nltk.tokenize import word_tokenize
import toolz
from nltk.probability import FreqDist
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.pipeline import make_pipeline
from numpy.random import RandomState
from sklearn.metrics import ConfusionMatrixDisplay, PredictionErrorDisplay
from sklearn.metrics import confusion_matrix 
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

# C_range = np.logspace(-2, 10, 5)
# gamma_range = np.logspace(-9, 3, 13)
stop_words =[".", "i", "ili", "(", ")", ";", ",", "u", "iz", "se", "koji", "na", "kao", "sa", "kojim", "koj"]
scorer = make_scorer(accuracy_score)
tg_names = ["No", "Yes"]


def apllyMLpipe (defData, pipe):
    X_train, X_test, y_train, y_test = train_test_split(defData["Text"],
                                                        defData["Sentiment"], stratify=defData["Sentiment"])
    # print ("Train " + str(sum (y_train)))
    # print ("Test " + str(sum (y_test)))
    
    
    pipe.fit(X_train, y_train, )
    y_pred = pipe.predict(X_test)
    RocCurveDisplay.from_predictions(y_test, y_pred)
    return roc_curve( y_test, y_pred)
def testSVM(data):
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("svm", SVC()),
        ]
    )
    parameters = {
        "vect__preprocessor": (None, stem_str),
        "vect__max_df": (0.5, 0.75, 1.0),
        "vect__stop_words": (None, stop_words),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        "svm__kernel": ('poly', 'rbf', 'sigmoid'),
        "svm__probability": (True,),
        "svm__class_weight": (None, "balanced"),
        "svm__C": C_range,
        "svm__cache_size": (1000,)
        
    }
    for i, d in enumerate(data):
        print ("Iteration : ", i)
        #positive
        print ("Positive sentiment")
        x, y =  d["Text"], d ["POS"]
        showGrid(x,y,pipeline, parameters)
        print ("Negative sentiment")
        x, y =  d["Text"], d ["NEG"]
        showGrid(x,y,pipeline, parameters)
def testCLF(data):
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", SGDClassifier()),
        ]
    )
    parameters = {
        "vect__preprocessor": (None, stem_str),
        "vect__max_df": (0.5, 0.75, 1.0),
        "vect__stop_words": (None, stop_words),
        'vect__max_features': (None, 5000, 10000, 50000),
        "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        "clf__loss": ('hinge', 'log_loss', 'perceptron'),
        "clf__early_stopping": (False, True),
        "clf__class_weight": (None, "balanced"),
        "clf__alpha": (0.00001, 0.000001),
        "clf__penalty": ("l2", "elasticnet")
    }
    for d in data:
        print ("Iteration : ", d["iteration"] )
        #positive
        print ("Positive sentiment")
        pom =  d ["pd"]
        x, y =  pom["Text"], pom ["Sentiment"]
        showGrid(x,y,pipeline, parameters)
        print ("Negative sentiment")
        pom =  d ["nd"]
        x, y =  pom["Text"], pom ["Sentiment"]
        showGrid(x,y,pipeline, parameters)


def showGrid(x, y, pipeline, parameters, title=""):
    """
    Perform grid serch for machine lerning and print and lots results.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    pipeline : TYPE
        DESCRIPTION.
    parameters : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(17, 17))
    fig.suptitle(title, fontsize=16)
    scorers = ['balanced_accuracy', "f1_weighted", 'precision_weighted',
               "recall_weighted", 'precision' , 'recall']
    X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y)
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1,
                               scoring=scorers, cv=5,
                               refit="balanced_accuracy")

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    # print("parameters:")
    # pprint(parameters)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()
    results = grid_search.cv_results_
    print("Best score: %0.3f" % grid_search.best_score_)
    for scorer in scorers:
        best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
        best_score = results["mean_test_%s" % scorer][best_index]
        print("Best score for %s: %0.3f" % (scorer, best_score))
    
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    est = grid_search.best_estimator_
    y_pred_test = est.predict(X_test)
    y_pred_train = est.predict(X_train)
    print("Test classifcation report:")
    print(classification_report(y_test, y_pred_test))
    # y_pred_all = est.predict(x)
    cm = confusion_matrix(y_test, y_pred_test)
    display = ConfusionMatrixDisplay(cm).plot(ax=axs[0])
    display.ax_.set_title(" Test")
    # cm = confusion_matrix(y_train, y_pred_train)
    # display = ConfusionMatrixDisplay(cm).plot(ax=axs[1, 0])
    # display.ax_.set_title("Train")
    # cm = confusion_matrix(y, y_pred_all)
    # display = ConfusionMatrixDisplay(cm).plot(ax=axs[2, 0])
    # display.ax_.set_title("All")
    prec, recall, _ = precision_recall_curve(y_test, y_pred_test)
    display = PrecisionRecallDisplay(precision=prec, recall=recall).plot(
        ax=axs[1])
    display.ax_.set_title(" Test")
    # prec, recall, _ = precision_recall_curve(y_train, y_pred_train)
    # display = PrecisionRecallDisplay(precision=prec, recall=recall).plot(
    #     ax=axs[1, 1])
    # display.ax_.set_title("Train")
    return grid_search
    # prec, recall, _ = precision_recall_curve(y, y_pred_all)
    # display = PrecisionRecallDisplay(precision=prec, recall=recall).plot(
    #     ax=axs[2, 1])
    # display.ax_.set_title("All")
    
def showGridReg(x, y, pipeline, parameters, title=""):
    """
    Perform grid serch for machine lerning and print and lots results.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    pipeline : TYPE
        DESCRIPTION.
    parameters : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(17, 17))
    fig.suptitle(title, fontsize=16)
    scorers = ['explained_variance', "max_error", 'r2',
               "neg_mean_absolute_percentage_error" 
               ]
    X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y)
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1,
                               scoring=scorers, cv=5,
                               refit="r2")

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    # print("parameters:")
    # pprint(parameters)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()
    results = grid_search.cv_results_
    print("Best score: %0.3f" % grid_search.best_score_)
    for scorer in scorers:
        best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
        best_score = results["mean_test_%s" % scorer][best_index]
        print("Best score for %s: %0.3f" % (scorer, best_score))
    
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    est = grid_search.best_estimator_
    y_pred_test = est.predict(X_test)
    print("Test r2 score:")
    print(r2_score(y_test, y_pred_test))
    display = PredictionErrorDisplay(y_test, y_pred_test).plot(ax=axs[0])

    display.ax_.set_title(" Test")
    return grid_search

def dic2panda (data):
    """
    Decreptet.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    pom : TYPE
        DESCRIPTION.

    """
    pom = list()
    for d in data:
        dfpos = pd.DataFrame(d["POS"],columns=["Text"] )
        dfpos.insert(0,"POS", 1)
        dfpos.insert(0,"NEG", 0)
        dfpos.insert(0,"OBJ", 0)
        dfneg = pd.DataFrame(d["NEG"],columns=["Text"] )
        dfneg.insert(0,"POS", 0)
        dfneg.insert(0,"NEG", 1)
        dfneg.insert(0,"OBJ", 0)
        dfobj = pd.DataFrame(d["OBJ"],columns=["Text"] )
        dfobj.insert(0,"POS", 0)
        dfobj.insert(0,"NEG", 0)
        dfobj.insert(0,"OBJ", 1)
        

        ret = pd.concat([dfpos, dfneg, dfobj])
        pom.append(ret)
    return pom

    
def dic2panda2 (data):
    pom = list()
    for d in data:
        ret = dict()

        allDef = d["OBJ"] | d["NEG"] | d["POS"]
        notposDef = d["OBJ"] | d["NEG"]
        notNegDef =  d["OBJ"] | d["POS"]
        dfpos = pd.DataFrame(d["POS"],columns=["Text"] )
        dfpos.insert(0,"Sentiment", 1)


        dfnpos = pd.DataFrame(notposDef,columns=["Text"] )
        dfnpos.insert(0,"Sentiment", 0)
        df = pd.concat([dfpos, dfnpos])
        dfneg = pd.DataFrame(d["NEG"],columns=["Text"] )
        dfneg.insert(0,"Sentiment", 1)

        dfnneg = pd.DataFrame(notNegDef, columns=["Text"])
        dfnneg.insert(0, "Sentiment", 0)
        df1 = pd.concat([dfneg, dfnneg])

        ret["iteration"] = d["iteration"]
        ret["alld"] = allDef
        ret["pd"] = df
        ret["nd"] = df1
        pom.append(ret)
    return pom



def PCA_visual(data):
    pass

def apllyToAllDEf (definions, function):
    ret = list()
    for d in definions:
        ret.append(toolz.valmap(lambda l: list(map(function, l)), d))
    return ret


# defi1 = dic2panda(defi)
# testSVM(defi1)

# pipeline = Pipeline(
#     [
#         ("gloss", SrbSynset2GlossTransformer()),
#         ("vect", CountVectorizer()),
#         ("tfidf", TfidfTransformer()),
#         ("svm", SVC()),
#     ]
# )
# parameters = {
#     "vect__preprocessor": (None, stem_str),
#     "vect__max_df": (0.5, 0.75, 1.0),
#     "vect__stop_words": (None, stop_words),
#     # 'vect__max_features': (None, 5000, 10000, 50000),
#     "vect__ngram_range": ((1, 1), (1, 2)),  # unigrams or bigrams
#     'tfidf__use_idf': (True, False),
#     'tfidf__norm': ('l1', 'l2'),
#     "svm__kernel": ('poly', 'rbf', 'sigmoid'),
#     "svm__probability": (True,),
#     "svm__class_weight": (None, "balanced"),
#     "svm__C": C_range,
#     "svm__cache_size": (1000,)
    
# }
# for syn in Synset_Sentiment:
#     X, y = syn.getXY() 
#     showGrid(X, y, pipeline, parameters)
    
# X, y = Synset_Sentiment[0].getXY()

# showGrid(X, y, pipeline, parameters)





#    defi1.append(toolz.valmap(lambda l: list(map(stem_arr, l)), d))
 