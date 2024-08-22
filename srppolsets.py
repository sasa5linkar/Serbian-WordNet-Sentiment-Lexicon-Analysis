# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 13:11:28 2022

@author: "Petalinkar Saša"

Constrstion of training sets for sentimant polarity from Serbain Wordnet

"""
import requests
from srpskiwordnet import SrbSynset
from srpskiwordnet import SrbWordNetReader
import pandas as pd
from nltk.corpus import sentiwordnet as swn
from sklearn.base import BaseEstimator, TransformerMixin

POLARITY = ["POS", "NEG", "OBJ"]


def getObjectiveIDfromWNop(wordnet_corpus_reader:SrbWordNetReader):
    """
    Get sysnst that exist in slected Serbian Wordent and are marked objective
    in WN-op corpus.

    Parameters
    ----------
    wordnet_corpus_reader : SrbWordNetReader
        Reder to Serbain Wornet corpus from which we take sysstes

    Returns
    -------
    Set of Srbsynst
        Set of objective serbain systens

    """
    IDs = list()
    URL = "https://raw.githubusercontent.com/aesuli/SentiWordNet/master/data/Micro-WNop-WN3.txt"
    response = requests.get(URL)
    data = response.text.split('\n')
    common = data[7:115]
    group1= data[118:597]
    group2= data[600:1068]
    for s in common:
        pom = s.split("\t")
        if (pom[0]=="0" and pom[1]=="0"):
            IDs.append(syn2ID(pom[2]))
    for s in group1:
        pom = s.split("\t")
        if (pom[0]=="0" and pom[1]=="0"and pom[2]=="0"and pom[3]=="0"
            and pom[4]=="0"and pom[5]=="0"):
            IDs.append(syn2ID(pom[6]))   
    for s in group2:
        pom = s.split("\t")
        if (pom[0]=="0" and pom[1]=="0"and pom[2]=="0"and pom[3]=="0"):
            IDs.append(syn2ID(pom[4]))        
            
    return set(map(wordnet_corpus_reader.synset_from_ID, IDs))

def getPositiveIDfromWNop(wordnet_corpus_reader:SrbWordNetReader):
    """
    Get sysnst that exist in slected Serbian Wordent and are marked positive
    in WN-op corpus.

    Parameters
    ----------
    wordnet_corpus_reader : SrbWordNetReader
        Reder to Serbain Wornet corpus from which we take sysstes

    Returns
    -------
    Set of Srbsynst
        Set of objective serbain systens

    """
    IDs = list()
    URL = "https://raw.githubusercontent.com/aesuli/SentiWordNet/master/data/Micro-WNop-WN3.txt"
    response = requests.get(URL)
    data = response.text.split('\n')
    common = data[7:115]
    group1= data[118:597]
    group2= data[600:1068]
    for s in common:
        pom = s.split("\t")
        if (pom[0]=="1" and pom[1]=="0"):
            IDs.append(syn2ID(pom[2]))
    for s in group1:
        pom = s.split("\t")
        if (pom[0]=="1" and pom[1]=="0"and pom[2]=="1"and pom[3]=="0"
            and pom[4]=="1"and pom[5]=="0"):
            IDs.append(syn2ID(pom[6]))   
    for s in group2:
        pom = s.split("\t")
        if (pom[0]=="1" and pom[1]=="0"and pom[2]=="1"and pom[3]=="0"):
            IDs.append(syn2ID(pom[4]))        
            
    return set(map(wordnet_corpus_reader.synset_from_ID, IDs))

def getNegativeIDfromWNop(wordnet_corpus_reader:SrbWordNetReader):
    """
    Get sysnst that exist in slected Serbian Wordent and are marked nagative
    in WN-op corpus.

    Parameters
    ----------
    wordnet_corpus_reader : SrbWordNetReader
        Reder to Serbain Wornet corpus from which we take sysstes

    Returns
    -------
    Set of Srbsynst
        Set of objective serbain systens

    """
    IDs = list()
    URL = "https://raw.githubusercontent.com/aesuli/SentiWordNet/master/data/Micro-WNop-WN3.txt"
    response = requests.get(URL)
    data = response.text.split('\n')
    common = data[7:115]
    group1= data[118:597]
    group2= data[600:1068]
    for s in common:
        pom = s.split("\t")
        if (pom[0]=="0" and pom[1]=="1"):
            IDs.append(syn2ID(pom[2]))
    for s in group1:
        pom = s.split("\t")
        if (pom[0]=="0" and pom[1]=="1"and pom[2]=="0"and pom[3]=="1"
            and pom[4]=="0"and pom[5]=="1"):
            IDs.append(syn2ID(pom[6]))   
    for s in group2:
        pom = s.split("\t")
        if (pom[0]=="0" and pom[1]=="1"and pom[2]=="0"and pom[3]=="1"):
            IDs.append(syn2ID(pom[4]))        
            
    return set(map(wordnet_corpus_reader.synset_from_ID, IDs))

def getObjectiveIDfromWSWN(wordnet_corpus_reader:SrbWordNetReader):
    """
    Get sysnst that exist in slected Serbian Wordent and are marked objective
    in SentiWordNet corpus.

    Parameters
    ----------
    wordnet_corpus_reader : SrbWordNetReader
        Reder to Serbain Wornet corpus from which we take sysstes

    Returns
    -------
    Set of Srbsynst
        Set of objective serbain systens

    """
    syns = list()
    URL = "https://raw.githubusercontent.com/aesuli/SentiWordNet/master/data/SentiWordNet_3.0.0.txt"
    response = requests.get(URL)
    data = response.text.split('\n')
    for line in data:
        if line.startswith("#"):
            continue
        pom = line.split("\t")
        if len(pom)<4:
            continue
        pos = pom[0]
        offset = pom[1]
        pos_score = pom[2]
        neg_score = pom[3]
        if pos_score=="0" and neg_score=="0":
            syn_id = syn2ID(pos+ offset)
            syn = wordnet_corpus_reader.synset_from_ID(syn_id)
            if syn is not None:
                if syn.is_definition_in_serbain():
                    syns.append(syn)    
    return syns

def getPositiveIDfromWSWN(wordnet_corpus_reader:SrbWordNetReader):
    """
    Get sysnst that exist in slected Serbian Wordent and are marked objective
    in SentiWordNet corpus.

    Parameters
    ----------
    wordnet_corpus_reader : SrbWordNetReader
        Reder to Serbain Wornet corpus from which we take sysstes

    Returns
    -------
    Set of Srbsynst
        Set of objective serbain systens

    """
    syns = list()
    URL = "https://raw.githubusercontent.com/aesuli/SentiWordNet/master/data/SentiWordNet_3.0.0.txt"
    response = requests.get(URL)
    data = response.text.split('\n')
    for line in data:
        if line.startswith("#"):
            continue
        pom = line.split("\t")
        if len(pom)<4:
            continue
        pos = pom[0]
        offset = pom[1]
        pos_score = pom[2]
        neg_score = pom[3]
        if pos_score=="1" and neg_score=="0":
            syn_id = syn2ID(pos+ offset)
            syn = wordnet_corpus_reader.synset_from_ID(syn_id)
            if syn is not None:
                if syn.is_definition_in_serbain():
                    syns.append(syn)    
    return syns
def getNegativeIDfromWSWN(wordnet_corpus_reader:SrbWordNetReader):
    """
    Get sysnst that exist in slected Serbian Wordent and are marked objective
    in SentiWordNet corpus.

    Parameters
    ----------
    wordnet_corpus_reader : SrbWordNetReader
        Reder to Serbain Wornet corpus from which we take sysstes

    Returns
    -------
    Set of Srbsynst
        Set of objective serbain systens

    """
    syns = list()
    URL = "https://raw.githubusercontent.com/aesuli/SentiWordNet/master/data/SentiWordNet_3.0.0.txt"
    response = requests.get(URL)
    data = response.text.split('\n')
    for line in data:
        if line.startswith("#"):
            continue
        pom = line.split("\t")
        if len(pom)<4:
            continue
        pos = pom[0]
        offset = pom[1]
        pos_score = pom[2]
        neg_score = pom[3]
        if pos_score=="0" and neg_score=="1":
            syn_id = syn2ID(pos+ offset)
            syn = wordnet_corpus_reader.synset_from_ID(syn_id)
            if syn is not None:
                if syn.is_definition_in_serbain():
                    syns.append(syn)    
    return syns


def syn2ID(syn):
    syn = syn.strip()
    ret = "ENG30-" + syn[1:] + "-" + syn[0] 
    return ret


def syn2gloss(syn):
    """
        The syn2gloss function takes a synset as input and returns its definition as a string.
        
        Parameters:
        
        syn: The synset for which the definition is required.
        Returns:
        
        str: The definition of the synset as a string.
    """
    return syn.definition()

def syn2ID2(syn):
    """
    Returns the identifier of a given WordNet Synset.
    
    Parameters:
    syn (WordNet Synset): A WordNet Synset
    
    Returns:
    str: A string representing the identifier of the Synset
    
    """
    return syn.ID()
    
class SrbSynset2GlossTransformer(TransformerMixin, BaseEstimator):
    """
    Class that tranform synst to their gloss.
    """

    def transform(self, X):
        return X.apply(syn2gloss)
  
    def fit(self, X, y):
        return self

# class PolarityDict():
#     def __init__(self, file):
#         head = ["Word", "POS", "NEG"]
#         self.table = pd.read_csv(file, names=head, sep=";")


class PolaritySets():
    """
    A class that stores sets of synsets marked by sentiment from WordNet. 
    Sentiment is divided into objective, positive, and negative. Positive and negative sets 
    are expanded in each iteration by relations between synsets.

    Attributes
    ----------
    _wordnet_corpus_reader : SrbWordNetReader
        A Serbian WordNet corpus reader object.
    _pos : set
        A set of synsets marked as positive.
    _neg : set
        A set of synsets marked as negative.
    _obj : set
        A set of synsets marked as objective.
    _k : int
        The number of iterations performed.

    Methods
    -------
    __init__(self, wordnet_corpus_reader, k=0)
        Initializes a PolaritySets object.
    addPOS(self, word)
        Adds synsets that contain a given lemma to the positive set.
    addPOSIDall(self, IDs)
        Adds all synsets with given IDs to the positive set.
    addPOSID(self, ID)
        Adds a synset with a given ID to the positive set.
    addPOSall(self, words)
        Adds all synsets that contain given lemmas to the positive set.
    addNEG(self, word)
        Adds synsets that contain a given lemma to the negative set.
    addNEGall(self, words)
        Adds all synsets that contain given lemmas to the negative set.
    addNEGID(self, ID)
        Adds a synset with a given ID to the negative set.
    addNEGIDall(self, IDs)
        Adds all synsets with given IDs to the negative set.
    addOBJSYN(self, syns)
        Adds a given set of synsets to the objective set.
    addPOSSYN(self, syns)
        Adds a given set of synsets to the positive set.
    addNEGSYN(self, syns)
        Adds a given set of synsets to the negative set.
    addWNop(self)
        Adds all synsets from WN-op corpus to the objective set.
    addWNopAll(self)
        Adds all synsets from WN-op corpus to the objective set.
    addWSWN(self)
        Adds all synsets from WSWN corpus to the objective set.
    addWSWNALL(self)
        Adds all synsets from WSWN corpus to the objective, positive, and negative sets.
    removeSyn(self, syn, polarity="OBJ")
        Removes a given synset from the stated polarity set (default is objective).
    removeSynID(self, ID, polarity="OBJ")
        Removes a synset with a given ID from the stated polarity set (default is objective).
    removeSynIDs(self, IDs, polarity="OBJ")
        Removes synsets with given IDs from the stated polarity set (default is objective).
    next_itteration(self)
        Returns a new PolaritySets object that has been updated with a new iteration.
    _expandPolarty(self)
        Expands the positive and negative sets based on the relationships between synsets.
    """
    def __init__(self, wordnet_corpus_reader:SrbWordNetReader, k = 0 ):
        """
        Initializes the PolaritySets object with empty sets of synsets for each sentiment category 
        (positive, negative, and objective), based on the provided wordnet corpus reader.

        The positive and negative synset sets are expanded in each iteration based on the relations 
        between the synsets within the WordNet structure.

        Parameters
        ----------
        wordnet_corpus_reader : SrbWordNetReader
            An instance of the Serbian WordNet Corpus Reader class. It provides the structure and 
            relations of the synsets for the Serbian language WordNet.

        k : Integer, optional
            The number of expansion iterations to perform upon initialization. Default is 0, meaning 
            no expansion is performed at the start.
        """
        self._wordnet_corpus_reader = wordnet_corpus_reader
        self._pos = set()
        self._neg = set()
        self._obj = set()
        self._k = k

    def addPOS(self, word):
        """
        Adds synsets containing the given literal to the positive sentiment set.

        Parameters
        ----------
        word : String
            The literal word to be used to find and add related synsets to the positive sentiment set.

        Returns
        -------
        None. The function operates in-place and modifies the object's state.
        """

        syns = self._wordnet_corpus_reader.synsets(word)
        for syn in syns:
            self.addPOSID(syn._ID)            
    def addPOSIDall(self, IDs):
        """
        Adds all synsets, whose IDs are contained within the provided iterable, to the positive sentiment set.

        Parameters
        ----------
        IDs : Iterable
            An iterable object (such as list or set) containing the IDs of synsets to be added to the positive sentiment set.

        Returns
        -------
        None. The function operates in-place and modifies the object's state.

        """
        for ID in IDs:
            self.addPOSID(ID)

    def addPOSID(self, ID):
        """
        Adds the synset with the given ID to the positive sentiment set, provided the synset has a Serbian definition.

        Parameters
        ----------
        ID : String
            The ID of the synset to be added to the positive sentiment set.

        Returns
        -------
        None. The function operates in-place and modifies the object's state.
        """        
        syn = self._wordnet_corpus_reader.synset_from_ID(ID)
        if syn.is_definition_in_serbain():
            self._pos.add(syn)
            self._obj.discard(syn)

    def addPOSall(self, words):
        """
        Adds all synsets related to the given words to the positive sentiment set.

        This function iterates through each word in the provided iterable and adds 
        the related synsets to the positive sentiment set by calling the addPOS method.

        Parameters
        ----------
        words : Iterable
            An iterable object (such as list or set) containing words. 
            The synsets related to these words are to be added to the positive sentiment set.

        Returns
        -------
        None. The function operates in-place and modifies the object's state.
        """
        for word in words:
            self.addPOS(word)

    def addNEG(self, word):
        """
        Adds synsets that contain the specified literal to the negative sentiment set.

        Parameters
        ----------
        word : str
            The literal string whose corresponding synsets are to be added to the negative sentiment set.

        Returns
        -------
        None. The function operates in-place and modifies the object's state.
        """

        syns = self._wordnet_corpus_reader.synsets(word)
        for syn in syns:
            self.addNEGID(syn._ID)  

    def addNEGall(self, words):
        """
        Adds all synsets related to the given words to the negative sentiment set.

        Parameters
        ----------
        words : Iterable
            An iterable object (such as list or set) containing words. 
            The synsets related to these words are to be added to the negative sentiment set.

        Returns
        -------
        None. The function operates in-place and modifies the object's state.
        """
        for word in words:
            self.addNEG(word)

    def addNEGID(self, ID):
        """
        Adds the synset with the specified ID to the negative sentiment set, 
        provided the definition of the synset is in Serbian.

        Parameters
        ----------
        ID : str
            The ID of the synset to be added to the negative sentiment set.

        Returns
        -------
        None. The function operates in-place and modifies the object's state.
        """
        syn = self._wordnet_corpus_reader.synset_from_ID(ID)
        if syn.is_definition_in_serbain():
            self._neg.add(syn)
            self._obj.discard(syn)
            
            
    def addNEGIDall(self, IDs):
        """
        Adds all synsets with IDs contained in the given iterable to the negative sentiment set.

        Parameters
        ----------
        IDs : Iterable
            An iterable object (such as list or set) containing the IDs of synsets 
            to be added to the negative sentiment set.

        Returns
        -------
        None. The function operates in-place and modifies the object's state.
        """
        for ID in IDs:
            self.addNEGID(ID)
    def addOBJSYN(self, syns):
        """
        Adds synsets to the objective sentiment set if they exist and are defined in Serbian.
        They also should not be present in the positive or negative sentiment sets.

        Parameters
        ----------
        syns : Iterable
            An iterable of synsets (such as a list or a set).

        Returns
        -------
        None. This function operates in-place and modifies the object's state.
        """
        
        for syn in syns:
            if syn is not None:
                if syn.is_definition_in_serbain():
                    if  not (syn in self._neg or syn in self._pos):
                        self._obj.add(syn)

    def addPOSSYN(self, syns):
        """
        Adds synsets to the objective sentiment set if they exist and are defined in Serbian.
        They also should not be present in the positive or negative sentiment sets.

        Parameters
        ----------
        syns : Iterable
            An iterable of synsets (such as a list or a set).

        Returns
        -------
        None. This function operates in-place and modifies the object's state.
        """

        for syn in syns:
            if syn is not None:
                if syn.is_definition_in_serbain():
                    self._pos.add(syn)
                    self._obj.discard(syn)

    def addNEGSYN(self, syns):
        """
        Adds synsets to the objective sentiment set if they exist and are defined in Serbian.
        They also should not be present in the positive or negative sentiment sets.

        Parameters
        ----------
        syns : Iterable
            An iterable of synsets (such as a list or a set).

        Returns
        -------
        None. This function operates in-place and modifies the object's state.
        """

        for syn in syns:
            if syn is not None:
                if syn.is_definition_in_serbain():
                    self._neg.add(syn)
                    self._obj.discard(syn)

    def addWNop(self):
        self.addOBJSYN(getObjectiveIDfromWNop(self._wordnet_corpus_reader))

    def addWNopAll(self):
        """
        Add all synstets from WN-op corpus.

        Returns
        -------
        None.

        """
        self.addOBJSYN(getObjectiveIDfromWNop(self._wordnet_corpus_reader))
        self.addPOSSYN(getPositiveIDfromWNop(self._wordnet_corpus_reader))
        self.addNEGSYN(getNegativeIDfromWNop(self._wordnet_corpus_reader))

    def addWSWN(self):
        self.addOBJSYN(getObjectiveIDfromWSWN(self._wordnet_corpus_reader))
        
    def addWSWNALL(self):
        self.addOBJSYN(getObjectiveIDfromWSWN(self._wordnet_corpus_reader))
        self.addPOSSYN(getPositiveIDfromWSWN(self._wordnet_corpus_reader))
        self.addNEGSYN(getNegativeIDfromWNop(self._wordnet_corpus_reader))
    def removeSyn (self, syn, polarity = "OBJ"):
        """
        Removes a synset from the specified sentiment set. By default, it removes from the objective sentiment set.

        Parameters
        ----------
        syn : Synset
            The synset to be removed.

        polarity : str, optional
            The sentiment set from which to remove the synset. Must be one of ["OBJ", "NEG", "POS"]. Default is "OBJ".

        Returns
        -------
        None. This function operates in-place and modifies the object's state.
        """
        if (polarity == "OBJ"):
           self._obj.discard(syn)
        elif (polarity == "NEG"):
           self._neg.discard(syn)
        elif (polarity == "POS"):
           self._pos.discard(syn)
    def removeSynID (self, ID, polarity = "OBJ"):
        """
        Removes a synset from the specified sentiment set by its ID. By default, it removes from the objective sentiment set.

        Parameters
        ----------
        ID : str
            The ID of the synset to be removed.

        polarity : str, optional
            The sentiment set from which to remove the synset. Must be one of ["OBJ", "NEG", "POS"]. Default is "OBJ".

        Returns
        -------
        None. This function operates in-place and modifies the object's state.
        """
        syn = self._wordnet_corpus_reader.synset_from_ID(ID)
        self.removeSyn (syn, polarity)
    def removeSynIDs (self, IDs, polarity = "OBJ"): 
        """
        Removes multiple synsets from the specified sentiment set by their IDs. By default, it removes from the objective sentiment set.

        Parameters
        ----------
        IDs : Iterable
            An iterable of synset IDs to be removed (such as a list or a set).

        polarity : str, optional
            The sentiment set from which to remove the synsets. Must be one of ["OBJ", "NEG", "POS"]. Default is "OBJ".

        Returns
        -------
        None. This function operates in-place and modifies the object's state.
        """
        for ID in IDs:
            self.removeSynID(ID, polarity)
        
    def next_iteration (self):
        """
        Produces the next iteration of the polarity sets. 
        
        This method copies the current sets of positive, negative, and objective synsets 
        to a new PolaritySets object, then expands the polarity sets of the new object, 
        updates its internal dataframe, and returns the new object. This method leaves 
        the original object unchanged.
        
        The iteration count (k) of the new object will be one greater than the iteration count 
        of the original object.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        PolaritySets
            The newly generated PolaritySets object with expanded polarity sets and updated internal dataframe.
        """
        ret = PolaritySets(self._wordnet_corpus_reader, self._k +1)
        ret._pos = self._pos.copy()
        ret._neg = self._neg.copy()
        ret._obj = self._obj.copy()
        ret._expandPolarty()
        ret.updateDataFrame()
        return ret

    def _expandPolarty(self):
        """
        Expand the positive and negative polarity sets based on defined relationships in WordNet.

        We are using certain relationships from WordNet for the expansion:

        - Antonyms: The antonyms of positive synsets are added to the negative set and vice versa.
        - Other relationships ("+", "=", "^"): The synsets related to positive or negative synsets through these relationships are added to the respective set.

        The expansion is performed iteratively and updates the existing positive and negative sets.
        """
        rel= {"+","=","^"}
        neg = self._neg.copy()
        pos = self._pos.copy()
        #reversed
        for syn in self._pos:
                for s in syn.antonyms():
                    if s is not None:
                        if s.is_definition_in_serbain():
                            neg.add(s)
        for syn in self._neg:
            for s in syn.antonyms():
                if s is not None:
                    if s.is_definition_in_serbain():
                        pos.add(s)
        #preserved
        for r in rel:
            for syn in self._pos:
                for s in syn._related(r):
                    if s is not None:
                        if s.is_definition_in_serbain():
                            pos.add(s)
            for syn in self._neg:
                for s in syn._related(r):
                    if s is not None:
                        if s.is_definition_in_serbain():
                            neg.add(s)
        pos.discard(None)
        neg.discard(None)
        self._neg =neg
        self._pos =pos

    def _getText(self, pol):
        """
        Generate a list of dictionaries containing information about synsets belonging to a particular polarity.
    
        Parameters
        ----------
        pol : string ("POS", "NEG", "OBJ")
            The polarity to retrieve synsets for.
    
        Returns
        -------
        ret : list of dicts
            Each dictionary represents a synset and includes the synset's ID, sentiment, lemma names, definition, and POS tag.
        """
        ret = list()
        if (pol == "POS"):
            syns = self._pos
        if (pol == "NEG"):
            syns = self._neg
        if (pol == "OBJ"):
            syns = self._obj
        for syn in syns:
            el = dict()
            el["ID"] = syn.ID()
            el["POS"], el["NEG"] = syn._sentiment
            el["Lemme"] = ",".join(syn._lemma_names)
            el["Definicija"] = syn.definition()
            el["Vrsta"] = syn._POS
            ret.append(el)
        return ret
    def getDef(self):
        """
        Create a dictionary containing lists of synsets for each polarity, as well as the current iteration number.
    
        Returns
        -------
        ret : dict
            The returned dictionary has keys for "POS", "NEG", "OBJ", and "iteration". 
            The values for each polarity are lists of synsets, while the value for "iteration" is an integer.
        """
        ret = dict()
        for t in POLARITY:
            ret[t] = self._getText(t)
        ret["iteration"] = self._k
        return ret

    def updateDataFrame(self):
        """
        Updates the internal dataframe based on the current status of positive (_pos), negative (_neg), 
        and objective (_obj) synset collections.
        
        This function should be called after all manual entries of synsets are completed. 
        However, it is automatically invoked after the expansion by polarity process.
        
        The internal dataframe is restructured with the columns: 'POS', 'NEG', 'OBJ', and 'Sysnet'. 
        Each row corresponds to a synset and its associated polarity labels (binary encoded), 
        where '1' indicates the synset belongs to the respective polarity (Positive, Negative, or Objective) 
        and '0' indicates it does not.
        
        Returns
        -------
        None. The function operates in-place and does not return any value.
        """

        dfpos = pd.DataFrame(self._pos, columns=["Sysnet"])
        dfpos.insert(0, "POS", 1)
        dfpos.insert(0, "NEG", 0)
        dfpos.insert(0, "OBJ", 0)
        dfneg = pd.DataFrame(self._neg, columns=["Sysnet"])
        dfneg.insert(0, "POS", 0)
        dfneg.insert(0, "NEG", 1)
        dfneg.insert(0, "OBJ", 0)
        dfobj = pd.DataFrame(self._obj, columns=["Sysnet"])
        dfobj.insert(0, "POS", 0)
        dfobj.insert(0, "NEG", 0)
        dfobj.insert(0, "OBJ", 1)
        self._df = pd.concat([dfpos, dfneg, dfobj])



    def getXY(self, polarity="POS", preprocess=[syn2gloss]):
        """
        Generates input features (X) and target labels (y) for machine learning models.
        The function operates on a DataFrame containing synsets and their respective polarities.
    
        Parameters
        ----------
        polarity : str, optional
            The sentiment polarity used to classify the target labels (y).
            Accepts either "POS" (positive) or "NEG" (negative). The default is "POS".
            
        preprocess : list of functions, optional
            A sequence of preprocessing functions to apply on each element of the synset data (X).
            By default, this list includes the `syn2gloss` function, which converts synsets to glosses.
    
        Returns
        -------
        X : pandas Series
            The input features for a machine learning model, each element being a Serbian Synset.
            
        y : pandas Series
            The target labels for a machine learning model. This represents a binary classification where 0 denotes
            that the synset does not correspond to the selected polarity, while 1 indicates it does.
        """
        X = self._df["Sysnet"]
        y = self._df[polarity]
        
        if preprocess is not None:
            for f in preprocess:
                X= X.apply(f)
        
        return X, y
