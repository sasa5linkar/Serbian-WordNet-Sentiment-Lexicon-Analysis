# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:06:18 2022

@author: "Petalinkar Saša"

NLTK interface for Serbian Wordnet
loaded from XML file 

Last modified on Wed Jun 15 15:06:18 2022
"""
import os
from xml.etree.ElementTree import Element, SubElement
import xml.etree.cElementTree as ET
from nltk.corpus.reader.wordnet import Lemma
from nltk.corpus.reader.wordnet import Synset
from nltk.corpus.reader.xmldocs import XMLCorpusReader
from itertools import chain, islice
import pandas as pd

######################################################################
# Table of Contents
######################################################################
# - Constants
# - Data Classes
#   - Serbian Lemma
#   - Serbian Synset
# - Serbian WordNet Corpus Reader
######################################################################
# Constants
######################################################################

#: Positive infinity (for similarity functions)
_INF = 1e300

# { Part-of-speech constants
ADJ, ADJ_SAT, ADV, NOUN, VERB = "a", "s", "r", "n", "v"
# }

POS_LIST = [NOUN, VERB, ADJ, ADV]

    #This maps symbols used for relations form SW xml to WN format
    # https://wordnet.princeton.edu/documentation/wninput5wn
REL_MAP = {
        "hypernym":"@",
        "hyponym":"~",
        "eng_derivative": "+",  #all are marked as Derivationally related form
        'holo_member': '#m', 
        'derived-vn':'+',       #all are marked as Derivationally related form
        'holo_member': '#m', 
        'particle':'p',         #not in wn5
        'instance_hypernym':'@i',
        'substanceHolonym':'#s',
        'attribute':'=', 
        'SubstanceMeronym':'%s',
        'verb_group':'$',
        'TopicDomain':';c',
        'usage_domain':';u',
        'similar_to':'&', 
        'category_domain':';r', #not in wn5
        'Entailment':'*', 
        'TopicDomainMember':'-c',
        'holo_part':'#p',
        'holo_portion':'#p',
        'mero_portion':'%p', 
        'mero_member':'%m', 
        'entailment':'*', 
        'partMeronym':'%p', 
        'region_domain':';r', 
        'InstanceHyponym':'~i', 
        'causes':'>',
        'be_in_state':"b",      #not in wn5 
        'RegionDomain':';r',
        'subevent':'e',         #not in wn5
        'pertainym':'\\',
        'derived-pos':'+',      #all are marked as Derivationally related form
        'near_antonym':'!',     #marked as antonym
        'DerivedFromAdjective':'\\',
        'specifiedBy':'~',      #this is the defintion of hyponym
        'substanceMeronym':'%s',
        'derived-gender':'+',   #all are marked as Derivationally related form 
        'also_see':'^',     
        'specificOf':'@',       #this is the defintion of hypernym
        'derived':'+',           #all are marked as Derivationally related form
        None:"?"                #None has appered when all possible 
                                #types were listed. This serves for rror check 
        }

######################################################################
# Data Classes
######################################################################

#Serbian Lemma
class SrbLemma(Lemma):
    """The lexical entry for a single morphological form of a
    sense-disambiguated word, for Serbina WordNet
    
    Create a Lemma from a <xml Element> 
    <LITERAL><SENSE></SENSE><LNOTE /></LITERAL>
    Name in PWN is in form
    <word>.<pos>.<number>.<lemma> since we lack that informarion SWNT
    we will just use literal for name of lemma
    """
    def __init__(self, xmlLemma, synset):
        self._lang = "srp"
        self._synset = synset
        self._wordnet_corpus_reader = synset._wordnet_corpus_reader
        if (xmlLemma.tag != "LITERAL"):
            raise Exception ("Not a word")
        self._name = xmlLemma.text
        def get_single(field):
            ret = None
            xml_chunk = xmlLemma.find(field)    
            if xml_chunk is not None:
                ret = xml_chunk.text
            return ret
        self.__annotations__ = get_single("LNOTE")
        self._sense = get_single("SENSE")
    def xml(self):
        """
        Return XML representation of lemma
        """
        xmlLemma = Element("LITERAL")
        xmlLemma.text = self._name
        xmlLemma.append(SubElement(xmlLemma, "SENSE"))
        xmlLemma.append(SubElement(xmlLemma, "LNOTE"))
        return xmlLemma    
    def __repr__(self):
        return "<SrbLemma name:%s synset:%s>" % (self._name, self._synset)

    def __str__(self):
        return "SrbLemma: a is %s, b is %s" % (self.a, self.b)


#Serbian Synset
class SrbSynset(Synset):
    """
    A sysnset from Seribian Wordnet.

    Based on Eurowordnet form
    Rewcoreded as XML
    Contain unqiue ID
    """

    def __init__(self, xmlSynset:Element,wordnet_corpus_reader):
        """
        Initilies synstet from XML element 

        Parameters
        ----------
        xmlSynset : Element
            XML element that conatins full desciption of perticualr synset
        wordnet_corpus_reader : SrbWordNetReader
            Corpuse reader linked XML file with Serbian Wornet

        Raises
        ------
        Exception
            "Not a synset" xml elent is not  a synstet
            "Synset lacks id" xml elemnts lack id fileds or its empnty
            
        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self._wordnet_corpus_reader = wordnet_corpus_reader
        if (xmlSynset.tag != "SYNSET"):
            raise Exception ("Not a synset")
        self._ID = xmlSynset.find("ID").text
        if (self._ID == ""):
            raise Exception ("Synset lacks id")
        
        #here we load literals. If they are missing we reaise na exeption
        xml_pom = xmlSynset.find("SYNONYM")
        self._lemmas = list()
        self._lemma_names =list()
        if (xml_pom is None):
            raise Exception ("Synset "+ self._ID + " lacks literals")
        for lit in xml_pom.findall("LITERAL"):
            pom = SrbLemma(lit, self)
            self._lemmas.append(pom)
            self._lemma_names.append(pom.name())
        #name of the fisrt literal is assigned name of the synset liken PWN
        self._name = self._ID
        
        self._all_hypernyms = None
        self._rel = dict()
        self._relwn5 = dict()
        for rel in xmlSynset.findall("ILR"):
            self._add_rel(rel)
        #fields with sigle text value     
        def get_single(field):
            ret = None
            xml_chunk = xmlSynset.find(field)    
            if xml_chunk is not None:
                ret = xml_chunk.text
            return ret

        self._POS = get_single("POS")
        self._stamp = get_single("STAMP")
        self._definition = get_single("DEF")
        self._definition_lemma = None
        self._domain = get_single("DOMAIN")
        self._NL = get_single("NL")
        self._BCS= get_single("BCS")
        self._SNOTE = get_single("SNOTE")
        #usage/not present in all synsets
        self._examples = list()
        for us in xmlSynset.findall("USAGE"):
            self._examples.append(us.text)
        #sentiment value- originaly in text. conevert to real (replace , with .)
        # we store sentiment as tuple (POS,NEG)
        def txt2real(text):
            return float(text.replace(",","."))
        sent = xmlSynset.find("SENTIMENT")
        self._sentiment = (txt2real(sent.find("POSITIVE").text),txt2real(sent.find("NEGATIVE").text))        

    def definition(self):
        """Return definition in serbian language."""
        return self._definition

    def examples(self):
        """Return examples in serbian language."""
        return self._examples

    def ID(self):
        """Return uniue ID of this synset.

        Returns
        -------
        String
            unquie ID
        """
        return self._ID
    def _add_rel(self, xmlRel:Element):
        """
        Add releshionship to synset, using part of its xml.

        Parameters
        ----------
        xmlRel : Element
            xml representing releshionship under IRL tag
            example:
                <ILR>ENG30-03297735-n<TYPE>hypernym</TYPE></ILR>
        Returns None
        -------
        None.

        """
        pom_id = xmlRel.text
        pom_type = xmlRel.find("TYPE").text
        if (pom_type not in self._rel.keys()):
            self._rel[pom_type] = set()
        self._rel[pom_type].add(pom_id)
        pom_typewn5 = REL_MAP[pom_type]
        if (pom_typewn5 not in self._relwn5.keys()):
            self._relwn5[pom_typewn5] = set()
        self._relwn5[pom_typewn5].add(pom_id)
        
    def _related(self, relation_symbol, sort=False):
        
        get_synset = self._wordnet_corpus_reader.synset_from_ID
        if relation_symbol not in self._relwn5.keys():
            return []
        pointer_ID = self._relwn5[relation_symbol]
        r = [get_synset(ID) for ID in pointer_ID]
        if sort:
            r.sort()
        return r        
    def get_relations_types(self):
        return list(self._rel.keys())
    def __repr__(self):
        return """<SrbSynset ID:%s>
        "Lemma - %s"
        "Definition- %s"
        
        """ % (self._ID, self._lemmas, self.definition() )
    def lemmas(self):
        """
        Returns the set of lemmas that are part of the synset.
        """
        return self._lemmas
    
    def lemma_names(self):
        """
        Returns the list of lemma names that are part of the synset.
        """
        return self._lemma_names
    
    def antonyms(self):
        """
        Returns a set of antonyms (lemmas with opposite meaning) of the synset.
        """
        return self._related("!")
    
    def derivationally_related_forms(self):
        """
        Returns a set of derivationally related forms (related to the synset by morphological derivation).
        """
        return self._related("+")
    
    def POS(self):
        """
        Returns the part-of-speech (POS) of the synset.
        """
        return self._POS


    def parse_definition(self, parser):
        """
        Parses definion of synstet. 

        Parameters
        ----------
        parser : funtion 
            Parser. 

        Returns
        -------
        None.

        """
        self._definition = parser (self._definition)
    def is_definition_in_serbain(self):
        """
        Checks if defintion of synste is in Serbian.
        
        Some sysnets have defintion temorary copied from Prinston WN. 
        That defiention start with "?"

        Returns
        -------
        Boolean
            true Serbian
            false: English or does not exists

        """
        if self.definition() is None:
            return False
        return not self.definition().startswith('?')     
    
    def _estimateSentiment (self, estimators, preprocessor):
        """
        Calculte sentiment using trained ML estimators 

        Parameters
        ----------
        estimators : (sklearn.base.BaseEstimator, sklearn.base.BaseEstimator)
            A tuple of estimators, first for positive, second for negative 
        preprocessor: function
            A pretprocesor for deinition before appling ML 

        Returns
        -------
        (POS, NEG)

        """
        est_POS, est_NEG = estimators
        tekst = preprocessor(self.definition())
        p = est_POS.predict(tekst)
        n = est_NEG.predict(tekst)
        return (p*(1-n), n(1-p))
    def xml(self):
        """Returns xml representation of synset
        same as one used to create synset
        """
        xmlSynset = Element("SYNSET")
        xmlSynset.append(SubElement(xmlSynset, "ID"))
        xmlSynset.find("ID").text = self._ID
        xmlSynset.append(SubElement(xmlSynset, "POS"))
        xmlSynset.find("POS").text = self._POS
        xmlSynset.append(SubElement(xmlSynset, "STAMP"))
        xmlSynset.find("STAMP").text = self._stamp
        xmlSynset.append(SubElement(xmlSynset, "DEF"))
        xmlSynset.find("DEF").text = self._definition
        xmlSynset.append(SubElement(xmlSynset, "NL"))
        xmlSynset.find("NL").text = self._NL
        xmlSynset.append(SubElement(xmlSynset, "BCS"))
        xmlSynset.find("BCS").text = self._BCS
        xmlSynset.append(SubElement(xmlSynset, "SNOTE"))
        xmlSynset.find("SNOTE").text = self._SNOTE
        xmlSynset.append(SubElement(xmlSynset, "DOMAIN"))
        xmlSynset.find("DOMAIN").text = self._domain
        xmlSynset.append(SubElement(xmlSynset, "SENTIMENT"))
        xmlSynset.find("SENTIMENT").append(SubElement(xmlSynset.find("SENTIMENT"), "POSITIVE"))
        xmlSynset.find("SENTIMENT").find("POSITIVE").text = str(self._sentiment[0])
        xmlSynset.find("SENTIMENT").append(SubElement(xmlSynset.find("SENTIMENT"), "NEGATIVE"))
        xmlSynset.find("SENTIMENT").find("NEGATIVE").text = str(self._sentiment[1])
        xmlSynset.append(SubElement(xmlSynset, "SYNONYM"))
        for lem in self._lemmas:
            xmlSynset.find("SYNONYM").append(lem.xml())
        for rel in self._relwn5:
            for ID in self._relwn5[rel]:
                xmlSynset.append(SubElement(xmlSynset, "ILR"))
                xmlSynset.find("ILR").text = ID
                xmlSynset.find("ILR").append(SubElement(xmlSynset.find("ILR"), "TYPE"))
                xmlSynset.find("ILR").find("TYPE").text = rel
        for us in self._examples:
            xmlSynset.append(SubElement(xmlSynset, "USAGE"))
            xmlSynset.find("USAGE").text = us
        return xmlSynset 
    def __repr__(self):
        return "<SrbSynset ID:%s, lemmas:%s, definition:%s>" % (self._ID, ",".join(self._lemma_names), self._definition)
                
    def __str__(self):
        return "Synset: ID is %s, lemmas are %s, definition is %s" % (self._ID, self._lemma_names, self._definition)
    def __hash__(self):
        return hash(self._ID)
    def __eq__(self, other):
        if not isinstance(other, SrbSynset):
            return False
        return self._ID == other._ID
    
# =============================================================================
# Serbian Wornet Reader class
# =============================================================================


class SrbWordNetReader(XMLCorpusReader):
    """Reader for Serbina Wornet based on XML reader."""

    def __init__(self, root, fileids, wrap_etree=False):
        """
        Initilize Wordnet reader

        Parameters
        ----------
        root : String
            Path to the folder where file is located.
        fileids : String    
            Name of the file.
        wrap_etree : Boolean, optional
            If true, wrap the ElementTree object in an XMLCorpusView. The default is False.
        
        Returns
        ------
        None.

        """
        super().__init__(root, fileids, wrap_etree)
        self._path = fileids
#       Here we initilize  dictonary of synsets, since all synsets in
#       Serbain Wordet have unique id, we will be using that as a key
        self._slex =None
        self._synset_dict = dict()
        #this just quick refence table between names and ids. 
        #Reminded name of synset is the firt literal
        self._synset_name = dict()
#       Here we initilize  dictonary of synsets reashionspits, the key will be 
#       tuple (synset id, relashion symbol) while value would resultnty synset
#       key
        self._synset_rel = dict()

        self.load_synsets()
    def save_wordnet_as_xml_file(self, root, fileids):
        """
        Save wordnet as xml file

        Parameters
        ----------
        root : String
            Path to the folder where file will be saved.
        fileids : String
            Name of the file.

        Returns
        -------
        None.

        """
        xmlRoot = Element("WN")
        xmlRoot.append(SubElement(xmlRoot, "SYNSET"))
        for ID in self._synset_dict:
            xmlRoot.find("SYNSET").append(self._synset_dict[ID].xml())
        tree = ET.ElementTree(xmlRoot)
        tree.write(os.path.join(root, fileids))

    def synset_from_ID(self, ID):
        """
        Rerurn synster by its unique ID string from wordnet

        Parameters
        ----------
        ID : String
            Uniqe ID assigned to each synset

        Returns
        -------
        SrbSynset
            Synst with required ID.

        """
        if ID in self._synset_dict.keys():
            return self._synset_dict[ID]
        else:
            return None
    def synset_from_name(self, name):
        """
        Rerurns synstet by its name string from wordnet

        Parameters
        ----------
        name : String
            Name of the synset.

        Returns
        -------
        SrbSynset
            Synst with required name.

        """
        ID = self._synset_name[name]
        return self._synset_dict[ID]

    def synsets(self, word, POS=None):
        """
        Load all synsets with a given lemma and part of speech tag.

        If no pos is specified, all synsets for all parts of speech
        will be loaded.
        """
        ret = list()
        for id in self._synset_dict:
            syn = self._synset_dict[id]
            if (word in syn.lemma_names()):
                if POS is not None:
                    if POS!= syn.POS():
                        continue
                ret.append(syn)
        return ret
    def load_synsets(self):
        """
        Load all synsets from XML file
        """
        for i, syn in enumerate(self.xml()):
            try:
                pom = SrbSynset(syn, self)
                self._synset_dict[pom.ID()]= pom
                self._synset_name[pom.name()] = pom.ID()
            except Exception as err:
                print(err, i)
        self.rel_types = set()
        for key in self._synset_dict:
            self.rel_types.update(self._synset_dict[key].get_relations_types())
    def synset_from_pos_and_offset(self, pos, offset):
        ID = "ENG30-" + offset +"-" +pos 
        return self.synset_from_ID(ID)

    def parse_all_defintions (self, parser):
        """
        Deprecated function that should not be used.
        """        
        for id in self._synset_dict:
            syn = self._synset_dict[id]
            syn.parse_definition(parser)
            

    def get_relations_types(self):
        """
        Returns all types of relations in Wordnet.
        """
        return self.rel_types            
    def morphy(self, form, pos=None):
        """
        Return the base form of the given word form and part-of-speech tag using the 
        loaded lexicon.
        
        Parameters
        ----------
        form : str
            The word form to be analyzed.
        pos : str or None, optional
            The part-of-speech tag for the given word form. If None, the function will
            return all possible analyses for the word form across all available
            part-of-speech tags. Default is None.
        
        Returns
        -------
        str or None
            The base form of the given word form based on the morphological analysis,
            or None if no analyses are available.
        """
        if self._slex is None:
            return None
        if pos is None:
             analyses = chain.from_iterable(self._morphy(form, p) 
                                           for p in self._slex["POS"].unique())

        else:
            analyses = self._morphy(form, pos)
        first = list(islice(analyses, 1))
        if len(first) == 1:
            return first[0]
        else:
            return None
    def _morphy(self, form, pos):
        """
        Return the base form of the given word form and part-of-speech tag using the
        loaded lexicon.

        Parameters
        ----------
        form : str
            The word form to be analyzed.
        pos : str
            The part-of-speech tag for the given word form.

        Returns
        -------
        list of str
            The base form of the given word form based on the morphological analysis.

        """

        if self._slex is None:
            return []
        else:
            matches = self._slex.loc[(self._slex["POS"] == pos) & (self._slex["Term"] == form)]
            if len(matches) == 0:
                return []
            else:
                return [matches["Lemma"].iloc[0]]
        

    def sentiment_df(self):
        """
        Returns a Pandas DataFrame containing sentiment information for each synset in the WordNet corpus.
        The DataFrame includes the synset ID, positive and negative sentiment scores, lemma names, and definition.
        The sentiment information is extracted from the synset's _sentiment attribute, which contains a tuple of
        two float values representing the synset's positive and negative sentiment scores, respectively. The
        lemma names are obtained from the synset's _lemma_names attribute and are returned as a comma-separated
        string in the DataFrame. The definition of the synset is obtained using the definition() method of the
        synset object. The function returns the DataFrame.
        """        
        syns_list = list()
        for sifra in self._synset_dict:
            syn = self._synset_dict[sifra]
            el = dict()
            el["ID"] = sifra
            el["POS"], el["NEG"] = syn._sentiment
            el["Lemme"] = ",".join(syn._lemma_names)
            el["Definicija"] = syn.definition()
            syns_list.append(el)
        return  pd.DataFrame(syns_list)
        
    def __repr__(self):
        return "<SrbWordNetReader with %s synsets>" % (len(self._synset_dict))
    def __str__(self):
        return "SrbWordNetReader with %s synsets" % (len(self._synset_dict))
    
    def load_lexicon (self, path):
        """
        Load a lexicon from a file located at `path`.
        
        Parameters
        ----------
        path : str
            Path to the file containing the lexicon. The file is expected to be
            tab-separated or space-separated, with three columns named "Term",
            "POS", and "Lemma".
        
        Returns
        -------
        None.
        
        """

        self._slex = pd.read_csv(path, sep = "\t| ", 
                   on_bad_lines='skip', names=["Term", "POS", "Lemma"],
                   engine='python')
    def load_new_sentiment(self, cvs_file):
        """
        Load new sentiment from cvs file

        Parameters
        ----------
        cvs_file : String
            Path to the file containing the sentiment. The file is expected to be
            tab-separated or space-separated, with three columns named "ID",
            "POS", and "NEG". I can have other colamnt thaty will be ignored.

        Returns
        -------
        None.

        """
        try:
            sentiment = pd.read_csv(cvs_file, sep = "\t| ", 
                    on_bad_lines='skip', names=["ID", "POS", "NEG"],
                    engine='python')        
            for index, row in sentiment.iterrows():
                self._synset_dict[row["ID"]]._sentiment = (row["POS"], row["NEG"])
        except Exception as err:
            print(err)
            print("Error in file", cvs_file)
            print("File should be tab separated with columns ID, POS, NEG")
            print("ID is synset ID")
            print("POS is positive sentiment")
            print("NEG is negative sentiment")
            print("Other columns will be ignored")

class SrbWordNetReaderUserInterface():
    """"
    Class that provides user interface for Serbian Wordnet
    """
    def __init__(self):
        self._swn = None
    def load_wordnet(self):
        """
        Load wordnet from xml file

        Returns
        -------
        None.

        """
        try:
            path = input("Enter path to the folder where file is located:") 
            file = input("Enter name of the file:") 
            self._swn = SrbWordNetReader(path, file)
        except Exception as err:
            print(err)
            print("Error in file", file)
            print("File should be xml file with Serbian Wordnet")
    def load_lexicon(self):
        """
        Load lexicon from file

                Returns
        -------
        None.

        """
        if self._swn is None:
            print("Wordnet not loaded")
            return
        try:
            path= input("Enter path to the file:")
            self._swn.load_lexicon(path)
        except Exception as err:
            print(err)
            print("Error in file", file)
            print("File should be csv file with Serbian lexicon")
    
    def load_new_sentiment(self):
        """
        Load new sentiment from cvs file

        Returns
        -------
        None.

        """
        if self._swn is None:
            print("Wordnet not loaded")
            return
        try:
            path= input("Enter path to the file:")
            self._swn.load_new_sentiment(path)
        except Exception as err:
            print(err)
            print("Error in file", file)
            print("File should be csv file with new sentiment")
            print("File should be tab separated with columns ID, POS, NEG")
            print("ID is synset ID")
            print("POS is positive sentiment")
            print("NEG is negative sentiment")
            print("Other columns will be ignored")

    def save_wordnet(self):
        """
        Save wordnet as xml file

        Returns

        -------
        None.

        """
        if self._swn is None:
            print("Wordnet not loaded")
            return
        path = input("Enter path to the folder where file will be saved:") 
        file = input("Enter name of the file:") 
        try:
            self._swn.save_wordnet_as_xml_file(path, file)
        except Exception as err:
            print(err)
            print("Error in file", file)
    def sentiment_analyze(self):
        """
        Create table with ID, sentiment, defintions, lemmas, and part of speech for further analysis
        of all synsets in SrbWordNet and saves the result in a csv file 

        Returns
        -------
        None.

        """
        path = input("Enter path to the folder where file will be saved:") 
        file = input("Enter name of the file:") 
        if self._swn is None:
            print("Wordnet not loaded")
            return
        
        df = self._swn.sentiment_df()
        df.to_csv(os.path.join(path, file))
    def get_synsets(self):
        """
        Asks user for word, and optionaly part of speech
        If there is lexicon loaded, it will try to find base form of word
        Note that part of speech is optinal, user may nor provide it
        Word is not, and should be checked before call
        Get all synsets from Wordnet that contain that word
        Print all synsets with their ID, definition, 
        id of related synsetes by type of relation
        lemmas and sentiment       
        Returns
        -------
        None.

        """
        if self._swn is None:
            print("Wordnet not loaded")
            return
        word = input("Enter word:")
        pos = input("Enter part of speech (optional):")
        #check if pos is valid
        if pos not in POS_LIST:
            pos = None
        if self._swn._slex is not None:
            word = self._swn.morphy(word, pos)
        if word is None:
            print("Word not found")
            return
        synsets = self._swn.synsets(word, pos)
        if len(synsets) == 0:
            print("No synsets found")
            return
        for syn in synsets:
            print(f"Synset: {syn.ID()}, Definition {syn.definition()}, {syn._sentiment}")
            print("Lemmas:", syn.lemma_names())
            print("Relations:")
            for rel in syn._relwn5:
                print(rel, syn._relwn5[rel])
            print("--------------------------------------------------")

    def get_synset_by_ID(self):
        """
        Asks user for ID of synset
        Get synset from Wordnet with that ID
        Print synset with its ID, definition, lemmas
        id f relation by type  and sentiment       
        Returns
        -------
        None.

        """
        if self._swn is None:
            print("Wordnet not loaded")
            return
        ID = input("Enter ID:")
        synset = self._swn.synset_from_ID(ID)
        if synset is None:
            print("Synset not found")
            return
        print(f"Synset: {synset.ID()}, Definition {synset.definition()}, {synset._sentiment}")
        print("Lemmas:", synset.lemma_names())
        print("Relations:")
        for rel in synset._relwn5:
            print(rel, synset._relwn5[rel])
        print("--------------------------------------------------")
    def get_synset_by_name(self):
        """
        Asks user for name of synset
        Get synset from Wordnet with that name
        Print synset with its ID, definition, lemmas
        id f relation by type  and sentiment       
        Returns
        -------
        None.

        """
        if self._swn is None:
            print("Wordnet not loaded")
            return
        name = input("Enter name:")
        synset = self._swn.synset_from_name(name)
        if synset is None:
            print("Synset not found")
            return
        print(f"Synset: {synset.ID()}, Definition {synset.definition()}, {synset._sentiment}")
        print("Lemmas:", synset.lemma_names())
        print("Relations:")
        for rel in synset._relwn5:
            print(rel, synset._relwn5[rel])
        print("--------------------------------------------------")
    def termiante(self):
        """
        Terminate program

        Returns
        -------
        None.

        """
        print("Terminating")
        exit(0)
    def print_menu(self):
        """
        Print menu for user
        
        Returns
        ------- 
        None.
        """
        print("1. Load Wordnet")
        print("2. Load lexicon")
        print("3. Load new sentiment")
        print("4. Save Wordnet")
        print("5. Sentiment analyze")
        print("6. Get synsets")
        print("7. Get synset by ID")
        print("8. Get synset by name")
        print("9. Terminate")
    def run(self):
        """
        Run user interface

        Returns 
        -------
        None.
        """
        while True:
            self.print_menu()
            #print wordent filename and path if esist or imform that it is not loaded
            if self._swn is None:
                print("Wordnet not loaded")
            else:
                print("Wordnet loaded from file", self._swn._path)
            #print lexicon filename and path if esist or imform that it is not loaded
            if self._swn is None:
                print("Lexicon not loaded")
            else:
                print("Lexicon loaded from file", self._swn._slex)
            try:
                option = int(input("Enter option:"))
            except Exception as err:
                print(err)
                print("Invalid option")
                continue
            if option == 1:
                self.load_wordnet()
            elif option == 2:
                self.load_lexicon()
            elif option == 3:
                self.load_new_sentiment()
            elif option == 4:
                self.save_wordnet()
            elif option == 5:
                self.sentiment_analyze()
            elif option == 6:
                self.get_synsets()
            elif option == 7:
                self.get_synset_by_ID()
            elif option == 8:
                self.get_synset_by_name()
            elif option == 9:
                self.termiante()
            else:
                print("Invalid option")
                continue
            print("--------------------------------------------------")
            print("--------------------------------------------------")
            print("--------------------------------------------------")
        
def main():
    """
    Main function
    """
    ui = SrbWordNetReaderUserInterface()
    ui.run()
if __name__ == "__main__":
    main()





