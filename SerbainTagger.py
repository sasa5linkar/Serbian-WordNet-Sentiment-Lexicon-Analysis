# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 14:15:57 2022

@author: "Petalinkar Saša"
"""

import treetaggerwrapper as ttpw

# path to the Serbian parameter file for TreeTagger
TTPARPATH = "C:\\TreeTagger\\bin\\TreeTagger_pos.par"

class SrbTreeTagger():
    """
    A wrapper for TreeTagger with the Serbian parameter file.
    """
    def __init__(self, ):
        self._tagger = ttpw.TreeTagger(TAGPARFILE=TTPARPATH)

    def lemmarizer (self, text):
        """
        Replaces all words in a string with their lemmas using TreeTagger.

        Parameters
        ----------
        text : str
            The string to lemmatize.

        Returns
        -------
        str
            The lemmatized string.
        """
        if text is None:
            return None
        tags = self._tagger.tag_text(text)
        tags2 = ttpw.make_tags(tags)
        ret = ""
        try:
            for tag in tags2:
                if tag.__class__.__name__ == "Tag":
                    ret += tag[2] + " "
        except IndexError:
            print(tags2)
        return ret
