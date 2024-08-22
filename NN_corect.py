# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:48:39 2023

@author: Korisnik
"""

import pandas as pd 
from srpskiwordnet import SrbWordNetReader

RES_DIR = ".\\resources\\"

it = ["0", "2" ,"4", "6"]
sten = dict()
for i in it:
    sten[i] = pd.read_csv(RES_DIR +"sentiment_RNN" + i + ".csv", index_col=0)

swordnet = SrbWordNetReader(RES_DIR,"wnsrp30.xml")

tablicaSentimenta =  pd.DataFrame()
tablicaSentimenta["ID"] = sten["0"]["ID"]
syns_list = list()
for sifra in tablicaSentimenta["ID"]:
    syn = swordnet._synset_dict[sifra]
    el = dict()
    el["ID"] = sifra
    el["Lemme"] = ",".join(syn._lemma_names)
    el["Definicija"] = syn.definition()
    el["Vrsta"] = syn.POS() 
    syns_list.append(el)
sword = pd.DataFrame(syns_list)

tablicaSentimenta["POS"] = (sten["0"]["POS"]+sten["2"]["POS"]+sten["4"]["POS"]
                            +sten["6"]["POS"])/4 
tablicaSentimenta["NEG"] = (sten["0"]["NEG"]+sten["2"]["NEG"]+sten["4"]["NEG"]
                            +sten["6"]["NEG"])/4 
round_par= {"POS": 3, "NEG" : 3}

tablicaSentimenta = tablicaSentimenta.round(round_par)

print((tablicaSentimenta["ID"]==sword["ID"]).all())
sword["POS"], sword["NEG"] = tablicaSentimenta["POS"], tablicaSentimenta["NEG"]

sword.to_csv(RES_DIR + "srbsentiwordnet4.cvs", columns = ["ID","POS","NEG","Lemme","Definicija"] )
sword.to_csv(RES_DIR + "srbsentiwordnet_a4.cvs", columns = ["ID","POS","NEG","Lemme","Definicija", "Vrsta"] )