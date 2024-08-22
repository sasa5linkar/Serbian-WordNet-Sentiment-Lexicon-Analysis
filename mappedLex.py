import pandas as pd
import os    
from srpskiwordnet import SrbWordNetReader

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
        el["POS"], el["NEG"] = syn._sentiment
        syns_list.append(el)

    return pd.DataFrame(syns_list)

ROOT_DIR = ""
RES_DIR = os.path.join(ROOT_DIR, "resources")
def main():
    swn = SrbWordNetReader(RES_DIR, "wnsrp30.xml")
    df = sentiment_analyze_df(swn)
    df.to_csv(os.path.join(RES_DIR, "swn30_sentiment.csv"), index=False)
if __name__ == "__main__":
    main() 
