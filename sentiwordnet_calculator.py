import os
from transformers import pipeline
from srpskiwordnet import SrbWordNetReader
import pandas as pd

# Define folder and file paths
ROOT_DIR = ""
RES_DIR = os.path.join(ROOT_DIR, "resources")
MOD_DIR = os.path.join(ROOT_DIR, "ml_models")


class SentimentPipeline:
    """
    This class defines a custom sentiment analysis pipeline using Hugging Face's Transformers.
    
    The pipeline uses two separate models for predicting positive/non-positive and 
    negative/non-negative sentiment respectively.

    Inputs:
        Single text string or a list of text strings for sentiment analysis.

    Returns:
        If a single text string is provided, a single dictionary is returned with POS, NEG, and OBJ scores.
        If a list of text strings is provided, a list of dictionaries is returned with each dictionary 
        representing POS, NEG, and OBJ scores for the corresponding text.

    Usage:
        sentiment_pipeline = SentimentPipeline(YOUR_POS_MODEL, YOUR_NEG_MODEL)
        result = sentiment_pipeline("Your glossed text here")
        results = sentiment_pipeline(["Your first glossed text here", "Your second glossed text here"])
    """

    def __init__(self, model_path_positive, model_path_negative):
        """
        Constructor for the SentimentPipeline class.
        Initializes two pipelines using Hugging Face's Transformers, one for positive and one for negative sentiment.
        """
        self.pos_pipeline = pipeline('text-classification', model=model_path_positive)
        self.neg_pipeline = pipeline('text-classification', model=model_path_negative)

    def __call__(self, texts):
        """
        Callable method for the SentimentPipeline class. Processes the given text(s) and returns sentiment scores.
        """
        
        # Check if input is a single string. If it is, convert it into a list.
        if isinstance(texts, str):
            texts = [texts]
        if texts is None:
            texts = [""]
        results = []
        # Run the text through the pipelines
        pos_results = self.pos_pipeline(texts)
        neg_results = self.neg_pipeline(texts)
        for pos_result, neg_result in zip(pos_results, neg_results):
            # Calculate probabilities for positive/non-positive and negative/non-negative.
            # If the label is POSITIVE/NEGATIVE, the score for positive/negative is the score returned by the model, 
            # and the score for non-positive/non-negative is 1 - the score returned by the model.
            # If the label is NON-POSITIVE/NON-NEGATIVE, the score for non-positive/non-negative is the score returned by the model,
            # and the score for positive/negative is 1 - the score returned by the model.
            Pt, Pn = (pos_result['score'], 1 - pos_result['score']) if pos_result['label'] == 'POSITIVE' else (1 - pos_result['score'], pos_result['score'])
            Nt, Nn = (neg_result['score'], 1 - neg_result['score']) if neg_result['label'] == 'NEGATIVE' else (1 - neg_result['score'], neg_result['score'])

            # Calculate POS, NEG, OBJ scores using the formulas provided
            POS = Pt * Nn
            NEG = Nt * Pn
            OBJ = 1 - POS - NEG

            # Append the scores to the results
            results.append({"POS": POS, "NEG": NEG, "OBJ": OBJ})

        # If the input was a single string, return a single dictionary. Otherwise, return a list of dictionaries.
        return results if len(results) > 1 else results[0]
class SentimentPipelineAvg:
    """
    This class defines a custom sentiment analysis pipeline using Hugging Face's Transformers.  
    The pipeline uses two separate models for predicting positive/non-positive and
    negative/non-negative sentiment respectively.
    Inputs:
        Single text string or a list of text strings for sentiment analysis.
    Returns:
        If a single text string is provided, a single dictionary is returned with POS, NEG, and OBJ scores.
        If a list of text strings is provided, a list of dictionaries is returned with each dictionary 
        representing POS, NEG, and OBJ scores for the corresponding text.

    """

    def __init__(self):
        """
        Constructor for the SentimentPipelineAvg class.
        Initializes a list of SentimentPipeline objects.
        """
        self.pipelines = [] #list of SentimentPipeline objects
    def add(self, model_path_positive, model_path_negative):
        """
        Method for the SentimentPipelineAvg class.
        Initializes a SentimentPipeline object and adds it to the list.
        """
        self.pipelines.append(SentimentPipeline(model_path_positive, model_path_negative))
    def addAll(self, model_paths):
        """
        Method for the SentimentPipelineAvg class.
        Initializes a list of SentimentPipeline objects and adds them to the list.
        """
        for model_path in model_paths:

            self.add(model_path[0], model_path[1])
    def __call__(self, texts):
        """
        Callable method for the SentimentPipelineAvg class. Processes the given text(s) and returns sentiment scores.
        """
        # Check if input is a single string. If it is, convert it into a list.
        if isinstance(texts, str):
            texts = [texts]
        if texts is None:
            texts = [""]
        results = []
        infered = []
        for pipe in self.pipelines:
            #run text trough pipelines
            infered.append(pipe(texts))
        for i in range(len(texts)):
            POS = sum([inf[i]["POS"] for inf in infered]) / len(self.pipelines)
            NEG = sum([inf[i]["NEG"] for inf in infered]) / len(self.pipelines)
            OBJ = sum([inf[i]["OBJ"] for inf in infered]) / len(self.pipelines)
            results.append({"POS": POS, "NEG": NEG, "OBJ": OBJ})



        # If the input was a single string, return a single dictionary. Otherwise, return a list of dictionaries.
        return results if len(results) > 1 else results[0]  

def get_definions_dataframe():
    """
    Function that returns a dataframe with sentiment analysis of all synsets in SrbWordNet
    :param swn: SrbWordNetReader object 
    :return: dataframe with sentiment analysis of all synsets in SrbWordNet
    """  
    swn = SrbWordNetReader(RES_DIR, "wnsrp30.xml");
    syns_list = list()
    for sifra in swn._synset_dict:
        syn = swn._synset_dict[sifra]
        el = dict()
        el["ID"] = sifra
        el["Lemme"] = ",".join(syn._lemma_names)
        el["Vrsta"] = syn.POS()
        el["Definicija"] = syn.definition()
        syns_list.append(el)
    return pd.DataFrame(syns_list)
def get_pipe_paths(base):
    """
    Function that returns a list of tuples with paths to positive and negative models
    :param base: base name of the models
    :return: list of tuples with paths to positive and negative models
    """

    pipies_names=[]
    DATASET_ITERATIONS = [0, 2, 4, 6]
    for i in DATASET_ITERATIONS:
        pipies_names.append((f"Tanor/{base}SENTPOS{i}", f"Tanor/{base}SENTNEG{i}"))
    return pipies_names
def inferLLM (base, output_file_number):
    """
    Function that infers sentiment of all synsets in SrbWordNet and saves it to a csv file
    :param base: base name of the models
    :param output_file: path to the output csv file
    """
    df = get_definions_dataframe()
    df["Definicija"].fillna("", inplace=True)
    df["Definicija"] = df["Definicija"].astype(str)
    pipe_paths = get_pipe_paths(base)
    pipe = SentimentPipelineAvg()
    pipe.addAll(pipe_paths)
    inf = pipe(df["Definicija"].tolist())
    dr_inf = pd.DataFrame(inf)
    df = pd.concat([df, dr_inf], axis=1)
    filename = f"srbsentiwordnet_a{output_file_number}.csv"
    #,ID,POS,NEG,Lemme,Definicija,Vrsta
    df = df[["ID","POS","NEG","Lemme","Definicija","Vrsta"]]
    df.to_csv(os.path.join(RES_DIR, filename), index=False)
    filename = f"srbsentiwordnet{output_file_number}.csv"
    #drop column Vrsta
    df = df.drop(columns=["Vrsta"])
    df.to_csv(os.path.join(RES_DIR, filename), index=False)

def main():
    print("Starting...")
    # print("Infering BERTic...")
    # inferLLM("BERTic", 5)
    # print("Setting model repo to public...")
    # set_model_repo_to_public()
    # print("Infering SRGPT...")
    # inferLLM("SRGPT", 6)
    print("Infering Jerteh355...")
    inferLLM("Jerteh355", 7)
    set_model_repo_to_public("Jerteh355")
    print("Done!")
def set_model_repo_to_public(basemodel):
    """
    Function that changes existing model repository from private to public
    :param name: name of the model repository
    """
    from huggingface_hub import update_repo_visibility
    ids = get_pipe_paths(basemodel)
    for repo_id in ids:
        repo_id_pos, repo_id_neg = repo_id
        update_repo_visibility(repo_id=repo_id_pos, private=False)
        update_repo_visibility(repo_id=repo_id_neg, private=False)




if __name__ == "__main__":
    main()