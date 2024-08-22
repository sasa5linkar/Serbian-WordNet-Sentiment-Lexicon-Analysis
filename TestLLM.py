import torch
from transformers import pipeline
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = ""
RES_DIR = os.path.join(ROOT_DIR, "resources")
MOD_DIR = os.path.join(ROOT_DIR, "ml_models")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_sets")
REP_DIR = os.path.join(ROOT_DIR, "reports", "CompareLLM")

def print_correctly_classified_instances(i, polarity, model ="BERTic"):
    # Empty GPU cache before testing model
    torch.cuda.empty_cache()

    # Construct file name
    name = f"UP{polarity}{i}.csv"
    
    # Load the test data
    X_test = pd.read_csv(os.path.join(TRAIN_DIR, f"X_test_{name}"))["Sysnet"]
    y_test = pd.read_csv(os.path.join(TRAIN_DIR, f"y_test_{name}"))[polarity]
    X_test = X_test.fillna("")        
    # Construct model name
    if (model=="BERTic"):
        model_name = f"Tanor/BERTicSENT{polarity}{i}"
    if (model=="BERTicovo"):
        model_name = f"Tanor/BERTicovoSENT{polarity}{i}"
    if (model=="SRGPT"):
        model_name = f"Tanor/SRGPTSENT{polarity}{i}"
    
    # Load model using pipeline
    pipe = pipeline("text-classification", model=model_name)
    
    # Define label to id mapping
    label2id = {"NON-POSITIVE": 0, "POSITIVE": 1}
    if (polarity =="NEG"):
        label2id = {"NON-NEGATIVE": 0, "NEGATIVE": 1}
    
    # Process test data through pipeline
    data = pipe(X_test.to_list())
    
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Convert the 'label' column into a series where 'NON-POSITIVE' is 0 and 'POSITIVE' is 1
    df['label'] = df['label'].map(label2id)
    
    # Convert 'label' column into a series
    series = df['label']
    predicted_classes = series.values
    
    # Create a DataFrame with test data, predicted classes and real classes
    table = pd.DataFrame({"X": X_test, "Predicted": predicted_classes, 
                          "Real": y_test})
    
    # Create a table of instances where the predicted class is 1 and the real class is also 1
    correct_class_1 = table[(table["Predicted"] == 1) & (table["Real"] == 1)]
    
    # Print the instances where both the predicted class and real class are 1
    print(correct_class_1["X"])
    
    # Delete the pipeline to free up memory
    del pipe
    torch.cuda.empty_cache()
    
def compare_models(i, polarity, model1, model2, compare_misclassified=False, compare_other_class=False):
    """
    Compare the performance of two models on a test dataset based on specified criteria.

    Parameters:
    - i (int): An index used to construct the file name for loading test data.
    - polarity (str): The sentiment polarity ("POS" or "NEG") used to construct the file name and model name.
    - model1 (str): The name of the first model to compare.
    - model2 (str): The name of the second model to compare.
    - compare_misclassified (bool, optional): If True, the function will focus on instances that 
                                              are misclassified by the models. Default is False.
    - compare_other_class (bool, optional): If True, the function will focus on the opposite class 
                                            (e.g., non-positive for "POS" polarity). Default is False.

    Outputs:
    - Prints the texts that match the specified criteria for both models, for the first model only, 
      and for the second model only.
      
    Note:
    This function relies on the 'run_model' and 'create_comparison_logic' functions to obtain model results 
    and construct the comparison logic, respectively.
    """
    # Construct file name
    name = f"UP{polarity}{i}.csv"
    
    # Load the test data
    X_test = pd.read_csv(os.path.join(TRAIN_DIR, f"X_test_{name}"))["Sysnet"].fillna("")
    y_test = pd.read_csv(os.path.join(TRAIN_DIR, f"y_test_{name}"))[polarity]
        
    # Define label to id mapping
    label2id = {"NON-POSITIVE": 0, "POSITIVE": 1} if polarity != "NEG" else {"NON-NEGATIVE": 0, "NEGATIVE": 1}

    # Construct model names
    model_name1 = f"Tanor/{model1}SENT{polarity}{i}"
    model_name2 = f"Tanor/{model2}SENT{polarity}{i}"
    
    # Create dataframes for each model to store the results
    results_model1 = run_model(X_test, y_test, model_name1, label2id)
    results_model2 = run_model(X_test, y_test, model_name2, label2id)

    # Flags Logic for results_model1
    comparison_logic_1, label = create_comparison_logic(results_model1, compare_misclassified, compare_other_class)

    # Flags Logic for results_model2
    comparison_logic_2, _ = create_comparison_logic(results_model2, compare_misclassified, compare_other_class)

    results_model1_filtered = results_model1[comparison_logic_1]
    results_model2_filtered = results_model2[comparison_logic_2]

    # Find texts based on comparison logic for both models
    matched_both = pd.merge(results_model1_filtered, results_model2_filtered, how='inner', on=['X'])
    # Find texts based on comparison logic for the first model but not the second
    matched_model1_only = results_model1_filtered[~results_model1_filtered.X.isin(results_model2_filtered.X)]
    # Find texts based on comparison logic for the second model but not the first
    matched_model2_only = results_model2_filtered[~results_model2_filtered.X.isin(results_model1_filtered.X)]
    
    print(f"{label} by both models:\n", matched_both["X"])
    print(f"{label} by the first model but not the second:\n", matched_model1_only["X"])
    print(f"{label} by the second model but not the first:\n", matched_model2_only["X"])

    
def run_model(X_test, y_test, model_name, label2id):
    """
    Run a model on a test dataset and return a DataFrame with the test data, 
    predicted classes, and real classes.

    Parameters:
    - X_test (pd.Series): The test data.
    - y_test (pd.Series): The real classes for the test data.
    - model_name (str): The name of the model to run.
    - label2id (dict): A dictionary mapping label names to integer IDs.

    Returns:
    - pd.DataFrame: A DataFrame with columns 'X' (test data), 'Predicted' (predicted classes), 
                    and 'Real' (real classes).
    """
    # Empty GPU cache before testing model
    torch.cuda.empty_cache()
    
    # Load model using pipeline
    pipe = pipeline("text-classification", model=model_name)
    
    # Process test data through pipeline
    data = pipe(X_test.to_list())
    
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Convert the 'label' column into a series where 'NON-POSITIVE' is 0 and 'POSITIVE' is 1
    df['label'] = df['label'].map(label2id)
    
    # Create a DataFrame with test data, predicted classes, and real classes
    table = pd.DataFrame({"X": X_test, "Predicted": df['label'], 
                          "Real": y_test})
    
    # Delete the pipeline to free up memory
    del pipe
    torch.cuda.empty_cache()

    return table

    
def test_model(i, polarity, model ="BERTic"):
    """
    Test a specified model on a given test dataset and print the confusion matrix and classification report.

    Parameters:
    - i (int): An index used to construct the file name for loading test data.
    - polarity (str): The sentiment polarity ("POS" or "NEG") used to construct the file name and model name.
    - model (str, optional): The name of the model to test. Default is "BERTic". 
                             Options are "BERTic", "BERTicovo", and "SRGPT".

    Outputs:
    - Prints the confusion matrix and classification report for the tested model on the provided test data.

    Notes:
    - The function assumes that the test data is stored in csv files with specific naming conventions.
    - It uses the HuggingFace's transformers pipeline to load and test the model.
    """
    # Empty GPU cache before testing model
    torch.cuda.empty_cache()

    # Construct file name
    name = f"UP{polarity}{i}.csv"
    
    # Load the test data
    X_test = pd.read_csv(os.path.join(TRAIN_DIR, f"X_test_{name}"))["Sysnet"]
    y_test = pd.read_csv(os.path.join(TRAIN_DIR, f"y_test_{name}"))[polarity]
    X_test = X_test.fillna("")    
    # Construct model name
    if (model=="BERTic"):
        model_name = f"Tanor/BERTicSENT{polarity}{i}"
    if (model=="BERTicovo"):
        model_name = f"Tanor/BERTicovoSENT{polarity}{i}"
    if (model=="SRGPT"):
        model_name = f"Tanor/SRGPTSENT{polarity}{i}"    
    # Load model using pipeline
    pipe = pipeline("text-classification", model=model_name)
    
    # Define label to id mapping
    label2id = {"NON-POSITIVE": 0, "POSITIVE": 1}
    if (polarity =="NEG"):
        label2id = {"NON-NEGATIVE": 0, "NEGATIVE": 1}
    
    # Process test data through pipeline
    data = pipe(X_test.to_list())
    
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Convert the 'label' column into a series where 'NON-POSITIVE' is 0 and 'POSITIVE' is 1
    df['label'] = df['label'].map(label2id)
    
    # Convert 'label' column into a series
    series = df['label']
    predicted_classes = series.values
    
    # Compute confusion matrix
    y_test_np = y_test.values
    confusion_mat = confusion_matrix(y_test_np, predicted_classes)
    
    print(confusion_mat)
    classification_rep = classification_report(y_test_np, predicted_classes)
    
    print(classification_rep)
    
    del pipe
    torch.cuda.empty_cache()
def create_comparison_logic(df, compare_misclassified, compare_other_class):
    """
    Create a boolean mask for a DataFrame based on the given comparison criteria.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame for which the mask is being generated. 
                         It should have columns 'Predicted' and 'Real' indicating 
                         the predicted and actual classes, respectively.
    - compare_misclassified (bool): If True, the mask will select instances that 
                                    are misclassified. If False, it will select 
                                    instances that are correctly classified.
    - compare_other_class (bool): If True, the mask will focus on class 0 (e.g., non-positive). 
                                  If False, it will focus on class 1 (e.g., positive).
    
    Returns:
    - tuple: A tuple containing the boolean mask and a descriptive label.
    """
    
    if not compare_misclassified and not compare_other_class:
        # Correctly classified as class 1
        comparison_logic = (df["Predicted"] == 1) & (df["Real"] == 1)
        label = "Texts correctly classified as class 1"
    elif compare_misclassified and not compare_other_class:
        # Misclassified for class 1
        comparison_logic = (df["Predicted"] != df["Real"]) & (df["Real"] == 1)
        label = "Texts misclassified for class 1"
    elif not compare_misclassified and compare_other_class:
        # Correctly classified as class 0
        comparison_logic = (df["Predicted"] == 0) & (df["Real"] == 0)
        label = "Texts correctly classified as class 0"
    else:
        # Misclassified for class 0
        comparison_logic = (df["Predicted"] != df["Real"]) & (df["Real"] == 0)
        label = "Texts misclassified for class 0"
    
    return comparison_logic, label



def plot_score_distribution(df, score_col, true_col, pred_col, title="Score Distribution"):
    """
    Plot the distribution of model scores.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - score_col (str): Name of the column containing the model scores.
    - true_col (str): Name of the column containing the true labels.
    - pred_col (str): Name of the column containing the predicted labels.
    - title (str): Title for the plot.
    """
    # Correctly Classified
    correct = df[df[true_col] == df[pred_col]][score_col]
    # Misclassified
    misclassified = df[df[true_col] != df[pred_col]][score_col]

    # Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(correct, shade=True, label="Correctly Classified", clip=(0,1))
    sns.kdeplot(misclassified, shade=True, label="Misclassified", clip=(0,1))
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

# Usage:
# Assuming 'results' is your dataframe with columns "Score", "Real" and "Predicted"
# plot_score_distribution(results, "Score", "Real", "Predicted")

def run_model_with_scores(X_test, y_test, model_name, label2id):
    """
    Run a model on a test dataset and return a DataFrame with the test data, 
    predicted classes, real classes, and scores indicating the model's confidence.

    Parameters:
    - X_test (pd.Series): The test data.
    - y_test (pd.Series): The real classes for the test data.
    - model_name (str): The name of the model to run.
    - label2id (dict): A dictionary mapping label names to integer IDs.

    Returns:
    - pd.DataFrame: A DataFrame with columns 'X' (test data), 'Predicted' (predicted classes), 
                    'Real' (real classes), and 'Score' (model's confidence).
    """
    # Empty GPU cache before testing model
    torch.cuda.empty_cache()
    
    # Load model using pipeline
    pipe = pipeline("text-classification", model=model_name)
    
    # Process test data through pipeline
    data = pipe(X_test.to_list())
    
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Convert the 'label' column into a series based on label2id mapping
    df['label'] = df['label'].map(label2id)
    
    # Adjust the score for label 0 if necessary
    df['score'] = df.apply(lambda row: row['score'] if row['label'] == 1 else 1 - row['score'], axis=1)
    
    # Create a DataFrame with test data, predicted classes, real classes, and scores
    table = pd.DataFrame({"X": X_test, "Predicted": df['label'], 
                          "Real": y_test, "Score": df['score']})
    
    # Delete the pipeline to free up memory
    del pipe
    torch.cuda.empty_cache()

    return table
def plot_model_distribution(i, polarity, model="BERTic"):
    """
    Plot the distribution of a model's confidence scores on a given test dataset,
    distinguishing between correctly classified and misclassified samples.

    Parameters:
    - i (int): An index used to construct the file name for loading test data.
    - polarity (str): The sentiment polarity ("POS" or "NEG") used to construct the file name and model name.
    - model (str, optional): The name of the model to test. Default is "BERTic". 
                             Options are "BERTic", "BERTicovo", and "SRGPT".

    Outputs:
    - A plot showing the distribution of the model's confidence scores.
    """
    
    # Empty GPU cache before testing model
    torch.cuda.empty_cache()

    # Construct file name
    name = f"UP{polarity}{i}.csv"
    
    # Load the test data
    X_test = pd.read_csv(os.path.join(TRAIN_DIR, f"X_test_{name}"))["Sysnet"].fillna("")
    y_test = pd.read_csv(os.path.join(TRAIN_DIR, f"y_test_{name}"))[polarity]
    
    # Define label to id mapping
    label2id = {"NON-POSITIVE": 0, "POSITIVE": 1} if polarity != "NEG" else {"NON-NEGATIVE": 0, "NEGATIVE": 1}

    # Construct model name
    if model == "BERTic":
        model_name = f"Tanor/BERTicSENT{polarity}{i}"
    elif model == "BERTicovo":
        model_name = f"Tanor/BERTicovoSENT{polarity}{i}"
    elif model == "SRGPT":
        model_name = f"Tanor/SRGPTSENT{polarity}{i}"

    # Get results table using run_model_with_scores function
    results = run_model_with_scores(X_test, y_test, model_name, label2id)

    # Use the plot_score_distribution function
    plot_score_distribution(df=results, 
                            score_col="Score", 
                            true_col="Real", 
                            pred_col="Predicted", 
                            title=f"Distribution of {model}'s Confidence Scores for {polarity} polarity")
