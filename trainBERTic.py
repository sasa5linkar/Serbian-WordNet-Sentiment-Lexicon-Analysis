# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:38:41 2023

@author: sasa5
"""
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import time
from datasets import Dataset
from transformers import DataCollatorWithPadding
import evaluate
from transformers import  TrainingArguments, Trainer, pipeline, EarlyStoppingCallback
from tqdm import tqdm
import torch


torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.8, 0)
ROOT_DIR = ""
RES_DIR = os.path.join(ROOT_DIR, "resources")
MOD_DIR = os.path.join(ROOT_DIR, "ml_models")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_sets")
REP_DIR = os.path.join(ROOT_DIR, "reports", "BERTic2")
maxlen = 300
# Create directory if not exists
if not os.path.exists(REP_DIR):
    os.makedirs(REP_DIR)
def train_model (i, polarity, eval = "accuracy", epochs=16):
    def preprocess_function(examples):
        return tokenizer(examples["text"], max_length=maxlen,truncation=True, padding=True)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    if eval == "f1":
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return f1_score.compute(predictions=predictions, references=labels)
 
    torch.cuda.empty_cache()
    BATCH_SIZE = 64
    
    # Construct file name
    name = f"UP{polarity}{i}.csv"
    
    # Read the training data from the CSV files
    X = pd.read_csv(os.path.join(TRAIN_DIR, f"X_train_{name}"))["Sysnet"]
    y = pd.read_csv(os.path.join(TRAIN_DIR, f"y_train_{name}"))[polarity]
    
   
    # Renaming is necessary to avoid errors in subsequent operations
    X.rename("text", inplace=True)
    X = X.fillna("")
    y.rename("labels", inplace=True)
    
    # Split the dataset into training and validation subsets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, 
                                                      stratify=y, random_state=42)

    # Define label mappings based on polarity
    id2label = {0: "NON-POSITIVE", 1: "POSITIVE"}
    label2id = {"NON-POSITIVE": 0, "POSITIVE": 1}
    if (polarity =="NEG"):
        id2label = {0: "NON-NEGATIVE", 1: "NEGATIVE"}
        label2id = {"NON-NEGATIVE": 0, "NEGATIVE": 1}
    
    # Construct the model name
    model_name = f"Tanor/BERTicSENT{polarity}{i}"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,  id2label=id2label, 
        label2id=label2id, )

    # Construct datasets for validation and training
    dataset_val = Dataset.from_pandas(pd.concat([X_val, y_val], axis=1))
    dataset_train = Dataset.from_pandas(pd.concat([X_train, y_train], axis=1))
    
    # Tokenize datasets
    tokenised_val=dataset_val.map(preprocess_function)
    tokenised_train =dataset_train.map(preprocess_function)
    
    # Define output directory
    outputdir = f"BERTicSENT{polarity}{i}"
    
    # Set up data collator, accuracy metric, and training arguments
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy = evaluate.load("accuracy")
    f1_score = evaluate.load("f1")
    training_args = TrainingArguments(
        output_dir=outputdir,
        overwrite_output_dir = True,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4, 
        gradient_checkpointing=True,
        optim="adafactor",
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_train,
        eval_dataset=tokenised_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)], # wait for '3' evaluation steps without improvement.

    )
    
    # Train model and push to hub
    trainer.train()
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Try to push the model to the hub
            trainer.push_to_hub()
    
            # If the push is successful, exit the loop
            break
    
        except Exception as e:
            print(f"Push attempt {attempt+1} failed with error: {e}")
    
            # If this wasn't the last attempt, wait before trying again
            if attempt < max_attempts - 1:
                time.sleep(10)  # Wait for 10 seconds
            else:
                print("All push attempts failed.")

    # Show GPU memory usage before and after deleting the model and datasets
    print(f"""Max memory allocated by tensors- before:
    {torch.cuda.max_memory_allocated(device=None) / (1024 ** 3):.2f} GB""")
    del model
    del tokenizer
    del tokenised_val
    del tokenised_train
    del dataset_val
    del dataset_train
    torch.cuda.empty_cache()
    print(f"""Max memory allocated by tensors- after:
    {torch.cuda.max_memory_allocated(device=None) / (1024 ** 3):.2f} GB""")

def test_model(i, polarity):
    """
    Function that tests a model saved on Hugging Face Hub on the test set and saves the results to a file
    :param i: Dataset iteration
    :param polarity: Polarity of the model
    :return: None
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
    model_name = f"Tanor/BERTicSENT{polarity}{i}"
    
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
    
    # Write confusion matrix and classification report to a file
    with open(os.path.join(REP_DIR, f"report_{name}.txt"), "w") as f:
        f.write(str(confusion_mat))
        f.write("\n\n")
        f.write(classification_rep)
    
    # Create a DataFrame with test data, predicted classes and real classes
    table = pd.DataFrame({"X": X_test, "Predicted": predicted_classes, 
                          "Real": y_test})
    
    # Create a table of misclassified instances
    misclassified_X = table[table["Predicted"] != table["Real"]]
    
    # Save the table of misclassified instances to a file
    misclassified_X.to_csv(os.path.join(REP_DIR, f"table_{name}.csv"), index=False)
    
    # Delete the pipeline to free up memory
    del pipe
    torch.cuda.empty_cache()

def test_model_local(i, polarity):
    """
    Function that tests a model saved on disk on the test set and saves the results to a file
    :param i: Dataset iteration
    :param polarity: Polarity of the model
    :return: None
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
    model_name = f"BERTicSENT{polarity}{i}"
    
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
    
    # Write confusion matrix and classification report to a file
    with open(os.path.join(REP_DIR, f"report_{name}.txt"), "w") as f:
        f.write(str(confusion_mat))
        f.write("\n\n")
        f.write(classification_rep)
    
    # Create a DataFrame with test data, predicted classes and real classes
    table = pd.DataFrame({"X": X_test, "Predicted": predicted_classes, 
                          "Real": y_test})
    
    # Create a table of misclassified instances
    misclassified_X = table[table["Predicted"] != table["Real"]]
    
    # Save the table of misclassified instances to a file
    misclassified_X.to_csv(os.path.join(REP_DIR, f"table_{name}.csv"), index=False)
    
    # Delete the pipeline to free up memory
    del pipe
    torch.cuda.empty_cache()    

#function that load model and tokenized and pushset ot hup, overwring the chekpoint if neceeary
def upload_local_model_to_hub(i, polarity):

    

    model_name_local = f"BERTicSENT{polarity}{i}"
    model_name_hub = f"Tanor/BERTicSENT{polarity}{i}"
    tokenizer = AutoTokenizer.from_pretrained(model_name_local)
    if (polarity =="NEG"):
        id2label = {0: "NON-NEGATIVE", 1: "NEGATIVE"}
        label2id = {"NON-NEGATIVE": 0, "NEGATIVE": 1}
    else:
        id2label = {0: "NON-POSITIVE", 1: "POSITIVE"}
        label2id = {"NON-POSITIVE": 0, "POSITIVE": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_local, num_labels=2,  id2label=id2label, 
        label2id=label2id, )
    
    max_attempts = 3

    for attempt in range(max_attempts):
           
        try:
            # Try to push the model to the hub
            model.push_to_hub(model_name_hub)
            tokenizer.push_to_hub(model_name_hub)

            # If the push is successful, exit the loop
            break
    
        except Exception as e:
            print(f"Push attempt {attempt+1} failed with error: {e}")
    
            # If this wasn't the last attempt, wait before trying again
            if attempt < max_attempts - 1:
                time.sleep(10)

def delete_model(i, polarity):
    from huggingface_hub import delete_repo
    model_name = f"BERTicSENT{polarity}{i}"
    delete_repo(model_name)
