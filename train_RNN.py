# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:27:05 2023

@author: Petalinkar Sasa

This script performs sentiment analysis using a transformer model.
It loads and preprocesses the data, trains the model, evaluates its performance,
and saves a report with the classification metrics and the misclassified examples.

The script is designed to be used with a specific data format and directory structure,
and it may need to be adapted for other use cases.
"""


import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

def polarity_correction(pos, neg):
    """
    This function corrects the polarity of the output of a two-output model 
    that outputs a positive sentiment score and a negative sentiment score. 

    Args:
        pos (tensor): Tensor with positive sentiment scores.
        neg (tensor): Tensor with negative sentiment scores.

    Returns:
        tuple: A tuple with tensors of corrected positive and negative sentiment scores.
    """
    one = tf.convert_to_tensor(1.0)  # Tensor of 1.0 for calculations

    # Subtract negative sentiment score from 1 and multiply it with the positive sentiment score
    ret_pos = pos * (one - neg)

    # Subtract positive sentiment score from 1 and multiply it with the negative sentiment score
    ret_neg = neg * (one - pos)

    return ret_pos, ret_neg

# Define directories
ROOT_DIR = ""
RES_DIR = os.path.join(ROOT_DIR, "resources")
MOD_DIR = os.path.join(ROOT_DIR, "ml_models")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_sets")
REP_DIR = os.path.join(ROOT_DIR, "reports", "RNN")

# Define constants
BUFFER_SIZE = 1000
BATCH_SIZE = 128
VOCAB_SIZE = 25000
MAX_LEN = 300  # Only consider the first 300 words
DATASET_ITERATIONS = [0, 2, 4, 6]  # Dataset iterations to process

# Create directory if not exists
if not os.path.exists(REP_DIR):
    os.makedirs(REP_DIR)

# Set a seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
def load_and_preprocess_data(polarity, iteration):
    """
    Load and preprocess the data for a given polarity and iteration.

    Args:
        polarity (str): The polarity of the sentiment ("POS" or "NEG").
        iteration (int): The iteration number.

    Returns:
        tuple: A tuple with the training and validation datasets.
    """
    # File name
    name = f"LM{polarity}{iteration}.csv"

    # Read the data from the CSV file
    X = pd.read_csv(os.path.join(TRAIN_DIR, f"X_train_{name}"))["Sysnet"]
    y = pd.read_csv(os.path.join(TRAIN_DIR, f"y_train_{name}"))[polarity]

    X_test = pd.read_csv(os.path.join(TRAIN_DIR, f"X_test_{name}"))["Sysnet"]
    y_test = pd.read_csv(os.path.join(TRAIN_DIR, f"y_test_{name}"))[polarity]

    # Split dataset into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)

    # Convert pandas dataframes to tensorflow tensors
    X_train = tf.convert_to_tensor(X_train, name="Definicija")
    y_train = tf.convert_to_tensor(y_train, name="Sentiment")

    X_val = tf.convert_to_tensor(X_val, name="Definicija")
    y_val = tf.convert_to_tensor(y_val, name="Sentiment")

    X_test = tf.convert_to_tensor(X_test, name="Definicija")
    y_test = tf.convert_to_tensor(y_test, name="Sentiment")

    # Create tf.data datasets for training and validation
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    # Shuffle, batch, and prefetch data for performance
    train_dataset = train_dataset.shuffle(BUFFER_SIZE, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, X_test, y_test

def create_model(encoder):
    """
    Create and compile the transformer model.

    Args:
        n_transformer_layers (int, optional): The number of transformer layers. Default is 1.

    Returns:
        keras.Model: The compiled model.
    """
    # Define the model architecture using Keras Functional API
    model= tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 128, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,  return_sequences=True)),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,  return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=['binary_accuracy'])

    return model
def train_model(model, train_dataset, validation_dataset):
    """
    Train the model on the given datasets.

    Args:
        model (keras.Model): The compiled model.
        train_dataset (tf.data.Dataset): The training dataset.
        validation_dataset (tf.data.Dataset): The validation dataset.

    Returns:
        History: The history object returned by model.fit().
    """
    history = model.fit(train_dataset, 
                        epochs=5, 
                        validation_data=validation_dataset, 
                        validation_steps=5)

    return history

def evaluate_model(model, X_test):
    """
    Evaluate the model on the test data.

    Args:
        model (keras.Model): The trained model.
        X_test (Tensor): The test input data.

    Returns:
        np.array: The predicted outputs.
    """
    y_pred = model.predict(X_test)

    # Assuming y_pred is a continuous value from 0 to 1, we need to convert it to binary.
    # Usually, we use a threshold of 0.5 for this conversion
    y_pred_binary = np.where(y_pred > 0.5, 1, 0)

    return y_pred_binary

def write_report(y_test, y_pred_binary, polarity, iteration):
    """
    Write the classification report and confusion matrix to a text file.

    Args:
        y_test (Tensor): The true labels.
        y_pred_binary (np.array): The predicted labels.
        polarity (str): The polarity of the sentiment ("POS" or "NEG").
        iteration (int): The iteration number.
    """
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_binary)
    print('Confusion Matrix:\n', conf_matrix)

    # Calculate the classification report
    class_report = classification_report(y_test, y_pred_binary)
    print('Classification Report:\n', class_report)

    # File name
    name = f"LM{polarity}{iteration}.csv"

    # Writing report
    with open(os.path.join(REP_DIR, f"report_{name}.txt"), "w") as f:
        f.write(str(conf_matrix))
        f.write("\n\n")
        f.write(class_report)
def save_misclassified(X_test, y_pred_binary, y_test, polarity, iteration):
    """
    Save the misclassified examples to a CSV file.

    Args:
        X_test (Tensor): The test input data.
        y_pred_binary (np.array): The predicted labels.
        y_test (Tensor): The true labels.
        polarity (str): The polarity of the sentiment ("POS" or "NEG").
        iteration (int): The iteration number.
    """
    # Convert tensors to pandas DataFrames for easier manipulation
    X_test_df = pd.DataFrame(X_test.numpy(), columns=["Sysnet"])
    y_test_df = pd.DataFrame(y_test.numpy(), columns=[polarity])

    # Concatenate the two DataFrames along the columns
    test_data = pd.concat([X_test_df, y_test_df], axis=1)

    # Add the predicted labels to the DataFrame
    test_data["Predicted"] = y_pred_binary

    # Select the misclassified examples
    misclassified = test_data[test_data[polarity] != test_data["Predicted"]]

    # File name
    name = f"LM{polarity}{iteration}.csv"

    # Save the misclassified examples to a CSV file
    misclassified.to_csv(os.path.join(REP_DIR, f"misclassified_{name}"), index=False)

def main():
    """
    The main function to execute the script.
    """
    # Record the start time for the total execution
    total_start_time = time.time()

    # Repeat the process for each dataset iteration and polarity
    for i in DATASET_ITERATIONS:
        # Record the start time for this iteration
        iter_start_time = time.time()

        for polarity in ["POS", "NEG"]:
            train_dataset, validation_dataset, X_test, y_test = load_and_preprocess_data(polarity, i)
            encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, output_mode="int")
            encoder.adapt(train_dataset.map(lambda text, label: text))


            # Create and train the model
            model = create_model(encoder)
            train_model(model, train_dataset, validation_dataset)

            # Evaluate the model
            y_pred_binary = evaluate_model(model, X_test)

            # Write the report and save the misclassified examples
            write_report(y_test, y_pred_binary, polarity, i)
            save_misclassified(X_test, y_pred_binary, y_test, polarity, i)
            # Save the model
            model_name = f"RNN_model_{polarity}_{i}"
            model_path = os.path.join(MOD_DIR, model_name)
            model.save(f"{model_path}.tf", save_format='tf')



        # Record the end time for this iteration
        iter_end_time = time.time()

        # Calculate the execution time for this iteration
        iter_total_time = iter_end_time - iter_start_time

        # Convert the iteration time to hours, minutes, and seconds
        hours, rem = divmod(iter_total_time, 3600)
        minutes, seconds = divmod(rem, 60)

        # Save the timing result for this iteration in a text file
        with open(os.path.join(REP_DIR, f'execution_time_{i}.txt'), 'w') as f:
            f.write("Execution Time for Iteration {}: {:0>2}:{:0>2}:{:05.2f}".format(i, int(hours), int(minutes), seconds))

    # Record the end time for the total execution
    total_end_time = time.time()

    # Calculate the total execution time
    total_total_time = total_end_time - total_start_time

    # Convert the total time to hours, minutes, and seconds
    hours, rem = divmod(total_total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Save the total timing result in a text file
    with open(os.path.join(REP_DIR, 'execution_time_total.txt'), 'w') as f:
        f.write("Total Execution Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

if __name__ == "__main__":
    main()
