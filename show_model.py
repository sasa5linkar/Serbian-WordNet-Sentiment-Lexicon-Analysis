import tensorflow as tf
from tensorflow.keras.utils import plot_model
import os

def plot_and_save_model(model, filename: str) -> None:
    """
    Plot the architecture of a Keras model and save it to a file.
    
    Parameters:
        model: The Keras model to plot.
        filename: The name of the file where the plot will be saved.
    """
    plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)

def adapt_encoder(train_dataset: tf.data.Dataset, vocab_size) -> tf.keras.layers.TextVectorization:
    """
    Create and adapt a TextVectorization layer based on the training dataset.
    """
    encoder = tf.keras.layers.TextVectorization(max_tokens=vocab_size, output_mode="int")
    encoder.adapt(train_dataset.map(lambda text, label: text))
    return encoder

def show_transformer() -> None:
    """
    Load data, create and show a summary of the Transformer model.
    """
    from train_tranformer import VOCAB_SIZE, create_model, load_and_preprocess_data, N_TRANSFORMER_LAYERS, REP_DIR
    train_dataset, _, _, _ = load_and_preprocess_data("POS", 0)
    encoder = adapt_encoder(train_dataset, VOCAB_SIZE)
    model = create_model(encoder, N_TRANSFORMER_LAYERS)
    model_plot_filename = os.path.join(REP_DIR, "model_plot.png")
    print(model.summary())
    plot_and_save_model(model,model_plot_filename)
def show_RNN() -> None:
    """
    Load data, create and show a summary of the RNN model.
    """
    from train_RNN import VOCAB_SIZE, create_model, load_and_preprocess_data, REP_DIR
    train_dataset, _, _, _ = load_and_preprocess_data("POS", 0)
    encoder = adapt_encoder(train_dataset, VOCAB_SIZE)
    model = create_model(encoder)
    model_plot_filename = os.path.join(REP_DIR, "model_plot.png")
    print(model.summary())
    plot_and_save_model(model,model_plot_filename)


def main():
    show_transformer()
    show_RNN()

if __name__ == "__main__":
    main()

