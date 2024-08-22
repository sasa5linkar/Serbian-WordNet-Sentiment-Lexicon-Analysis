from tensorflow.keras import layers, models, Sequential
import tensorflow as tf

# Define the TransformerBlock custom layer for testing
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config

        @classmethod
        def from_config(cls, config):
            return cls(**config)

# Create a simple model using only the TransformerBlock layer
model_transformer_block = Sequential([
    layers.Input(shape=(None, 64)),  # Dummy input shape
    TransformerBlock(64, 4, 128)
])

# Attempt to save the model
try:
    model_transformer_block.save('transformer_block_model')
    print("Successfully saved the model with TransformerBlock.")
except Exception as e:
    print(f"Failed to save the model with TransformerBlock. Error: {str(e)}")
