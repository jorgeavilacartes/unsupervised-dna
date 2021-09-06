"""
This script is a template to add a new architecture.

The class ModelLoader is in charge to load the architecture 
defined here to the training pipeline.

MANDATORY: 
1. Define the architecture in a function called get_model()
2. This function must return the compiled model 

SUGGESTIONS:
If you need a custom loss or layer, is preferable to add this functions to 
custom_layer.py or custom_losses.py, then import it to your script,
so it can be used in future architectures.
"""

from pathlib import Path

# Architectures are implemented using tensorflow
# You could use pytorch if you want
import tensorflow as tf

# Layers needed for a Variational Autoencoder
from .custom_layers import Sampling
from .custom_losses import (
    kl_divergence,
    reconstruction_loss
)

# Reference name of model
MODEL_NAME = str(Path(__file__).resolve().stem)

# config
k = 7 # (128x128)
latent_dim = 3
intermediate_dim = 64

def get_model():

    # Encoder
    encoder_input = tf.keras.Input(shape=(2**k, 2**k,1))

    x = tf.keras.layers.Conv2D(32, 3, strides = 2,
                            activation="relu", padding="same")(encoder_input)
    x = tf.keras.layers.Conv2D(64, 3, strides = 2,
                            activation="relu", padding="same")(x)
    x = tf.keras.layers.Flatten()(x)


    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()((z_mean, z_log_var))

    encoder = tf.keras.Model(inputs=encoder_input, outputs=[z_mean, z_log_var, z], name="encoder")
    print(encoder.summary())
    encoder.add_loss(kl_divergence(z_mean, z_log_var))

    # Decoder (recreate the image)
    latent_input=tf.keras.Input(shape=(latent_dim,), name="z_sampling")
    y = tf.keras.layers.Dense(32 * 32 * intermediate_dim, activation="relu")(latent_input)
    y = tf.keras.layers.Reshape((32, 32, intermediate_dim))(y)
    y = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(y)
    y = tf.keras.layers.Conv2DTranspose(1, 3, strides=2, activation="sigmoid", padding="same")(y)

    decoder = tf.keras.Model(latent_input, y)
    print(decoder.summary())

    # VAE
    z_mean, z_lg_var, z = encoder(encoder_input)
    decoder_output      = decoder(z)

    vae = tf.keras.Model(encoder_input, decoder_output)

    vae.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=reconstruction_loss,
                metrics=["mse"]
    )

    return vae    