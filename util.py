import os

import keras
import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt

def reset_tf_session():
    """Reset TF Session

    Resets the TensorFlow session for Keras `backend`.

    Returns:
        tf.Session: New tensorflow session object.
    """
    default_session = tf.get_default_session()
    if default_session is not None:
        default_session.close()
    K.clear_session()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    sess = tf.InteractiveSession(config=config)
    K.set_session(sess)
    return sess

def apply_gaussian_noise(X, sigma=0.1):
    """Apply Gaussian Noise

    Adds noise from standard normal distribution with standard deviation sigma.

    Args:
        X (numpy.ndarray):
            Input array (image data).
        sigma (float):
            Standard deviation of the normal 
            distribution. Defaults to 0.1.

    Returns:
        numpy.ndarray: Input array after applying gaussian noise.
    """
    noise = np.random.normal(loc=0.0, scale=sigma, size=X.shape)
    return X + noise

def show_image(X):
    """Show Image

    Displays image using `matplotlib`.

    Args:
        X (numpy.ndarray):
            Input array (image data).
    """
    plt.imshow(np.clip(X + 0.5, 0, 1))

def visualize(X, encoder, decoder):
    """Visualize

    Draws original, encoded and decoded images.
    
    Args:
        X (numpy.ndarray):
            Input array (image data).
        encoder (keras.models.Sequential):
            Encoder part of the autoencoder.
        decoder (keras.models.Sequential):
            Decoder part of the autoencoder.
    """
    emb = encoder.predict(X[None])[0]
    reco = decoder.predict(emb[None])[0]

    plt.subplot(1, 3, 1)
    plt.title("Original")
    show_image(X)

    plt.subplot(1, 3, 2)
    plt.title("Embedding")
    plt.imshow(emb.reshape([emb.shape[-1]//2,-1]))

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed")
    show_image(reco)
    plt.show()