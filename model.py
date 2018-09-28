import keras
import keras.layers as L
import numpy as np

def build_autoencoder(img_shape, encoder, decoder, lr=0.002):
    """Builds a custom Autoencoder using Encoder and Decoder models.

    Args:
        img_shape (tuple):
            Shape of input image.
        encoder (keras.models.Sequential):
            Encoder part of the autoencoder.
        decoder (keras.models.Sequential):
            Decoder part of the autoencoder.
        lr (float, optional):
            Learning rate of the optimizer. 
            If unspecified, defaults to 0.002.
            
    Returns:
        keras.model.Model:
            Autoencoder model consisting of encoder and 
            decoder sequential models.
    """
    
    # Inputs and Outputs
    inp = L.Input(img_shape)
    embedding = encoder(inp)
    reconstruction = decoder(embedding)
    
    # Build Model
    autoencoder = keras.models.Model(inputs=inp, outputs=reconstruction)
    optimizer = keras.optimizers.Adamax(lr=lr)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    
    return autoencoder

def create_pca_autoencoder(img_shape, lr, emb_size):
    """Creates a Linear PCA Autoencoder.

    Args:
        img_shape (tuple):
            Shape of input image.
        lr (float):
            Learning rate of the optimizer for training 
            the autoencoder model.
        emb_size (int):
            No. of embedding dims for encoder output.

    Returns:
        keras.model.Model:
            The autoencoder model.
    """

    # Encoder (Image -> Embedding)
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Flatten())
    encoder.add(L.Dense(emb_size))
    
    # Decoder (Embedding -> Image)
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((emb_size,)))
    decoder.add(L.Dense(np.prod(img_shape)))
    decoder.add(L.Reshape(img_shape))
    
    return build_autoencoder(img_shape, encoder, decoder, lr=lr)

def create_deep_conv_autoencoder(img_shape, lr, emb_size):
    """Creates a Deep Convolutional Denoising Autoencoder.

    Args:
        img_shape (tuple):
            Shape of input image.
        lr (float):
            Learning rate of the optimizer for training 
            the autoencoder model.
        emb_size (int):
            No. of embedding dims for encoder output.

    Returns:
        keras.model.Model:
            The autoencoder model.
    """

    # Encoder (Image -> Embedding)
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Conv2D(32, kernel_size=(3, 3), padding='same', activation='elu'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='same'))
    encoder.add(L.Conv2D(64, kernel_size=(3, 3), padding='same', activation='elu'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='same'))
    encoder.add(L.Conv2D(128, kernel_size=(3, 3), padding='same', activation='elu'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='same'))
    encoder.add(L.Conv2D(256, kernel_size=(3, 3), padding='same', activation='elu'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2), padding='same'))
    encoder.add(L.Flatten())
    encoder.add(L.Dense(emb_size, activation='elu'))

    # Decoder (Embedding -> Image)
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((emb_size,)))
    decoder.add(L.Dense(np.prod(img_shape), activation='elu'))
    decoder.add(L.Reshape(img_shape))
    decoder.add(L.Conv2DTranspose(128, kernel_size=(3, 3), padding='same', activation='elu'))
    decoder.add(L.Conv2DTranspose(64, kernel_size=(3, 3), padding='same', activation='elu'))
    decoder.add(L.Conv2DTranspose(32, kernel_size=(3, 3), padding='same', activation='elu'))
    decoder.add(L.Conv2DTranspose(3, kernel_size=(3, 3), padding='same', activation=None))
    
    return build_autoencoder(img_shape, encoder, decoder, lr=lr)
