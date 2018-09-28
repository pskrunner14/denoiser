import os
import click
import logging

import keras
import keras.layers as L
import tensorflow as tf

from data import load_dataset
from util import reset_tf_session, apply_gaussian_noise, visualize
from model import create_pca_autoencoder, create_deep_conv_autoencoder

@click.command(name='Training Configuration')
@click.option(
    '-mt', 
    '--model-type', 
    default='deep', 
    help='Type of model for training [pca(linear)/deep(conv)]'
)
@click.option(
    '-lr', 
    '--learning-rate', 
    default=0.002, 
    help='Learning rate for minimizing loss during training'
)
@click.option(
    '-bz',
    '--batch-size',
    default=32,
    help='Batch size of minibatches to use during training'
)
@click.option(
    '-ne', 
    '--num-epochs', 
    default=10, 
    help='Number of epochs for training model'
)
@click.option(
    '-se',
    '--save-every',
    default=1,
    help='Epoch interval to save model checkpoints during training'
)
@click.option(
    '-tb',
    '--tensorboard-vis',
    is_flag=True,
    help='Flag for TensorBoard Visualization'
)
def train(model_type, learning_rate, batch_size, num_epochs, save_every, tensorboard_vis):
    """Trains the Autoencoder and saves best model.

    Args:
        model_type (str, optional): 
            Type of autoencoder [pca(linear)/deep(conv)]. 
            If unspecified, defaults to `deep`.
        learning_rate (float, optional): 
            Learning rate for minimizing loss during training. 
            If unspecified, defaults to 0.002.
        batch_size (int, optional):
            Batch size of minibatches to use during training. 
            If unspecified, defauls to 32.
        num_epochs (int, optional):
            Number of epochs for training model. If unspecified, 
            defaults to 10.
        save_every (int, optional):
            Epoch interval to save model checkpoints during training. 
            If unspecified, defaults to 1.
        tensorboard_vis (bool, optional):
            Flag for TensorBoard Visualization. If unspecified, 
            defaults to `False`.
    """
    setup_paths()

    logging.info('loading data from LFW dataset directory')
    X_train, X_test, IMG_SHAPE, attr = load_dataset(use_raw=True, dimx=32, dimy=32)

    reset_tf_session()

    if model_type == 'pca':
        logging.info('creating PCA autoencoder')
        autoencoder = create_pca_autoencoder(IMG_SHAPE, learning_rate, 32)

        logging.info('training PCA autoencoder')
        autoencoder.fit(
            x=X_train, 
            y=X_train,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_data=(X_test, X_test),
            verbose=1
        )

        logging.info('evaluating PCA autoencoder')
        mse = autoencoder.evaluate(
            X_test,
            X_test,
            verbose=1
        )
        logging.info("MSE: {:.6f}".format(mse))

        logging.info('visualizing reconstructions of faces in test set')
        for i in range(5):
            img = X_test[i]
            visualize(img, autoencoder.layers[1], autoencoder.layers[2])

    elif model_type == 'deep':
        if os.path.exists('models/autoencoder.h5'):
            logging.info('loading pre-trained DeepConv denoising autoencoder')
            autoencoder = keras.models.load_model('models/autoencoder.h5')
        else:
            logging.info('creating DeepConv denoising autoencoder')
            autoencoder = create_deep_conv_autoencoder(IMG_SHAPE, learning_rate, 512)

        logging.info('training DeepConv denoising autoencoder on noisy images')
        callbacks = configure_callbacks(save_every, tensorboard_vis)
        for epoch in range(num_epochs):
            X_train_noise = apply_gaussian_noise(X_train)
            X_test_noise = apply_gaussian_noise(X_test)
            # save training data in history object
            history = autoencoder.fit(
                x=X_train_noise, 
                y=X_train, 
                epochs=1,
                validation_data=(X_test_noise, X_test),
                callbacks=callbacks,
                verbose=1
            )
            logging.info('Epoch {}/{}  -  loss: {:.6f}  -  val loss: {:.6f}\n'
                .format(epoch + 1, num_epochs, history.history['loss'][0], history.history['val_loss'][0]))

        logging.info('saving DeepConv denoising autoencoder to `models/autoencoder.h5`')
        autoencoder.save('models/autoencoder.h5')

        logging.info('evaluating DeepConv denoising autoencoder')
        X_test_noise = apply_gaussian_noise(X_test)
        denoising_mse = autoencoder.evaluate(
            X_test_noise, 
            X_test, 
            verbose=1
        )
        logging.info('Denoising MSE: {:.6f}'.format(denoising_mse))

        logging.info('visualizing reconstructions of faces in test set')
        for i in range(5):
            img = X_test_noise[i]
            visualize(img, autoencoder.layers[1], autoencoder.layers[2])
    else:
        raise UserWarning('Unrecognized model type!')

def configure_callbacks(save_every=1, tensorboard_vis=False):
    """Configures callbacks for training model with `keras`.

    Args:
        Refer to `train`.

    Returns:
        list: List containing configured callbacks.
    """
    saver = keras.callbacks.ModelCheckpoint(
        'models/ckpts/model.ckpt',
        monitor='val_loss',
        save_best_only=True,    # checkpoint models only when `val_loss` impoves
        period=save_every,
        verbose=1
    )
    callbacks = [saver]
    if tensorboard_vis:
        tensorboard_cb = keras.callbacks.TensorBoard(
            log_dir='./logs',
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard_cb)
    return callbacks

def setup_paths():
    """Sets up directory paths for training/checkpointing."""
    if not os.path.isdir('models/ckpts'):
        if not os.path.isdir('models'):
            os.mkdir('models')
        os.mkdir('models/ckpts')

def main():
    LOG_FORMAT = '%(levelname)s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level='INFO')
    try:
        train()
    except KeyboardInterrupt:
        print('EXIT')

if __name__ == '__main__':
    main()