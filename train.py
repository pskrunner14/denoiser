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
    help='Type of model for training [pca(linear)/deep(conv)] - please provide the same flag when training a pre-trained model'
)
@click.option(
    '-lr', 
    '--learning-rate', 
    default=0.005, 
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
    """Train Model [optional args]

    Trains the Autoencoder and saves best model.

    Args:
        model_type (str): 
            Type of autoencoder [pca(linear)/deep(conv)]. Defaults to `deep`.
        learning_rate (float): 
            Learning rate for minimizing loss during training. Defaults to 0.001.
        batch_size (int):
            Batch size of minibatches to use during training. Defauls to 32.
        num_epochs (int):
            Number of epochs for training model. Defaults to 10.
        save_every (int):
            Epoch interval to save model checkpoints during training. Defaults to 1.
        tensorboard_vis (bool):
            Flag for TensorBoard Visualization. Defaults to `False`.
    """
    setup_paths()

    logging.info('loading data from LFW dataset directory')
    X_train, X_test, IMG_SHAPE, attr = load_dataset(use_raw=True, dimx=32, dimy=32)

    reset_tf_session()

    if os.path.exists('models/autoencoder.h5'):
        logging.info('loading pre-trained {} autoencoder'.format(model_type.upper()))
        autoencoder = keras.models.load_model('models/autoencoder.h5')
    elif model_type == 'pca':
        logging.info('creating PCA autoencoder')
        autoencoder = create_pca_autoencoder(IMG_SHAPE, emb_size=32)
    elif model_type == 'deep':
        logging.info('creating DeepConv autoencoder')
        autoencoder = create_deep_conv_autoencoder(IMG_SHAPE, emb_size=32)
    else:
        raise UserWarning('Unrecognized model type!')

    callbacks = configure_callbacks(save_every, tensorboard_vis)

    logging.info('training {} autoencoder'.format(model_type.upper()))
    autoencoder.fit(
        x=X_train, 
        y=X_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(X_test, X_test),
        callbacks=callbacks,
        verbose=1
    )
    logging.info('saving {} autoencoder model to `models/autoencoder.h5`'.format(model_type.upper()))
    autoencoder.save('models/autoencoder.h5')

    logging.info('evaluating {} autoencoder'.format(model_type.upper()))
    score = autoencoder.evaluate(
        X_test,
        X_test,
        verbose=0
    )
    logging.info("MSE: {}".format(score))

    logging.info('visualizing reconstructions of faces in test set')
    for i in range(5):
        img = X_test[i]
        visualize(img, autoencoder.layers[1], autoencoder.layers[2])

def configure_callbacks(save_every=1, tensorboard_vis=False):
    """Configure Callbacks 
    
    Configures callbacks for training model with `keras`.

    Args:
        Refer to `train`.

    Returns:
        list: List containing configured callbacks.
    """
    # checkpoint models only when `val_loss` impoves
    saver = keras.callbacks.ModelCheckpoint(
        'models/ckpts/model.ckpt',
        monitor='val_loss',
        save_best_only=True,
        period=save_every,
        verbose=1
    )
    callbacks = [saver]

    if tensorboard_vis:
        # tensorboard visualization callback
        tensorboard_cb = keras.callbacks.TensorBoard(
            log_dir='./logs',
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard_cb)
    
    return callbacks

def setup_paths():
    if not os.path.isdir('models/ckpts'):
        if not os.path.isdir('models'):
            os.mkdir('models')
        os.mkdir('models/ckpts')

def main():
    LOG_FORMAT = '%(levelname)s %(message)s'
    logging.basicConfig(
        format=LOG_FORMAT, 
        level='INFO'
    )
    try:
        train()
    except KeyboardInterrupt:
        print('EXIT')

if __name__ == '__main__':
    main()