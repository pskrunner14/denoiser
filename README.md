# Denoiser

Denoising Images using Convolutional Autoencoder. The principle behind denoising autoencoders is to be able to reconstruct data from an input of corrupted data. After giving the autoencoder the corrupted data, we force the hidden layer to learn only the more robust features, rather than just the identity. The output will then be a more refined version of the input data. We can train a denoising autoencoder by stochastically corrupting data sets and inputting them into a neural network. The autoencoder can then be trained against the original data. One way to corrupt the data would be simply to randomly remove some parts of the data or in our case add some random uniform noise to the data, so that the autoencoder is trying to predict the original input.

![Denoising Autoencoder](./images/ae.png)

## Dataset

I've used the [Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) Dataset that contains more than 13,000 images of faces collected from the web which is more than enough for this task. Each face has been labeled with the name of the person pictured. 1680 of the people pictured have two or more distinct photos in the data set.

## Getting Started

In order to train the model, you will need to install the required python packages using:

```bash
pip install -r requirements.txt
```

Once you're done with that, you can open up a terminal and start training the model:

```bash
python train.py -lr 0.002 --num-epochs 10 --batch-size 32 --save-every 5 --tensorboard-vis
```

Passing the `--tensorboard-vis` flag allows you to view the training/validation loss in your browser using:

```bash
tensorboard --logdir=./logs
```

## Built with

* Python
* Keras
* TensorFlow