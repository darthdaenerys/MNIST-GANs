# MNIST Generative Adverserial Network

Tensorflow implementation of Generative Adversarial Networks for MNIST dataset.

![](DCGAN\animation.gif)

Mnist-gan is a GAN(Generative Adversarial Network) that learns how to generate images that look like mnist digits. Separate networks are trained for each digit. The idea is to train a generator network which when fed noise (in my case a 128 dimensional random vector) will generate an image that looks like an mnist style 8 (for example). In essence the network is finding a function of 128 variables that returns a matrix of pixels that look like an eight. Mind bending! You can see how the digit starts to take shape from the noise.

## DCGAN
---
This is what the generated actually generated before training.

![](DCGAN\noise.png)

### Results

These are the results after training for 500 epochs.

![](DCGAN/DCGAN%20generated-1.png)

### Generator and Discriminator loss curves

![](DCGAN\plot.png)

There are some bumps both for generator and discriminator but on an average both stays almost at the same loss.

### Implementation details

    - Batch size: 128
    - Learning rate: 1e-4
    - Epochs: 500
    - Optimizer: Adam

### Learning time

    - 7 secs per epoch
    - Total time: almost 1 hr

## WGAN
---
### Pre-training images

![](WGAN/noise.gif)

### Results after 120 epochs

![](WGAN/wgan-generated.gif)

### Generator and critic loss curves

![](WGAN/loss_trend.png)

### Implementation details

    - Batch size: 400
    - Learning rate: 0.00002
    - Epochs: 120
    - Optimizer: Adam

### Learning time

    - 116 secs per epoch
    - Training time: 4 hours 21 minutes 43 seconds


## Dataset

    MNIST

![](mnist.png)

**You can download the dataset by clicking** [here](http://yann.lecun.com/exdb/mnist/)

## Development Environment

- Windows 10
- Tensorflow v2.9.1
- Keras API v2.9.0
- Python v3.9.7
- Numpy v1.22.2
- Matplotlib v3.2.2