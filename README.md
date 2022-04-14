# Rad-cGAN v1.0: Radar-based precipitation nowcasting model with conditional Generative Adversarial Networks for multiple dam domains



## Introduction

Here is the cGAN-based precipitation nowcasting model, named Rad-cGAN, was trained with a radar reflectivity map of the Soyang-gang Dam region in South Korea provided by Korea Meteorological Administration (KMA) .

The source code of Rad-cGAN and reference models (U-net and ConvLSTM) were written using [Keras](https://keras.io/)  functional API is in the folder `model`.

The pre-trained Rad-cGAN model is available in Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6460012.svg)](https://doi.org/10.5281/zenodo.6460012)


## Model architecture

We developed a Rad-cGAN model using a cGAN framework. Whole architecture is represented in following figure:
<img src="images/model.png" alt="Rad-cGAN architecture" width="100%"/>

# Generator

The model consists of nine convolutional layers, two max-pooling layers, two up-sampling layers, and an output convolutional layer. Each convolutional layer, except for the output layer, is composed of the following operations: 3 × 3 2D convolution with zero-padding, batch normalization, and activation function of ReLU. In the contracting part of the generator, a 2 × 2 2D max-pooling operation was used to down-sample the input images. 

A 2 × 2 2D up-sampling operation was further applied in the expanding part after skip connection to increase the resolution of featured images that contain both high- and low-level information. Finally, the output convolutional layer had a 1 × 1 2D convolution that used a linear function for activation to obtain future prediction of the radar reflectivity image.

# Discriminator

The model consists of three convolutional layers and an output layer. The first two convolutional layers were composed of 4 × 4 2D convolution with strides of two and zero-padding, batch normalization, and ReLU activation function, which was leaky and had a 0.2 slope. The third convolutional layer had the same configuration as the previous layers, except that its stride was 1. To distinguish the input pair in the image form, the output layer consisted of 4 × 4 2D convolution with zero-padding and sigmoid activation functions. 

## Optimization procedure
To optimize Rad-cGAN, we followed the training procedure suggested by Isola et al. (2017). First, we randomly selected samples that consisted of four consecutive radar reflectivity images (t-30, t-20, t-10 min, and t) and the image at t+10 min. Then, we created a training sample for the discriminator by adding labels to classify whether the samples were real (image at t+10 min from observation) or fake (t+10 image from the generator) pairs. Next, we updated the parameters of the discriminator using the minibatch stochastic gradient descent (SGD) method for one step. Binary cross-entropy was used as a loss function, and we applied the ADAM optimizer (Kingma and Ba, 2015) with a learning rate of 0.0002 and momentum parameters β_1 = 0.5 and β_2 = 0.999. 

