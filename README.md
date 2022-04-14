# Rad-cGAN v1.0: Radar-based precipitation nowcasting model with conditional Generative Adversarial Networks for multiple dam domains



## Introduction

Here is the cGAN-based precipitation nowcasting model, named Rad-cGAN, was trained with a radar reflectivity map of the Soyang-gang Dam region in South Korea provided by Korea Meteorological Administration (KMA) .

The source code of Rad-cGAN and reference models (U-net and ConvLSTM) were written using [Keras](https://keras.io/)  functional API is in the folder `model`.

The pre-trained Rad-cGAN model is avilable in Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6459876.svg)](https://doi.org/10.5281/zenodo.6459876)


## Model architecture

We developed a Rad-cGAN model using a cGAN framework. Whole architecture is represented following figure:
<img src="images/model.png" alt="Rad-cGAN architecture" width="100%"/>
