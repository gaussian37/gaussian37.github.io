---
layout: post
title: Overview of VAE(Variational AutoEncoder)  
date: 2018-10-19 08:42:00
img: gan/concept/vae/vae.png
categories: [gan-concept] 
tags: [autoencoder, ae, vae, variational autoencoder] # add tag
---

+ Code
    - [Keras VAE with CNN (MNIST)](http://nbviewer.jupyter.org/github/gaussian37/Deep-Learning/blob/master/Library/Keras/VAE/VAE-CNN-MNIST-Keras.ipynb)
    - [Keras VAE with CNN (Fashion MNIST)](http://nbviewer.jupyter.org/github/gaussian37/Deep-Learning/blob/master/Library/Keras/VAE/VAE-CNN-Fashion-Keras.ipynb)

Before starting to talk about **VAE**, Check the notation.

+ $$ x $$ : Observed data
+ $$ p(x) $$  : Evidence
+ $$ z $$ : Latent vector
+ $$ p(z) $$ : Prior
+ $$ p(x\|z) $$ : Likelihood
+ $$ p(z||x) $$ : Posterior
+ $$ p(x,z) $$ : Probability Model Defined as Joint Distribution of x, z







  