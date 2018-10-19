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
+ $$ p(z\|x) $$ : Posterior
+ $$ p(x,z) $$ : Probability Model Defined as Joint Distribution of x, z

We are going to use this model.

+ $$ p(x,z) = p(x\|z)p(z)

If you know AE(`AutoEncoder`) then, you exactly know what `latent vector` is.

### our goal is to find out `latent vector z` which can represent `x` well.
In order to meet the goal, we need to focus on `Posterior`.

$$ p(z\|x) : z(latent vector), x(observable data) $$

$$ p(z\|x) = \frac{p(z\|x)p(z)}{p(x)} $$ : Infer Good Value of `z` given `x` (**Bayesian**)

$$ p(x) = \int p(x,z)dz = \int p(z\|x)p(z) dz $$ (**Product Rule**)













  