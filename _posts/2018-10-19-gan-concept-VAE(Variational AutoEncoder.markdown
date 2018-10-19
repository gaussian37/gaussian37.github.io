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

+ $$ p(x,z) = p(x\|z)p(z) $$

If you know AE(`AutoEncoder`) then, you exactly know what `latent vector` is.

### our goal is to find out `latent vector z` which can represent `x` well.

In order to meet the goal, we need to focus on `Posterior`.
But, Why do we focus on posterior? Because we need $$ p(x) $$. but it's hard to caculate. (**Intractable**)
Thus, we are going to access approximate `posterior`.

$$ p(z\|x) $$ : $$ z $$ is latent vector, $$ x $$ is observable data.

$$ p(z\|x) = \frac{p(z\|x)p(z)}{p(x)} $$ : Infer Good Value of `z` given `x` (**Bayesian**)

$$ p(x) = \int p(x,z)dz = \int p(z\|x)p(z) dz $$ (**Product Rule**)

Among few ways, we will use `Variational Inference`. 

+ Variational Inference
    - pick a family of distributions over the latent variables with its own variational parameters
        - $$ q_{\phi}(z\|x) $$ : $$ \phi $$ is `distribution` such as gaussian, uniform...
    - Find $$ \phi $$ that makes $$ q $$ close to the posterior of interest. 


















  