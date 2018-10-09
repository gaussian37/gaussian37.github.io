---
layout: post
title: 8-5. Introduction to generative adversarial networks
date: 2018-10-01 09:28:00
img: dl/chollet/chollet.png
categories: [dl-chollet] 
tags: [deep learning, chollet, gan] # add tag
---

Generative adversarial networks (GANs), introduced in 2014 by Goodfellow et al., are an alternative to VAEs for learning latent spaces of images.
They enable the generation of fairly realistic synthetic images 
by forcing the generated images to be statistically almost indistinguishable from real ones.

Ian Goodfellow et al., [“Generative Adversarial Networks,” arXiv (2014)](https://arxiv.org/abs/1406.2661)

An intuitive way to understand GANs is to imagine a forger trying to create a fake Picasso painting.
At first, the forger is pretty bad at the task.
He mixes some of his fakes with authentic Picassos and shows them all to an art dealer. 
The art dealer makes an authenticity assessment for each painting and gives the forger feedback about what makes a Picasso look like a Picasso.
The forger goes back to his studio to prepare some new fakes. As times goes on, the forger becomes increasingly competent at imitating the style of Picasso, and the art dealer becomes increasingly expert at spotting fakes. 
In the end, they have on their hands some excellent fake Picassos.

That’s what a GAN is: a forger network and an expert network, each being trained to best the other. As such, a GAN is made of two parts:

+ **Generator network** - Takes as input a random vector (a random point in the latent space), and decodes it into a synthetic image
+ **Discriminator network(or adversary)** - Takes as input an image (real or synthetic), and predicts whether the image came from the training set or was created by the generator network.

The generator network is trained to be able to fool the discriminator network, 
and thus it evolves toward generating increasingly realistic images as training goes on:
artificial images that look indistinguishable from real ones,
to the extent that it’s impossible for the discriminator network to tell the two apart
Meanwhile, the discriminator is constantly adapting to the gradually improving capabilities of the generator,
setting a high bar of realism for the generated images. 
  


