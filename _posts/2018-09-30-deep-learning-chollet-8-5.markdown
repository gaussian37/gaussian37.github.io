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
Once training is over, the generator is capable of turning any point in its input space into a believable image.
Unlike VAEs, this latent space has fewer explicit guarantees of meaningful structure; in particular, it isn’t continuous.

+ A generator transforms random latent vectors into images, and a discriminator seeks to tell real images from generated ones. The generator is trained to fool the discriminator.

![8.15](../assets/img/dl/chollet/08-5/08fig15_alt.jpg)

Remarkably, a GAN is a system where the optimization minimum isn’t fixed, unlike in any other training setup you’ve encountered.
Normally, gradient descent consists of rolling down hills in a static loss landscape.
But with a GAN, every step taken down the hill changes the entire landscape a little.
It’s a dynamic system where the **optimization process is seeking not a minimum, but an equilibrium between two forces.** 
For this reason, GANs are notoriously difficult to train—getting a GAN to work requires lots of careful tuning of the model architecture and training parameters.

### 8.5.1. A schematic GAN implementation

In this section, we’ll explain how to implement a GAN in Keras.
The specific implementation is a deep convolutional GAN (**DCGAN**):
a GAN where the generator and discriminator are deep convnets. In particular, it uses a `Conv2DTranspose` layer for image upsampling in the generator.

You’ll train the GAN on images from CIFAR10, a dataset of 50,000 32 × 32 RGB images belonging to 10 classes (5,000 images per class).
To make things easier, you’ll only use images belonging to the class “frog.”

Schematically, the GAN looks like this: <br>

1. A `generator` network maps vectors of shape `(latent_dim,)` to images of shape `(32, 32, 3)`.
2. A `discriminator` network maps images of shape `(32, 32, 3)` to a binary score estimating the probability that the image is real.
3. A `gan` network chains the generator and the discriminator together: `gan(x) = discriminator(generator(x))`. Thus this `gan` network maps latent space vectors to the discriminator’s assessment of the realism of these latent vectors as decoded by the generator.
4. You train the discriminator using examples of real and fake images along with “real”/“fake” labels, just as you train any regular image-classification model.
5. To train the generator, you use the gradients of the generator’s weights with regard to the loss of the gan model. This means, at every step, you move the weights of the generator in a direction that makes the discriminator more likely to classify as “real” the images decoded by the generator. In other words, you train the generator to fool the discriminator. 
 


 
 
  


