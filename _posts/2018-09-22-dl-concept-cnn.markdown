---
layout: post
title: What is Convolution Neural Network?  
date: 2018-09-22 03:49:00
img: dl/concept/what-is-cnn/cnn.png
categories: [dl-concept] 
tags: [deep learning, cnn, convolution] # add tag
---

# Introduction

In a regular neural network, the input is transformed through a series of hidden layers having multiple neurons. 
Each neuron is connected to all the neurons in the previous and the following layers.
This arrangement is called a `fully connected layer` and the last layer is the `output layer`.
In Computer Vision applications where the input is an image, we use convolutional neural network because the regular fully connected neural networks don’t work well. 
This is because if each pixel of the image is an input then as we add more layers the amount of parameters increases exponentially.

![convnet](../assets/img/dl/concept/what-is-cnn/convnet.jpg)

<br>

Consider an example where we are using a three color channel image with size 1 megapixel (1000 height X 1000 width) then our input will have 1000 X 1000 X 3 (3 Million) features.
If we use a fully connected hidden layer with 1000 hidden units then the weight matrix will have 3 Billion (3 Million X 1000) parameters.
So, the regular neural network is not scalable for image classification as processing such a large input is computationally very expensive and not feasible.
The other challenge is that a large number of parameters can lead to over-fitting.
However, when it comes to images, there seems to be little correlation between two closely situated individual pixels. This leads to the idea of convolution.

# What is Convolution?

<video width="320" height="240" controls autoplay>
  <source src="../assets/img/dl/concept/what-is-cnn/Convolution-Operation.mp4" type="video/mp4">
</video>
