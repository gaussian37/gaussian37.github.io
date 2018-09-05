---
layout: post
title: 5-1 Deep learning for computer vision, Introduction to convnets
date: 2018-09-06 02:40:00
img: deep-learning/chollet/chollet-book.png
categories: [deep-learning-chollet] 
tags: [deep learning, chollet, convnet] # add tag
---

This chapter (Deep learning for computer vision) covers

- Understanding convolutional neural networks (convnets)
- Using data augmentation to mitigate overfitting
- Using a pretrained convnet to do feature extraction
- Fine-tuning a pretrained convnet
- Visualizing what convnets learn and how they make classification decisions

### 5.1 Introduction to convnets

The following lines of code show you what a basic convnet looks like. <br> 
It’s a stack of `Conv2D` and `MaxPooling2D` layers. You’ll see in a minute exactly what they do.


``` python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
```

A convnet takes as input tensors of shape (image_height, image_width, image_channels). <br>
In this case, we’ll configure the convnet to process inputs of size (28, 28, 1), which is the format of MNIST images. 
We’ll do this by passing the argument input_shape=(28, 28, 1) to the first layer.


```python
>>> model.summary()
________________________________________________________________
Layer (type)                     Output Shape          Param #
================================================================
conv2d_1 (Conv2D)                (None, 26, 26, 32)    320
________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 13, 13, 32)    0
________________________________________________________________
conv2d_2 (Conv2D)                (None, 11, 11, 64)    18496
________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 5, 5, 64)      0
________________________________________________________________
conv2d_3 (Conv2D)                (None, 3, 3, 64)      36928
================================================================
Total params: 55,744
Trainable params: 55,744
Non-trainable params: 0
``` 


