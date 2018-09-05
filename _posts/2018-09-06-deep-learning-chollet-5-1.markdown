---
layout: post
title: 5-1 Deep learning for computer vision : Introduction to convnets
date: 2018-09-06 02:40:00
img: deep-learning/chollet/chollet.png
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


from keras import layers
from keras import models


``` python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
```
